use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use serde_json::json;
use tokio::sync::Mutex;

use super::{
    agent::Agent,
    context_engineering::{ModelRequest, ModelResponse},
    message_repair,
    middleware::Middleware,
    middleware::MiddlewareContext,
    runtime::{Runtime, RuntimeRequest},
    state::AgentState,
    AgentError,
};
use crate::schemas::{LogTools, Message, MessageType};
use crate::{
    chain::{chain_trait::Chain, ChainError},
    language_models::GenerateResult,
    memory::SimpleMemory,
    prompt::PromptArgs,
    schemas::{
        agent::{AgentAction, AgentEvent},
        memory::BaseMemory,
        StructuredOutputStrategy,
    },
    tools::{FileBackend, Tool, ToolContext, ToolRuntime, ToolStore},
};

/// Convert message-based input format to standard prompt args.
fn convert_messages_to_prompt_args(input_variables: PromptArgs) -> Result<PromptArgs, ChainError> {
    let messages_value = input_variables
        .get("messages")
        .ok_or_else(|| ChainError::OtherError("Missing 'messages' key".to_string()))?;

    let messages: Vec<Message> = serde_json::from_value(messages_value.clone())
        .map_err(|e| ChainError::OtherError(format!("Failed to parse messages: {}", e)))?;

    // Extract the last user/human message as input
    let input = messages
        .iter()
        .rev()
        .find(|m| matches!(m.message_type, MessageType::HumanMessage))
        .map(|m| m.content.clone())
        .unwrap_or_else(|| {
            messages
                .last()
                .map(|m| m.content.clone())
                .unwrap_or_default()
        });

    let mut prompt_args = PromptArgs::new();
    prompt_args.insert("input".to_string(), json!(input));

    // Preserve chat history if it exists, otherwise use messages
    if input_variables.contains_key("chat_history") {
        prompt_args.insert(
            "chat_history".to_string(),
            input_variables["chat_history"].clone(),
        );
    } else {
        prompt_args.insert("chat_history".to_string(), json!(messages));
    }

    // Copy any other keys
    for (key, value) in input_variables {
        if key != "messages" && key != "chat_history" {
            prompt_args.insert(key, value);
        }
    }

    Ok(prompt_args)
}

pub struct AgentExecutor<A>
where
    A: Agent,
{
    agent: A,
    max_iterations: Option<i32>,
    break_if_error: bool,
    pub memory: Option<Arc<Mutex<dyn BaseMemory>>>,
    state: Arc<Mutex<AgentState>>,
    context: Arc<dyn ToolContext>,
    pub(crate) store: Arc<dyn ToolStore>,
    response_format: Option<Box<dyn StructuredOutputStrategy>>,
    middleware: Vec<Arc<dyn Middleware>>,
    file_backend: Option<Arc<dyn FileBackend>>,
}

impl<A> AgentExecutor<A>
where
    A: Agent,
{
    pub fn from_agent(agent: A) -> Self {
        Self {
            agent,
            max_iterations: Some(10),
            break_if_error: false,
            memory: None,
            state: Arc::new(Mutex::new(AgentState::new())),
            context: Arc::new(crate::tools::EmptyContext),
            store: Arc::new(crate::tools::InMemoryStore::new()),
            response_format: None,
            middleware: Vec::new(),
            file_backend: None,
        }
    }

    pub fn with_file_backend(mut self, file_backend: Option<Arc<dyn FileBackend>>) -> Self {
        self.file_backend = file_backend;
        self
    }

    pub fn with_context(mut self, context: Arc<dyn ToolContext>) -> Self {
        self.context = context;
        self
    }

    pub fn with_store(mut self, store: Arc<dyn ToolStore>) -> Self {
        self.store = store;
        self
    }

    pub fn with_state(mut self, state: Arc<Mutex<AgentState>>) -> Self {
        self.state = state;
        self
    }

    pub fn with_max_iterations(mut self, max_iterations: i32) -> Self {
        self.max_iterations = Some(max_iterations);
        self
    }

    pub fn with_memory(mut self, memory: Arc<Mutex<dyn BaseMemory>>) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn with_break_if_error(mut self, break_if_error: bool) -> Self {
        self.break_if_error = break_if_error;
        self
    }

    pub fn with_response_format(
        mut self,
        response_format: Box<dyn StructuredOutputStrategy>,
    ) -> Self {
        self.response_format = Some(response_format);
        self
    }

    pub fn with_middleware(mut self, middleware: Vec<Arc<dyn Middleware>>) -> Self {
        self.middleware = middleware;
        self
    }

    fn get_name_to_tools(&self) -> HashMap<String, Arc<dyn Tool>> {
        let mut name_to_tool = HashMap::new();
        for tool in self.agent.get_tools().iter() {
            log::debug!("Loading Tool:{}", tool.name());
            name_to_tool.insert(tool.name().trim().replace(" ", "_"), tool.clone());
        }
        name_to_tool
    }

    async fn handle_command(
        &self,
        command: crate::agent::state::Command,
    ) -> Result<(), ChainError> {
        let mut state = self.state.lock().await;
        match command {
            crate::agent::state::Command::UpdateState { fields } => {
                for (key, value) in fields {
                    state.set_field(key, value);
                }
            }
            crate::agent::state::Command::RemoveMessages { ids } => {
                state.messages.retain(|msg| {
                    !ids.contains(
                        &msg.id
                            .as_ref()
                            .map(|s| s.as_str())
                            .unwrap_or("")
                            .to_string(),
                    )
                });
            }
            crate::agent::state::Command::ClearMessages => {
                state.messages.clear();
            }
            crate::agent::state::Command::ClearState => {
                state.messages.clear();
                state.custom_fields.clear();
            }
        }
        Ok(())
    }
}

#[async_trait]
impl<A> Chain for AgentExecutor<A>
where
    A: Agent + Send + Sync,
{
    async fn call(&self, input_variables: PromptArgs) -> Result<GenerateResult, ChainError> {
        // Check if input is message-based format and convert if needed
        let mut input_variables = if input_variables.contains_key("messages") {
            convert_messages_to_prompt_args(input_variables)?
        } else {
            input_variables.clone()
        };
        let name_to_tools = self.get_name_to_tools();
        let mut steps: Vec<(AgentAction, String)> = Vec::new();
        let mut middleware_context = MiddlewareContext::new();
        log::debug!("steps: {:?}", steps);
        if let Some(memory) = &self.memory {
            let memory = memory.lock().await;
            let mut history = memory.messages();
            history = message_repair::repair_dangling_tool_calls(history);
            input_variables.insert("chat_history".to_string(), json!(history));
        } else {
            let mut history = SimpleMemory::new().messages();
            history = message_repair::repair_dangling_tool_calls(history);
            input_variables.insert("chat_history".to_string(), json!(history));
        }

        // Create runtime for middleware
        let runtime = Arc::new(Runtime::new(
            Arc::clone(&self.context),
            Arc::clone(&self.store),
        ));

        loop {
            middleware_context.increment_iteration();

            // Create runtime request
            let runtime_request =
                RuntimeRequest::new(input_variables.clone(), Arc::clone(&self.state))
                    .with_runtime(Arc::clone(&runtime));

            // Apply before_agent_plan hooks (try runtime-aware version first)
            let mut plan_input = input_variables.clone();
            for mw in &self.middleware {
                // Try runtime-aware hook first
                let modified = mw
                    .before_agent_plan_with_runtime(
                        &runtime_request,
                        &steps,
                        &mut middleware_context,
                    )
                    .await
                    .map_err(|e| ChainError::AgentError(format!("Middleware error: {}", e)))?;

                if let Some(modified_input) = modified {
                    plan_input = modified_input;
                } else {
                    // Fallback to non-runtime hook
                    if let Some(modified_input) = mw
                        .before_agent_plan(&plan_input, &steps, &mut middleware_context)
                        .await
                        .map_err(|e| ChainError::AgentError(format!("Middleware error: {}", e)))?
                    {
                        plan_input = modified_input;
                    }
                }
            }

            // Create ModelRequest for context engineering
            // Extract messages from plan_input (if available)
            let mut messages = Vec::new();
            if let Some(chat_history) = plan_input.get("chat_history") {
                if let Ok(msgs) = serde_json::from_value::<Vec<Message>>(chat_history.clone()) {
                    messages = msgs;
                }
            }

            // Get tools from agent
            let tools = self.agent.get_tools();

            // Create ModelRequest
            let mut model_request = ModelRequest::new(messages, tools, Arc::clone(&self.state))
                .with_runtime(Arc::clone(&runtime));

            // Apply before_model_call hooks
            for mw in &self.middleware {
                if let Some(modified_request) = mw
                    .before_model_call(&model_request, &mut middleware_context)
                    .await
                    .map_err(|e| ChainError::AgentError(format!("Middleware error: {}", e)))?
                {
                    model_request = modified_request;

                    // Update plan_input with modified messages
                    if !model_request.messages.is_empty() {
                        plan_input.insert(
                            "chat_history".to_string(),
                            serde_json::json!(model_request.messages),
                        );
                    }

                    // Note: Tool filtering would need to be handled at agent level
                    // For now, we'll pass the filtered tools through metadata
                    // In practice, you'd need to modify the agent interface or use a wrapper
                }
            }

            let mut agent_event = self
                .agent
                .plan(&steps, plan_input.clone())
                .await
                .map_err(|e| ChainError::AgentError(format!("Error in agent planning: {}", e)))?;

            // Create ModelResponse (simplified - actual response comes from agent)
            let model_response = ModelResponse::new(GenerateResult {
                generation: String::new(),
                ..Default::default()
            });

            // Apply after_model_call hooks
            for mw in &self.middleware {
                if let Some(_modified_response) = mw
                    .after_model_call(&model_request, &model_response, &mut middleware_context)
                    .await
                    .map_err(|e| ChainError::AgentError(format!("Middleware error: {}", e)))?
                {
                    // Response modifications would be applied here
                }
            }

            // Apply after_agent_plan hooks (try runtime-aware version first)
            let mut runtime_request =
                RuntimeRequest::new(plan_input.clone(), Arc::clone(&self.state))
                    .with_runtime(Arc::clone(&runtime));

            for mw in &self.middleware {
                // Try runtime-aware hook first
                let modified = mw
                    .after_agent_plan_with_runtime(
                        &runtime_request,
                        &agent_event,
                        &mut middleware_context,
                    )
                    .await
                    .map_err(|e| ChainError::AgentError(format!("Middleware error: {}", e)))?;

                if let Some(modified_event) = modified {
                    agent_event = modified_event;
                } else {
                    // Fallback to non-runtime hook
                    if let Some(modified_event) = mw
                        .after_agent_plan(&plan_input, &agent_event, &mut middleware_context)
                        .await
                        .map_err(|e| ChainError::AgentError(format!("Middleware error: {}", e)))?
                    {
                        agent_event = modified_event;
                    }
                }
            }
            match agent_event {
                AgentEvent::Action(actions) => {
                    for mut action in actions {
                        // Apply before_tool_call hooks (try runtime-aware version first)
                        for mw in &self.middleware {
                            // Try runtime-aware hook first
                            let modified = mw
                                .before_tool_call_with_runtime(
                                    &action,
                                    Some(&*runtime),
                                    &mut middleware_context,
                                )
                                .await
                                .map_err(|e| {
                                    ChainError::AgentError(format!("Middleware error: {}", e))
                                })?;

                            if let Some(modified_action) = modified {
                                action = modified_action;
                            } else {
                                // Fallback to non-runtime hook
                                if let Some(modified_action) = mw
                                    .before_tool_call(&action, &mut middleware_context)
                                    .await
                                    .map_err(|e| {
                                        ChainError::AgentError(format!("Middleware error: {}", e))
                                    })?
                                {
                                    action = modified_action;
                                }
                            }
                        }

                        log::debug!("Action: {:?}", action.tool_input);
                        middleware_context.increment_tool_call_count();

                        let tool = name_to_tools
                            .get(&action.tool.trim().replace(" ", "_"))
                            .ok_or_else(|| {
                                AgentError::ToolError(format!("Tool {} not found", action.tool))
                            })
                            .map_err(|e| ChainError::AgentError(e.to_string()))?;

                        // Create ToolRuntime for tools that need it
                        let tool_call_id = format!("call_{}", steps.len());
                        let mut tool_runtime = ToolRuntime::new(
                            Arc::clone(&self.state),
                            Arc::clone(&self.context),
                            Arc::clone(&self.store),
                            tool_call_id,
                        );
                        if let Some(ref fb) = self.file_backend {
                            tool_runtime = tool_runtime.with_file_backend(Arc::clone(fb));
                        }

                        // Check if tool requires runtime
                        let observation_result: Result<String, String> = if tool.requires_runtime()
                        {
                            let input = tool.parse_input(&action.tool_input).await;
                            tool.run_with_runtime(input, &tool_runtime)
                                .await
                                .map(|result| result.into_string())
                                .map_err(|e| e.to_string())
                        } else {
                            tool.call(&action.tool_input)
                                .await
                                .map_err(|e| e.to_string())
                        };

                        let mut observation = match observation_result {
                            Ok(result) => result,
                            Err(error_msg) => {
                                log::info!("The tool return the following error: {}", error_msg);
                                if self.break_if_error {
                                    return Err(ChainError::AgentError(
                                        AgentError::ToolError(error_msg.clone()).to_string(),
                                    ));
                                } else {
                                    format!("The tool return the following error: {}", error_msg)
                                }
                            }
                        };

                        // Apply after_tool_call hooks (try runtime-aware version first)
                        for mw in &self.middleware {
                            // Try runtime-aware hook first
                            let modified = mw
                                .after_tool_call_with_runtime(
                                    &action,
                                    &observation,
                                    Some(&*runtime),
                                    &mut middleware_context,
                                )
                                .await
                                .map_err(|e| {
                                    ChainError::AgentError(format!("Middleware error: {}", e))
                                })?;

                            if let Some(modified_observation) = modified {
                                observation = modified_observation;
                            } else {
                                // Fallback to non-runtime hook
                                if let Some(modified_observation) = mw
                                    .after_tool_call(&action, &observation, &mut middleware_context)
                                    .await
                                    .map_err(|e| {
                                        ChainError::AgentError(format!("Middleware error: {}", e))
                                    })?
                                {
                                    observation = modified_observation;
                                }
                            }
                        }

                        steps.push((action, observation));
                    }
                }
                AgentEvent::Finish(mut finish) => {
                    // Apply before_finish hooks (try runtime-aware version first)
                    for mw in &self.middleware {
                        // Try runtime-aware hook first
                        let modified = mw
                            .before_finish_with_runtime(
                                &finish,
                                Some(&*runtime),
                                &mut middleware_context,
                            )
                            .await
                            .map_err(|e| {
                                ChainError::AgentError(format!("Middleware error: {}", e))
                            })?;

                        if let Some(modified_finish) = modified {
                            finish = modified_finish;
                        } else {
                            // Fallback to non-runtime hook
                            if let Some(modified_finish) = mw
                                .before_finish(&finish, &mut middleware_context)
                                .await
                                .map_err(|e| {
                                    ChainError::AgentError(format!("Middleware error: {}", e))
                                })?
                            {
                                finish = modified_finish;
                            }
                        }
                    }

                    if let Some(memory) = &self.memory {
                        let mut memory = memory.lock().await;

                        memory.add_user_message(match &input_variables["input"] {
                            // This avoids adding extra quotes to the user input in the history.
                            serde_json::Value::String(s) => s,
                            x => x, // this the json encoded value.
                        });

                        let mut tools_ai_message_seen: HashMap<String, ()> = HashMap::default();
                        for (action, observation) in steps {
                            let LogTools { tool_id, tools } = serde_json::from_str(&action.log)?;
                            let tools_value: serde_json::Value = serde_json::from_str(&tools)?;
                            if tools_ai_message_seen.insert(tools, ()).is_none() {
                                memory.add_message(
                                    Message::new_ai_message("").with_tool_calls(tools_value),
                                );
                            }
                            memory.add_message(Message::new_tool_message(observation, tool_id));
                        }

                        memory.add_ai_message(&finish.output);
                    }

                    let result = GenerateResult {
                        generation: finish.output.clone(),
                        ..Default::default()
                    };

                    // Apply after_finish hooks (try runtime-aware version first)
                    for mw in &self.middleware {
                        // Try runtime-aware hook first
                        mw.after_finish_with_runtime(
                            &finish,
                            &result,
                            Some(&*runtime),
                            &mut middleware_context,
                        )
                        .await
                        .map_err(|e| ChainError::AgentError(format!("Middleware error: {}", e)))?;
                    }

                    return Ok(result);
                }
            }

            if let Some(max_iterations) = self.max_iterations {
                if steps.len() >= max_iterations as usize {
                    return Ok(GenerateResult {
                        generation: "Max iterations reached".to_string(),
                        ..Default::default()
                    });
                }
            }
        }
    }

    async fn invoke(&self, input_variables: PromptArgs) -> Result<String, ChainError> {
        let result = self.call(input_variables).await?;
        Ok(result.generation)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_messages_to_prompt_args() {
        let mut input_vars = PromptArgs::new();
        input_vars.insert(
            "messages".to_string(),
            json!([
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"}
            ]),
        );

        let result = convert_messages_to_prompt_args(input_vars);
        assert!(result.is_ok());
        let args = result.unwrap();
        assert!(args.contains_key("input"));
        assert!(args.contains_key("chat_history"));
        assert_eq!(args["input"], json!("Hello"));
    }

    #[test]
    fn test_convert_messages_preserves_other_keys() {
        let mut input_vars = PromptArgs::new();
        input_vars.insert(
            "messages".to_string(),
            json!([{"role": "user", "content": "Hello"}]),
        );
        input_vars.insert("custom_key".to_string(), json!("custom_value"));

        let result = convert_messages_to_prompt_args(input_vars);
        assert!(result.is_ok());
        let args = result.unwrap();
        assert!(args.contains_key("custom_key"));
        assert_eq!(args["custom_key"], json!("custom_value"));
    }
}
