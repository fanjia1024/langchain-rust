mod agent;
pub use agent::*;

mod executor;
pub use executor::*;

mod chat;
pub use chat::*;

mod open_ai_tools;
pub use open_ai_tools::*;

mod error;
pub use error::*;

mod model_detector;
pub use model_detector::*;

mod unified_agent;
pub use unified_agent::*;

mod state;
pub use state::*;

mod structured_output;
pub use structured_output::*;

mod middleware;
pub use middleware::*;

mod multi_agent;
pub use multi_agent::*;

mod runtime;
pub use runtime::*;

pub mod context_engineering;
pub use context_engineering::*;

use std::sync::Arc;

use crate::{
    agent::chat::ConversationalAgentBuilder,
    agent::middleware::Middleware,
    language_models::llm::LLM,
    schemas::StructuredOutputStrategy,
    tools::{Tool, ToolContext, ToolStore},
};

/// Create an agent with a simplified API, similar to LangChain Python's `create_agent`.
///
/// This function provides a simple way to create an agent with:
/// - Model specified as a string (e.g., "gpt-4o-mini", "claude-3-sonnet")
/// - Tools as a slice
/// - Optional system prompt
/// - Optional middleware
///
/// # Example
/// ```rust,ignore
/// use langchain_rust::agent::create_agent;
/// use langchain_rust::agent::middleware::LoggingMiddleware;
///
/// let middleware = vec![Arc::new(LoggingMiddleware::new())];
/// let agent = create_agent(
///     "gpt-4o-mini",
///     &[],
///     Some("You are a helpful assistant"),
///     Some(middleware),
/// )?;
/// ```
pub fn create_agent(
    model: &str,
    tools: &[Arc<dyn Tool>],
    system_prompt: Option<&str>,
    middleware: Option<Vec<Arc<dyn Middleware>>>,
) -> Result<UnifiedAgent, AgentError> {
    create_agent_with_runtime(model, tools, system_prompt, None, None, None, middleware)
}

/// Create an agent with structured output support.
pub fn create_agent_with_structured_output(
    model: &str,
    tools: &[Arc<dyn Tool>],
    system_prompt: Option<&str>,
    response_format: Option<Box<dyn StructuredOutputStrategy>>,
    middleware: Option<Vec<Arc<dyn Middleware>>>,
) -> Result<UnifiedAgent, AgentError> {
    create_agent_with_runtime(
        model,
        tools,
        system_prompt,
        None,
        None,
        response_format,
        middleware,
    )
}

/// Create an agent with runtime support (context and store).
pub fn create_agent_with_runtime(
    model: &str,
    tools: &[Arc<dyn Tool>],
    system_prompt: Option<&str>,
    context: Option<Arc<dyn ToolContext>>,
    store: Option<Arc<dyn ToolStore>>,
    response_format: Option<Box<dyn StructuredOutputStrategy>>,
    middleware: Option<Vec<Arc<dyn Middleware>>>,
) -> Result<UnifiedAgent, AgentError> {
    // Detect and create LLM from model string
    let llm = detect_and_create_llm(model)?;

    // Create agent with system prompt if provided
    let prefix = system_prompt
        .map(|s| s.to_string())
        .unwrap_or_else(|| "You are a helpful assistant.".to_string());

    let agent: Box<dyn Agent> = Box::new(
        ConversationalAgentBuilder::new()
            .tools(tools)
            .prefix(prefix)
            .build(llm)?,
    );

    let mut unified_agent = UnifiedAgent::new(agent);

    if let Some(ctx) = context {
        unified_agent = unified_agent.with_context(ctx);
    }

    if let Some(s) = store {
        unified_agent = unified_agent.with_store(s);
    }

    if let Some(rf) = response_format {
        unified_agent = unified_agent.with_response_format(rf);
    }

    if let Some(mw) = middleware {
        unified_agent = unified_agent.with_middleware(mw);
    }

    Ok(unified_agent)
}

/// Create an agent from an existing LLM instance.
///
/// This function is useful when you want to use a custom LLM configuration
/// instead of the model string detection.
pub fn create_agent_from_llm<L: Into<Box<dyn LLM>>>(
    llm: L,
    tools: &[Arc<dyn Tool>],
    system_prompt: Option<&str>,
) -> Result<UnifiedAgent, AgentError> {
    let prefix = system_prompt
        .map(|s| s.to_string())
        .unwrap_or_else(|| "You are a helpful assistant.".to_string());

    let agent: Box<dyn Agent> = Box::new(
        ConversationalAgentBuilder::new()
            .tools(tools)
            .prefix(prefix)
            .build(llm)?,
    );

    Ok(UnifiedAgent::new(agent))
}
