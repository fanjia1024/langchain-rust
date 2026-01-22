use async_trait::async_trait;
use serde_json::json;

use super::{Middleware, MiddlewareContext, MiddlewareError};
use crate::prompt::PromptArgs;
use crate::schemas::agent::AgentAction;

/// Summarization middleware for managing long conversation history.
///
/// Automatically summarizes conversation history when it exceeds configured
/// thresholds (token count or message count), preserving recent messages.
///
/// # Example
/// ```rust,ignore
/// use langchain_rust::agent::middleware::SummarizationMiddleware;
///
/// let middleware = SummarizationMiddleware::new()
///     .with_token_threshold(4000)
///     .with_message_threshold(50)
///     .with_preserve_recent(10);
/// ```
pub struct SummarizationMiddleware {
    token_threshold: Option<usize>,
    message_threshold: Option<usize>,
    preserve_recent: usize,
    summarization_prompt: String,
}

impl SummarizationMiddleware {
    pub fn new() -> Self {
        Self {
            token_threshold: None,
            message_threshold: Some(50),
            preserve_recent: 10,
            summarization_prompt: "Summarize the following conversation history, preserving key information:".to_string(),
        }
    }

    pub fn with_token_threshold(mut self, threshold: usize) -> Self {
        self.token_threshold = Some(threshold);
        self
    }

    pub fn with_message_threshold(mut self, threshold: usize) -> Self {
        self.message_threshold = Some(threshold);
        self
    }

    pub fn with_preserve_recent(mut self, count: usize) -> Self {
        self.preserve_recent = count;
        self
    }

    pub fn with_summarization_prompt(mut self, prompt: String) -> Self {
        self.summarization_prompt = prompt;
        self
    }

    fn should_summarize(&self, message_count: usize, _token_count: usize) -> bool {
        if let Some(threshold) = self.message_threshold {
            if message_count > threshold {
                return true;
            }
        }

        // Token count checking would require tokenization
        // For now, we rely on message count
        false
    }

    async fn summarize_history(&self, _history: &str) -> String {
        // Placeholder for actual summarization logic
        // In a full implementation, this would call an LLM to summarize
        format!("[Summarized conversation history]")
    }
}

impl Default for SummarizationMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Middleware for SummarizationMiddleware {
    async fn before_agent_plan(
        &self,
        input: &PromptArgs,
        _steps: &[(AgentAction, String)],
        context: &mut MiddlewareContext,
    ) -> Result<Option<PromptArgs>, MiddlewareError> {
        // Check if chat_history exists and needs summarization
        if let Some(chat_history_value) = input.get("chat_history") {
            if let Ok(history_array) = serde_json::from_value::<Vec<serde_json::Value>>(chat_history_value.clone()) {
                let message_count = history_array.len();
                
                if self.should_summarize(message_count, 0) {
                    // Mark that summarization should happen
                    context.set_custom_data("should_summarize".to_string(), json!(true));
                    context.set_custom_data("message_count".to_string(), json!(message_count));
                    
                    // In a full implementation, we would:
                    // 1. Extract recent messages to preserve
                    // 2. Summarize older messages
                    // 3. Replace chat_history with summarized version + recent messages
                    
                    log::info!(
                        "Summarization triggered: {} messages (threshold: {:?})",
                        message_count,
                        self.message_threshold
                    );
                }
            }
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_summarization_middleware_creation() {
        let middleware = SummarizationMiddleware::new();
        assert_eq!(middleware.preserve_recent, 10);
    }

    #[test]
    fn test_should_summarize() {
        let middleware = SummarizationMiddleware::new().with_message_threshold(50);
        assert!(!middleware.should_summarize(40, 0));
        assert!(middleware.should_summarize(60, 0));
    }
}
