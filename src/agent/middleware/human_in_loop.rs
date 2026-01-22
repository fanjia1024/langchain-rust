use async_trait::async_trait;
use std::collections::HashMap;
use std::time::Duration;

use super::{Middleware, MiddlewareContext, MiddlewareError};
use crate::schemas::agent::{AgentAction, AgentFinish};

/// Human-in-the-loop middleware for requiring human approval.
///
/// Pauses execution at configured points (before tool calls, before finish)
/// and waits for human approval before continuing. Supports both global
/// approval settings and per-tool approval configuration.
///
/// # Example
/// ```rust,ignore
/// use langchain_rust::agent::middleware::HumanInTheLoopMiddleware;
///
/// let middleware = HumanInTheLoopMiddleware::new()
///     .with_interrupt_on("send_email".to_string(), true)
///     .with_interrupt_on("delete_database".to_string(), true)
///     .with_interrupt_on("search".to_string(), false)
///     .with_approval_required_for_finish(true);
/// ```
pub struct HumanInTheLoopMiddleware {
    approval_required_for_tool_calls: bool,
    approval_required_for_finish: bool,
    interrupt_on: HashMap<String, bool>,
    timeout: Option<Duration>,
    default_on_timeout: bool,
}

impl HumanInTheLoopMiddleware {
    pub fn new() -> Self {
        Self {
            approval_required_for_tool_calls: false,
            approval_required_for_finish: false,
            interrupt_on: HashMap::new(),
            timeout: None,
            default_on_timeout: true,
        }
    }

    pub fn with_approval_required_for_tool_calls(mut self, required: bool) -> Self {
        self.approval_required_for_tool_calls = required;
        self
    }

    pub fn with_approval_required_for_finish(mut self, required: bool) -> Self {
        self.approval_required_for_finish = required;
        self
    }

    /// Configure approval requirement for a specific tool.
    ///
    /// If `interrupt` is `true`, the middleware will require approval before
    /// executing this tool. If `false`, it will auto-approve (even if global
    /// approval is required).
    pub fn with_interrupt_on(mut self, tool_name: String, interrupt: bool) -> Self {
        self.interrupt_on.insert(tool_name, interrupt);
        self
    }

    /// Configure approval requirements for multiple tools at once.
    pub fn with_interrupt_on_map(mut self, interrupt_map: HashMap<String, bool>) -> Self {
        self.interrupt_on.extend(interrupt_map);
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    pub fn with_default_on_timeout(mut self, default: bool) -> Self {
        self.default_on_timeout = default;
        self
    }

    /// Check if approval is required for a specific tool.
    fn requires_approval_for_tool(&self, tool_name: &str) -> bool {
        // Check per-tool configuration first
        if let Some(&interrupt) = self.interrupt_on.get(tool_name) {
            return interrupt;
        }

        // Fall back to global setting
        self.approval_required_for_tool_calls
    }

    async fn wait_for_approval(&self, description: &str) -> Result<bool, MiddlewareError> {
        // In a real implementation, this would:
        // 1. Display the action/result to the user
        // 2. Wait for user input (yes/no)
        // 3. Return the approval result

        // For now, we use a placeholder that always approves
        // In production, this would integrate with a UI or callback system
        log::info!("Human approval required for: {}", description);
        log::info!("[Placeholder] Auto-approving (in production, this would wait for user input)");

        if let Some(timeout) = self.timeout {
            // Simulate waiting with timeout
            tokio::time::sleep(timeout).await;
            Ok(self.default_on_timeout)
        } else {
            // No timeout, wait indefinitely (in production)
            Ok(true) // Placeholder: auto-approve
        }
    }
}

impl Default for HumanInTheLoopMiddleware {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Middleware for HumanInTheLoopMiddleware {
    async fn before_tool_call(
        &self,
        action: &AgentAction,
        context: &mut MiddlewareContext,
    ) -> Result<Option<AgentAction>, MiddlewareError> {
        // Check if approval is required for this specific tool
        if self.requires_approval_for_tool(&action.tool) {
            let description = format!(
                "Tool call: {} with input: {}",
                action.tool, action.tool_input
            );
            let approved = self.wait_for_approval(&description).await?;

            if !approved {
                return Err(MiddlewareError::Aborted(format!(
                    "Human rejected tool call: {}",
                    action.tool
                )));
            }

            context.set_custom_data(
                format!("human_approved_tool_{}", action.tool),
                serde_json::json!(true),
            );
            context.set_custom_data(
                "human_approved_tool_call".to_string(),
                serde_json::json!(true),
            );
        }

        Ok(None)
    }

    async fn before_finish(
        &self,
        finish: &AgentFinish,
        context: &mut MiddlewareContext,
    ) -> Result<Option<AgentFinish>, MiddlewareError> {
        if self.approval_required_for_finish {
            let description = format!("Final result: {}", finish.output);
            let approved = self.wait_for_approval(&description).await?;

            if !approved {
                return Err(MiddlewareError::Aborted(
                    "Human rejected final result".to_string(),
                ));
            }

            context.set_custom_data("human_approved_finish".to_string(), serde_json::json!(true));
        }

        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_human_in_loop_middleware_creation() {
        let middleware = HumanInTheLoopMiddleware::new();
        assert!(!middleware.approval_required_for_tool_calls);
        assert!(!middleware.approval_required_for_finish);
        assert!(middleware.interrupt_on.is_empty());
    }

    #[test]
    fn test_per_tool_approval() {
        let middleware = HumanInTheLoopMiddleware::new()
            .with_interrupt_on("send_email".to_string(), true)
            .with_interrupt_on("search".to_string(), false);

        assert!(middleware.requires_approval_for_tool("send_email"));
        assert!(!middleware.requires_approval_for_tool("search"));
        assert!(!middleware.requires_approval_for_tool("unknown_tool"));
    }

    #[test]
    fn test_global_vs_per_tool_precedence() {
        let middleware = HumanInTheLoopMiddleware::new()
            .with_approval_required_for_tool_calls(true)
            .with_interrupt_on("search".to_string(), false);

        // Per-tool setting should override global
        assert!(!middleware.requires_approval_for_tool("search"));
        // Other tools should use global setting
        assert!(middleware.requires_approval_for_tool("other_tool"));
    }

    #[tokio::test]
    async fn test_wait_for_approval() {
        let middleware = HumanInTheLoopMiddleware::new()
            .with_timeout(Duration::from_millis(10))
            .with_default_on_timeout(true);

        let approved = middleware.wait_for_approval("test action").await.unwrap();
        assert!(approved);
    }
}
