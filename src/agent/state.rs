use std::collections::HashMap;

use serde_json::Value;

use crate::schemas::messages::Message;

/// Mutable agent state that flows through execution.
///
/// This includes messages, custom fields, counters, and any other
/// state that tools may need to access or modify.
#[derive(Clone, Debug)]
pub struct AgentState {
    /// Conversation messages
    pub messages: Vec<Message>,
    /// Custom fields that can be set and accessed by tools
    pub custom_fields: HashMap<String, Value>,
    /// Structured output response (when using structured output)
    pub structured_response: Option<Value>,
}

impl Default for AgentState {
    fn default() -> Self {
        Self {
            messages: Vec::new(),
            custom_fields: HashMap::new(),
            structured_response: None,
        }
    }
}

impl AgentState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_messages(messages: Vec<Message>) -> Self {
        Self {
            messages,
            custom_fields: HashMap::new(),
            structured_response: None,
        }
    }

    pub fn get_field(&self, key: &str) -> Option<&Value> {
        self.custom_fields.get(key)
    }

    pub fn set_field(&mut self, key: String, value: Value) {
        self.custom_fields.insert(key, value);
    }

    pub fn remove_field(&mut self, key: &str) -> Option<Value> {
        self.custom_fields.remove(key)
    }
}

/// Command pattern for updating agent state or controlling execution flow.
///
/// Tools can return commands to update state, remove messages, or control
/// the agent's execution flow.
#[derive(Debug, Clone)]
pub enum Command {
    /// Update custom state fields
    UpdateState { fields: HashMap<String, Value> },
    /// Remove specific messages by ID
    RemoveMessages { ids: Vec<String> },
    /// Clear all messages
    ClearMessages,
    /// Clear all state (messages and custom fields)
    ClearState,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_agent_state() {
        let mut state = AgentState::new();

        // Test field operations
        state.set_field("test_key".to_string(), serde_json::json!("test_value"));
        assert_eq!(
            state.get_field("test_key"),
            Some(&serde_json::json!("test_value"))
        );

        let removed = state.remove_field("test_key");
        assert_eq!(removed, Some(serde_json::json!("test_value")));
        assert_eq!(state.get_field("test_key"), None);
    }

    #[test]
    fn test_agent_state_with_messages() {
        let messages = vec![
            Message::new_human_message("Hello"),
            Message::new_ai_message("Hi there!"),
        ];

        let state = AgentState::with_messages(messages.clone());
        assert_eq!(state.messages.len(), 2);
        assert_eq!(state.custom_fields.len(), 0);
    }

    #[test]
    fn test_command_creation() {
        let mut fields = HashMap::new();
        fields.insert("key1".to_string(), serde_json::json!("value1"));

        let command = Command::UpdateState { fields };
        match command {
            Command::UpdateState { fields } => {
                assert_eq!(fields.len(), 1);
            }
            _ => panic!("Wrong command type"),
        }
    }
}
