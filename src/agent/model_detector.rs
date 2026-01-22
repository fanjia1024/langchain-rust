use crate::{
    language_models::{
        llm::LLM,
        model_parser::parse_model_string,
    },
    llm::{
        claude::Claude,
        deepseek::Deepseek,
        openai::{OpenAI, OpenAIConfig},
        qwen::Qwen,
    },
};

#[cfg(feature = "ollama")]
use crate::llm::ollama::Ollama;

use super::AgentError;

/// Detects the LLM provider from a model string and returns an appropriate LLM instance.
///
/// Supported patterns:
/// - Simple format: "gpt-4o-mini" (auto-detects provider)
/// - Provider:model format: "openai:gpt-4o-mini", "azure_openai:gpt-4.1"
/// - OpenAI: "gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"
/// - Claude: "claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-3-5-sonnet", "claude-sonnet-4-5-20250929"
/// - Ollama: "llama3", "mistral", "qwen", etc. (any non-prefixed model name)
/// - Qwen: "qwen-*", "qwen-plus", "qwen-max", etc.
/// - Deepseek: "deepseek-*", "deepseek-chat", "deepseek-reasoner"
///
/// # Example
/// ```rust,ignore
/// let llm = detect_and_create_llm("gpt-4o-mini")?;
/// let llm_with_provider = detect_and_create_llm("openai:gpt-4o-mini")?;
/// ```
pub fn detect_and_create_llm(model: &str) -> Result<Box<dyn LLM>, AgentError> {
    let parsed = parse_model_string(model)
        .map_err(|e| AgentError::OtherError(format!("Failed to parse model string: {}", e)))?;
    
    let provider = parsed.provider.as_deref();
    let model_name = parsed.model.as_str();
    let model_lower = model_name.to_lowercase();

    // OpenAI models
    if provider == Some("openai") || (provider.is_none() && model_lower.starts_with("gpt-")) {
        let llm: OpenAI<OpenAIConfig> = OpenAI::default().with_model(model_name);
        return Ok(Box::new(llm));
    }

    // Claude models
    if provider == Some("anthropic") || provider == Some("claude")
        || (provider.is_none() && (model_lower.starts_with("claude-") || model_lower.starts_with("claude")))
    {
        let llm = Claude::default().with_model(model_name);
        return Ok(Box::new(llm));
    }

    // Qwen models
    if provider == Some("qwen")
        || (provider.is_none() && (model_lower.starts_with("qwen-") || model_lower == "qwen"))
    {
        let llm = Qwen::default().with_model(model_name);
        return Ok(Box::new(llm));
    }

    // Deepseek models
    if provider == Some("deepseek")
        || (provider.is_none() && (model_lower.starts_with("deepseek-") || model_lower == "deepseek"))
    {
        let llm = Deepseek::default().with_model(model_name);
        return Ok(Box::new(llm));
    }

    // Ollama models (default for unrecognized patterns)
    // Ollama models are typically just the model name without prefix
    // Examples: "llama3", "mistral", "codellama", etc.
    #[cfg(feature = "ollama")]
    {
        if provider == Some("ollama") || provider.is_none() {
            let llm = Ollama::default().with_model(model_name);
            return Ok(Box::new(llm));
        }
    }
    
    #[cfg(not(feature = "ollama"))]
    {
        Err(AgentError::OtherError(format!(
            "Unrecognized model: {}. Supported providers: openai, anthropic, qwen, deepseek. Ollama support requires the 'ollama' feature.",
            model
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_openai_models() {
        let models = vec![
            "gpt-3.5-turbo",
            "gpt-4",
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
        ];

        for model in models {
            let result = detect_and_create_llm(model);
            assert!(result.is_ok(), "Failed to detect OpenAI model: {}", model);
        }
    }

    #[test]
    fn test_detect_claude_models() {
        let models = vec![
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
            "claude-3-5-sonnet-20240620",
            "claude-sonnet-4-5-20250929",
        ];

        for model in models {
            let result = detect_and_create_llm(model);
            assert!(result.is_ok(), "Failed to detect Claude model: {}", model);
        }
    }

    #[test]
    fn test_detect_qwen_models() {
        let models = vec!["qwen-plus", "qwen-max", "qwen-turbo", "qwen"];

        for model in models {
            let result = detect_and_create_llm(model);
            assert!(result.is_ok(), "Failed to detect Qwen model: {}", model);
        }
    }

    #[test]
    fn test_detect_deepseek_models() {
        let models = vec!["deepseek-chat", "deepseek-reasoner", "deepseek"];

        for model in models {
            let result = detect_and_create_llm(model);
            assert!(result.is_ok(), "Failed to detect Deepseek model: {}", model);
        }
    }

    #[test]
    fn test_detect_ollama_models() {
        let models = vec!["llama3", "mistral", "codellama", "phi"];

        for model in models {
            let result = detect_and_create_llm(model);
            assert!(result.is_ok(), "Failed to detect Ollama model: {}", model);
        }
    }

    #[test]
    fn test_detect_with_provider() {
        let result = detect_and_create_llm("openai:gpt-4o-mini");
        assert!(result.is_ok(), "Failed to detect model with provider prefix");
    }

    #[test]
    fn test_detect_anthropic_provider() {
        let result = detect_and_create_llm("anthropic:claude-3-5-sonnet-20240620");
        assert!(result.is_ok(), "Failed to detect Claude model with provider prefix");
    }
}
