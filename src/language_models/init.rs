use crate::{
    language_models::{
        llm::LLM,
        model_parser::parse_model_string,
        options::CallOptions,
        LLMError,
    },
    llm::{
        claude::Claude,
        deepseek::Deepseek,
        openai::{AzureConfig, OpenAI, OpenAIConfig},
        qwen::Qwen,
    },
};

#[cfg(feature = "ollama")]
use crate::llm::ollama::Ollama;

/// Initialize a chat model from a model string with optional parameters.
///
/// This is the unified interface for creating LLM instances, similar to
/// LangChain Python's `init_chat_model()` function.
///
/// # Supported Formats
///
/// - Simple format: "gpt-4o-mini" (auto-detects provider)
/// - Provider:model format: "openai:gpt-4o-mini", "azure_openai:gpt-4.1"
///
/// # Supported Models
///
/// - OpenAI: "gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4-turbo"
/// - Claude: "claude-3-opus", "claude-3-sonnet", "claude-3-haiku", "claude-3-5-sonnet", "claude-sonnet-4-5-20250929"
/// - Qwen: "qwen-plus", "qwen-max", "qwen-turbo", "qwen"
/// - Deepseek: "deepseek-chat", "deepseek-reasoner", "deepseek"
/// - Ollama: "llama3", "mistral", "codellama", etc. (requires "ollama" feature)
///
/// # Parameters
///
/// - `model`: Model identifier string
/// - `temperature`: Optional temperature (0.0-2.0 for most models)
/// - `max_tokens`: Optional maximum tokens in response
/// - `timeout`: Optional request timeout in seconds
/// - `max_retries`: Optional maximum retry attempts
/// - `api_key`: Optional API key override
/// - `base_url`: Optional base URL override (for proxies)
/// - `azure_deployment`: Optional Azure deployment ID (for Azure OpenAI)
///
/// # Example
/// ```rust,ignore
/// use langchain_rust::language_models::init_chat_model;
///
/// let model = init_chat_model(
///     "gpt-4o-mini",
///     Some(0.7),
///     Some(1000),
///     None,
///     None,
///     None,
///     None,
///     None,
/// )?;
///
/// let response = model.invoke("Hello!").await?;
/// ```
pub fn init_chat_model(
    model: &str,
    temperature: Option<f32>,
    max_tokens: Option<u32>,
    timeout: Option<u64>,
    max_retries: Option<u32>,
    api_key: Option<String>,
    base_url: Option<String>,
    azure_deployment: Option<String>,
) -> Result<Box<dyn LLM>, LLMError> {
    let parsed = parse_model_string(model)?;
    let provider = parsed.provider.as_deref();

    // Build CallOptions
    let mut options = CallOptions::new();
    if let Some(temp) = temperature {
        options = options.with_temperature(temp);
    }
    if let Some(max) = max_tokens {
        options = options.with_max_tokens(max);
    }
    if let Some(to) = timeout {
        options = options.with_timeout(to);
    }
    if let Some(retries) = max_retries {
        options = options.with_max_retries(retries);
    }
    if let Some(key) = api_key {
        options = options.with_api_key(key);
    }
    if let Some(url) = base_url {
        options = options.with_base_url(url);
    }

    // Determine provider and create model
    let model_name = parsed.model.as_str();
    let model_lower = model_name.to_lowercase();

    // Handle Azure OpenAI
    if provider == Some("azure_openai") || (provider.is_none() && model_lower.starts_with("gpt-") && azure_deployment.is_some()) {
        let mut config = AzureConfig::default();
        if let Some(key) = &options.api_key {
            config = config.with_api_key(key.clone());
        }
        if let Some(url) = &options.base_url {
            config = config.with_api_base(url.clone());
        }
        if let Some(deployment) = azure_deployment {
            config = config.with_deployment_id(deployment);
        }
        let llm: OpenAI<AzureConfig> = OpenAI::new(config)
            .with_model(model_name)
            .with_options(options);
        return Ok(Box::new(llm));
    }

    // OpenAI models
    if provider == Some("openai") || (provider.is_none() && model_lower.starts_with("gpt-")) {
        let mut config = OpenAIConfig::default();
        if let Some(key) = &options.api_key {
            config = config.with_api_key(key.clone());
        }
        // Note: OpenAIConfig from async_openai doesn't support base_url directly
        // Base URL can be set via environment variable OPENAI_API_BASE
        let llm: OpenAI<OpenAIConfig> = OpenAI::new(config)
            .with_model(model_name)
            .with_options(options);
        return Ok(Box::new(llm));
    }

    // Claude models
    if provider == Some("anthropic") || provider == Some("claude")
        || (provider.is_none() && (model_lower.starts_with("claude-") || model_lower.starts_with("claude")))
    {
        let mut llm = Claude::default().with_model(model_name);
        if let Some(key) = &options.api_key {
            llm = llm.with_api_key(key.clone());
        }
        // Note: Claude doesn't support base_url override in current implementation
        llm = llm.with_options(options);
        return Ok(Box::new(llm));
    }

    // Qwen models
    if provider == Some("qwen")
        || (provider.is_none() && (model_lower.starts_with("qwen-") || model_lower == "qwen"))
    {
        let mut llm = Qwen::default().with_model(model_name);
        if let Some(key) = &options.api_key {
            llm = llm.with_api_key(key.clone());
        }
        if let Some(url) = &options.base_url {
            llm = llm.with_base_url(url.clone());
        }
        llm = llm.with_options(options);
        return Ok(Box::new(llm));
    }

    // Deepseek models
    if provider == Some("deepseek")
        || (provider.is_none() && (model_lower.starts_with("deepseek-") || model_lower == "deepseek"))
    {
        let mut llm = Deepseek::default().with_model(model_name);
        if let Some(key) = &options.api_key {
            llm = llm.with_api_key(key.clone());
        }
        if let Some(url) = &options.base_url {
            llm = llm.with_base_url(url.clone());
        }
        llm = llm.with_options(options);
        return Ok(Box::new(llm));
    }

    // Ollama models (default for unrecognized patterns)
    #[cfg(feature = "ollama")]
    {
        let mut llm = Ollama::default().with_model(model_name);
        if let Some(url) = &options.base_url {
            llm = llm.with_base_url(url.clone());
        }
        llm = llm.with_options(options);
        return Ok(Box::new(llm));
    }

    #[cfg(not(feature = "ollama"))]
    {
        Err(LLMError::OtherError(format!(
            "Unsupported model: {}. Supported providers: openai, azure_openai, anthropic, qwen, deepseek. Enable 'ollama' feature for Ollama models.",
            model
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_chat_model_openai() {
        let result = init_chat_model("gpt-4o-mini", None, None, None, None, None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_init_chat_model_with_provider() {
        let result = init_chat_model("openai:gpt-4o-mini", None, None, None, None, None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_init_chat_model_with_params() {
        let result = init_chat_model(
            "gpt-4o-mini",
            Some(0.7),
            Some(1000),
            Some(30),
            Some(3),
            None,
            None,
            None,
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_init_chat_model_claude() {
        let result = init_chat_model("claude-3-5-sonnet-20240620", None, None, None, None, None, None, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_init_chat_model_invalid() {
        let result = init_chat_model("invalid-model-xyz", None, None, None, None, None, None, None);
        assert!(result.is_err());
    }
}
