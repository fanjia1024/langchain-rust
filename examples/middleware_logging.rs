use std::sync::Arc;
use langchain_rust::agent::{create_agent, middleware::LoggingMiddleware};
use langchain_rust::schemas::Message;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Create logging middleware
    let logging_middleware = LoggingMiddleware::new()
        .with_log_level(langchain_rust::agent::middleware::logging::LogLevel::Info)
        .with_structured_logging(false);

    // Create agent with middleware
    let agent = create_agent(
        "gpt-4o-mini",
        &[],
        Some("You are a helpful assistant that provides concise answers."),
        Some(vec![Arc::new(logging_middleware)]),
    )?;

    // Invoke the agent
    let result = agent
        .invoke(json!({
            "messages": [
                Message::new_human_message("What is the capital of France?")
            ]
        }))
        .await?;

    println!("Agent response: {}", result);
    Ok(())
}
