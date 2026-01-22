# Agent Guidelines for langchain-rust

## Build, Lint & Test Commands

### Core Commands
```bash
# Build with all features (release mode)
cargo build --verbose --all --release --all-features

# Run all tests
cargo test --release --all-features

# Run a single test by name
cargo test --release --all-features test_function_name

# Run tests in a specific module
cargo test --release --all-features -- module_name

# Format check (CI requirement)
cargo fmt --all -- --check

# Run clippy lints
cargo clippy --all-features -- -D warnings
```

### Features
The project uses Cargo features for optional dependencies:
- `postgres`, `qdrant`, `surrealdb`, `sqlite-vss`, `sqlite-vec` - vector stores
- `ollama`, `mistralai` - LLM providers
- `pdf-extract`, `lopdf`, `html-to-markdown` - document loaders
- `tree-sitter` - code parsing

## Code Style Guidelines

### Error Handling
- Use `thiserror` for all error types
- Pattern: `#[derive(Error, Debug)]` with `#[error("...")]` attributes
- Use `#[from]` for automatic error conversion
- Example:
```rust
#[derive(Error, Debug)]
pub enum ChainError {
    #[error("LLM error: {0}")]
    LLMError(#[from] LLMError),
    #[error("Missing input variable: {0}")]
    MissingInputVariable(String),
}
```

### Builder Pattern
- Use builder pattern for configuration: `with_*` methods returning `self`
- Always implement `Default` when sensible
- Example:
```rust
impl OpenAI<OpenAIConfig> {
    pub fn new(config: C) -> Self { ... }
    pub fn with_model<S: Into<String>>(mut self, model: S) -> Self { ... }
    pub fn with_config(mut self, config: C) -> Self { ... }
}
```

### Naming Conventions
- **Snake case** for functions and variables: `fn process_document()`
- **CamelCase** for types and traits: `struct OpenAI`, `trait LLM`
- **SCREAMING_SNAKE_CASE** for constants: `const MAX_RETRIES: u32 = 3`
- Prefix boolean getters with `is_` or `has_`: `is_valid()`, `has_content()`

### Imports
- Use `crate::` for internal imports
- Group external imports together, internal imports separately
- Use `pub use module::*` for re-exports in `mod.rs`
- Avoid absolute paths like `langchain_rust::chain`

### Async/Await Patterns
- Use `#[async_trait]` for trait methods
- Mark async recursive functions with `#[async_recursion]`
- Use `tokio::test` for async tests
- Example:
```rust
#[async_trait]
impl LLM for OpenAI<C> {
    async fn generate(&self, prompt: &[Message]) -> Result<GenerateResult, LLMError> { ... }
}
```

### Shared State
- Use `Arc<Mutex<T>>` for shared mutable state in async contexts
- Use `Arc<RwLock<T>>` for read-heavy shared state
- Prefer ownership transfer over excessive cloning

### Testing
- Unit tests in same file using `#[cfg(test)]` module
- Integration tests in `tests/` directory
- Mark integration tests with `#[ignore]` if they require external services
- Use `tokio_test` for async test assertions
- Example:
```rust
#[tokio::test]
async fn test_llm_invoke() {
    let llm = OpenAI::new(OpenAIConfig::default());
    let result = llm.invoke("hello").await;
    assert!(result.is_ok());
}
```

### Documentation
- Document public APIs with `///` comments
- Include examples in doc comments where helpful
- Use `//!` for module-level documentation

### Type Conversions
- Implement `Into<T>` for ergonomic conversions
- Implement `ToString` explicitly when custom formatting needed
- Use `From::from` for error conversions (not `Into`)