# Agent Guidelines for langchain-ai-rs

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
- Avoid absolute paths like `langchain_ai_rs::chain`

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

### Human-in-the-Loop (HILP)
- Deep agent HILP: use `interrupt_on` (per-tool `InterruptConfig` with `allowed_decisions`: approve, edit, reject), **checkpointer required**, same `thread_id` when resuming.
- On interrupt, result is `AgentInvokeResult::Interrupt { interrupt_value }` (action_requests, review_configs); resume with `AgentInput::Resume(serde_json::json!({ "decisions": [...] }))` and same config.
- Decision order must match `action_requests`. Subagents can override `interrupt_on` via `with_subagent_and_interrupt_on`. See [Human-in-the-loop](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop) and example `deep_agent_human_in_the_loop`.

### Skills (progressive disclosure)
- Directory-based skills: each skill is a directory with `SKILL.md` and YAML frontmatter (`name`, `description`). See [Skills](https://docs.langchain.com/oss/python/deepagents/skills).
- Use `with_skill_dir(path)` / `with_skill_dirs(dirs)` on [DeepAgentConfig]. At startup only frontmatter is read; when the user message matches a skill (by description/name), its full content is loaded and injected as a "## Skills" system message for that turn.
- Types and helpers in `langchain_ai_rs::agent::deep_agent::skills`: `SkillMeta`, `load_skill_index`, `match_skills`, `load_skill_full_content`. Middleware: `SkillsMiddleware`, `build_skills_middleware`. Example: `deep_agent_skills`.
- Legacy: `skill_paths` and `skill_contents` still load at build time and append to the system prompt (no matching).

### Deep Agent middleware alignment
- Aligns with [Deep Agents Middleware](https://docs.langchain.com/oss/python/deepagents/middleware). Use `with_planning_system_prompt(Some(...))` for extra instructions when planning is enabled (TodoListMiddleware-style). Use `with_filesystem_system_prompt(Some(...))` when filesystem is enabled (FilesystemMiddleware-style). Use `with_custom_tool_description("ls", "...")` or `with_custom_tool_descriptions(map)` to override built-in tool descriptions (e.g. "ls", "read_file", "write_todos"). Subagent config (`enable_task_tool`, `default_subagent_model`, `subagents`, general-purpose) corresponds to SubAgentMiddleware.

### Long-term memory
- Use `with_long_term_memory("/memories/")` on [DeepAgentConfig] so that paths under that prefix are stored in the [ToolStore] and persist across threads; other paths use the default backend (workspace or custom file_backend). See [Long-term memory](https://docs.langchain.com/oss/python/deepagents/long-term-memory). Example: `deep_agent_long_term_memory`.

### Type Conversions
- Implement `Into<T>` for ergonomic conversions
- Implement `ToString` explicitly when custom formatting needed
- Use `From::from` for error conversions (not `Into`)