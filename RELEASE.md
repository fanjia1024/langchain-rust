# langchain-ai-rust v5.0.0

Build LLM applications in Rust with type safety: chains, agents, RAG, LangGraph, embeddings, vector stores, and 20+ document loaders. This release includes compilation fixes and significant architecture improvements.

---

## Compilation Fixes (2025-01-27)

### LangGraph
- Added `S: State + 'static` / `SubState: State + 'static` bounds to `StateGraph`, `CompiledGraph`, `SuperStepExecutor`, `SubgraphNode`, `SubgraphNodeWithTransform` for correct lifetimes and `Send`
- Fixed E0505 in `compile_with_persistence`: acquire `self.nodes` before calling `propagate_persistence_to_subgraphs` to avoid move while borrowed
- Added correct lifetime `'a` to `stream`, `stream_with_options`, `stream_internal`, `stream_with_mode`, `stream_with_modes`, `astream_with_config_and_mode` so streams legally capture `&self`
- In `stream_internal`, replaced `?` with explicit `match` where the expression yields a `Result`, so the stream item type is `StreamEvent<S>`
- Added `#[serde(bound = "S: Serialize + serde::de::DeserializeOwned")]` to `StateSnapshot<S>` (fixes E0283)
- Added `S: State + 'static` to `save_checkpoint`
- SQLite persistence: `SqliteSaver` uses `PhantomData<S>`; map serde errors to `rusqlite::Error` inside `query_row`/`query_map`; implemented `From<rusqlite::Error>` for `PersistenceError`

### VectorStore
- Added `Send + Sync` to `type Options` in the trait so default batch impl futures are `Send`
- Implemented `From<Box<dyn Error + Send + Sync>>` for `VectorStoreError`
- All vector store implementations (faiss, weaviate, chroma, pinecone, mongodb, surrealdb, sqlite_vss, sqlite_vec, opensearch, pgvector) now return `VectorStoreError` with internal `.map_err(...)` or `?` conversion

### Retrievers
- `get_relevant_documents` now returns `Result<Vec<Document>, RetrieverError>` for: FlashRankReranker, ContextualAIReranker, CohereReranker, TFIDFRetriever, BM25Retriever, SVMRetriever

### Document loaders
- **toml_loader / xml_loader**: Switched to `async_stream::stream!` instead of `futures::stream`
- **excel_loader**: Use temp file for `open_workbook_auto`; match cells via `calamine::Data`, removed non-existent `Duration` branch and duplicate `Cursor` import
- **github_loader**: Use octocrab's `Content` fields (`type`, `content`, etc.) to distinguish file vs directory instead of non-existent `File`/`Dir` enums
- **aws_s3_loader**: Replaced `response.is_truncated()` with `.unwrap_or(false)` for correct `bool` type

---

## Architecture Optimization (2025-01-24)

### Error handling
- **Unified error system** (`src/error/utils.rs`)
  - `ErrorCode` enum for shared error codes (1000–9999)
  - `ErrorContext` for contextual error data
  - Helpers: `error_info()`, `error_context()`
  - `LangChainError` extended to support `AgentError`, `RAGError`, `MultiAgentError` conversion

### Utility modules (`src/utils/`)
- **similarity.rs**: `cosine_similarity_f64/f32`, `batch_cosine_similarity_*`, `text_similarity` (Jaccard)
- **vectors.rs**: `mean_embedding_*`, `sum_vectors_*`
- **builder.rs**: `Builder`, `ValidatedBuilder`, `simple_builder!` macro
- **async_utils.rs**: `join_all`, `try_join_all`, `batch_process` / `batch_process_result`, `spawn_all`

### VectorStore base
- **`src/vectorstore/base.rs`**: `VectorStoreBaseConfig`, `VectorStoreHelpers` (extract_texts, validate_documents_vectors, get_embedder, apply_score_threshold, sort_by_score), `VectorStoreInitializable`, `VectorStoreBatch`

### LLM interface
- **`src/language_models/common_config.rs`**: `LLMConfig`, `LLMBuilder`, `LLMHelpers` (validate_model_name, get_api_key_from_env, merge_options), `LLMInitConfig`, `StreamingLLM`

### Middleware chain
- **`src/agent/middleware/chain.rs`**: `MiddlewareChainExecutor` with hooks for before/after agent_plan, model_call, tool_call, finish; `MiddlewareResult<T>`, `MiddlewareChainConfig`

### Type aliases (in `src/lib.rs`)
- `Tool`, `Tools`, `ToolContext`, `ToolStore`, `AgentState`, `Memory`, `MiddlewareList`, `Messages`, `Embedding`, `EmbeddingF32`, `Documents`

### Documentation
- `docs/ARC_MUTEX_OPTIMIZATION.md` – Arc/Mutex usage
- `docs/ASYNC_OPTIMIZATION.md` – async patterns
- `docs/ARCHITECTURE_OPTIMIZATION_SUMMARY.md` – architecture overview

### Testing
- **`tests/architecture.rs`**: Error unification, utility functions, type alias tests

### Other improvements
- Unified `mod` + `pub use` in `src/agent/mod.rs` and `src/chain/mod.rs`
- Shared similarity helpers used in semantic_router, long_term_memory, guardrail_utils
- Improved docs for `create_agent` and public API

**Stats:** ~1,661 new lines, 13 new files, 7 modified files. No breaking changes; fully backward compatible.

---

## Installation

```bash
cargo add langchain-ai-rust
# With vector store (e.g. postgres)
cargo add langchain-ai-rust --features postgres
```

**Full documentation:** [docs.rs/langchain-ai-rust](https://docs.rs/langchain-ai-rust)
