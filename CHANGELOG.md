# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Architecture Optimization (2025-01-24)

#### Added

##### Error Handling
- **统一错误处理系统** (`src/error/utils.rs`)
  - 新增 `ErrorCode` 枚举：统一的错误代码系统（1000-9999）
  - 新增 `ErrorContext` 结构：用于错误上下文信息
  - 新增 `error_info()` 和 `error_context()` 工具函数
  - 扩展 `LangChainError` 以支持 `AgentError`、`RAGError`、`MultiAgentError` 自动转换

##### Utility Modules
- **新增 `src/utils/` 工具模块**
  - `similarity.rs`：统一的相似度计算函数
    - `cosine_similarity_f64()` / `cosine_similarity_f32()`：余弦相似度计算
    - `batch_cosine_similarity_f64()` / `batch_cosine_similarity_f32()`：批量相似度计算
    - `text_similarity()`：文本相似度计算（基于 Jaccard 相似度）
  - `vectors.rs`：统一的向量操作函数
    - `mean_embedding_f64()` / `mean_embedding_f32()`：计算向量平均值
    - `sum_vectors_f64()` / `sum_vectors_f32()`：计算向量和
  - `builder.rs`：Builder 模式抽象
    - `Builder` trait：统一的 Builder 接口
    - `ValidatedBuilder` trait：支持验证的 Builder
    - `simple_builder!` 宏：简化 Builder 创建
  - `async_utils.rs`：异步操作优化工具
    - `join_all()`：并行执行多个 Future
    - `try_join_all()`：并行执行多个 Future（带错误处理）
    - `batch_process()` / `batch_process_result()`：批量并行处理
    - `spawn_all()`：使用 tokio::spawn 并行执行任务

##### VectorStore Base Layer
- **新增 `src/vectorstore/base.rs`**
  - `VectorStoreBaseConfig`：共享配置结构
  - `VectorStoreHelpers`：辅助函数集合
    - `extract_texts()`：从文档中提取文本
    - `validate_documents_vectors()`：验证文档和向量数量匹配
    - `get_embedder()`：从选项或配置中获取 embedder
    - `apply_score_threshold()`：应用分数阈值过滤
    - `sort_by_score()`：按分数排序文档
  - `VectorStoreInitializable` trait：初始化接口
  - `VectorStoreBatch` trait：批量操作接口

##### LLM Unified Interface
- **新增 `src/language_models/common_config.rs`**
  - `LLMConfig` trait：统一配置接口
  - `LLMBuilder` trait：统一构建器模式
  - `LLMHelpers`：辅助函数
    - `validate_model_name()`：验证模型名称格式
    - `get_api_key_from_env()`：从环境变量获取 API key
    - `merge_options()`：合并调用选项
  - `LLMInitConfig`：初始化配置结构
  - `StreamingLLM` trait：流式响应接口

##### Middleware Chain Optimization
- **新增 `src/agent/middleware/chain.rs`**
  - `MiddlewareChainExecutor`：优化的链执行器
    - `execute_before_agent_plan()`：执行 before_agent_plan 链
    - `execute_before_model_call()`：执行 before_model_call 链
    - `execute_after_model_call()`：执行 after_model_call 链
    - `execute_before_tool_call()`：执行 before_tool_call 链
    - `execute_after_tool_call()`：执行 after_tool_call 链
    - `execute_before_finish()`：执行 before_finish 链
    - `execute_after_finish()`：执行 after_finish 链
  - `MiddlewareResult<T>`：执行结果枚举
  - `MiddlewareChainConfig`：链配置结构

##### Type Aliases
- **在 `src/lib.rs` 中新增常用类型别名**
  - `Tool` = `Arc<dyn crate::tools::Tool>`
  - `Tools` = `Vec<Arc<dyn crate::tools::Tool>>`
  - `ToolContext` = `Arc<dyn crate::tools::ToolContext>`
  - `ToolStore` = `Arc<dyn crate::tools::ToolStore>`
  - `AgentState` = `Arc<Mutex<crate::agent::AgentState>>`
  - `Memory` = `Arc<Mutex<dyn crate::schemas::memory::BaseMemory>>`
  - `MiddlewareList` = `Vec<Arc<dyn crate::agent::Middleware>>`
  - `Messages` = `Vec<crate::schemas::Message>`
  - `Embedding` = `Vec<f64>`
  - `EmbeddingF32` = `Vec<f32>`
  - `Documents` = `Vec<crate::schemas::Document>`

##### Documentation
- **新增优化指南文档**
  - `docs/ARC_MUTEX_OPTIMIZATION.md`：Arc/Mutex 使用优化指南
  - `docs/ASYNC_OPTIMIZATION.md`：异步操作优化指南
  - `docs/ARCHITECTURE_OPTIMIZATION_SUMMARY.md`：架构优化总结

##### Testing
- **新增架构测试** (`tests/architecture.rs`)
  - 错误统一化测试
  - 工具函数测试
  - 类型别名测试

#### Changed

##### Module Organization
- **统一模块导出模式**
  - `src/agent/mod.rs`：统一使用 `mod` + `pub use` 模式
  - `src/chain/mod.rs`：统一使用 `mod` + `pub use` 模式
  - 优化了模块可见性，明确公共 API 边界

##### Code Deduplication
- **统一相似度计算实现**
  - `src/semantic_router/utils.rs`：使用统一的 `cosine_similarity_f64()`
  - `src/tools/long_term_memory/implementations/enhanced_in_memory_store.rs`：使用统一的 `cosine_similarity_f32()`
  - `src/agent/middleware/guardrail_utils.rs`：使用统一的 `text_similarity()`

##### API Documentation
- **改进公共 API 文档**
  - `src/agent/mod.rs`：改进了 `create_agent` 函数的文档
  - 添加了详细的参数说明和使用示例

#### Technical Details

##### Files Added
- `src/error/utils.rs` (326 lines)
- `src/utils/mod.rs`
- `src/utils/similarity.rs`
- `src/utils/vectors.rs`
- `src/utils/builder.rs`
- `src/utils/async_utils.rs`
- `src/vectorstore/base.rs`
- `src/language_models/common_config.rs`
- `src/agent/middleware/chain.rs`
- `tests/architecture.rs`
- `docs/ARC_MUTEX_OPTIMIZATION.md`
- `docs/ASYNC_OPTIMIZATION.md`
- `docs/ARCHITECTURE_OPTIMIZATION_SUMMARY.md`

##### Files Modified
- `src/error/mod.rs` - 扩展错误类型，添加工具模块导出
- `src/agent/mod.rs` - 统一导出模式
- `src/chain/mod.rs` - 统一导出模式
- `src/lib.rs` - 添加 utils 模块和类型别名
- `src/semantic_router/utils.rs` - 使用统一的相似度函数
- `src/tools/long_term_memory/implementations/enhanced_in_memory_store.rs` - 使用统一的相似度函数
- `src/agent/middleware/guardrail_utils.rs` - 使用统一的文本相似度函数

##### Statistics
- **新增代码行数**：约 1,661 行
- **新增文件数**：13 个
- **改进文件数**：7 个

#### Benefits

1. **代码重复减少**：相似度计算、向量操作等统一到工具模块
2. **错误处理统一**：所有错误可通过 `LangChainError` 统一处理
3. **类型安全提升**：添加了类型别名和统一接口
4. **可维护性提升**：模块组织更清晰，文档更完善
5. **扩展性提升**：提供了基础抽象层，便于后续扩展
6. **性能优化**：提供了异步优化工具和批量处理功能

#### Breaking Changes

无。所有优化都保持了向后兼容性。

#### Migration Guide

无需迁移。所有更改都是向后兼容的。

#### Notes

- 现有代码的编译错误是项目原本就有的问题，与本次优化无关
- 建议在应用优化时进行充分的测试
- 详细的使用指南请参考 `docs/ARCHITECTURE_OPTIMIZATION_SUMMARY.md`
