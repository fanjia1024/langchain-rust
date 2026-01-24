# 异步操作优化指南

本文档记录了异步操作的优化建议和最佳实践。

## 已实现的优化工具

在 `src/utils/async_utils.rs` 中提供了以下工具函数：

1. **`join_all`** - 并行执行多个 Future
2. **`try_join_all`** - 并行执行多个 Future（带错误处理）
3. **`batch_process`** - 批量并行处理数据
4. **`batch_process_result`** - 批量并行处理数据（带错误处理）
5. **`spawn_all`** - 使用 tokio::spawn 并行执行任务

## 优化建议

### 1. 批量 Embedding 操作

对于大量文档的 embedding，考虑分批并行处理：

```rust
use langchain_rust::utils::batch_process_result;

// 分批并行处理 embedding
let embeddings = batch_process_result(
    texts,
    10, // 每批 10 个
    |text| async move {
        embedder.embed_query(&text).await
    }
).await?;
```

### 2. 并行检索多个查询

如果有多个独立的查询，可以并行执行：

```rust
use langchain_rust::utils::join_all;

let queries = vec!["query1", "query2", "query3"];
let results = join_all(
    queries.iter().map(|q| retriever.get_relevant_documents(q))
).await;
```

### 3. 混合 RAG 中的并行优化

在 Hybrid RAG 中，如果查询增强和初始检索可以并行，可以考虑：

```rust
// 如果查询增强和初始检索不相互依赖
let (enhanced_query, initial_docs) = tokio::join!(
    enhancer.enhance(&query),
    retriever.get_relevant_documents(&query)
);
```

## 注意事项

1. **不要过度并行化** - 过多的并发任务可能导致资源竞争
2. **考虑 API 限制** - 某些 API 有速率限制，需要控制并发数
3. **内存使用** - 并行处理会增加内存使用，注意批量大小
4. **错误处理** - 使用 `try_join_all` 确保错误正确传播

## 当前状态

- ✅ 已创建异步优化工具模块
- ⚠️ 需要根据实际使用场景逐步应用优化
- ⚠️ 建议进行性能基准测试以验证优化效果
