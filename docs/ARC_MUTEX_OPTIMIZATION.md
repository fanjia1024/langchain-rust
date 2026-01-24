# Arc/Mutex 使用优化指南

本文档记录了 Arc/Mutex 使用的最佳实践和优化建议。

## 当前使用情况分析

### 合理使用 Arc<Mutex> 的场景

1. **AgentState** - 需要并发读写访问
   - 位置: `src/agent/executor.rs`, `src/agent/state.rs`
   - 原因: 多个异步任务可能同时修改状态
   - 建议: 保持使用 `Arc<Mutex<AgentState>>`

2. **BaseMemory** - 需要并发读写访问
   - 位置: `src/agent/executor.rs`
   - 原因: 需要线程安全的读写操作
   - 建议: 保持使用 `Arc<Mutex<dyn BaseMemory>>`

### 可以优化的场景

1. **Runtime** - 只读数据，不需要 Mutex
   - 当前: `Arc<Runtime>` (已优化)
   - Runtime 包含的 `context` 和 `store` 都是 `Arc<dyn Trait>`，已经是只读的

2. **ToolContext** - 只读数据
   - 当前: `Arc<dyn ToolContext>` (已优化)
   - ToolContext trait 只提供只读方法

3. **ToolStore** - 异步 trait，内部处理并发
   - 当前: `Arc<dyn ToolStore>` (已优化)
   - ToolStore 是 async trait，内部实现负责并发安全

## 优化建议

### 1. 使用 RwLock 替代 Mutex（读多写少场景）

对于读多写少的场景，考虑使用 `Arc<RwLock<T>>`：

```rust
// 如果 AgentState 主要是读取操作
use tokio::sync::RwLock;

pub struct AgentExecutor {
    state: Arc<RwLock<AgentState>>,  // 允许多个并发读取
}
```

### 2. 减少不必要的 Arc 克隆

在函数参数中，优先使用引用而非 Arc：

```rust
// 好的做法
fn process(&self, state: &AgentState) { }

// 避免不必要的 Arc 克隆
fn process(state: Arc<Mutex<AgentState>>) { }
```

### 3. 使用 OnceCell/LazyStatic 替代 Arc<Mutex>（单次初始化）

对于只需要初始化一次的数据：

```rust
use std::sync::OnceLock;

static CONFIG: OnceLock<Config> = OnceLock::new();
```

## 性能考虑

- `Mutex` 适合：写操作频繁或读写操作相当
- `RwLock` 适合：读操作远多于写操作
- `Arc` 用于：需要共享所有权
- 引用用于：不需要共享所有权，只需要临时访问

## 当前状态

经过分析，当前的 Arc/Mutex 使用基本合理：
- AgentState 和 Memory 需要 Mutex 保护
- Context 和 Store 使用 Arc 即可（只读或内部处理并发）
- Runtime 使用 Arc 即可（包含的都是只读数据）
