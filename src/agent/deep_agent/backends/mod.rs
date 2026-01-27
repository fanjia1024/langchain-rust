//! Pluggable file backends for deep agent filesystem tools.
//!
//! See [Backends](https://docs.langchain.com/oss/python/deepagents/backends).
//!
//! - **Default**: When no backend is set, FS tools use the workspace root from config/context (same as [WorkspaceBackend]).
//! - [WorkspaceBackend]: Local disk under a root directory (sandboxed).
//! - [StoreBackend]: Files stored in [crate::tools::ToolStore] (e.g. InMemoryStore).
//! - [CompositeBackend]: Routes requests by path prefix to different backends (longest-prefix match).

mod composite;
mod store;
mod workspace;

pub use composite::CompositeBackend;
pub use crate::tools::{EditResult, FileBackend, FileInfo, GrepMatch, WriteResult};
pub use store::StoreBackend;
pub use workspace::WorkspaceBackend;
