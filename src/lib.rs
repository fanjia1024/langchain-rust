#![allow(dead_code)]
pub mod agent;
pub mod chain;
pub mod document_loaders;
pub mod embedding;
pub mod error;
pub mod langgraph;
pub mod language_models;
pub mod llm;
pub mod memory;
pub mod output_parsers;
pub mod prompt;
pub mod rag;
pub mod retrievers;
pub mod schemas;
pub mod semantic_router;
pub mod text_splitter;
pub mod tools;
pub mod utils;
pub mod vectorstore;

pub use url;

// ============================================================================
// Type Aliases for Common Type Combinations
// ============================================================================

use std::sync::Arc;
use tokio::sync::Mutex;

/// Type alias for a tool wrapped in Arc
pub type Tool = Arc<dyn crate::tools::Tool>;

/// Type alias for a list of tools
pub type Tools = Vec<Arc<dyn crate::tools::Tool>>;

/// Type alias for tool context
pub type ToolContext = Arc<dyn crate::tools::ToolContext>;

/// Type alias for tool store
pub type ToolStore = Arc<dyn crate::tools::ToolStore>;

/// Type alias for agent state
pub type AgentState = Arc<Mutex<crate::agent::AgentState>>;

/// Type alias for memory
pub type Memory = Arc<Mutex<dyn crate::schemas::memory::BaseMemory>>;

/// Type alias for middleware list
pub type MiddlewareList = Vec<Arc<dyn crate::agent::Middleware>>;

/// Type alias for message list
pub type Messages = Vec<crate::schemas::Message>;

/// Type alias for embedding vector (f64)
pub type Embedding = Vec<f64>;

/// Type alias for embedding vector (f32)
pub type EmbeddingF32 = Vec<f32>;

/// Type alias for document list
pub type Documents = Vec<crate::schemas::Document>;
