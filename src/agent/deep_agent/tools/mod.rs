//! Built-in tools for Deep Agent: write_todos, file system (ls, read_file, write_file, edit_file, glob, grep), task.

mod tool_wrapper;
mod write_todos;
mod task;

pub use tool_wrapper::ToolWithCustomDescription;
pub use write_todos::{TodoItem, TodoStatus, WriteTodosTool};
pub use task::TaskTool;

pub mod fs;
pub use fs::{
    EditFileTool, GlobTool, GrepTool, LsTool, ReadFileTool, WriteFileTool,
};
