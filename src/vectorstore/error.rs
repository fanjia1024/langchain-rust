use thiserror::Error;

#[derive(Error, Debug)]
pub enum VectorStoreError {
    #[error("This vector store does not support delete")]
    DeleteNotSupported,
}
