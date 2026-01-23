// To run: cargo run --example vector_store_mongodb --features mongodb
// Requires: MongoDB Atlas with a Vector Search index on the collection.
// Set MONGODB_URI and create an index with $vectorSearch (see MongoDB Atlas docs).
// OPENAI_API_KEY for embedder.

#[cfg(feature = "mongodb")]
use langchain_rust::{
    embedding::openai::openai_embedder::OpenAiEmbedder, schemas::Document,
    vectorstore::mongodb::StoreBuilder, vectorstore::VecStoreOptions, vectorstore::VectorStore,
};
#[cfg(feature = "mongodb")]
use mongodb::Client;

#[cfg(feature = "mongodb")]
#[tokio::main]
async fn main() {
    let uri = std::env::var("MONGODB_URI").unwrap_or_else(|_| "mongodb://localhost:27017".into());
    let client = Client::with_uri_str(&uri).await.unwrap();
    let db = client.database("langchain_rs");
    let collection = db.collection::<mongodb::bson::Document>("vectors");

    let embedder = OpenAiEmbedder::default();
    let store = StoreBuilder::new()
        .collection(collection)
        .embedder(embedder)
        .index_name("vector_index") // your Atlas Vector Search index name
        .build()
        .unwrap();

    let doc1 = Document::new("langchain-rust is a port of the langchain python library to rust.");
    let doc2 = Document::new("langchaingo is a port of langchain to the go language.");
    let doc3 = Document::new("Capital of USA is Washington D.C. Capital of France is Paris.");

    let opt = VecStoreOptions::default();
    let _ids = store
        .add_documents(&[doc1, doc2, doc3], &opt)
        .await
        .unwrap();

    let results = store
        .similarity_search("capital of France", 2, &opt)
        .await
        .unwrap();
    for r in &results {
        println!("  {}", r.page_content);
    }
}

#[cfg(not(feature = "mongodb"))]
fn main() {
    println!("Run: cargo run --example vector_store_mongodb --features mongodb");
}
