# 🦜️🔗LangChain Rust

[![Latest Version]][crates.io]

[Latest Version]: https://img.shields.io/crates/v/langchain-rust.svg
[crates.io]: https://crates.io/crates/langchain-rust

⚡ Building applications with LLMs through composability, with Rust! ⚡

[![Discord](https://dcbadge.vercel.app/api/server/JJFcTFbanu?style=for-the-badge)](https://discord.gg/JJFcTFbanu)
[![Docs: Tutorial](https://img.shields.io/badge/docs-tutorial-success?style=for-the-badge&logo=appveyor)](https://langchain-rust.sellie.tech/get-started/quickstart)

## 🤔 What is this?

This is the Rust language implementation of [LangChain](https://github.com/langchain-ai/langchain), providing a powerful and type-safe way to build LLM applications in Rust.

## ✨ Key Features

- 🚀 **Multiple LLM Providers**: Support for OpenAI, Azure OpenAI, Anthropic Claude, MistralAI, Google Gemini, AWS Bedrock, HuggingFace, Alibaba Qwen, DeepSeek, and Ollama
- 🔗 **Chains**: LLM chains, conversational chains, sequential chains, Q&A chains, SQL chains, and more
- 🤖 **Agents**: Chat agents with tools, multi-agent systems (router, subagents, skills, handoffs)
- 📚 **RAG**: Agentic RAG, Hybrid RAG, and two-step RAG implementations
- 🧠 **Memory**: Simple memory, conversational memory, and long-term memory with metadata
- 🛠️ **Tools**: Search tools, command line, Wolfram Alpha, text-to-speech, and more
- 📄 **Document Loaders**: PDF, HTML, CSV, Git commits, source code, and more
- 🗄️ **Vector Stores**: PostgreSQL (pgvector), Qdrant, SQLite (VSS/Vec), SurrealDB, OpenSearch, In-Memory, Chroma, FAISS (hnsw_rs), MongoDB Atlas, Pinecone, Weaviate
- 🎯 **Embeddings**: OpenAI, Azure OpenAI, Ollama, FastEmbed, MistralAI
- 🔧 **Middleware**: Logging, PII detection, content filtering, rate limiting, retry, and custom middleware
- 🎨 **Structured Output**: JSON schema validation and structured response generation
- ⚙️ **Runtime Context**: Dynamic prompts, typed context, and runtime-aware middleware

## 📦 Installation

This library heavily relies on `serde_json` for its operation.

### Step 1: Add `serde_json`

First, ensure `serde_json` is added to your Rust project.

```bash
cargo add serde_json
```

### Step 2: Add `langchain-rust`

Then, you can add `langchain-rust` to your Rust project.

#### Simple install

```bash
cargo add langchain-rust
```

#### With Vector Stores

##### PostgreSQL (pgvector)

```bash
cargo add langchain-rust --features postgres
```

##### Qdrant

```bash
cargo add langchain-rust --features qdrant
```

##### SQLite (VSS)

Download additional sqlite_vss libraries from <https://github.com/asg017/sqlite-vss>

```bash
cargo add langchain-rust --features sqlite-vss
```

##### SQLite (Vec)

Download additional sqlite_vec libraries from <https://github.com/asg017/sqlite-vec>

```bash
cargo add langchain-rust --features sqlite-vec
```

##### SurrealDB

```bash
cargo add langchain-rust --features surrealdb
```

##### OpenSearch

```bash
cargo add langchain-rust --features opensearch
```

##### In-Memory

```bash
cargo add langchain-rust --features in-memory
```

##### Chroma

```bash
cargo add langchain-rust --features chroma
```

##### FAISS (hnsw_rs)

```bash
cargo add langchain-rust --features faiss
```

##### MongoDB Atlas Vector Search

```bash
cargo add langchain-rust --features mongodb
```

##### Pinecone

```bash
cargo add langchain-rust --features pinecone
```

##### Weaviate

```bash
cargo add langchain-rust --features weaviate
```

#### With LLM Providers

##### Ollama

```bash
cargo add langchain-rust --features ollama
```

##### MistralAI

```bash
cargo add langchain-rust --features mistralai
```

##### Google Gemini

```bash
cargo add langchain-rust --features gemini
```

##### AWS Bedrock

```bash
cargo add langchain-rust --features bedrock
```

#### With Document Loaders

##### PDF (pdf-extract)

```bash
cargo add langchain-rust --features pdf-extract
```

##### PDF (lopdf)

```bash
cargo add langchain-rust --features lopdf
```

##### HTML to Markdown

```bash
cargo add langchain-rust --features html-to-markdown
```

#### With Code Parsing

##### Tree-sitter (for source code parsing, requires 0.26+)

```bash
cargo add langchain-rust --features tree-sitter
```

#### With FastEmbed (Local Embeddings)

```bash
cargo add langchain-rust --features fastembed
```

## 🚀 Quick Start

### Simple LLM Invocation

```rust
use langchain_rust::llm::openai::{OpenAI, OpenAIModel};

#[tokio::main]
async fn main() {
    let llm = OpenAI::default().with_model(OpenAIModel::Gpt4oMini.to_string());
    let response = llm.invoke("What is Rust?").await.unwrap();
    println!("{}", response);
}
```

### Using init_chat_model (Recommended)

The `init_chat_model` function provides a unified interface to initialize any supported LLM:

```rust
use langchain_rust::language_models::init_chat_model;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize any supported model
    let model = init_chat_model("gpt-4o-mini", None, None, None, None, None, None, None).await?;
    
    let response = model.invoke("Hello, world!").await?;
    println!("{}", response);
    
    Ok(())
}
```

Supported model formats:
- `gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo` (OpenAI)
- `claude-3-5-sonnet-20241022` (Anthropic)
- `mistralai/mistral-large-latest` (MistralAI)
- `gemini-1.5-pro` (Google Gemini)
- `anthropic.claude-3-5-sonnet-20241022-v2:0` (AWS Bedrock)
- `meta-llama/Llama-3.1-8B-Instruct` (HuggingFace)
- `qwen-plus` (Alibaba Qwen)
- `deepseek-chat` (DeepSeek)
- `llama3` (Ollama)

### Conversational Chain

```rust
use langchain_rust::{
    chain::{Chain, LLMChainBuilder},
    fmt_message, fmt_placeholder, fmt_template,
    llm::openai::{OpenAI, OpenAIModel},
    message_formatter,
    prompt::HumanMessagePromptTemplate,
    prompt_args,
    schemas::messages::Message,
    template_fstring,
};

#[tokio::main]
async fn main() {
    let open_ai = OpenAI::default().with_model(OpenAIModel::Gpt4oMini.to_string());

    let prompt = message_formatter![
        fmt_message!(Message::new_system_message(
            "You are a helpful assistant."
        )),
        fmt_placeholder!("history"),
        fmt_template!(HumanMessagePromptTemplate::new(template_fstring!(
            "{input}", "input"
        ))),
    ];

    let chain = LLMChainBuilder::new()
        .prompt(prompt)
        .llm(open_ai)
        .build()
        .unwrap();

    match chain
        .invoke(prompt_args! {
            "input" => "What is Rust?",
            "history" => vec![
                Message::new_human_message("Hello"),
                Message::new_ai_message("Hi there!"),
            ],
        })
        .await
    {
        Ok(result) => println!("Result: {:?}", result),
        Err(e) => panic!("Error: {:?}", e),
    }
}
```

### Creating an Agent with Tools

```rust
use std::sync::Arc;
use langchain_rust::{
    agent::create_agent,
    schemas::messages::Message,
    tools::CommandExecutor,
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let command_executor = Arc::new(CommandExecutor::default());
    
    let agent = create_agent(
        "gpt-4o-mini",
        &[command_executor],
        Some("You are a helpful assistant that can execute commands."),
        None,
    )?;

    let result = agent
        .invoke_messages(vec![Message::new_human_message(
            "What files are in the current directory?",
        )])
        .await?;

    println!("{}", result);
    Ok(())
}
```

## 📚 Current Features

### LLMs

- [x] [OpenAI](examples/llm_openai.rs)
- [x] [Azure OpenAI](examples/llm_azure_open_ai.rs)
- [x] [Anthropic Claude](examples/llm_anthropic_claude.rs)
- [x] [MistralAI](examples/llm_mistralai.rs)
- [x] [Google Gemini](examples/llm_gemini.rs)
- [x] [AWS Bedrock](examples/llm_bedrock.rs)
- [x] [HuggingFace](examples/llm_huggingface.rs)
- [x] [Alibaba Qwen](examples/llm_alibaba_qwen.rs)
- [x] [DeepSeek](examples/llm_deepseek.rs)
- [x] [Ollama](examples/llm_ollama.rs)
- [x] [Unified Model Initialization](examples/init_chat_model.rs)

### Embeddings

- [x] [OpenAI](examples/embedding_openai.rs)
- [x] [Azure OpenAI](examples/embedding_azure_open_ai.rs)
- [x] [Ollama](examples/embedding_ollama.rs)
- [x] [Local FastEmbed](examples/embedding_fastembed.rs)
- [x] [MistralAI](examples/embedding_mistralai.rs)

### Vector Stores

- [x] [PostgreSQL (pgvector)](examples/vector_store_postgres.rs)
- [x] [Qdrant](examples/vector_store_qdrant.rs)
- [x] [SQLite VSS](examples/vector_store_sqlite_vss.rs)
- [x] [SQLite Vec](examples/vector_store_sqlite_vec.rs)
- [x] [SurrealDB](examples/vector_store_surrealdb/src/main.rs)
- [x] [OpenSearch](examples/vector_store_opensearch.rs)
- [x] [In-Memory](examples/vector_store_in_memory.rs)
- [x] [Chroma](examples/vector_store_chroma.rs)
- [x] [FAISS](examples/vector_store_faiss.rs)
- [x] [MongoDB Atlas](examples/vector_store_mongodb.rs)
- [x] [Pinecone](examples/vector_store_pinecone.rs)
- [x] [Weaviate](examples/vector_store_weaviate.rs)

### Chains

- [x] [LLM Chain](examples/llm_chain.rs)
- [x] [Conversational Chain](examples/conversational_chain.rs)
- [x] [Conversational Retriever Simple](examples/conversational_retriever_simple_chain.rs)
- [x] [Conversational Retriever With Vector Store](examples/conversational_retriever_chain_with_vector_store.rs)
- [x] [Sequential Chain](examples/sequential_chain.rs)
- [x] [Q&A Chain](examples/qa_chain.rs)
- [x] [SQL Chain](examples/sql_chain.rs)
- [x] [Streaming Chain](examples/streaming_from_chain.rs)

### Agents

- [x] [Simple Agent](examples/create_agent_simple.rs)
- [x] [Chat Agent with Tools](examples/agent.rs)
- [x] [OpenAI Compatible Tools Agent](examples/open_ai_tools_agent.rs)
- [x] [Multi-Agent Router](examples/multi_agent_router.rs)
- [x] [Multi-Agent Subagents](examples/multi_agent_subagents.rs)
- [x] [Multi-Agent Skills](examples/multi_agent_skills.rs)
- [x] [Multi-Agent Handoffs](examples/multi_agent_handoffs.rs)

### Text Splitters

#### Text Structure-Based
- [x] [RecursiveCharacterTextSplitter](examples/text_splitter_recursive_character.rs) - Recommended default, splits recursively by separators
- [x] CharacterTextSplitter - Simple character-based splitting with single separator
- [x] PlainTextSplitter - Basic text splitting
- [x] TokenSplitter - Token-based splitting (Tiktoken)

#### Document Structure-Based
- [x] MarkdownSplitter - Split Markdown by structure
- [x] [HTMLSplitter](examples/text_splitter_html.rs) - Split HTML by tags
- [x] [JsonSplitter](examples/text_splitter_json.rs) - Split JSON by objects/arrays
- [x] CodeSplitter - Split code by syntax tree (tree-sitter 0.26+, requires `tree-sitter` feature)

### RAG (Retrieval-Augmented Generation)

- [x] [Agentic RAG](examples/rag_agentic.rs) - Agent decides when to retrieve
- [x] [Hybrid RAG](examples/rag_hybrid.rs) - Combines multiple retrieval strategies
- [x] [Two-Step RAG](examples/rag_two_step.rs) - Two-stage retrieval process

### Retrievers

#### External Index Retrievers
- [x] [Wikipedia Retriever](examples/retriever_wikipedia.rs) - Retrieve Wikipedia articles
- [x] Arxiv Retriever - Retrieve academic papers from arXiv
- [x] Tavily Search API Retriever - Real-time web search

#### Algorithm-Based Retrievers
- [x] BM25 Retriever - BM25 algorithm for text retrieval
- [x] TF-IDF Retriever - TF-IDF based retrieval
- [x] SVM Retriever - Support Vector Machine based retrieval

#### Rerankers
- [x] Cohere Reranker - Rerank using Cohere API
- [x] FlashRank Reranker - Local ONNX model reranking
- [x] Contextual AI Reranker - Contextual AI API reranking

#### Hybrid Retrievers
- [x] [Merger Retriever](examples/retriever_merger.rs) - Combine multiple retrievers
- [x] Ensemble Retriever - Voting mechanism from multiple retrievers

#### Query Enhancement Retrievers
- [x] RePhrase Query Retriever - LLM-based query rephrasing
- [x] Multi Query Retriever - Generate multiple query variations

#### Document Compression Retrievers
- [x] Embeddings Redundant Filter - Filter redundant documents by similarity

### Tools

- [x] Serpapi/Google Search
- [x] DuckDuckGo Search
- [x] [Wolfram Alpha](examples/wolfram_tool.rs)
- [x] Command Line Executor
- [x] [Text-to-Speech](examples/text_to_speech.rs)
- [x] [Speech-to-Text](examples/speech2text_openai.rs)
- [x] [Advanced Tools](examples/advanced_tools.rs)

### Middleware

- [x] [Logging Middleware](examples/middleware_logging.rs)
- [x] [PII Detection](examples/guardrails_pii.rs)
- [x] [Content Filtering](examples/guardrails_combined.rs)
- [x] [Custom Middleware](examples/middleware_custom.rs)
- [x] [Runtime-Aware Middleware](examples/runtime_middleware.rs)
- [x] [Dynamic Prompt Middleware](examples/runtime_dynamic_prompt.rs)

### Memory

- [x] Simple Memory
- [x] Conversational Memory
- [x] [Long-Term Memory (Basic)](examples/long_term_memory_basic.rs)
- [x] [Long-Term Memory (Search)](examples/long_term_memory_search.rs)
- [x] [Long-Term Memory (Tool)](examples/long_term_memory_tool.rs)

### Runtime & Context

- [x] [Typed Context](examples/runtime_typed_context.rs)
- [x] [Dynamic Tools](examples/context_engineering_dynamic_tools.rs)
- [x] [Dynamic Prompts](examples/context_engineering_dynamic_prompt.rs)
- [x] [Message Injection](examples/context_engineering_message_injection.rs)
- [x] [Complete Context Engineering](examples/context_engineering_complete.rs)

### Structured Output

- [x] [Structured Output](examples/structured_output.rs)
- [x] [Structured Output Provider](examples/structured_output_provider.rs)

### Advanced Features

- [x] [Configurable Models](examples/configurable_model.rs)
- [x] [Invocation Config](examples/invocation_config.rs)
- [x] [Semantic Routing](examples/semantic_routes.rs)
- [x] [Dynamic Semantic Routing](examples/dynamic_semantic_routes.rs)
- [x] [Vision LLM Chain](examples/vision_llm_chain.rs)
- [x] [Tool Runtime](examples/tool_runtime.rs)

### Document Loaders

#### Common File Types
- [x] PDF (pdf-extract or lopdf)
- [x] HTML
- [x] HTML to Markdown
- [x] CSV
- [x] TSV (Tab-Separated Values)
- [x] JSON (including JSONL)
- [x] Markdown
- [x] TOML (with `toml` feature)
- [x] YAML (with `yaml` feature)
- [x] XML (with `xml` feature)

#### Office Documents
- [x] Excel (.xlsx, .xls) (with `excel` feature)
- [x] Word, PowerPoint, and more via PandocLoader

#### Web Loaders
- [x] WebBaseLoader - Load content from URLs
- [x] RecursiveURLLoader - Recursively crawl websites
- [x] SitemapLoader - Load all URLs from sitemap.xml (with `xml` feature)

#### Cloud Storage
- [x] AWS S3 (with `aws-s3` feature)

#### Productivity Tools
- [x] GitHub (with `github` feature)
- [x] Git Commits (with `git` feature)

#### Other
- [x] Source Code (with tree-sitter feature)
- [x] Pandoc (various formats: docx, epub, html, ipynb, markdown, etc.)

See the [examples](examples/) directory for complete examples of each feature.

## 🔧 Configuration

### Environment Variables

For OpenAI:
```bash
export OPENAI_API_KEY="your-api-key"
```

For Anthropic:
```bash
export ANTHROPIC_API_KEY="your-api-key"
```

For MistralAI:
```bash
export MISTRAL_API_KEY="your-api-key"
```

For Google Gemini:
```bash
export GOOGLE_API_KEY="your-api-key"
```

For AWS Bedrock:
```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION="us-east-1"
```

## 📖 Documentation

- [Official Documentation](https://langchain-rust.sellie.tech/get-started/quickstart)
- [Examples Directory](examples/)
- [API Documentation](https://docs.rs/langchain-rust)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) - The original Python implementation
- All contributors and users of this library

## 🔗 Links

- [Crates.io](https://crates.io/crates/langchain-rust)
- [Documentation](https://langchain-rust.sellie.tech)
- [Discord](https://discord.gg/JJFcTFbanu)
- [GitHub Repository](https://github.com/Abraxas-365/langchain-rust)
