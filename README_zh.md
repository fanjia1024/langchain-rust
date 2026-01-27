# 🦜️🔗LangChain AI Rust

[![Latest Version]][crates.io]

[Latest Version]: https://img.shields.io/crates/v/langchain-ai-rs.svg
[crates.io]: https://crates.io/crates/langchain-ai-rs

⚡ 使用 Rust 通过组合性构建 LLM 应用程序！⚡

[![Discord](https://dcbadge.vercel.app/api/server/JJFcTFbanu?style=for-the-badge)](https://discord.gg/JJFcTFbanu)

## 🤔 这是什么？

这是 [LangChain](https://github.com/langchain-ai/langchain) 的 Rust 语言实现，为在 Rust 中构建 LLM 应用程序提供了强大且类型安全的方式。

## ✨ 核心特性

- 🚀 **多种 LLM 提供商**：支持 OpenAI、Azure OpenAI、Anthropic Claude、MistralAI、Google Gemini、AWS Bedrock、HuggingFace、阿里巴巴通义千问、DeepSeek 和 Ollama
- 🔗 **链式调用**：LLM 链、对话链、顺序链、问答链、SQL 链等
- 🤖 **智能体**：带工具的聊天智能体、多智能体系统（路由器、子智能体、技能、交接）
- 📚 **RAG**：智能体 RAG、混合 RAG 和两步 RAG 实现
- 🧠 **记忆**：简单记忆、对话记忆和带元数据的长期记忆
- 🛠️ **工具**：搜索工具、命令行、Wolfram Alpha、文本转语音等
- 📄 **文档加载器**：PDF、HTML、CSV、Git 提交、源代码等
- 🗄️ **向量存储**：PostgreSQL (pgvector)、Qdrant、SQLite (VSS/Vec)、SurrealDB、OpenSearch、In-Memory、Chroma、FAISS (hnsw_rs)、MongoDB Atlas、Pinecone、Weaviate
- 🎯 **嵌入模型**：OpenAI、Azure OpenAI、Ollama、FastEmbed、MistralAI
- 🔧 **中间件**：日志记录、PII 检测、内容过滤、速率限制、重试和自定义中间件
- 🎨 **结构化输出**：JSON 模式验证和结构化响应生成
- ⚙️ **运行时上下文**：动态提示、类型化上下文和运行时感知中间件
- 📊 **LangGraph**：状态图、流式、持久化（SQLite/内存）、中断、子图与时间旅行调试
- 🤖 **Deep Agent**：规划（write_todos）、文件系统工具（ls、read_file、write_file、edit_file）、技能、长期记忆与人机协同

## 📦 安装

本库严重依赖 `serde_json` 进行运行。

### 步骤 1：添加 `serde_json`

首先，确保将 `serde_json` 添加到您的 Rust 项目中。

```bash
cargo add serde_json
```

### 步骤 2：添加 `langchain-ai-rs`

然后，您可以将 `langchain-ai-rs` 添加到您的 Rust 项目中。

#### 简单安装

```bash
cargo add langchain-ai-rs
```

#### 使用向量存储

##### PostgreSQL (pgvector)

```bash
cargo add langchain-ai-rs --features postgres
```

##### Qdrant

```bash
cargo add langchain-ai-rs --features qdrant
```

##### SQLite (VSS)

从 <https://github.com/asg017/sqlite-vss> 下载额外的 sqlite_vss 库

```bash
cargo add langchain-ai-rs --features sqlite-vss
```

##### SQLite (Vec)

从 <https://github.com/asg017/sqlite-vec> 下载额外的 sqlite_vec 库

```bash
cargo add langchain-ai-rs --features sqlite-vec
```

##### SurrealDB

```bash
cargo add langchain-ai-rs --features surrealdb
```

##### OpenSearch

```bash
cargo add langchain-ai-rs --features opensearch
```

##### In-Memory

```bash
cargo add langchain-ai-rs --features in-memory
```

##### Chroma

```bash
cargo add langchain-ai-rs --features chroma
```

##### FAISS (hnsw_rs)

```bash
cargo add langchain-ai-rs --features faiss
```

##### MongoDB Atlas Vector Search

```bash
cargo add langchain-ai-rs --features mongodb
```

##### Pinecone

```bash
cargo add langchain-ai-rs --features pinecone
```

##### Weaviate

```bash
cargo add langchain-ai-rs --features weaviate
```

#### 使用 LLM 提供商

##### Ollama

```bash
cargo add langchain-ai-rs --features ollama
```

##### MistralAI

```bash
cargo add langchain-ai-rs --features mistralai
```

##### Google Gemini

```bash
cargo add langchain-ai-rs --features gemini
```

##### AWS Bedrock

```bash
cargo add langchain-ai-rs --features bedrock
```

#### 使用文档加载器

##### PDF (pdf-extract)

```bash
cargo add langchain-ai-rs --features pdf-extract
```

##### PDF (lopdf)

```bash
cargo add langchain-ai-rs --features lopdf
```

##### HTML 转 Markdown

```bash
cargo add langchain-ai-rs --features html-to-markdown
```

#### 使用代码解析

##### Tree-sitter（用于源代码解析，需要 0.26+）

```bash
cargo add langchain-ai-rs --features tree-sitter
```

#### 使用 FastEmbed（本地嵌入）

```bash
cargo add langchain-ai-rs --features fastembed
```

## 🚀 快速开始

### 简单的 LLM 调用

```rust
use langchain_ai_rs::llm::openai::{OpenAI, OpenAIModel};

#[tokio::main]
async fn main() {
    let llm = OpenAI::default().with_model(OpenAIModel::Gpt4oMini.to_string());
    let response = llm.invoke("什么是 Rust？").await.unwrap();
    println!("{}", response);
}
```

### 使用 init_chat_model（推荐）

`init_chat_model` 函数提供了统一的接口来初始化任何支持的 LLM：

```rust
use langchain_ai_rs::language_models::init_chat_model;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 初始化任何支持的模型
    let model = init_chat_model("gpt-4o-mini", None, None, None, None, None, None, None).await?;

    let response = model.invoke("你好，世界！").await?;
    println!("{}", response);

    Ok(())
}
```

支持的模型格式：

- `gpt-4o-mini`, `gpt-4o`, `gpt-4-turbo` (OpenAI)
- `claude-3-5-sonnet-20241022` (Anthropic)
- `mistralai/mistral-large-latest` (MistralAI)
- `gemini-1.5-pro` (Google Gemini)
- `anthropic.claude-3-5-sonnet-20241022-v2:0` (AWS Bedrock)
- `meta-llama/Llama-3.1-8B-Instruct` (HuggingFace)
- `qwen-plus` (阿里巴巴通义千问)
- `deepseek-chat` (DeepSeek)
- `llama3` (Ollama)

### 对话链

```rust
use langchain_ai_rs::{
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
            "你是一个有用的助手。"
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
            "input" => "什么是 Rust？",
            "history" => vec![
                Message::new_human_message("你好"),
                Message::new_ai_message("你好！"),
            ],
        })
        .await
    {
        Ok(result) => println!("结果: {:?}", result),
        Err(e) => panic!("错误: {:?}", e),
    }
}
```

### 创建带工具的智能体

```rust
use std::sync::Arc;
use langchain_ai_rs::{
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
        Some("你是一个可以执行命令的有用助手。"),
        None,
    )?;

    let result = agent
        .invoke_messages(vec![Message::new_human_message(
            "当前目录下有哪些文件？",
        )])
        .await?;

    println!("{}", result);
    Ok(())
}
```

### LangGraph（Hello World）

使用 `MessagesState` 构建状态图，用 `function_node` 添加节点，连接 START → 节点 → END，编译后调用：

```rust
use langchain_ai_rs::langgraph::{function_node, MessagesState, StateGraph, END, START};
use langchain_ai_rs::schemas::messages::Message;

let mock_llm = function_node("mock_llm", |_state: &MessagesState| async move {
    use std::collections::HashMap;
    let mut update = HashMap::new();
    update.insert(
        "messages".to_string(),
        serde_json::to_value(vec![Message::new_ai_message("hello world")])?,
    );
    Ok(update)
});

let mut graph = StateGraph::<MessagesState>::new();
graph.add_node("mock_llm", mock_llm)?;
graph.add_edge(START, "mock_llm");
graph.add_edge("mock_llm", END);

let compiled = graph.compile()?;
let initial_state = MessagesState::with_messages(vec![Message::new_human_message("hi!")]);
let final_state = compiled.invoke(initial_state).await?;
```

更多见 [LangGraph Hello World](examples/langgraph_hello_world.rs) 与 [LangGraph 流式](examples/langgraph_streaming.rs)。

### Deep Agent（基础）

使用 `create_deep_agent` 开启规划与文件系统；智能体获得工作区及内置工具（write_todos、ls、read_file、write_file、edit_file）：

```rust
use langchain_ai_rs::{
    agent::{create_deep_agent, DeepAgentConfig},
    chain::Chain,
    prompt_args,
    schemas::messages::Message,
};

let workspace = std::env::temp_dir().join("my_agent_workspace");
std::fs::create_dir_all(&workspace)?;

let config = DeepAgentConfig::new()
    .with_planning(true)
    .with_filesystem(true)
    .with_workspace_root(workspace);

let agent = create_deep_agent(
    "gpt-4o-mini",
    &[],
    Some("You are a helpful assistant with planning and file tools."),
    config,
)?;

let result = agent
    .invoke(prompt_args! {
        "messages" => vec![Message::new_human_message("List files in the workspace.")]
    })
    .await?;
```

更多见 [Deep Agent 基础](examples/deep_agent_basic.rs) 与 [Deep Agent 自定义](examples/deep_agent_customization.rs)。

## 📚 当前功能

### LLM 模型

- [x] [OpenAI](examples/llm_openai.rs)
- [x] [Azure OpenAI](examples/llm_azure_open_ai.rs)
- [x] [Anthropic Claude](examples/llm_anthropic_claude.rs)
- [x] [MistralAI](examples/llm_mistralai.rs)
- [x] [Google Gemini](examples/llm_gemini.rs)
- [x] [AWS Bedrock](examples/llm_bedrock.rs)
- [x] [HuggingFace](examples/llm_huggingface.rs)
- [x] [阿里巴巴通义千问](examples/llm_alibaba_qwen.rs)
- [x] [DeepSeek](examples/llm_deepseek.rs)
- [x] [Ollama](examples/llm_ollama.rs)
- [x] [统一模型初始化](examples/init_chat_model.rs)

### 嵌入模型

- [x] [OpenAI](examples/embedding_openai.rs)
- [x] [Azure OpenAI](examples/embedding_azure_open_ai.rs)
- [x] [Ollama](examples/embedding_ollama.rs)
- [x] [本地 FastEmbed](examples/embedding_fastembed.rs)
- [x] [MistralAI](examples/embedding_mistralai.rs)

### 向量存储

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

### 链式调用

- [x] [LLM 链](examples/llm_chain.rs)
- [x] [对话链](examples/conversational_chain.rs)
- [x] [简单对话检索器](examples/conversational_retriever_simple_chain.rs)
- [x] [带向量存储的对话检索器](examples/conversational_retriever_chain_with_vector_store.rs)
- [x] [顺序链](examples/sequential_chain.rs)
- [x] [问答链](examples/qa_chain.rs)
- [x] [SQL 链](examples/sql_chain.rs)
- [x] [流式链](examples/streaming_from_chain.rs)

### 智能体

- [x] [简单智能体](examples/create_agent_simple.rs)
- [x] [带工具的聊天智能体](examples/agent.rs)
- [x] [OpenAI 兼容工具智能体](examples/open_ai_tools_agent.rs)
- [x] [多智能体路由器](examples/multi_agent_router.rs)
- [x] [多智能体子智能体](examples/multi_agent_subagents.rs)
- [x] [多智能体技能](examples/multi_agent_skills.rs)
- [x] [多智能体交接](examples/multi_agent_handoffs.rs)

### LangGraph

- [x] [Hello World](examples/langgraph_hello_world.rs)
- [x] [流式](examples/langgraph_streaming.rs)
- [x] [持久化基础](examples/langgraph_persistence_basic.rs)、[持久化 SQLite](examples/langgraph_persistence_sqlite.rs)、[持久化回放](examples/langgraph_persistence_replay.rs)
- [x] [中断](examples/langgraph_interrupts.rs)、[中断审批](examples/langgraph_interrupts_approval.rs)、[中断审核](examples/langgraph_interrupts_review.rs)
- [x] [子图共享状态](examples/langgraph_subgraph_shared_state.rs)、[子图流式](examples/langgraph_subgraph_streaming.rs)
- [x] [记忆存储](examples/langgraph_memory_store.rs)、[记忆基础](examples/langgraph_memory_basic.rs)
- [x] [智能体工作流](examples/langgraph_agent_workflow.rs)、[并行执行](examples/langgraph_parallel_execution.rs)、[时间旅行](examples/langgraph_time_travel.rs)、[任务示例](examples/langgraph_task_example.rs)

### Deep Agent

- [x] [基础（规划 + 文件系统）](examples/deep_agent_basic.rs)
- [x] [自定义](examples/deep_agent_customization.rs)
- [x] [技能](examples/deep_agent_skills.rs)
- [x] [规划](examples/deep_agent_planning.rs)
- [x] [文件系统](examples/deep_agent_filesystem.rs)
- [x] [人机协同](examples/deep_agent_human_in_the_loop.rs)
- [x] [长期记忆](examples/deep_agent_long_term_memory.rs)
- [x] [任务工具](examples/deep_agent_with_task.rs)

### 文本分割器 (Text Splitters)

#### 基于文本结构

- [x] [递归字符分割器](examples/text_splitter_recursive_character.rs) - 推荐默认，按分隔符递归分割
- [x] 字符分割器 - 使用单个分隔符的简单字符分割
- [x] 纯文本分割器 - 基础文本分割
- [x] Token 分割器 - 基于 Token 的分割（Tiktoken）

#### 基于文档结构

- [x] Markdown 分割器 - 按 Markdown 结构分割
- [x] [HTML 分割器](examples/text_splitter_html.rs) - 按 HTML 标签分割
- [x] [JSON 分割器](examples/text_splitter_json.rs) - 按 JSON 对象/数组分割
- [x] 代码分割器 - 按语法树分割代码（tree-sitter 0.26+，需要 `tree-sitter` 特性）

### RAG（检索增强生成）

- [x] [智能体 RAG](examples/rag_agentic.rs) - 智能体决定何时检索
- [x] [混合 RAG](examples/rag_hybrid.rs) - 结合多种检索策略
- [x] [两步 RAG](examples/rag_two_step.rs) - 两阶段检索过程

### 检索器 (Retrievers)

#### 外部索引检索器

- [x] [Wikipedia 检索器](examples/retriever_wikipedia.rs) - 检索 Wikipedia 文章
- [x] Arxiv 检索器 - 从 arXiv 检索学术论文
- [x] Tavily 搜索 API 检索器 - 实时网络搜索

#### 基于算法的检索器

- [x] BM25 检索器 - BM25 算法文本检索
- [x] TF-IDF 检索器 - 基于 TF-IDF 的检索
- [x] SVM 检索器 - 基于支持向量机的检索

#### 重排序器

- [x] Cohere 重排序器 - 使用 Cohere API 重排序
- [x] FlashRank 重排序器 - 本地 ONNX 模型重排序
- [x] Contextual AI 重排序器 - Contextual AI API 重排序

#### 混合检索器

- [x] [合并检索器](examples/retriever_merger.rs) - 合并多个检索器结果
- [x] 集成检索器 - 多个检索器的投票机制

#### 查询增强检索器

- [x] 查询重写检索器 - 基于 LLM 的查询重写
- [x] 多查询检索器 - 生成多个查询变体

#### 文档压缩检索器

- [x] 嵌入冗余过滤器 - 基于相似度过滤冗余文档

### 工具

- [x] Serpapi/Google 搜索
- [x] DuckDuckGo 搜索
- [x] [Wolfram Alpha](examples/wolfram_tool.rs)
- [x] 命令行执行器
- [x] [文本转语音](examples/text_to_speech.rs)
- [x] [语音转文本](examples/speech2text_openai.rs)
- [x] [高级工具](examples/advanced_tools.rs)

### 中间件

- [x] [日志中间件](examples/middleware_logging.rs)
- [x] [PII 检测](examples/guardrails_pii.rs)
- [x] [内容过滤](examples/guardrails_combined.rs)
- [x] [自定义中间件](examples/middleware_custom.rs)
- [x] [运行时感知中间件](examples/runtime_middleware.rs)
- [x] [动态提示中间件](examples/runtime_dynamic_prompt.rs)

### 记忆

- [x] 简单记忆
- [x] 对话记忆
- [x] [长期记忆（基础）](examples/long_term_memory_basic.rs)
- [x] [长期记忆（搜索）](examples/long_term_memory_search.rs)
- [x] [长期记忆（工具）](examples/long_term_memory_tool.rs)

### 运行时和上下文

- [x] [类型化上下文](examples/runtime_typed_context.rs)
- [x] [动态工具](examples/context_engineering_dynamic_tools.rs)
- [x] [动态提示](examples/context_engineering_dynamic_prompt.rs)
- [x] [消息注入](examples/context_engineering_message_injection.rs)
- [x] [完整上下文工程](examples/context_engineering_complete.rs)

### 结构化输出

- [x] [结构化输出](examples/structured_output.rs)
- [x] [结构化输出提供者](examples/structured_output_provider.rs)

### 高级功能

- [x] [可配置模型](examples/configurable_model.rs)
- [x] [调用配置](examples/invocation_config.rs)
- [x] [语义路由](examples/semantic_routes.rs)
- [x] [动态语义路由](examples/dynamic_semantic_routes.rs)
- [x] [视觉 LLM 链](examples/vision_llm_chain.rs)
- [x] [工具运行时](examples/tool_runtime.rs)

### 文档加载器

#### 常见文件类型

- [x] PDF (pdf-extract 或 lopdf)
- [x] HTML
- [x] HTML 转 Markdown
- [x] CSV
- [x] TSV（制表符分隔值）
- [x] JSON（包括 JSONL）
- [x] Markdown
- [x] TOML（需要 `toml` 特性）
- [x] YAML（需要 `yaml` 特性）
- [x] XML（需要 `xml` 特性）

#### Office 文档

- [x] Excel (.xlsx, .xls)（需要 `excel` 特性）
- [x] Word、PowerPoint 等（通过 PandocLoader）

#### 网页加载器

- [x] WebBaseLoader - 从 URL 加载内容
- [x] RecursiveURLLoader - 递归抓取网站
- [x] SitemapLoader - 从 sitemap.xml 加载所有 URL（需要 `xml` 特性）

#### 云存储

- [x] AWS S3（需要 `aws-s3` 特性）

#### 生产力工具

- [x] GitHub（需要 `github` 特性）
- [x] Git 提交（需要 `git` 特性）

#### 其他

- [x] 源代码（需要 tree-sitter 特性）
- [x] Pandoc（各种格式：docx、epub、html、ipynb、markdown 等）

查看 [examples](examples/) 目录以获取每个功能的完整示例。

## 🔧 配置

### 环境变量

对于 OpenAI：

```bash
export OPENAI_API_KEY="your-api-key"
```

对于 Anthropic：

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

对于 MistralAI：

```bash
export MISTRAL_API_KEY="your-api-key"
```

对于 Google Gemini：

```bash
export GOOGLE_API_KEY="your-api-key"
```

对于 AWS Bedrock：

```bash
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_REGION="us-east-1"
```

## 📖 文档

- [示例目录](examples/)
- [API 文档](https://docs.rs/langchain-ai-rs)

## 🤝 贡献

欢迎贡献！请随时提交 Pull Request。

## 📝 许可证

本项目采用 MIT 许可证 - 有关详细信息，请参阅 LICENSE 文件。

## 🙏 致谢

- [LangChain](https://github.com/langchain-ai/langchain) - 原始 Python 实现
- 本库的所有贡献者和用户

## 🔗 链接

- [Crates.io](https://crates.io/crates/langchain-ai-rs)
- [Discord](https://discord.gg/JJFcTFbanu)
- [GitHub 仓库](https://github.com/fanjia1024/langchain-rust)
