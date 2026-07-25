# RAG 设置概览详解

本文档详细解释了 RAG（Retrieval-Augmented Generation）系统的基本设置和组件，基于 notebooks/[1]_rag_setup_overview.ipynb。

## 什么是 RAG？

RAG（检索增强生成）是一种结合信息检索和语言模型的技术。它允许大型语言模型（LLM）在生成响应时访问外部知识源，从而提高响应的准确性和相关性。

基本流程：
1. 用户提出问题
2. 系统检索相关的上下文信息
3. 将检索到的上下文与原始问题一起发送给语言模型
4. 语言模型生成最终回答

## RAG 系统的组成部分

### 1. 数据加载器 (Document Loaders)
负责将不同格式的数据加载为 LangChain 可以处理的文档对象。

常见数据源包括：
- PDF 文件 (`PyPDFLoader`)
- 文本文件 (`TextLoader`)
- HTML 网页 (`WebBaseLoader`)
- YouTube 视频转录 (`YoutubeLoader`)
- 以及数十种其他格式

示例：
```python
from langchain.document_loaders import PyPDFLoader
loader = PyPDFLoader("document.pdf")
docs = loader.load()
```

### 2. 文档分割器 (Text Splitters)
由于语言模型有输入长度限制，需要将长文档分割成较小的块。

常用分割器：
- `RecursiveCharacterTextSplitter`: 递归地按字符类型分割
- `CharacterTextSplitter`: 按特定字符分割
- `TokenTextSplitter`: 按 token 数量分割

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(documents)
```

### 3. 嵌入模型 (Embedding Models)
将文本转换为高维向量表示，以便进行语义相似性比较。

主要嵌入服务提供商：
- **OpenAI Embeddings**: `text-embedding-ada-002`
- **Cohere Embeddings**: 高质量的嵌入模型
- **Hugging Face Transformers**: 开源嵌入模型
- **Vertex AI**: Google Cloud 嵌入服务

```python
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
```

### 4. 向量存储 (Vector Stores)
高效存储和检索嵌入向量的数据结构。

流行的向量数据库：
- **Pinecone**: 托管向量数据库
- **Chroma**: 轻量级开源解决方案
- **FAISS**: Meta 开发的高效相似性搜索库
- **Weaviate**: 云原生开源知识图谱
- **Milvus**: 高性能向量数据库

```python
from langchain.vectorstores import Chroma
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings
)
```

### 5. 检索器 (Retrievers)
从向量存储中查找与查询最相关的内容。

常见的检索策略：
- **相似性检索**: 最基本的检索方式
- **MMR (最大边际相关性)**: 平衡相关性和多样性
- **自查询检索器**: 支持元数据过滤
- **压缩检索器**: 使用 LLM 对检索结果进行后处理

```python
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={'k': 3, 'fetch_k': 10}
)
```

### 6. 提示工程 (Prompt Engineering)
设计合适的提示模板，告诉语言模型如何处理检索到的上下文。

```python
from langchain_core.prompts import ChatPromptTemplate

template = """请仅根据以下上下文回答问题:
{context}

问题: {question}"""

prompt = ChatPromptTemplate.from_template(template)
```

### 7. 语言模型 (Language Model)
最终生成自然语言回答的模型。

可用选项：
- **OpenAI**: GPT-3.5-turbo, GPT-4
- **Anthropic**: Claude 2, Claude 3
- **Google**: PaLM 2, Gemini
- **Meta**: Llama 2, Llama 3
- **Hugging Face**: 开源模型集合

## 完整的 RAG 流程示例

```python
# 1. 加载和分割文档
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("document.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

# 2. 创建嵌入并存储在向量存储中
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

# 3. 设置检索器
retriever = vectorstore.as_retriever()

# 4. 设置语言模型
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# 5. 创建提示模板
from langchain_core.prompts import ChatPromptTemplate

template = """请仅根据以下上下文回答问题:
{context}

问题: {question}"""

prompt = ChatPromptTemplate.from_template(template)

# 6. 设置 RAG 链
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# 7. 运行查询
result = rag_chain.invoke("你的问题")
```

## RAG 系统的优势

1. **减少幻觉**: 通过提供具体事实，降低 LLM 编造答案的可能性
2. **实时信息**: 访问最新更新的信息，不受训练日期限制
3. **领域专业知识**: 整合特定领域的知识
4. **可验证性**: 用户可以查看用于生成回答的来源文档
5. **成本效益**: 比重新训练模型更便宜、更灵活

## 潜在挑战和考虑因素

1. **检索质量**: 检索相关文档的能力直接影响最终结果
2. **延迟**: 多步过程可能增加响应时间
3. **上下文管理**: 必须在上下文窗口内管理所有信息
4. **成本**: 每次查询都可能产生多次 API 调用费用

## 最佳实践

1. **文档分块策略**: 根据内容类型调整块大小和重叠
2. **嵌入选择**: 根据准确性需求和预算选择合适的嵌入模型
3. **检索策略**: 根据应用场景选择适当的检索方法
4. **提示工程**: 设计清晰、具体的提示模板
5. **性能监控**: 跟踪延迟、成本和用户满意度

## 总结

这个基础设置概述展示了构建 RAG 系统所需的核心组件。理解每个组件的作用对于后续的高级功能（如多查询、路由和重排序）至关重要。