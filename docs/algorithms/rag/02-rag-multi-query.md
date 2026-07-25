# RAG 多查询技术详解

本文档详细解释了 RAG（Retrieval-Augmented Generation）系统中的多查询技术，基于 notebooks/[2]_rag_with_multi_query.ipynb。

## 什么是多查询技术？

多查询技术是一种改进 RAG 系统检索效果的方法，其核心思想是将用户的一个查询转换为多个不同的查询，然后使用这些查询从向量存储中检索相关文档。这种方法能够：

- 从不同角度获取相关信息
- 提高检索的召回率
- 减少因单一查询表述不准确导致的信息遗漏

## 多查询技术的核心原理

### 1. 查询扩展
通过提示工程，让语言模型基于原始查询生成多个语义相似但表达不同的查询。

示例：
- 原始查询："公司最近的财务表现如何？"
- 扩展查询：
  - "公司的营收情况怎么样？"
  - "公司的盈利能力如何？"
  - "公司的股票价格趋势？"
  - "公司的季度财报摘要？"

### 2. 多查询生成
使用专门的提示模板生成多个查询变体：

```python
from langchain.prompts import PromptTemplate

multi_query_prompt = PromptTemplate(
    input_variables=["question"],
    template="""你是一个AI助手，擅长从不同角度重新表述问题。任务是生成给定问题的{num_queries}个版本，以便从向量数据库中检索相关信息。通过改变句式、焦点和词汇，确保捕捉到各种可能的检索策略。

原始问题: {question}"""
)

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(temperature=0)
multi_query_chain = LLMChain(llm=llm, prompt=multi_query_prompt)
```

### 3. 查询执行
对生成的每个查询独立执行向量相似性搜索，获得多个结果集。

## 多查询实现策略

### 1. 使用 MultiQueryRetriever
LangChain 提供了专门的 MultiQueryRetriever 来简化多查询的实现：

```python
from langchain.retrievers import MultiQueryRetriever

retriever_from_llm = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(),
    llm=llm
)

# 这会自动生成多个查询并检索结果
docs = retriever_from_llm.get_relevant_documents("原始查询")
```

### 2. 自定义多查询实现
也可以手动实现多查询逻辑以获得更多控制：

```python
import re

def get_unique_union(documents: list[list]):
    """合并多个文档列表并去重"""
    combined_docs = []
    seen_contents = set()

    for docs in documents:
        for doc in docs:
            if doc.page_content not in seen_contents:
                combined_docs.append(doc)
                seen_contents.add(doc.page_content)

    return combined_docs

# 生成多个查询
queries = generate_queries(question)
all_docs = []

# 为每个查询检索文档
for query in queries:
    docs = vectorstore.similarity_search(query, k=k_value)
    all_docs.append(docs)

# 合并并去重
unique_docs = get_unique_union(all_docs)
```

## 检索融合技术

### 1. RRF (Reciprocal Rank Fusion)
将来自不同查询的检索结果按相关性重新排序的技术：

```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank

def reciprocal_rank_fusion(results: list, k=60):
    """实现倒数排名融合算法"""
    fused_scores = {}

    for result in results:
        for rank, doc in enumerate(result):
            doc_str = str(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            fused_scores[doc_str] += 1.0 / (rank + k)

    reranked_results = [
        (doc, score) for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return reranked_results
```

### 2. 相似性分数融合
将多个查询返回的相似性分数合并：

```python
def similarity_score_fusion(results: list):
    """合并相似性分数"""
    combined_scores = {}

    for result in results:
        for doc, score in result:
            doc_str = str(doc)
            if doc_str not in combined_scores:
                combined_scores[doc_str] = []
            combined_scores[doc_str].append(score)

    # 平均分数
    avg_scores = {
        doc: sum(scores) / len(scores)
        for doc, scores in combined_scores.items()
    }

    return sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
```

## 高级嵌入技术

### 1. 多模型嵌入
使用多个嵌入模型以提高检索覆盖范围：

```python
from langchain.embeddings import OpenAIEmbeddings, CohereEmbeddings
from langchain.vectorstores import FAISS

# 创建多个嵌入实例
openai_embeddings = OpenAIEmbeddings()
cohere_embeddings = CohereEmbeddings()

# 为同一文档集创建多个向量存储
openai_store = FAISS.from_documents(docs, openai_embeddings)
cohere_store = FAISS.from_documents(docs, cohere_embeddings)

# 使用 EnsembleRetriever 组合多个检索器
from langchain.retrievers import EnsembleRetriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[openai_store.as_retriever(), cohere_store.as_retriever()],
    weights=[0.5, 0.5]
)
```

### 2. 查询特定嵌入
根据查询类型使用最适合的嵌入模型：

```python
def adaptive_embedding_retriever(query: str, docs: list):
    """根据查询内容选择合适的嵌入模型"""
    if any(word in query.lower() for word in ["数学", "计算", "公式"]):
        # 使用数学优化的嵌入
        return mathematical_embeddings
    elif any(word in query.lower() for word in ["法律", "法规", "合同"]):
        # 使用法律领域优化的嵌入
        return legal_embeddings
    else:
        # 使用通用嵌入
        return openai_embeddings
```

## 性能比较分析

### 单查询 vs 多查询

| 方面 | 单查询 | 多查询 |
|------|--------|--------|
| 速度 | 更快 | 较慢（需多次查询） |
| 准确性 | 依赖查询表述 | 更高的召回率 |
| 覆盖面 | 可能遗漏相关信息 | 从多角度获取信息 |
| 成本 | 较低 | 较高（更多API调用） |

### 评估指标

1. **召回率 (Recall)**：找到的相关文档数量 / 所有相关文档数量
2. **精确率 (Precision)**：找到的相关文档数量 / 找到的总文档数量
3. **MRR (Mean Reciprocal Rank)**：平均倒数排名
4. **响应时间**：端到端查询处理时间

## 最佳实践

### 1. 查询数量优化
- 通常 3-5 个查询效果最好
- 过多查询会增加延迟和成本
- 过少查询可能无法显著改善检索效果

### 2. 生成提示优化
- 确保生成的查询保持原始意图
- 鼓励生成不同角度的查询
- 避免生成重复或无关的查询

### 3. 融合策略选择
- 对于精度要求高的场景，使用 RRF
- 对于速度要求高的场景，使用简单合并
- 根据数据特点调整权重

### 4. 成本控制
- 缓存多查询生成的结果
- 根据查询复杂度决定是否使用多查询
- 监控 API 调用频率和成本

## 应用场景

### 适合使用多查询的场景：
1. **复杂查询**：用户使用模糊或复杂的表述
2. **专业领域**：术语多样化的领域（如医学、法律）
3. **开放域问答**：需要高召回率的场景
4. **研究用途**：需要全面覆盖相关信息

### 不适合的场景：
1. **简单查询**：明确且直白的查询
2. **实时应用**：对响应时间要求极高的应用
3. **预算有限**：API 调用成本敏感的应用

## 总结

多查询技术是提高 RAG 系统检索效果的重要方法，通过生成多个语义相关的查询来增加信息发现的概率。正确实施这项技术可以在保持合理成本的同时显著提高检索质量和系统性能。