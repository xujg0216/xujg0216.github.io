# RAG 检索与重排序详解

本文档详细解释了 RAG（Retrieval-Augmented Generation）系统中的检索与重排序技术，基于 notebooks/[5]_rag_retrieval_and_reranking.ipynb。

## 什么是检索与重排序？

检索与重排序是 RAG 系统中的关键优化技术，旨在提升检索结果的相关性。它包括文档加载、多查询生成、重排序算法等多个环节，通过一系列技术手段确保最终传递给语言模型的信息是最相关的。

## 文档加载与分割

### 1. 多样化文档加载器

```python
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredHTMLLoader,
    WebBaseLoader,
    YoutubeLoader
)
from langchain.text_splitter import RecursiveCharacterTextSplitter

class AdvancedDocumentLoader:
    """高级文档加载器"""

    def __init__(self):
        self.loaders = {
            "pdf": PyPDFLoader,
            "txt": TextLoader,
            "html": UnstructuredHTMLLoader,
            "web": WebBaseLoader,
            "youtube": YoutubeLoader
        }

    def load_document(self, source: str, source_type: str):
        """根据源类型加载文档"""
        if source_type == "web":
            loader = self.loaders[source_type]([source])
        elif source_type == "youtube":
            loader = self.loaders[source_type](source)
        else:
            loader = self.loaders[source_type](source)

        documents = loader.load()
        return self._chunk_documents(documents)

    def _chunk_documents(self, documents):
        """分割文档"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        return text_splitter.split_documents(documents)
```

### 2. 智能分割策略

```python
class SmartTextSplitter:
    """智能文本分割器"""

    def __init__(self):
        self.default_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def adaptive_split(self, content: str, content_type: str = "generic"):
        """根据内容类型自适应分割"""
        if content_type == "code":
            return self._code_split(content)
        elif content_type == "academic":
            return self._academic_split(content)
        elif content_type == "legal":
            return self._legal_split(content)
        else:
            return self.default_splitter.split_text(content)

    def _code_split(self, content: str):
        """代码分割策略"""
        # 按函数、类等代码结构分割
        import re

        # 查找函数定义
        function_pattern = r'(def\s+\w+\s*\([^)]*\):|class\s+\w+\s*:)'
        splits = re.split(function_pattern, content)

        chunks = []
        for i in range(len(splits)-1):
            if re.match(function_pattern, splits[i]):
                chunk = splits[i] + splits[i+1]
                if len(chunk.strip()) > 0:
                    chunks.append(chunk)

        return chunks

    def _academic_split(self, content: str):
        """学术文档分割策略"""
        # 按章节、小节分割
        import re

        # 查找标题（通常是单独一行的大写字母开头）
        title_pattern = r'\n\s*[A-Z][A-Za-z\s]{5,}\n'
        splits = re.split(title_pattern, content)

        return [split.strip() for split in splits if len(split.strip()) > 100]

    def _legal_split(self, content: str):
        """法律文档分割策略"""
        # 按条款、章节分割
        import re

        # 查找条款编号（如 "第1条", "Article 1" 等）
        clause_pattern = r'((?:第\s*[一二三四五六七八九十\d]+\s*条)|(?:Article\s+\d+))'
        splits = re.split(clause_pattern, content)

        chunks = []
        for i in range(1, len(splits), 2):
            if i+1 < len(splits):
                chunk = splits[i] + splits[i+1]
                if len(chunk.strip()) > 50:
                    chunks.append(chunk)

        return chunks
```

## 多查询生成与 RAG-Fusion

### 1. RAG-Fusion 概念

RAG-Fusion 是一种通过生成多个查询来提高检索覆盖率的技术，然后使用重排序算法融合结果。

```python
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import re

class RagFusionGenerator:
    """RAG-Fusion 查询生成器"""

    def __init__(self, llm):
        self.llm = llm

        # RAG-Fusion 查询生成提示
        self.rag_fusion_prompt = PromptTemplate(
            input_variables=["original_query"],
            template="""
            你是一个AI助手，负责为给定的原始查询生成多个搜索查询。
            这些查询将用于从向量数据库中检索相关信息。

            原始查询: {original_query}

            请生成3-5个不同角度的搜索查询，考虑以下方面：
            1. 直接相关的关键词查询
            2. 同义词和相关概念查询
            3. 更具体的细节查询
            4. 更宽泛的概念查询
            5. 从不同角度的表达

            每行输出一个查询，不要编号，不要解释。

            查询列表:
            """
        )

    def generate_queries(self, original_query: str):
        """生成多个查询"""
        chain = LLMChain(llm=self.llm, prompt=self.rag_fusion_prompt)
        response = chain.run(original_query=original_query)

        # 解析生成的查询
        queries = [q.strip() for q in response.strip().split('\n') if q.strip()]

        # 添加原始查询
        queries.insert(0, original_query)

        return queries[:6]  # 最多返回6个查询（包括原始查询）
```

### 2. 多查询检索与结果合并

```python
import numpy as np
from typing import List, Dict
from langchain.docstore.document import Document

class MultiQueryRetriever:
    """多查询检索器"""

    def __init__(self, rag_fusion_generator: RagFusionGenerator, vector_store):
        self.rag_fusion_generator = rag_fusion_generator
        self.vector_store = vector_store

    def retrieve_multiple_queries(self, original_query: str, k_per_query: int = 5):
        """使用多个查询检索"""
        # 生成查询
        queries = self.rag_fusion_generator.generate_queries(original_query)

        all_results = {}
        for i, query in enumerate(queries):
            results = self.vector_store.similarity_search(query, k=k_per_query)
            all_results[f"query_{i}"] = {
                "query": query,
                "results": results
            }

        return all_results

    def deduplicate_results(self, all_results: Dict):
        """去重结果"""
        unique_docs = {}
        doc_scores = {}

        for query_key, query_data in all_results.items():
            for doc in query_data["results"]:
                doc_content_hash = hash(doc.page_content)

                if doc_content_hash not in unique_docs:
                    unique_docs[doc_content_hash] = doc

                # 累积分数（这里简单计数，实际可使用相似度分数）
                if doc_content_hash not in doc_scores:
                    doc_scores[doc_content_hash] = 0
                doc_scores[doc_content_hash] += 1

        # 按出现次数排序
        sorted_docs = sorted(
            unique_docs.values(),
            key=lambda x: doc_scores[hash(x.page_content)],
            reverse=True
        )

        return sorted_docs
```

## Reciprocal Rank Fusion (RRF)

### 1. RRF 算法实现

Reciprocal Rank Fusion (RRF) 是一种常用的重排序算法，用于融合多个检索结果列表。

```python
def reciprocal_rank_fusion(
    result_lists: List[List[Document]],
    k: int = 60,
    top_k: int = 10
) -> List[tuple]:
    """
    Reciprocal Rank Fusion 算法

    Args:
        result_lists: 多个检索结果列表
        k: RRF 参数，影响排名靠后的项目权重
        top_k: 返回前k个结果

    Returns:
        重排序后的文档及分数列表
    """
    all_docs = set()
    doc_scores = {}

    # 收集所有文档
    for result_list in result_lists:
        for doc in result_list:
            all_docs.add(doc)

    # 计算每个文档的 RRF 分数
    for result_list in result_lists:
        for rank, doc in enumerate(result_list):
            doc_hash = str(hash(doc.page_content))

            if doc_hash not in doc_scores:
                doc_scores[doc_hash] = 0

            # RRF 公式: 1 / (k + rank)
            rank_score = 1.0 / (k + rank)
            doc_scores[doc_hash] += rank_score

    # 排序并返回结果
    sorted_docs = sorted(
        doc_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    return sorted_docs

class RRFReRanker:
    """RRF 重排序器"""

    def __init__(self, k: int = 60):
        self.k = k

    def rerank(self, result_lists: List[List[Document]], top_k: int = 10):
        """使用 RRF 重排序"""
        rrf_scores = reciprocal_rank_fusion(result_lists, self.k, top_k)

        # 重构文档列表（这里需要从原始文档中找回文档对象）
        ranked_docs = []
        doc_map = {}

        # 创建文档哈希到文档对象的映射
        for result_list in result_lists:
            for doc in result_list:
                doc_hash = str(hash(doc.page_content))
                if doc_hash not in doc_map:
                    doc_map[doc_hash] = doc

        for doc_hash, score in rrf_scores:
            if doc_hash in doc_map:
                ranked_docs.append((doc_map[doc_hash], score))

        return ranked_docs
```

### 2. 集成 RAG-Fusion 和 RRF

```python
class FusionRerankRetriever:
    """融合 RAG-Fusion 和 RRF 的检索器"""

    def __init__(self, rag_fusion_generator, vector_store, rrf_re_ranker):
        self.rag_fusion_generator = rag_fusion_generator
        self.vector_store = vector_store
        self.rrf_re_ranker = rrf_re_ranker

    def retrieve_with_fusion_and_rerank(self, query: str, k_per_query: int = 5, top_k: int = 10):
        """执行融合查询和重排序"""
        # 生成多个查询
        queries = self.rag_fusion_generator.generate_queries(query)

        # 对每个查询执行检索
        result_lists = []
        for query_var in queries:
            results = self.vector_store.similarity_search(query_var, k=k_per_query)
            result_lists.append(results)

        # 使用 RRF 重排序
        ranked_results = self.rrf_re_ranker.rerank(result_lists, top_k)

        return ranked_results
```

## Cohere 重排序

### 1. Cohere 重排序集成

```python
import cohere
from typing import List, Tuple

class CohereReRanker:
    """Cohere 重排序器"""

    def __init__(self, api_key: str):
        self.co = cohere.Client(api_key)

    def rerank_with_cohere(self, query: str, documents: List[str], top_n: int = 10):
        """
        使用 Cohere API 进行重排序

        Args:
            query: 查询字符串
            documents: 文档列表
            top_n: 返回前n个结果

        Returns:
            重排序后的文档索引和相关性分数
        """
        response = self.co.rerank(
            query=query,
            documents=documents,
            top_n=top_n
        )

        return response.results

    def rerank_documents(self, query: str, doc_list: List[Document], top_n: int = 10):
        """对 Document 对象列表进行重排序"""
        # 提取文档内容
        document_texts = [doc.page_content for doc in doc_list]

        # 调用 Cohere 重排序
        results = self.rerank_with_cohere(query, document_texts, top_n)

        # 重构结果，包含原始文档对象
        reranked_docs = []
        for result in results:
            original_doc = doc_list[result.index]
            reranked_docs.append({
                'document': original_doc,
                'relevance_score': result.relevance_score,
                'index': result.index
            })

        return reranked_docs
```

### 2. 混合重排序策略

```python
class HybridReRanker:
    """混合重排序器"""

    def __init__(self, rrf_re_ranker: RRFReRanker, cohere_re_ranker: CohereReRanker = None):
        self.rrf_re_ranker = rrf_re_ranker
        self.cohere_re_ranker = cohere_re_ranker

    def hybrid_rerank(self, query: str, result_lists: List[List[Document]],
                     strategy: str = "rrf_then_cohere", top_k: int = 10):
        """
        混合重排序策略

        Args:
            query: 查询字符串
            result_lists: 多个结果列表
            strategy: 重排序策略 ("rrf_then_cohere", "cohere_then_rrf", "weighted_average")
            top_k: 返回前k个结果
        """
        if strategy == "rrf_then_cohere":
            # 先使用 RRF 重排序
            rrf_results = self.rrf_re_ranker.rerank(result_lists, top_k * 2)  # 获取更多结果用于Cohere
            top_docs = [item[0] for item in rrf_results]  # 提取文档对象

            if self.cohere_re_ranker:
                # 再使用 Cohere 重排序
                final_results = self.cohere_re_ranker.rerank_documents(query, top_docs, top_k)
            else:
                final_results = rrf_results[:top_k]

        elif strategy == "weighted_average":
            # 结合 RRF 和 Cohere 的分数（如果都有）
            rrf_scores = reciprocal_rank_fusion(result_lists, k=60, top_k=len(result_lists[0]))

            if self.cohere_re_ranker:
                # 提取所有唯一文档
                all_docs = set()
                for result_list in result_lists:
                    all_docs.update(result_list)

                all_docs_list = list(all_docs)

                # 获取 Cohere 分数
                cohere_results = self.cohere_re_ranker.rerank_documents(query, all_docs_list, len(all_docs_list))

                # 加权平均
                combined_scores = {}
                doc_to_obj = {hash(doc.page_content): doc for doc in all_docs}

                # 归一化 RRF 分数
                max_rrf = max([score for _, score in rrf_scores]) if rrf_scores else 1
                rrf_norm = [(doc_hash, score/max_rrf) for doc_hash, score in rrf_scores]

                # 归一化 Cohere 分数
                max_cohere = max([r['relevance_score'] for r in cohere_results]) if cohere_results else 1
                cohere_norm = [(str(hash(r['document'].page_content)), r['relevance_score']/max_cohere)
                              for r in cohere_results]

                # 组合分数（各占50%权重）
                for doc_hash, rrf_score in rrf_norm:
                    combined_scores[doc_hash] = rrf_score * 0.5

                for doc_hash, cohere_score in cohere_norm:
                    if doc_hash in combined_scores:
                        combined_scores[doc_hash] += cohere_score * 0.5
                    else:
                        combined_scores[doc_hash] = cohere_score * 0.5

                # 排序并返回结果
                sorted_combined = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

                final_results = []
                for doc_hash, score in sorted_combined:
                    doc = doc_to_obj.get(int(doc_hash.replace('\'', '').replace('"', '')))
                    if doc:
                        final_results.append({'document': doc, 'combined_score': score})
            else:
                final_results = rrf_results[:top_k]

        else:
            # 默认使用 RRF
            final_results = self.rrf_re_ranker.rerank(result_lists, top_k)

        return final_results
```

## 检索链设置

### 1. 完整的 RAG 链

```python
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

class AdvancedRagChain:
    """高级 RAG 链"""

    def __init__(self, retriever, llm, prompt_template):
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template

    def format_docs(self, docs):
        """格式化文档"""
        return "\n\n".join([doc.page_content for doc in docs])

    def create_chain(self):
        """创建 RAG 链"""
        prompt = ChatPromptTemplate.from_template(self.prompt_template)

        rag_chain = (
            {"context": self.retriever | self.format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        return rag_chain

# 使用示例
template = """
基于以下上下文回答问题：

{context}

问题: {question}

如果上下文没有相关信息，请明确说明你不知道答案，不要编造信息。
"""

# 假设我们已经有了上述组件
# advanced_rag_chain = AdvancedRagChain(retriever, llm, template)
# rag_chain = advanced_rag_chain.create_chain()
```

## CRAG 和 Self-RAG 检索

### 1. CRAG (Corrective RAG)

CRAG 通过验证检索到的信息并进行纠正来改进标准 RAG。

```python
class CRAGRetriever:
    """CRAG 检索器"""

    def __init__(self, base_retriever, llm):
        self.base_retriever = base_retriever
        self.llm = llm

    def retrieve_and_correct(self, query: str):
        """检索并纠正信息"""
        # 首先进行标准检索
        initial_docs = self.base_retriever.get_relevant_documents(query)

        # 验证检索到的文档与查询的相关性
        verified_docs = []
        for doc in initial_docs:
            if self._verify_relevance(query, doc):
                corrected_content = self._correct_content(query, doc)
                corrected_doc = doc.copy()
                corrected_doc.page_content = corrected_content
                verified_docs.append(corrected_doc)
            else:
                # 如果不相关，尝试获取更相关的文档
                additional_docs = self._get_alternative_docs(query, doc)
                verified_docs.extend(additional_docs)

        return verified_docs[:len(initial_docs)]  # 保持原始数量

    def _verify_relevance(self, query: str, document) -> bool:
        """验证文档与查询的相关性"""
        verification_prompt = f"""
        请判断以下文档内容是否与查询相关：

        查询: {query}
        文档: {document.page_content}

        回答"相关"或"不相关"。
        """

        response = self.llm.predict(verification_prompt)
        return "相关" in response.lower()

    def _correct_content(self, query: str, document):
        """根据查询纠正文档内容"""
        correction_prompt = f"""
        请根据查询需求，精炼和纠正以下文档内容：

        查询: {query}
        原文档: {document.page_content}

        请提供与查询更相关的精炼版本：
        """

        corrected_content = self.llm.predict(correction_prompt)
        return corrected_content

    def _get_alternative_docs(self, query: str, irrelevant_doc):
        """获取替代文档"""
        # 基于不相关文档生成新的查询
        alternative_query_prompt = f"""
        原查询: {query}
        不相关文档: {irrelevant_doc.page_content}

        请生成一个新的查询，能够找到与原始查询更相关的信息：
        """

        new_query = self.llm.predict(alternative_query_prompt)
        return self.base_retriever.get_relevant_documents(new_query)
```

### 2. Self-RAG

Self-RAG 通过在检索和生成过程中引入反思机制来提高准确性。

```python
class SelfRAGRetriever:
    """Self-RAG 检索器"""

    def __init__(self, base_retriever, llm):
        self.base_retriever = base_retriever
        self.llm = llm

    def retrieve_with_reflection(self, query: str):
        """带反思的检索"""
        # 1. 检索初始文档
        docs = self.base_retriever.get_relevant_documents(query)

        # 2. 对每个文档进行反思评估
        reflected_docs = []
        for doc in docs:
            reflection_result = self._reflect_on_document(query, doc)

            if reflection_result["useful"]:
                # 根据反思结果调整文档
                adjusted_doc = self._adjust_document(doc, reflection_result)
                reflected_docs.append(adjusted_doc)

        return reflected_docs

    def _reflect_on_document(self, query: str, document):
        """反思文档"""
        reflection_prompt = f"""
        请对以下文档进行反思评估：

        查询: {query}
        文档: {document.page_content}

        请提供以下信息：
        1. 文档是否有用？(有用/无用)
        2. 如果有用，好在哪里？
        3. 如果无用，为什么？
        4. 如何改进？

        以JSON格式回答：
        {{
            "useful": true/false,
            "strengths": "...",
            "weaknesses": "...",
            "suggestions": "..."
        }}
        """

        response = self.llm.predict(reflection_prompt)

        # 这里应该解析JSON响应，为简化示例返回模拟结果
        return {
            "useful": True,
            "strengths": "包含相关信息",
            "weaknesses": "信息不够具体",
            "suggestions": "需要更具体的细节"
        }

    def _adjust_document(self, document, reflection_result):
        """根据反思结果调整文档"""
        if reflection_result["useful"]:
            # 可以根据反思结果对文档进行微调
            adjusted_doc = document.copy()
            # 这里可以根据建议进行调整
            return adjusted_doc
        else:
            return None
```

## 长上下文影响考虑

### 1. 长文档处理策略

```python
class LongContextHandler:
    """长上下文处理器"""

    def __init__(self, max_context_length: int = 3000):
        self.max_context_length = max_context_length

    def handle_long_context(self, docs: List[Document], query: str):
        """处理长上下文"""
        # 计算总长度
        total_length = sum(len(doc.page_content) for doc in docs)

        if total_length <= self.max_context_length:
            # 如果长度合适，直接返回
            return docs
        else:
            # 如果太长，需要裁剪
            return self._compress_context(docs, query)

    def _compress_context(self, docs: List[Document], query: str):
        """压缩上下文"""
        # 策略1: 基于相关性分数排序，保留最重要的部分
        scored_docs = []
        for doc in docs:
            relevance_score = self._calculate_relevance(doc.page_content, query)
            scored_docs.append((doc, relevance_score))

        # 按相关性排序
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 逐步添加，直到达到最大长度
        compressed_docs = []
        current_length = 0

        for doc, score in scored_docs:
            if current_length + len(doc.page_content) <= self.max_context_length:
                compressed_docs.append(doc)
                current_length += len(doc.page_content)
            else:
                # 如果添加这个文档会超过限制，尝试截断文档
                remaining_space = self.max_context_length - current_length
                if remaining_space > 0:
                    truncated_content = doc.page_content[:remaining_space]
                    truncated_doc = Document(
                        page_content=truncated_content,
                        metadata=doc.metadata
                    )
                    compressed_docs.append(truncated_doc)
                break

        return compressed_docs

    def _calculate_relevance(self, content: str, query: str) -> float:
        """计算内容与查询的相关性（简化的实现）"""
        # 简单的关键词匹配分数
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())

        intersection = query_words.intersection(content_words)
        union = query_words.union(content_words)

        if len(union) == 0:
            return 0.0

        return len(intersection) / len(union)
```

## 完整的 RAG 系统集成示例

```python
class CompleteRAGSystem:
    """完整的 RAG 系统"""

    def __init__(self, llm, vector_store, cohere_api_key: str = None):
        self.llm = llm
        self.vector_store = vector_store

        # 初始化组件
        self.rag_fusion_gen = RagFusionGenerator(llm)
        self.rrf_reranker = RRFReRanker(k=60)

        if cohere_api_key:
            self.cohere_reranker = CohereReRanker(cohere_api_key)
            self.hybrid_reranker = HybridReRanker(self.rrf_reranker, self.cohere_reranker)
        else:
            self.cohere_reranker = None
            self.hybrid_reranker = HybridReRanker(self.rrf_reranker)

        self.multi_query_retriever = MultiQueryRetriever(self.rag_fusion_gen, vector_store)
        self.long_context_handler = LongContextHandler()

    def retrieve_and_answer(self, query: str, use_cohere: bool = False):
        """执行完整的检索和回答流程"""

        # 1. 使用多查询生成
        queries = self.rag_fusion_gen.generate_queries(query)

        # 2. 对每个查询进行检索
        result_lists = []
        for q in queries:
            results = self.vector_store.similarity_search(q, k=5)
            result_lists.append(results)

        # 3. 重排序
        if use_cohere and self.cohere_reranker:
            # 使用混合重排序（RRF + Cohere）
            reranked_results = self.hybrid_reranker.hybrid_rerank(
                query, result_lists, strategy="rrf_then_cohere", top_k=10
            )
            # 提取文档
            docs = [r['document'] if isinstance(r, dict) else r[0] for r in reranked_results]
        else:
            # 仅使用 RRF 重排序
            reranked_results = self.rrf_reranker.rerank(result_lists, top_k=10)
            docs = [r[0] for r in reranked_results]

        # 4. 处理长上下文
        final_docs = self.long_context_handler.handle_long_context(docs, query)

        # 5. 构建回答
        context = "\n\n".join([doc.page_content for doc in final_docs])

        answer_prompt = f"""
        基于以下上下文回答问题：

        上下文:
        {context}

        问题: {query}

        请提供详细、准确的回答。如果上下文没有相关信息，请明确说明。
        """

        answer = self.llm.predict(answer_prompt)

        return {
            "answer": answer,
            "sources": final_docs,
            "queries_used": queries
        }

# 使用示例
# rag_system = CompleteRAGSystem(llm, vector_store, cohere_api_key)
# result = rag_system.retrieve_and_answer("你的查询")
```

## 性能评估与监控

### 1. 评估指标

```python
class RAGEvaluator:
    """RAG 系统评估器"""

    def __init__(self, llm):
        self.llm = llm

    def evaluate_relevance(self, query: str, retrieved_docs: List[Document], ground_truth: str = None):
        """评估检索结果的相关性"""
        scores = []
        for doc in retrieved_docs:
            relevance_score = self._assess_relevance(query, doc.page_content)
            scores.append(relevance_score)

        return {
            "average_relevance": sum(scores) / len(scores) if scores else 0,
            "scores": scores
        }

    def _assess_relevance(self, query: str, content: str) -> float:
        """评估单个文档的相关性"""
        assessment_prompt = f"""
        评估以下文档内容与查询的相关性：

        查询: {query}
        文档内容: {content}

        请给出 0-1 之间的相关性分数，其中 1 表示高度相关，0 表示完全不相关。
        只返回数字分数。
        """

        response = self.llm.predict(assessment_prompt)

        try:
            score = float(response.strip())
            return max(0, min(1, score))  # 确保分数在 0-1 范围内
        except:
            return 0.5  # 默认中等相关性

    def evaluate_answer_quality(self, query: str, answer: str, ground_truth: str = None):
        """评估答案质量"""
        evaluation_prompt = f"""
        评估以下答案的质量：

        查询: {query}
        答案: {answer}

        请从以下方面评估：
        1. 准确性 (0-10)
        2. 完整性 (0-10)
        3. 相关性 (0-10)
        4. 清晰度 (0-10)

        以JSON格式返回：
        {{
            "accuracy": score,
            "completeness": score,
            "relevance": score,
            "clarity": score,
            "overall": average_score
        }}
        """

        response = self.llm.predict(evaluation_prompt)
        # 在实际实现中，应该解析 JSON 响应
        return {"overall": 7.5}  # 模拟返回值
```

## 总结

检索与重排序是 RAG 系统中至关重要的环节，通过以下关键技术实现了性能提升：

1. **文档加载与分割**：支持多种格式的智能分割
2. **RAG-Fusion**：通过多查询生成提高检索覆盖率
3. **RRF**：有效融合多个检索结果列表
4. **Cohere 重排序**：利用专用模型进行深度重排序
5. **CRAG 和 Self-RAG**：引入验证和反思机制提升准确性
6. **长上下文处理**：优化大量检索结果的处理

这些技术的综合应用能够显著提升 RAG 系统的检索质量和最终回答的准确性。