# RAG 路由与查询构建详解

本文档详细解释了 RAG（Retrieval-Augmented Generation）系统中的路由和查询构建技术，基于 notebooks/[3]_rag_routing_and_query_construction.ipynb。

## 什么是路由与查询构建？

路由与查询构建是 RAG 系统中的高级技术，用于：
- 根据查询内容智能地将请求路由到最适合的数据源
- 构建结构化查询以利用元数据过滤
- 基于语义相似性动态选择查询策略

## 路由策略

### 1. 逻辑路由 (Logical Routing)

逻辑路由基于预定义规则将查询分类并路由到相应数据源。

#### 基于函数的路由
```python
def route_by_language(query: str) -> str:
    """根据编程语言路由查询"""
    if any(lang in query.lower() for lang in ["python", "pandas", "numpy"]):
        return "python_docs"
    elif any(lang in query.lower() for lang in ["javascript", "react", "node"]):
        return "js_docs"
    elif any(lang in query.lower() for lang in ["java", "spring", "hibernate"]):
        return "java_docs"
    else:
        return "general_docs"
```

#### 条件路由示例
```python
from typing import Dict, Any

def conditional_router(query: str) -> Dict[str, Any]:
    """条件路由函数"""
    routing_map = {
        "mathematics": lambda q: any(keyword in q.lower() for keyword in
                                   ["equation", "formula", "calculate", "derive"]),
        "physics": lambda q: any(keyword in q.lower() for keyword in
                               ["force", "energy", "motion", "gravity"]),
        "programming": lambda q: any(keyword in q.lower() for keyword in
                                   ["code", "function", "debug", "algorithm"])
    }

    for category, condition in routing_map.items():
        if condition(query):
            return {"category": category, "target_db": f"{category}_db"}

    return {"category": "general", "target_db": "general_db"}
```

### 2. 语义路由 (Semantic Routing)

语义路由基于向量相似性匹配将查询路由到最相关的数据源。

```python
from langchain.embeddings import OpenAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class SemanticRouter:
    def __init__(self, embeddings, route_descriptions):
        """
        初始化语义路由器

        Args:
            embeddings: 嵌入模型
            route_descriptions: 路由描述字典
        """
        self.embeddings = embeddings
        self.route_descriptions = route_descriptions
        self.route_vectors = self._generate_route_vectors()

    def _generate_route_vectors(self):
        """为每个路由类别生成嵌入向量"""
        descriptions = list(self.route_descriptions.values())
        return np.array([
            self.embeddings.embed_query(desc)
            for desc in descriptions
        ])

    def route(self, query: str) -> str:
        """根据语义相似性路由查询"""
        query_vector = np.array(self.embeddings.embed_query(query)).reshape(1, -1)

        similarities = cosine_similarity(query_vector, self.route_vectors)[0]
        best_match_idx = np.argmax(similarities)

        return list(self.route_descriptions.keys())[best_match_idx]

# 使用示例
route_descriptions = {
    "math_db": "Mathematical concepts, equations, formulas, calculations, proofs",
    "physics_db": "Physics concepts, laws, experiments, theories, measurements",
    "coding_db": "Programming languages, algorithms, code examples, debugging"
}

router = SemanticRouter(OpenAIEmbeddings(), route_descriptions)
target_db = router.route("How to solve quadratic equations?")
```

## 查询构建技术

### 1. 元数据过滤查询结构

元数据过滤允许在查询时指定额外的过滤条件，以缩小检索范围。

```python
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document

# 示例：YouTube视频元数据过滤
def create_filtered_youtube_retriever(vectorstore):
    """创建支持元数据过滤的YouTube检索器"""

    # 可以按以下元数据过滤：
    # - 发布日期 (publish_date)
    # - 观看次数 (view_count)
    # - 播放时长 (duration)
    # - 视频类别 (category)

    retriever = vectorstore.as_retriever(
        search_kwargs={
            "filter": {
                "publish_date": {"$gte": "2023-01-01"},
                "view_count": {"$gte": 10000},
                "category": "Education"
            }
        }
    )

    return retriever

# 使用结构化查询模式
def build_structured_query(user_query: str, filters: dict = None) -> dict:
    """构建结构化查询"""
    structured_query = {
        "query_text": user_query,
        "filters": filters or {},
        "k": 5  # 返回前5个结果
    }

    return structured_query
```

### 2. 结构化搜索提示

使用LLM生成结构化数据库查询：

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

structured_search_prompt = PromptTemplate(
    template="""
    将自然语言查询转换为结构化搜索查询。

    输入查询: {query}

    输出应包含:
    1. 关键词搜索项
    2. 元数据过滤条件
    3. 优先级排序

    以JSON格式输出:
    {{
        "keywords": [...],
        "metadata_filters": {{
            "field_name": "value"
        }},
        "sort_by": "relevance",
        "limit": 10
    }}
    """,
    input_variables=["query"]
)

# 集成到查询链中
structured_search_chain = LLMChain(
    llm=llm,
    prompt=structured_search_prompt
)
```

## 向量存储集成

### 1. 多向量存储管理

管理多个向量存储实例：

```python
from langchain.vectorstores import Chroma
from typing import Dict, List

class MultiVectorStoreRouter:
    """多向量存储路由器"""

    def __init__(self, vector_stores: Dict[str, Chroma]):
        self.vector_stores = vector_stores

    def route_and_retrieve(self, query: str, routing_strategy: str = "auto"):
        """根据路由策略检索结果"""

        if routing_strategy == "semantic":
            target_store = self._semantic_route(query)
        elif routing_strategy == "logical":
            target_store = self._logical_route(query)
        else:  # auto
            target_store = self._hybrid_route(query)

        return target_store.similarity_search(query, k=5)

    def _semantic_route(self, query: str) -> Chroma:
        """语义路由"""
        # 实现语义路由逻辑
        pass

    def _logical_route(self, query: str) -> Chroma:
        """逻辑路由"""
        # 实现逻辑路由逻辑
        pass

    def _hybrid_route(self, query: str) -> Chroma:
        """混合路由"""
        # 实现混合路由逻辑
        pass
```

### 2. 统一检索接口

创建统一的检索接口来处理不同类型的路由：

```python
class UnifiedRetriever:
    """统一检索器"""

    def __init__(self, semantic_router=None, logical_router=None,
                 multi_store_router=None):
        self.semantic_router = semantic_router
        self.logical_router = logical_router
        self.multi_store_router = multi_store_router

    def retrieve(self, query: str, strategy: str = "hybrid", **kwargs):
        """
        统一检索方法

        Args:
            query: 查询字符串
            strategy: 路由策略 ("semantic", "logical", "hybrid")
            **kwargs: 额外参数
        """
        if strategy == "semantic":
            return self._semantic_retrieve(query, **kwargs)
        elif strategy == "logical":
            return self._logical_retrieve(query, **kwargs)
        elif strategy == "hybrid":
            return self._hybrid_retrieve(query, **kwargs)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _semantic_retrieve(self, query: str, **kwargs):
        """语义检索"""
        target_store = self.semantic_router.route(query)
        filters = kwargs.get('filters', {})
        return target_store.similarity_search(query, k=kwargs.get('k', 5),
                                           filter=filters)

    def _logical_retrieve(self, query: str, **kwargs):
        """逻辑检索"""
        target_store = self.logical_router.route(query)
        return target_store.similarity_search(query, k=kwargs.get('k', 5))

    def _hybrid_retrieve(self, query: str, **kwargs):
        """混合检索"""
        semantic_results = self._semantic_retrieve(query, **kwargs)
        logical_results = self._logical_retrieve(query, **kwargs)

        # 合并结果
        combined_results = self._merge_results(semantic_results, logical_results)
        return combined_results

    def _merge_results(self, results1, results2):
        """合并不同路由策略的结果"""
        # 实现结果合并逻辑
        all_results = results1 + results2
        # 去重并排序
        unique_results = []
        seen_content = set()

        for result in all_results:
            content_hash = hash(result.page_content)
            if content_hash not in seen_content:
                unique_results.append(result)
                seen_content.add(content_hash)

        return unique_results[:len(all_results)//2]  # 返回一半数量的结果
```

## 高级路由技术

### 1. 上下文感知路由

根据会话上下文进行路由：

```python
class ContextAwareRouter:
    """上下文感知路由器"""

    def __init__(self, base_router, context_window: int = 3):
        self.base_router = base_router
        self.context_window = context_window
        self.conversation_context = []

    def route_with_context(self, query: str, metadata: dict = None):
        """结合对话上下文进行路由"""
        # 添加当前查询到上下文
        self.conversation_context.append({
            "query": query,
            "timestamp": datetime.now(),
            "metadata": metadata
        })

        # 保留最近的对话上下文
        if len(self.conversation_context) > self.context_window:
            self.conversation_context = self.conversation_context[-self.context_window:]

        # 构建带上下文的查询
        context_query = self._build_contextual_query(query)

        # 使用基路由器进行路由
        return self.base_router.route(context_query)

    def _build_contextual_query(self, current_query: str) -> str:
        """构建包含上下文的查询"""
        if not self.conversation_context:
            return current_query

        context_parts = []
        for item in self.conversation_context[:-1]:  # 除当前查询外的所有历史
            context_parts.append(f"Previous: {item['query']}")

        context_parts.append(f"Current: {current_query}")

        return " ".join(context_parts)
```

### 2. 动态路由更新

根据反馈动态调整路由策略：

```python
class AdaptiveRouter:
    """自适应路由器"""

    def __init__(self, initial_routes: dict):
        self.routes = initial_routes
        self.performance_log = {}

    def route(self, query: str) -> str:
        """路由查询"""
        best_route = self._select_best_route(query)
        return self.routes[best_route]

    def update_with_feedback(self, query: str, route_used: str,
                           effectiveness_score: float):
        """根据反馈更新路由策略"""
        if route_used not in self.performance_log:
            self.performance_log[route_used] = []

        self.performance_log[route_used].append(effectiveness_score)

        # 如果某个路由持续表现不佳，则调整权重
        avg_performance = np.mean(self.performance_log[route_used])
        if avg_performance < 0.3:  # 阈值可调
            self._adjust_route_weights(route_used)

    def _select_best_route(self, query: str) -> str:
        """选择最佳路由（基于历史性能）"""
        # 实现基于历史性能的路由选择
        pass

    def _adjust_route_weights(self, poor_performing_route: str):
        """调整路由权重"""
        # 实现权重调整逻辑
        pass
```

## 最佳实践

### 1. 路由策略选择
- **逻辑路由**：适用于有明显分类边界的场景
- **语义路由**：适用于需要理解查询意图的场景
- **混合路由**：适用于复杂多样的查询场景

### 2. 性能优化
- 缓存路由决策结果
- 预计算向量相似性
- 异步处理多个路由候选

### 3. 错误处理
- 提供默认路由选项
- 实现降级机制
- 记录路由失败案例

### 4. 监控与评估
- 路由准确率监控
- 查询响应时间
- 用户满意度评分

## 应用场景

### 适合使用路由的场景：
1. **多领域知识库**：包含不同主题的知识库
2. **多模态数据**：文本、图像、表格等多种数据类型
3. **层次化组织**：按照部门、产品线等组织的信息
4. **时间敏感数据**：需要按时间或其他元数据过滤

### 查询构建适用场景：
1. **复杂过滤需求**：需要多个过滤条件
2. **结构化数据检索**：包含丰富元数据的文档
3. **精确检索**：需要严格限定检索范围的场景

## 总结

路由与查询构建是提升RAG系统效率和准确性的关键技术。通过智能路由可以将查询定向到最合适的数据源，而结构化查询构建则能充分利用元数据信息，两者结合可以显著提升检索效果。