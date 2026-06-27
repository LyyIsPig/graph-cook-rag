## HNSW 索引


HNSW（Hierarchical Navigable Small World）是一种基于图的索引算法，广泛应用于高维向量的近似最近邻（ANN）搜索。它通过构建分层图结构，实现了高效的搜索性能，兼具高召回率和低延迟，但需要较高的内存开销来维护其分层图。

### 工作原理

HNSW通过分层导航小世界图实现搜索优化。每一层的节点通过边连接，表示节点间的接近程度。上层图用于快速跳跃，底层图则进行精细搜索。搜索过程包括以下步骤：

1. 从顶层的入口点开始，利用贪婪搜索找到与查询向量最近的节点。
2. 达到局部最优后，跳转到下一层并重复搜索。
3. 在最底层完成最终的精细搜索，返回最近邻结果。

### 核心参数

HNSW的性能由以下关键参数控制：

- M：每个节点的最大连接数。较大的M值提高召回率，但增加内存和构建时间。
- efConstruction：索引构建时的候选邻居数量。值越高，索引质量越好，但构建时间更长。
- efSearch：搜索时的候选邻居数量。较大的值提高搜索精度，但增加查询延迟。

### 索引构建示例

以下是使用Python构建HNSW索引的代码示例：
```python
from pymilvus import MilvusClient

# 配置HNSW索引参数
index_params = {
"index_type": "HNSW",
"metric_type": "L2", # 欧几里得距离
"params": {
"M": 64, # 每个节点的最大连接数
"efConstruction": 100 # 构建时的候选邻居数量
}
}

# 创建索引
client = MilvusClient()
client.create_index(
collection_name="your_collection_name",
field_name="vector_field",
index_params=index_params
)
```

### 搜索示例

在构建索引后，可以执行相似性搜索：

```python
search_params = {
"params": {
"ef": 50 # 搜索时的候选邻居数量
}
}

results = client.search(
collection_name="your_collection_name",
anns_field="vector_field",
data=[[0.1, 0.2, 0.3, 0.4, 0.5]], # 查询向量
limit=10, # 返回Top-K结果
search_params=search_params
)
```

### 参数调优与性能权衡

- M值：较大的M值提高召回率，但会增加内存使用和构建时间。推荐范围为5到100。

- efConstruction：高值提升索引质量，适合对召回率要求较高的场景。

- efSearch：在查询时调整efSearch以平衡召回率和查询延迟。

### 优化建议

- 内存优化：通过使用乘积量化（PQ）技术减少内存占用。

- 搜索加速：结合倒排文件（IVF）减少搜索空间。

- 混合索引：结合多种索引技术以满足不同场景需求。

HNSW索引因其灵活性和高效性，已成为向量相似性搜索的主流选择。通过合理调整参数，可以在召回率、查询速度和内存使用之间找到最佳平衡。