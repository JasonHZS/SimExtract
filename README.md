# SimExtract
**相似度归因研究平台 | Similarity Attribution Research Platform**

从文本B中细粒度地识别并提取对文本A相似度贡献最高的关键词或片段（Span）。

本项目旨在系统化地研究和对比不同的相似度归因方法，为文本相似度分析提供可解释的细粒度归因结果。

## 核心目标

给定两个文本A和B，当检测到它们具有较高的语义相似度时，**从B中提取出对该相似度贡献最高的关键词或片段**。

例如：
- 文本A: "AI正在改变医疗行业"
- 文本B: "机器学习在医学诊断中的应用正在revolutionizing患者护理，同时天气很好"
- **归因结果**: "机器学习在医学诊断中的应用" (高贡献度片段)

## 4种研究方法

项目探索并对比以下4种归因方法：

### 1. 分段向量化 (Segmented Vectorization)
将文本B分段（句子/固定长度/语义单元），计算每个片段与文本A的向量相似度。

- **优势**: 实现简单，计算高效
- **状态**: 🔲 骨架已建立，待实现

### 2. Cross-Encoder 注意力分析
使用BERT Cross-Encoder处理`[CLS] A [SEP] B [SEP]`，分析注意力矩阵，提取A的token对B的token的注意力权重。

- **优势**: 捕捉深层交互，理论基础强
- **状态**: 🔲 骨架已建立，待实现

### 3. Token Wise
生成token级别权重/嵌入，通过 sparce embeddings 或 Late Interaction机制（MaxSim）计算每个token的归因得分。

#### 实现

- 基于 sparse embedding 的 token 权重：支持。bge-m3在生成稠密向量的同时可返回每个词项的“lexical weights”（近似 BM25 风格的 learned sparse 权重），并能计算文档间的词项匹配得分与逐词贡献。
- 基于 ColBERT 的 token 权重：支持获取每个 token 的向量（multi-vector/late interaction），可用这些向量计算 token 对齐的相似度贡献；官方 API给出整体 ColBERT 分数，但你可以用返回的 token 向量自行得到逐 token 的权重或对齐矩阵。

- **优势**: Token级精细度，保留上下文
- **状态**: 🔲 骨架已建立，待实现

### 4. Late Chunking
先生成全文上下文嵌入，再在嵌入层面进行智能分块和聚合。

- **优势**: 保留全局上下文，语义分块
- **状态**: 🔲 骨架已建立，待实现
