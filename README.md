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

## 归因方法

项目探索并对比以下4种归因方法，目前已完整实现3种：

### 1. 分段向量化 (Segmented Vectorization) ✅

**原理**：将文本B分段（句子/固定长度/语义单元），对每个片段独立向量化，然后计算每个片段与文本A的余弦相似度，从而识别哪些片段对整体相似度贡献最大。

**实现细节**：
- **分段策略**：支持两种分段方式
  - `fixed_sentences`：按句子数量分组（默认3句一组），适合处理文章、新闻等结构化文本
  - `fixed_length`：按 token 数量分段（默认30 tokens，overlap 10），智能处理中英文混合
- **向量化**：通过 TEI (Text Embeddings Inference) 服务调用 embedding 模型，支持批量处理提升效率
- **相似度计算**：标准余弦相似度，公式为 `cos(θ) = (A · B) / (||A|| × ||B||)`
- **智能分词**：自动识别 CJK 字符（每字一 token）、英文单词（连续字母）、数字序列

**优势**：
- 实现简洁，易于理解和调试
- 对向量化服务的依赖最小（只需标准 dense embedding）
- 支持任意 embedding 模型，无需特定模型能力

**工程特性**：
- **计算复杂度**：`O(n_segments)` 次向量化调用
- **延迟**：中等（取决于 TEI 服务响应时间和分段数量）
- **内存占用**：低（每次只保存分段的 embedding）
- **适用场景**：长文档归因、段落级粗粒度分析

---

### 2. Sparse Embedding Attribution (Token-wise) ✅

**原理**：基于 BGE-M3 模型的 lexical weights（learned sparse representation），类似 BM25 但通过神经网络学习得到。计算公式为：

```
s_lex = Σ(t∈A∩B) w_a(t) × w_b(t)
```

其中 `w_a(t)` 和 `w_b(t)` 分别是 token `t` 在文本A和B中的稀疏权重。通过识别交集 token 的贡献，提取高贡献区域。

**实现细节**：
- **模型**：使用 `BAAI/bge-m3` 的 sparse embedding 能力
- **Token 贡献计算**：对每个 token 计算 `w_a × w_b`，得到逐 token 的贡献分数
- **滑动窗口**：在 token 序列上应用滑动窗口（默认50 tokens，overlap 40），计算窗口内平均贡献分数
- **Top-K 提取**：返回得分最高的 K 个窗口作为 attribution spans
- **去重机制**：使用 LRU 缓存避免重复加载模型，支持多进程复用

**优势**：
- Token 级精细度，能定位到具体词汇
- 可解释性强（直接显示每个 token 的贡献权重）
- 适合关键词匹配类任务（如检索、问答）

**工程特性**：
- **计算复杂度**：`O(1)` 次模型调用（单次 encode 即可获得所有 token 权重）
- **延迟**：低（本地模型推理，无需外部服务调用）
- **内存占用**：中等（需加载完整 BGE-M3 模型，约2GB）
- **GPU 加速**：支持 FP16，显著降低显存和提升速度
- **适用场景**：关键词定位、精确匹配、检索归因分析

---

### 3. ColBERT Attribution (Token-wise Late Interaction) ✅

**原理**：基于 BGE-M3 的 ColBERT multi-vector representation，每个 token 都有独立的 embedding。使用 MaxSim 机制计算相似度：

```
s_colbert = (1/N_q) × Σ(i=1..N_q) max(j=1..N_d) E_q[i] · E_d[j]
```

对于每个 query token，找到与 document 中最相似的 token，然后对所有 query tokens 求平均。

**实现细节**：
- **模型**：使用 `BAAI/bge-m3` 的 colbert_vecs 输出
- **交互矩阵**：计算 query 和 document 所有 token 对之间的余弦相似度矩阵 `[N_q × N_d]`
- **滑动窗口 MaxSim**：对 document 应用滑动窗口（默认50 tokens），在每个窗口内计算 MaxSim 分数
- **1D Max Pooling**：使用 `F.max_pool1d` 高效计算窗口内的最大值
- **Topic Keywords 提取**：额外提取全局高贡献的 topic keywords 作为补充信息
- **向量归一化**：所有向量在计算前进行 L2 归一化，确保余弦相似度计算准确

**优势**：
- Token 级精细度 + 上下文感知（每个 token 的 embedding 包含上下文信息）
- MaxSim 机制能捕捉语义对齐而非精确匹配
- 适合处理改写、同义词替换等语义相似但词汇不同的情况

**工程特性**：
- **计算复杂度**：`O(N_q × N_d)` 矩阵乘法 + `O(N_windows)` 窗口聚合
- **延迟**：中（比 Sparse 稍慢，因为需要构建完整交互矩阵）
- **内存占用**：高（需存储完整交互矩阵，对长文本可能达到 GB 级别）
- **GPU 加速**：强烈建议使用 GPU，矩阵运算可充分利用并行计算
- **适用场景**：语义对齐分析、改写检测、跨语言归因

---

### 4. Cross-Encoder 注意力分析

使用 BERT Cross-Encoder 处理 `[CLS] A [SEP] B [SEP]`，分析注意力矩阵，提取 A 的 token 对 B 的 token 的注意力权重。

- **优势**: 捕捉深层交互，理论基础强
- **状态**: 🔲 骨架已建立，待实现

---

## 工程对比总结

| 方法 | 延迟 | 内存占用 | GPU需求 | Token粒度 | 语义理解 | 适用场景 |
|------|------|----------|---------|-----------|----------|----------|
| **Segmented** | 中 | 低 | 可选 | 段落级 | 高 | 长文档、段落归因 |
| **Sparse** | 低 | 中 | 推荐 | Token级 | 中 | 关键词定位、精确匹配 |
| **ColBERT** | 中 | 高 | 强烈推荐 | Token级 | 高 | 语义对齐、改写检测 |

**选择建议**：
- 需要快速响应且关注关键词匹配 → **Sparse Attribution**
- 需要理解语义对齐和同义改写 → **ColBERT Attribution**
- 处理长文档或需要段落级归因 → **Segmented Attribution**
- 追求极致性能且硬件有限 → **Sparse Attribution**（无需构建大矩阵）
- 追求最佳效果且有 GPU 资源 → **ColBERT Attribution**（语义理解最强）
