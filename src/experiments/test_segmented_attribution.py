"""测试脚本：使用 SegmentedAttribution 提取相似文档中的关键片段

这个脚本展示：
1. 使用 RandomQuerySampler 随机抽取一个文档作为查询
2. 获取最相似的 Top 3 文档
3. 使用 SegmentedAttribution 提取每个相似文档中与查询文档最相似的片段（Top 3 片段）

注意：
- 需要先启动 TEI 服务：
  docker run -p 8080:80 --rm -v $PWD/data:/data \
    ghcr.io/huggingface/text-embeddings-inference:cpu-1.2 \
    --model-id sentence-transformers/all-MiniLM-L6-v2
- ChromaDB 仅用于向量存储和检索，不用于向量化
- SegmentedAttribution 通过 TEI 服务进行向量化
"""

import sys
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.stores.chroma_store import ChromaStore
from src.data_pipeline.samplers import RandomQuerySampler
from src.attribution.segmented.method import SegmentedAttribution
from src.data_pipeline.vectorizers.tei_vectorizer import TEIVectorizer


def print_separator(char="=", length=100):
    """打印分隔线"""
    print(char * length)


def print_query_document(doc_id, document, metadata):
    """打印查询文档信息

    Args:
        doc_id: 文档ID
        document: 文档文本
        metadata: 文档元数据
    """
    print_separator("=")
    print("🎯 查询文档（随机抽取）")
    print_separator("=")
    print(f"文档ID: {doc_id}")
    print(f"文档长度: {len(document)} 字符")

    # 打印元数据
    print(f"\n元数据:")
    for key, value in metadata.items():
        print(f"  - {key}: {value}")

    # 打印完整文档文本
    print(f"\n文档文本:")
    print_separator("·", 100)
    print(document)
    print_separator("·", 100)
    print()


def print_similar_document_with_segments(
    doc_index,
    doc_id,
    document,
    metadata,
    distance,
    attribution_result
):
    """打印相似文档及其最相似片段

    Args:
        doc_index: 文档索引
        doc_id: 文档ID
        document: 文档文本
        metadata: 文档元数据
        distance: 相似度距离
        attribution_result: SegmentedAttribution结果
    """
    print_separator("=")
    print(f"📄 相似文档 #{doc_index}")
    print_separator("=")
    print(f"文档ID: {doc_id}")
    print(f"文档长度: {len(document)} 字符")
    print(f"相似度距离: {distance:.6f}")

    # 计算相似度分数
    similarity_score = (1 - distance / 2) * 100
    print(f"相似度分数: {similarity_score:.2f}%")

    # 打印元数据
    print(f"\n元数据:")
    for key, value in metadata.items():
        print(f"  - {key}: {value}")

    # 打印完整文档文本
    print(f"\n文档文本:")
    print_separator("·", 100)
    print(document)
    print_separator("·", 100)

    # 打印归因片段
    print(f"\n🔍 最相似的片段 (Top 3):")
    print_separator("-", 100)

    top_spans = attribution_result.spans[:3]
    for i, span in enumerate(top_spans, 1):
        print(f"\n片段 {i}:")
        print(f"  相似度分数: {span.score:.4f}")
        print(f"  位置: [{span.start_idx}:{span.end_idx}]")
        print(f"  长度: {len(span.text)} 字符")
        print(f"  内容:")
        print_separator("·", 100)
        # 打印完整片段内容
        print(f"  {span.text}")
        print_separator("·", 100)

    print()


def main():
    """主函数"""

    # ============== 配置参数 ==============
    PERSIST_DIR = "./chroma_db"
    COLLECTION_NAME = "xingqiu_chuangye"  # 可修改为其他集合
    N_RESULTS = 1  # 返回最相似的3个文档
    RANDOM_SEED = None  # 设置为None则每次随机，设置数字则可重复

    # TEI 服务配置
    TEI_API_URL = "http://localhost:8080/embed"
    TEI_BATCH_SIZE = 64
    TEI_TIMEOUT = 60
    TEI_DIMENSION = 384  # all-MiniLM-L6-v2 的维度是 384

    # SegmentedAttribution 配置
    SEGMENTATION_METHOD = "fixed_length"  # "fixed_length" 或 "fixed_sentences"
    CHUNK_SIZE = 50  # token数量（中文≈50字，英文≈50词）
    CHUNK_OVERLAP = 10  # 重叠token数
    NUM_SENTENCES = 3  # 句子分段时每段句子数

    print_separator("=")
    print("📊 SegmentedAttribution 相似片段提取测试")
    print_separator("=")
    print(f"配置:")
    print(f"  集合名称: {COLLECTION_NAME}")
    print(f"  相似文档数量: {N_RESULTS}")
    print(f"  随机种子: {RANDOM_SEED if RANDOM_SEED else '随机'}")
    print(f"  TEI 服务: {TEI_API_URL}")
    print(f"  分段方法: {SEGMENTATION_METHOD}")
    print(f"  分块大小: {CHUNK_SIZE} tokens")
    print(f"  分块重叠: {CHUNK_OVERLAP} tokens")
    if SEGMENTATION_METHOD == "fixed_sentences":
        print(f"  每段句子数: {NUM_SENTENCES}")
    print_separator("=")
    print()

    try:
        # ============== 1. 初始化 ChromaStore ==============
        chroma_store = ChromaStore(persist_directory=PERSIST_DIR)

        # 验证集合存在
        collections = chroma_store.list_collections()
        if COLLECTION_NAME not in collections:
            print(f"✗ 错误: 集合 '{COLLECTION_NAME}' 不存在!")
            print(f"可用集合: {', '.join(collections)}")
            return

        # 获取集合
        collection = chroma_store.get_collection(COLLECTION_NAME)
        total_docs = collection.count()

        # ============== 2. 创建采样器 ==============
        sampler = RandomQuerySampler(chroma_store, random_seed=RANDOM_SEED)

        # ============== 3. 执行采样和查询 ==============
        results = sampler.sample_and_query(
            collection_name=COLLECTION_NAME,
            n_results=N_RESULTS
        )

        # ============== 4. 提取查询结果 ==============
        ids = results['ids'][0]
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]

        # 查询文档（第一个）
        query_id = ids[0]
        query_doc = documents[0]
        query_metadata = metadatas[0]

        # 打印查询文档
        print_query_document(query_id, query_doc, query_metadata)

        # ============== 5. 初始化 TEI Vectorizer ==============
        vectorizer = TEIVectorizer(
            api_url=TEI_API_URL,
            batch_size=TEI_BATCH_SIZE,
            max_retries=3,
            timeout=TEI_TIMEOUT,
            dimension=TEI_DIMENSION
        )

        # 健康检查
        if not vectorizer.health_check():
            print(f"✗ 错误: TEI 服务健康检查失败！")
            print(f"请确保 TEI 服务运行在 {TEI_API_URL}")
            return

        # ============== 6. 初始化 SegmentedAttribution ==============
        attribution_config = {
            "segmentation_method": SEGMENTATION_METHOD,
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "num_sentences": NUM_SENTENCES,
            "vectorizer": vectorizer  # 使用 TEI vectorizer
        }

        attribution = SegmentedAttribution(attribution_config)

        # ============== 7. 对每个相似文档提取相似片段 ==============
        print_separator("=")
        print(f"🔬 分析 Top {N_RESULTS} 相似文档的最相似片段")
        print_separator("=")
        print()

        for i in range(1, len(ids)):  # 从第二个文档开始（第一个是查询文档）
            doc_id = ids[i]
            document = documents[i]
            metadata = metadatas[i]
            distance = distances[i]

            # 使用 SegmentedAttribution 提取相似片段
            attribution_result = attribution.extract(query_doc, document)

            # 打印文档及其最相似片段
            print_similar_document_with_segments(
                doc_index=i,
                doc_id=doc_id,
                document=document,
                metadata=metadata,
                distance=distance,
                attribution_result=attribution_result
            )

        # ============== 8. 总结 ==============
        print_separator("=")
        print("✅ 分析完成")
        print_separator("=")
        print(f"查询文档ID: {query_id}")
        print(f"查询文档长度: {len(query_doc)} 字符")
        print(f"相似文档数量: {len(ids) - 1}")
        print(f"距离范围: {distances[1]:.6f} ~ {distances[-1]:.6f}")
        print_separator("=")

    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
