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
import argparse
from pathlib import Path
from typing import List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.stores.chroma_store import ChromaStore
from src.data_pipeline.samplers import RandomQuerySampler
from src.attribution.segmented.method import SegmentedAttribution
from src.data_pipeline.vectorizers.tei_vectorizer import TEIVectorizer


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="使用 SegmentedAttribution 提取相似文档中的关键片段",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认参数
  uv run python src/experiments/test_segmented_attribution.py

  # 自定义分段参数
  uv run python src/experiments/test_segmented_attribution.py --chunk-size 100 --chunk-overlap 20

  # 使用句子分段
  uv run python src/experiments/test_segmented_attribution.py --segmentation-method fixed_sentences --num-sentences 3

  # 自定义集合和结果数量
  uv run python src/experiments/test_segmented_attribution.py --collection my_collection --n 5
        """
    )

    # 基本配置参数
    parser.add_argument(
        "--collection",
        type=str,
        default="xingqiu_chuangye",
        help="ChromaDB 集合名称 (默认: xingqiu_chuangye)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="返回最相似的文档数量 (默认: 1)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="随机种子，用于可重复的结果 (默认: None，每次随机)"
    )
    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./chroma_db",
        help="ChromaDB 持久化目录 (默认: ./chroma_db)"
    )

    # SegmentedAttribution 配置
    parser.add_argument(
        "--segmentation-method",
        type=str,
        choices=["fixed_length", "fixed_sentences"],
        default="fixed_length",
        help="分段方法 (默认: fixed_length)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="分块大小（token数量，中文≈50字，英文≈50词） (默认: 50)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=10,
        help="分块重叠的token数量 (默认: 10)"
    )
    parser.add_argument(
        "--num-sentences",
        type=int,
        default=3,
        help="句子分段时每段的句子数量（仅在 segmentation-method=fixed_sentences 时使用） (默认: 3)"
    )

    return parser.parse_args()


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

    # 解析命令行参数
    args = parse_arguments()

    # TEI 服务配置（硬编码）
    TEI_API_URL = "http://localhost:8080/embed"
    TEI_BATCH_SIZE = 64
    TEI_TIMEOUT = 60
    TEI_DIMENSION = 384  # all-MiniLM-L6-v2 的维度是 384

    print_separator("=")
    print("📊 SegmentedAttribution 相似片段提取测试")
    print_separator("=")
    print(f"配置:")
    print(f"  集合名称: {args.collection}")
    print(f"  相似文档数量: {args.n}")
    print(f"  随机种子: {args.random_seed if args.random_seed else '随机'}")
    print(f"  TEI 服务: {TEI_API_URL}")
    print(f"  分段方法: {args.segmentation_method}")
    print(f"  分块大小: {args.chunk_size} tokens")
    print(f"  分块重叠: {args.chunk_overlap} tokens")
    if args.segmentation_method == "fixed_sentences":
        print(f"  每段句子数: {args.num_sentences}")
    print_separator("=")
    print()

    try:
        # ============== 1. 初始化 ChromaStore ==============
        chroma_store = ChromaStore(persist_directory=args.persist_dir)

        # 验证集合存在
        collections = chroma_store.list_collections()
        if args.collection not in collections:
            print(f"✗ 错误: 集合 '{args.collection}' 不存在!")
            print(f"可用集合: {', '.join(collections)}")
            return

        # 获取集合
        collection = chroma_store.get_collection(args.collection)
        total_docs = collection.count()

        # ============== 2. 创建采样器 ==============
        sampler = RandomQuerySampler(chroma_store, random_seed=args.random_seed)

        # ============== 3. 执行采样和查询 ==============
        results = sampler.sample_and_query(
            collection_name=args.collection,
            n_results=args.n
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
            "segmentation_method": args.segmentation_method,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "num_sentences": args.num_sentences,
            "vectorizer": vectorizer  # 使用 TEI vectorizer
        }

        attribution = SegmentedAttribution(attribution_config)

        # ============== 7. 对每个相似文档提取相似片段 ==============
        print_separator("=")
        print(f"🔬 分析 Top {args.n} 相似文档的最相似片段")
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
