"""测试脚本：使用 SparseAttribution 在 Chroma 相似检索结果上做归因

这个脚本对齐 `src/experiments/test_segmented_attribution.py` 的测试方式：
1) 从 Chroma collection 随机抽取一个文档作为查询（Query）
2) 使用同一 collection 的向量检索拿到 Top-N 相似文档（通常包含 Query 自身）
3) 对每个相似文档（除 Query）执行 SparseAttribution，输出：
   - token-level 的 top contributing tokens
   - sliding window 的 top spans

注意：
- 需要安装 FlagEmbedding: pip install FlagEmbedding
- 首次运行会自动下载/加载 BGE-M3 模型（体积较大）
- 默认从 `config/attribution.yaml` 读取 sparse 配置作为默认值（CLI 会覆盖）
"""

import sys
import argparse
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_pipeline.stores.chroma_store import ChromaStore
from src.data_pipeline.samplers import RandomQuerySampler
from src.attribution.token_wise import SparseAttribution


def load_config():
    """加载配置文件"""
    config_path = project_root / "config" / "attribution.yaml"
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('sparse', {})
    return {}


def parse_arguments(config_defaults=None):
    """解析命令行参数
    
    Args:
        config_defaults: 从配置文件读取的默认值字典
    """
    if config_defaults is None:
        config_defaults = {}
    
    parser = argparse.ArgumentParser(
        description="使用 SparseAttribution 在 Chroma 相似检索结果上做归因",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认参数运行（从 Chroma collection 抽取数据）
  uv run python src/experiments/test_sparse_attribution.py

  # 指定 collection / 返回相似文档数量
  uv run python src/experiments/test_sparse_attribution.py --collection xingqiu_chuangye --n 5

  # 调整 sparse sliding window 参数
  uv run python src/experiments/test_sparse_attribution.py --window-size 50 --window-overlap 40

  # 显示更多 top tokens（打印用）
  uv run python src/experiments/test_sparse_attribution.py --top-n 20
        """
    )

    # 基本配置参数（对齐 segmented test）
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
        help="返回最相似的文档数量 (默认: 1，通常会包含 query 自身)"
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

    # SparseAttribution 配置（使用配置文件的值作为默认值）
    parser.add_argument(
        "--model-name",
        type=str,
        default=config_defaults.get("model_name", "BAAI/bge-m3"),
        help=f"BGE-M3 model name or path (默认: {config_defaults.get('model_name', 'BAAI/bge-m3')})"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=config_defaults.get("window_size", 50),
        help=f"滑动窗口 token 数量 (默认: {config_defaults.get('window_size', 50)})"
    )
    parser.add_argument(
        "--window-overlap",
        type=int,
        default=config_defaults.get("window_overlap", 40),
        help=f"滑动窗口重叠 token 数量 (默认: {config_defaults.get('window_overlap', 40)})"
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="打印 Top-N 贡献最高的 token (默认: 10)"
    )
    parser.add_argument(
        "--top-k-spans",
        type=int,
        default=config_defaults.get("top_k_spans", 5),
        help=f"返回 Top-K 贡献最高的片段 (默认: {config_defaults.get('top_k_spans', 5)})"
    )
    parser.add_argument(
        "--no-fp16",
        action="store_true",
        help=f"禁用半精度计算（默认{'启用' if config_defaults.get('use_fp16', True) else '禁用'} fp16）"
    )

    return parser.parse_args()


def print_separator(char="=", length=100):
    """打印分隔线"""
    print(char * length)


def print_query_document(doc_id, document, metadata):
    """打印查询文档信息"""
    print_separator("=")
    print("🎯 查询文档（随机抽取）")
    print_separator("=")
    print(f"文档ID: {doc_id}")
    print(f"文档长度: {len(document)} 字符")

    print("\n元数据:")
    for key, value in metadata.items():
        print(f"  - {key}: {value}")

    print("\n文档文本:")
    print_separator("·", 100)
    print(document)
    print_separator("·", 100)
    print()


def print_similar_document_with_sparse_attribution(
    doc_index,
    doc_id,
    document,
    metadata,
    distance,
    attribution_result,
    top_tokens,
):
    """打印相似文档及其 sparse attribution 结果"""
    print_separator("=")
    print(f"📄 相似文档 #{doc_index}")
    print_separator("=")
    print(f"文档ID: {doc_id}")
    print(f"文档长度: {len(document)} 字符")
    print(f"相似度距离: {distance:.6f}")

    similarity_score = (1 - distance / 2) * 100
    print(f"相似度分数: {similarity_score:.2f}%")

    print("\n元数据:")
    for key, value in metadata.items():
        print(f"  - {key}: {value}")

    print("\n文档文本:")
    print_separator("·", 100)
    print(document)
    print_separator("·", 100)

    print("\n🏆 Top Contributing Tokens:")
    print_separator("-", 100)
    if top_tokens:
        print(f"\n{'排名':<6}{'Token':<20}{'Score':<15}{'Normalized':<15}")
        print_separator("-", 60)
        for i, token_info in enumerate(top_tokens, 1):
            print(
                f"{i:<6}"
                f"{token_info['token']:<20}"
                f"{token_info['score']:<15.6f}"
                f"{token_info['normalized_score']:<15.4f}"
            )
    else:
        print("⚠️ 没有找到共同的 token")

    print("\n📑 Top Attribution Spans (Sliding Window):")
    print_separator("-", 100)
    print("归因元信息:")
    print(f"  - total_lexical_score: {attribution_result.metadata.get('total_lexical_score', 0.0):.6f}")
    print(f"  - num_contributing_tokens: {attribution_result.metadata.get('num_contributing_tokens', 0)}")
    print(f"  - total_windows_analyzed: {attribution_result.metadata.get('total_windows_analyzed', 0)}")
    print(f"  - window_size: {attribution_result.metadata.get('window_size', None)}")
    print(f"  - window_overlap: {attribution_result.metadata.get('window_overlap', None)}")

    if not attribution_result.spans:
        print("\n⚠️ 未提取到 spans（可能没有共同 token 或文本过短）")
        print()
        return

    for i, span in enumerate(attribution_result.spans, 1):
        print(f"\n【片段 {i}】")
        print(f"  位置: [{span.start_idx}:{span.end_idx}]")
        print(f"  归一化分数: {span.score:.4f}")
        if span.metadata and "raw_score" in span.metadata:
            print(f"  原始分数: {span.metadata['raw_score']:.6f}")
        if span.metadata and "token_count" in span.metadata:
            print(f"  窗口 Token 数: {span.metadata['token_count']}")
        if span.metadata and "contributing_tokens" in span.metadata:
            print(f"  贡献 Token 数: {span.metadata['contributing_tokens']}")
        print("  内容:")
        print_separator("·", 100)
        print(f"  {span.text}")
        print_separator("·", 100)

    print()


def main():
    """主函数"""
    # 加载配置文件
    config_defaults = load_config()
    
    # 解析命令行参数（命令行参数会覆盖配置文件）
    args = parse_arguments(config_defaults)

    print_separator("=")
    print("🔬 SparseAttribution (Chroma) 相似文档归因测试")
    print_separator("=")
    print(f"配置:")
    print(f"  滑动窗口大小: {args.window_size} tokens")
    print(f"  滑动窗口重叠: {args.window_overlap} tokens")
    print(f"  打印 Top-N tokens: {args.top_n}")
    print(f"  Top-K 片段: {args.top_k_spans}")
    print(f"  使用 FP16: {not args.no_fp16}")
    print_separator("=")
    print()

    try:
        # ============== 1. 初始化 ChromaStore ==============
        chroma_store = ChromaStore(persist_directory=args.persist_dir)

        collections = chroma_store.list_collections()
        if args.collection not in collections:
            print(f"✗ 错误: 集合 '{args.collection}' 不存在!")
            print(f"可用集合: {', '.join(collections)}")
            return

        collection = chroma_store.get_collection(args.collection)
        total_docs = collection.count()
        if total_docs <= 0:
            print(f"✗ 错误: 集合 '{args.collection}' 为空（count=0）")
            return

        # ============== 2. 创建采样器并执行采样/检索 ==============
        sampler = RandomQuerySampler(chroma_store, random_seed=args.random_seed)
        results = sampler.sample_and_query(
            collection_name=args.collection,
            n_results=args.n
        )

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        if not ids or not documents:
            print("✗ 错误: 检索结果为空")
            return

        query_id = ids[0]
        query_doc = documents[0] or ""
        query_metadata = metadatas[0] or {}

        if not query_doc.strip():
            print(f"✗ 错误: Query 文档内容为空 (id={query_id})")
            return

        print_query_document(query_id, query_doc, query_metadata)

        # ============== 3. 初始化 SparseAttribution ==============
        print("📥 正在加载 BGE-M3 模型...")
        
        # 构建配置：优先使用命令行参数，回退到配置文件
        config = {
            "model_name": args.model_name,
            "use_fp16": config_defaults.get("use_fp16", True) if not args.no_fp16 else False,
            "window_size": args.window_size,
            "window_overlap": args.window_overlap,
            "top_k_spans": args.top_k_spans,
        }
        
        # 如果配置文件指定了 device，也加入配置
        if "device" in config_defaults:
            config["device"] = config_defaults["device"]
        
        attribution = SparseAttribution(config)
        print("✓ 模型加载完成")
        print()

        # ============== 4. 对每个相似文档执行 sparse attribution ==============
        print_separator("=")
        print(f"🔬 分析 Top {args.n} 相似文档的 sparse attribution")
        print_separator("=")
        print()

        # i=0 是 query 本身，从 i=1 开始才是相似文档
        for i in range(1, len(ids)):
            doc_id = ids[i]
            document = documents[i] or ""
            metadata = metadatas[i] or {}
            distance = distances[i]

            if not document.strip():
                print(f"⚠️ 跳过空文档: id={doc_id}")
                continue

            # token-level 打印用 top tokens
            top_tokens = attribution.get_top_contributing_tokens(
                query_doc, document, top_n=args.top_n
            )

            # spans：sliding window attribution
            attribution_result = attribution.extract(query_doc, document)

            print_similar_document_with_sparse_attribution(
                doc_index=i,
                doc_id=doc_id,
                document=document,
                metadata=metadata,
                distance=distance,
                attribution_result=attribution_result,
                top_tokens=top_tokens,
            )

        # ============== 5. 总结 ==============
        print_separator("=")
        print("✅ 分析完成")
        print_separator("=")
        print(f"collection: {args.collection}")
        print(f"query_id: {query_id}")
        print(f"query_length: {len(query_doc)} chars")
        print(f"similar_docs_analyzed: {max(0, len(ids) - 1)}")
        if len(distances) > 1:
            print(f"distance_range: {distances[1]:.6f} ~ {distances[-1]:.6f}")
        print_separator("=")

    except Exception as e:
        print(f"\n✗ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
