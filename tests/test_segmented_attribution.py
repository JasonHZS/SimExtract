"""Test script for SegmentedAttribution implementation.

This script demonstrates the SegmentedAttribution method with a mock vectorizer.
"""

import sys
import os
import logging
from typing import List

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.attribution.segmented.method import SegmentedAttribution
from src.attribution.base import AttributionResult


class MockVectorizer:
    """Mock vectorizer for testing without TEI service."""

    def __init__(self, dimension=1024):
        self.dimension = dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return mock embeddings based on text length."""
        import random
        embeddings = []
        for text in texts:
            # Generate deterministic "embeddings" based on text
            random.seed(hash(text) % (2**32))
            embedding = [random.random() for _ in range(self.dimension)]
            embeddings.append(embedding)
        return embeddings

    def get_dimension(self) -> int:
        return self.dimension

    def health_check(self) -> bool:
        return True


def test_chinese_fixed_length():
    """Test with Chinese text using fixed_length segmentation."""
    print("\n=== Test 1: Chinese Text with Fixed-Length Segmentation ===")

    config = {
        "segmentation_method": "fixed_length",
        "chunk_size": 30,
        "chunk_overlap": 10,
        "vectorizer": MockVectorizer()
    }

    attribution = SegmentedAttribution(config)

    text_a = "人工智能正在改变医疗行业"
    text_b = """机器学习在医学诊断中的应用正在革新患者护理。
今天天气很好，阳光明媚。
人工智能技术帮助医生做出更准确的诊断。
深度学习模型可以分析医学影像数据。
未来医疗将更加智能化和个性化。"""

    result = attribution.extract(text_a, text_b)

    print(f"Source text: {text_a}")
    print(f"Target text length: {len(text_b)} chars")
    print(f"Number of segments: {len(result.spans)}")
    print(f"Best segment score: {result.spans[0].score:.4f}")
    print(f"\nTop 3 segments:")
    for i, span in enumerate(result.spans[:3]):
        print(f"  {i+1}. Score: {span.score:.4f}")
        print(f"     Position: [{span.start_idx}:{span.end_idx}]")
        print(f"     Text: {span.text[:50]}...")


def test_english_fixed_length():
    """Test with English text using fixed_length segmentation."""
    print("\n=== Test 2: English Text with Fixed-Length Segmentation (Word-based) ===")

    config = {
        "segmentation_method": "fixed_length",
        "chunk_size": 30,  # 30 words
        "chunk_overlap": 10,  # 10 words
        "vectorizer": MockVectorizer()
    }

    attribution = SegmentedAttribution(config)

    text_a = "Artificial intelligence is transforming the healthcare industry"
    text_b = """Machine learning applications in medical diagnosis are revolutionizing patient care.
The weather is nice today.
Doctors are increasingly relying on AI-powered tools.
Deep learning models can analyze medical imaging data with high accuracy.
The future of healthcare will be more intelligent and personalized."""

    result = attribution.extract(text_a, text_b)

    print(f"Source text: {text_a}")
    print(f"Target text length: {len(text_b)} chars")
    print(f"Number of segments: {len(result.spans)}")
    print(f"Best segment score: {result.spans[0].score:.4f}")
    print(f"\nTop 3 segments:")
    for i, span in enumerate(result.spans[:3]):
        print(f"  {i+1}. Score: {span.score:.4f}")
        print(f"     Position: [{span.start_idx}:{span.end_idx}]")
        print(f"     Text: {span.text[:80]}...")


def test_sentence_segmentation():
    """Test with sentence-based segmentation."""
    print("\n=== Test 3: Chinese Text with Sentence-Based Segmentation ===")

    config = {
        "segmentation_method": "fixed_sentences",
        "num_sentences": 3,
        "vectorizer": MockVectorizer()
    }

    attribution = SegmentedAttribution(config)

    text_a = "人工智能在医疗领域的应用"
    text_b = """机器学习正在革新医疗诊断。人工智能帮助医生分析病例。深度学习模型处理医学影像。
今天天气很好。阳光明媚。适合户外活动。
智能医疗系统提高诊断准确性。AI技术支持个性化治疗方案。未来医疗将更加智能化。"""

    result = attribution.extract(text_a, text_b)

    print(f"Source text: {text_a}")
    print(f"Target text length: {len(text_b)} chars")
    print(f"Number of segments: {len(result.spans)}")
    print(f"Best segment score: {result.spans[0].score:.4f}")
    print(f"\nAll segments:")
    for i, span in enumerate(result.spans):
        print(f"  {i+1}. Score: {span.score:.4f}")
        print(f"     Text: {span.text[:60]}...")


def test_batch_processing():
    """Test batch processing."""
    print("\n=== Test 4: Batch Processing ===")

    config = {
        "segmentation_method": "fixed_sentences",
        "num_sentences": 3,
        "vectorizer": MockVectorizer()
    }

    attribution = SegmentedAttribution(config)

    pairs = [
        ("人工智能改变医疗", "机器学习在医学诊断中应用。AI帮助医生。深度学习处理影像。"),
        ("气候变化影响环境", "全球变暖导致海平面上升。极端天气频发。生态系统受损。"),
        ("科技发展很快", "人工智能快速发展。量子计算取得突破。区块链技术应用广泛。"),
    ]

    results = attribution.batch_extract(pairs)

    print(f"Processed {len(results)} text pairs")
    for i, result in enumerate(results):
        print(f"\n  Pair {i+1}:")
        print(f"    Source: {result.text_a}")
        print(f"    Best segment score: {result.spans[0].score:.4f}")
        print(f"    Best segment: {result.spans[0].text[:40]}...")


def test_mixed_language():
    """Test with mixed Chinese and English text."""
    print("\n=== Test 5: Mixed Language Text ===")

    config = {
        "segmentation_method": "fixed_length",
        "chunk_size": 30,
        "chunk_overlap": 10,
        "vectorizer": MockVectorizer()
    }

    attribution = SegmentedAttribution(config)

    text_a = "Deep learning for medical image analysis"
    text_b = """深度学习用于医学影像分析。Deep neural networks can classify images with high accuracy.
这是计算机视觉的突破。AI models process medical scans efficiently.
人工智能提高诊断准确性。"""

    result = attribution.extract(text_a, text_b)

    print(f"Source text: {text_a}")
    print(f"Target text (mixed language): {len(text_b)} chars")
    print(f"Number of segments: {len(result.spans)}")
    print(f"\nTop 3 segments:")
    for i, span in enumerate(result.spans[:3]):
        print(f"  {i+1}. Score: {span.score:.4f}")
        print(f"     Text: {span.text[:60]}...")


def main():
    """Run all tests."""
    print("=" * 70)
    print("SegmentedAttribution Implementation Test")
    print("=" * 70)

    try:
        test_chinese_fixed_length()
        test_english_fixed_length()
        test_sentence_segmentation()
        test_batch_processing()
        test_mixed_language()

        print("\n" + "=" * 70)
        print("All tests completed successfully! ✓")
        print("=" * 70)

    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
