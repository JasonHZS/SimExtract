"""Tests for SegmentedAttribution implementation.

This module tests the SegmentedAttribution method with a mock vectorizer,
validating segmentation strategies and similarity computation.

Run with:
    python -m pytest tests/test_segmented_attribution.py -v
"""

import pytest
import random
from typing import List

from src.attribution.segmented.method import SegmentedAttribution
from src.attribution.base import AttributionResult


# ============== Mock Classes ==============

class MockVectorizer:
    """Mock vectorizer for testing without TEI service."""

    def __init__(self, dimension=1024):
        self.dimension = dimension

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return mock embeddings based on text hash for determinism."""
        embeddings = []
        for text in texts:
            random.seed(hash(text) % (2**32))
            embedding = [random.random() for _ in range(self.dimension)]
            embeddings.append(embedding)
        return embeddings

    def get_dimension(self) -> int:
        return self.dimension

    def health_check(self) -> bool:
        return True


# ============== Fixtures ==============

@pytest.fixture
def mock_vectorizer():
    """Create a mock vectorizer."""
    return MockVectorizer()


@pytest.fixture
def fixed_length_attribution(mock_vectorizer):
    """Create SegmentedAttribution with fixed_length segmentation."""
    config = {
        "segmentation_method": "fixed_length",
        "chunk_size": 30,
        "chunk_overlap": 10,
        "vectorizer": mock_vectorizer
    }
    return SegmentedAttribution(config)


@pytest.fixture
def sentence_attribution(mock_vectorizer):
    """Create SegmentedAttribution with sentence-based segmentation."""
    config = {
        "segmentation_method": "fixed_sentences",
        "num_sentences": 3,
        "vectorizer": mock_vectorizer
    }
    return SegmentedAttribution(config)


# ============== Tests ==============

class TestFixedLengthSegmentation:
    """Tests for fixed-length segmentation."""

    def test_chinese_text(self, fixed_length_attribution):
        """Test with Chinese text using fixed_length segmentation."""
        text_a = "人工智能正在改变医疗行业"
        text_b = """机器学习在医学诊断中的应用正在革新患者护理。
今天天气很好，阳光明媚。
人工智能技术帮助医生做出更准确的诊断。
深度学习模型可以分析医学影像数据。
未来医疗将更加智能化和个性化。"""

        result = fixed_length_attribution.extract(text_a, text_b)

        print(f"\nTest: Chinese fixed-length segmentation")
        print(f"  Source text: {text_a}")
        print(f"  Target text length: {len(text_b)} chars")
        print(f"  Number of segments: {len(result.spans)}")
        print(f"  Best segment score: {result.spans[0].score:.4f}")

        # Validate result structure
        assert isinstance(result, AttributionResult)
        assert result.text_a == text_a
        assert result.text_b == text_b
        assert len(result.spans) > 0

        # Validate spans
        for span in result.spans:
            assert span.start_idx >= 0
            assert span.end_idx <= len(text_b)
            assert span.score >= 0.0
            assert span.text == text_b[span.start_idx:span.end_idx]

        # Show top segments
        print(f"\n  Top 3 segments:")
        for i, span in enumerate(result.spans[:3]):
            print(f"    {i+1}. Score: {span.score:.4f}")
            print(f"       Position: [{span.start_idx}:{span.end_idx}]")
            print(f"       Text: {span.text[:50]}...")

    def test_english_text(self, fixed_length_attribution):
        """Test with English text using fixed_length segmentation."""
        text_a = "Artificial intelligence is transforming the healthcare industry"
        text_b = """Machine learning applications in medical diagnosis are revolutionizing patient care.
The weather is nice today.
Doctors are increasingly relying on AI-powered tools.
Deep learning models can analyze medical imaging data with high accuracy.
The future of healthcare will be more intelligent and personalized."""

        result = fixed_length_attribution.extract(text_a, text_b)

        print(f"\nTest: English fixed-length segmentation")
        print(f"  Source text: {text_a}")
        print(f"  Target text length: {len(text_b)} chars")
        print(f"  Number of segments: {len(result.spans)}")
        print(f"  Best segment score: {result.spans[0].score:.4f}")

        assert isinstance(result, AttributionResult)
        assert len(result.spans) > 0

        print(f"\n  Top 3 segments:")
        for i, span in enumerate(result.spans[:3]):
            print(f"    {i+1}. Score: {span.score:.4f}")
            print(f"       Position: [{span.start_idx}:{span.end_idx}]")
            print(f"       Text: {span.text[:80]}...")

    def test_mixed_language(self, fixed_length_attribution):
        """Test with mixed Chinese and English text."""
        text_a = "Deep learning for medical image analysis"
        text_b = """深度学习用于医学影像分析。Deep neural networks can classify images with high accuracy.
这是计算机视觉的突破。AI models process medical scans efficiently.
人工智能提高诊断准确性。"""

        result = fixed_length_attribution.extract(text_a, text_b)

        print(f"\nTest: Mixed language text")
        print(f"  Source text: {text_a}")
        print(f"  Target text (mixed language): {len(text_b)} chars")
        print(f"  Number of segments: {len(result.spans)}")

        assert isinstance(result, AttributionResult)
        assert len(result.spans) > 0

        print(f"\n  Top 3 segments:")
        for i, span in enumerate(result.spans[:3]):
            print(f"    {i+1}. Score: {span.score:.4f}")
            print(f"       Text: {span.text[:60]}...")


class TestSentenceSegmentation:
    """Tests for sentence-based segmentation."""

    def test_chinese_sentences(self, sentence_attribution):
        """Test with Chinese text using sentence segmentation."""
        text_a = "人工智能在医疗领域的应用"
        text_b = """机器学习正在革新医疗诊断。人工智能帮助医生分析病例。深度学习模型处理医学影像。
今天天气很好。阳光明媚。适合户外活动。
智能医疗系统提高诊断准确性。AI技术支持个性化治疗方案。未来医疗将更加智能化。"""

        result = sentence_attribution.extract(text_a, text_b)

        print(f"\nTest: Chinese sentence segmentation")
        print(f"  Source text: {text_a}")
        print(f"  Target text length: {len(text_b)} chars")
        print(f"  Number of segments: {len(result.spans)}")
        print(f"  Best segment score: {result.spans[0].score:.4f}")

        assert isinstance(result, AttributionResult)
        assert len(result.spans) > 0

        print(f"\n  All segments:")
        for i, span in enumerate(result.spans):
            print(f"    {i+1}. Score: {span.score:.4f}")
            print(f"       Text: {span.text[:60]}...")


class TestBatchProcessing:
    """Tests for batch processing."""

    def test_batch_extract(self, sentence_attribution):
        """Test batch processing of multiple text pairs."""
        pairs = [
            ("人工智能改变医疗", "机器学习在医学诊断中应用。AI帮助医生。深度学习处理影像。"),
            ("气候变化影响环境", "全球变暖导致海平面上升。极端天气频发。生态系统受损。"),
            ("科技发展很快", "人工智能快速发展。量子计算取得突破。区块链技术应用广泛。"),
        ]

        results = sentence_attribution.batch_extract(pairs)

        print(f"\nTest: Batch processing")
        print(f"  Processed {len(results)} text pairs")

        assert len(results) == len(pairs)

        for i, result in enumerate(results):
            print(f"\n  Pair {i+1}:")
            print(f"    Source: {result.text_a}")
            print(f"    Best segment score: {result.spans[0].score:.4f}")
            print(f"    Best segment: {result.spans[0].text[:40]}...")

            assert isinstance(result, AttributionResult)
            assert result.text_a == pairs[i][0]
            assert result.text_b == pairs[i][1]
            assert len(result.spans) > 0


class TestResultStructure:
    """Tests for result structure validation."""

    def test_spans_sorted_by_score(self, fixed_length_attribution):
        """Test that spans are sorted by score in descending order."""
        text_a = "人工智能"
        text_b = "机器学习在医学诊断中的应用。人工智能技术帮助医生。深度学习模型分析数据。"

        result = fixed_length_attribution.extract(text_a, text_b)

        # Verify spans are sorted by score (descending)
        scores = [span.score for span in result.spans]
        assert scores == sorted(scores, reverse=True), "Spans should be sorted by score descending"

    def test_span_positions_valid(self, fixed_length_attribution):
        """Test that span positions are valid within text_b."""
        text_a = "测试查询"
        text_b = "这是一段测试文本，用于验证分段功能是否正常工作。"

        result = fixed_length_attribution.extract(text_a, text_b)

        for span in result.spans:
            assert 0 <= span.start_idx < len(text_b), f"Invalid start_idx: {span.start_idx}"
            assert 0 < span.end_idx <= len(text_b), f"Invalid end_idx: {span.end_idx}"
            assert span.start_idx < span.end_idx, "start_idx should be less than end_idx"

    def test_method_name_correct(self, fixed_length_attribution):
        """Test that method_name is correctly set."""
        text_a = "查询"
        text_b = "文档内容"

        result = fixed_length_attribution.extract(text_a, text_b)

        assert result.method_name == "SegmentedAttribution"


class TestEdgeCases:
    """Tests for edge cases."""

    def test_short_text_b(self, fixed_length_attribution):
        """Test with very short text_b."""
        text_a = "人工智能"
        text_b = "AI技术"

        result = fixed_length_attribution.extract(text_a, text_b)

        assert isinstance(result, AttributionResult)
        assert len(result.spans) >= 1

    def test_single_segment(self, mock_vectorizer):
        """Test when text_b results in single segment."""
        config = {
            "segmentation_method": "fixed_length",
            "chunk_size": 100,  # Large chunk size
            "chunk_overlap": 0,
            "vectorizer": mock_vectorizer
        }
        attribution = SegmentedAttribution(config)

        text_a = "查询"
        text_b = "短文本"

        result = attribution.extract(text_a, text_b)

        assert isinstance(result, AttributionResult)
        assert len(result.spans) >= 1
