"""Integration tests for ColBERTAttribution with real BGE-M3 model.

This test validates that ColBERTAttribution works correctly with the actual
BGE-M3 model. Unlike the mock tests in test_colbert_attribution.py, these tests
verify:

1. Model initialization and ColBERT vector generation
2. MaxSim score computation
3. Sliding window span extraction
4. Topic keyword extraction
5. Character position mapping accuracy

Requirements:
- FlagEmbedding library installed
- Local BGE-M3 model (configured in config/attribution.yaml)

Run with:
    CUDA_VISIBLE_DEVICES=1 python -m pytest tests/test_colbert_integration.py -v -s
"""

import pytest
import torch

from src.attribution.token_wise.colbert import ColBERTAttribution, WindowScoreResult


# ============== Fixtures ==============

@pytest.fixture(scope="module")
def colbert_config():
    """Configuration for ColBERT integration tests."""
    return {
        "model_name": "models/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
        "use_fp16": True,
        "device": "cuda:0",
        "window_size": 10,
        "window_overlap": 5,
        "top_k_spans": 3,
        "top_k_topic_keywords": 5,
    }


@pytest.fixture(scope="module")
def attribution(colbert_config):
    """Load ColBERTAttribution model once for all tests in this module."""
    print("\n" + "=" * 80)
    print("Loading BGE-M3 model for ColBERT integration tests...")
    print("=" * 80)

    try:
        attr = ColBERTAttribution(colbert_config)
        print(f"✓ Model loaded successfully")
        print(f"  Model: {colbert_config['model_name']}")
        print(f"  Window size: {attr.window_size}")
        print(f"  Window overlap: {attr.window_overlap}")
        print("=" * 80 + "\n")
        return attr
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        print("\nNOTE: This test requires:")
        print("  1. FlagEmbedding installed: pip install FlagEmbedding")
        print("  2. Local BGE-M3 model in models/ directory")
        print("=" * 80 + "\n")
        pytest.skip(f"Model not available: {e}")


# ============== Tests ==============

class TestModelInitialization:
    """Tests for model initialization."""

    def test_model_and_tokenizer_initialized(self, attribution):
        """Verify model and tokenizer are properly initialized."""
        assert attribution.model is not None
        assert attribution.tokenizer is not None
        assert hasattr(attribution.model, 'encode')
        assert hasattr(attribution.tokenizer, 'decode')


class TestColBERTEncoding:
    """Tests for ColBERT encoding."""

    def test_encoding_produces_correct_output_format(self, attribution):
        """Verify ColBERT encoding produces correct output format."""
        texts = ["机器学习", "machine learning"]

        colbert_vecs = attribution._encode_colbert(texts)

        print(f"\nTest: ColBERT encoding")
        print(f"  Input texts: {texts}")
        print(f"  Output vectors: {len(colbert_vecs)}")

        assert len(colbert_vecs) == 2, "Should return vectors for both texts"

        for idx, vecs in enumerate(colbert_vecs):
            print(f"\n  Text[{idx}]: '{texts[idx]}'")
            print(f"    Vector shape: {vecs.shape}")
            print(f"    Num tokens: {vecs.shape[0]}")
            print(f"    Embedding dim: {vecs.shape[1]}")

            assert len(vecs.shape) == 2, "Should be 2D tensor"
            assert vecs.shape[0] > 0, "Should have tokens"
            assert vecs.shape[1] > 0, "Should have embedding dim"


class TestColBERTScore:
    """Tests for ColBERT score computation."""

    def test_identical_texts_high_score(self, attribution):
        """Verify ColBERT score for identical texts is high."""
        text = "人工智能是重要的技术"

        score = attribution.compute_colbert_score(text, text)

        print(f"\nTest: ColBERT score for identical texts")
        print(f"  Text: '{text}'")
        print(f"  Score: {score:.6f}")

        assert score > 0.95, "Identical texts should have score > 0.95"
        assert score <= 1.0 + 1e-5, "Score should not exceed 1.0"

    def test_similar_texts_reasonable_score(self, attribution):
        """Verify ColBERT score for similar texts is reasonably high."""
        text_a = "机器学习技术"
        text_b = "机器学习方法和应用"

        score = attribution.compute_colbert_score(text_a, text_b)

        print(f"\nTest: ColBERT score for similar texts")
        print(f"  Text A: '{text_a}'")
        print(f"  Text B: '{text_b}'")
        print(f"  Score: {score:.6f}")

        assert score > 0.5, "Similar texts should have score > 0.5"

    def test_different_texts_lower_score(self, attribution):
        """Verify ColBERT score for different texts is lower."""
        text_a = "人工智能技术"
        text_b = "今天天气很好"

        score = attribution.compute_colbert_score(text_a, text_b)

        print(f"\nTest: ColBERT score for different texts")
        print(f"  Text A: '{text_a}'")
        print(f"  Text B: '{text_b}'")
        print(f"  Score: {score:.6f}")

        assert score < 0.8, "Different texts should have score < 0.8"


class TestExtractEndToEnd:
    """Tests for full extract() pipeline."""

    def test_extract_returns_valid_result(self, attribution):
        """Test full extract() pipeline with real model."""
        text_a = "人工智能和机器学习"
        text_b = "人工智能技术的发展推动了机器学习的进步，深度学习是重要方向。"

        result = attribution.extract(text_a, text_b)

        print(f"\nTest: Extract end-to-end")
        print(f"  Text A: '{text_a}'")
        print(f"  Text B: '{text_b}'")
        print(f"  Method: {result.method_name}")
        print(f"  Spans: {len(result.spans)}")
        print(f"  Metadata:")
        for key, value in result.metadata.items():
            if key != 'topic_keywords':
                print(f"    {key}: {value}")

        # Validate result structure
        assert result.text_a == text_a
        assert result.text_b == text_b
        assert result.method_name == "ColBERTAttribution"
        assert len(result.spans) > 0, "Should extract spans"
        assert len(result.spans) <= attribution.top_k_spans

        # Show extracted spans
        print(f"\n  Extracted spans:")
        for i, span in enumerate(result.spans):
            print(f"\n    Span {i+1}:")
            print(f"      Score: {span.score:.4f}")
            print(f"      Position: [{span.start_idx}:{span.end_idx}]")
            print(f"      Text: '{span.text}'")
            print(f"      Window rank: {span.metadata.get('window_rank')}")

            # Validate span
            assert 0.0 <= span.score <= 1.0
            assert span.start_idx >= 0
            assert span.end_idx <= len(text_b)
            assert text_b[span.start_idx:span.end_idx] == span.text

        # Validate metadata
        assert 'colbert_score' in result.metadata
        assert 'num_query_tokens' in result.metadata
        assert 'num_doc_tokens' in result.metadata
        assert 'topic_keywords' in result.metadata


class TestTopicKeywords:
    """Tests for topic keyword extraction."""

    def test_topic_keywords_extracted_correctly(self, attribution):
        """Test topic keywords are extracted correctly."""
        text_a = "机器学习"
        text_b = "机器学习是人工智能的一个重要分支，深度学习是机器学习的子领域。"

        result = attribution.extract(text_a, text_b)

        print(f"\nTest: Topic keywords extraction")
        print(f"  Text A: '{text_a}'")
        print(f"  Text B: '{text_b}'")

        keywords = result.metadata.get('topic_keywords', [])
        print(f"\n  Topic keywords ({len(keywords)}):")
        for kw in keywords:
            print(f"    '{kw['text']}': score={kw['score']:.4f}, "
                  f"pos=[{kw['start_idx']}:{kw['end_idx']}]")

        assert len(keywords) > 0, "Should extract topic keywords"
        assert len(keywords) <= attribution.top_k_topic_keywords

        # Validate keyword structure
        for kw in keywords:
            assert 'text' in kw
            assert 'score' in kw
            assert 'start_idx' in kw
            assert 'end_idx' in kw
            assert 0.0 <= kw['score'] <= 1.0
            assert text_b[kw['start_idx']:kw['end_idx']] == kw['text']


class TestLanguageSupport:
    """Tests for different language support."""

    def test_english_text(self, attribution):
        """Test with English text."""
        text_a = "machine learning"
        text_b = "Machine learning is a subset of artificial intelligence that enables systems to learn."

        result = attribution.extract(text_a, text_b)

        print(f"\nTest: English text")
        print(f"  Text A: '{text_a}'")
        print(f"  Text B: '{text_b}'")
        print(f"  ColBERT score: {result.metadata['colbert_score']:.6f}")
        print(f"  Spans: {len(result.spans)}")

        assert result.text_a == text_a
        assert result.text_b == text_b
        assert len(result.spans) > 0

        for i, span in enumerate(result.spans[:3]):
            print(f"    Span {i+1}: '{span.text}' (score={span.score:.4f})")

    def test_mixed_language(self, attribution):
        """Test with mixed Chinese-English text."""
        text_a = "AI人工智能"
        text_b = "AI技术与machine learning机器学习正在改变世界。"

        result = attribution.extract(text_a, text_b)

        print(f"\nTest: Mixed language text")
        print(f"  Text A: '{text_a}'")
        print(f"  Text B: '{text_b}'")
        print(f"  ColBERT score: {result.metadata['colbert_score']:.6f}")
        print(f"  Spans: {len(result.spans)}")

        assert result.text_a == text_a
        assert result.text_b == text_b

        for i, span in enumerate(result.spans[:3]):
            print(f"    Span {i+1}: '{span.text}' (score={span.score:.4f})")


class TestSemanticAttribution:
    """Tests for semantic attribution quality."""

    def test_semantic_related_content_higher_scores(self, attribution):
        """Test that semantically related content gets higher scores."""
        text_a = "人工智能正在改变医疗行业"
        text_b = "机器学习用于医学诊断正在彻底改变患者护理，尽管今天天气很好。"

        result = attribution.extract(text_a, text_b)

        print(f"\nTest: Semantic attribution")
        print(f"  Text A: '{text_a}'")
        print(f"  Text B: '{text_b}'")
        print(f"  ColBERT score: {result.metadata['colbert_score']:.6f}")

        print(f"\n  Extracted spans:")
        for i, span in enumerate(result.spans):
            print(f"    {i+1}. '{span.text}' (score={span.score:.4f})")

        # The top span should contain medical/AI related content
        if result.spans:
            top_span = result.spans[0]
            assert "天气" not in top_span.text, "Top span should not be about weather"


class TestWindowMetadata:
    """Tests for window metadata."""

    def test_window_metadata_correctly_populated(self, attribution):
        """Verify window metadata is correctly populated."""
        text_a = "测试查询"
        text_b = "这是一个测试文档，包含多个词语用于验证窗口元数据。"

        result = attribution.extract(text_a, text_b)

        print(f"\nTest: Window metadata")
        print(f"  Window size: {result.metadata['window_size']}")
        print(f"  Window overlap: {result.metadata['window_overlap']}")
        print(f"  Stride: {result.metadata['stride']}")
        print(f"  Total windows: {result.metadata['total_windows']}")

        # Validate metadata values
        assert result.metadata['window_size'] == min(
            attribution.window_size, result.metadata['num_doc_tokens']
        )
        assert result.metadata['window_overlap'] == attribution.window_overlap
        assert result.metadata['total_windows'] > 0

        # Validate span metadata
        for span in result.spans:
            assert 'window_idx' in span.metadata
            assert 'token_start' in span.metadata
            assert 'token_end' in span.metadata
            assert 'window_rank' in span.metadata
            assert 'raw_score' in span.metadata

            print(f"\n    Span: '{span.text}'")
            print(f"      Window idx: {span.metadata['window_idx']}")
            print(f"      Token range: [{span.metadata['token_start']}:{span.metadata['token_end']}]")
            print(f"      Raw score: {span.metadata['raw_score']:.4f}")


class TestEdgeCases:
    """Tests for edge cases."""

    def test_minimal_overlap_texts(self, attribution):
        """Test behavior when texts have minimal semantic overlap."""
        text_a = "量子物理学基础"
        text_b = "烹饪美食的艺术和技巧，如何做好一道菜。"

        result = attribution.extract(text_a, text_b)

        print(f"\nTest: Minimal overlap texts")
        print(f"  Text A: '{text_a}'")
        print(f"  Text B: '{text_b}'")
        print(f"  ColBERT score: {result.metadata['colbert_score']:.6f}")
        print(f"  Spans: {len(result.spans)}")

        assert result.text_a == text_a
        assert result.text_b == text_b
        assert result.metadata['colbert_score'] < 0.7, "Unrelated texts should have lower score"


class TestInternalMethods:
    """Tests for internal computation methods."""

    def test_compute_interaction_matrix(self, attribution):
        """Test _compute_interaction_matrix with real model vectors."""
        text_a = "机器学习"
        text_b = "深度学习"

        colbert_vecs = attribution._encode_colbert([text_a, text_b])
        query_vecs = torch.tensor(colbert_vecs[0]) if not isinstance(colbert_vecs[0], torch.Tensor) else colbert_vecs[0]
        doc_vecs = torch.tensor(colbert_vecs[1]) if not isinstance(colbert_vecs[1], torch.Tensor) else colbert_vecs[1]

        sim_matrix = attribution._compute_interaction_matrix(query_vecs, doc_vecs)

        print(f"\nTest: Interaction matrix")
        print(f"  Query shape: {query_vecs.shape}")
        print(f"  Doc shape: {doc_vecs.shape}")
        print(f"  Sim matrix shape: {sim_matrix.shape}")
        print(f"  Value range: [{sim_matrix.min().item():.4f}, {sim_matrix.max().item():.4f}]")

        assert sim_matrix.shape[0] == query_vecs.shape[0]
        assert sim_matrix.shape[1] == doc_vecs.shape[0]
        assert torch.all(sim_matrix >= -1.0 - 1e-5)
        assert torch.all(sim_matrix <= 1.0 + 1e-5)

    def test_compute_window_scores(self, attribution):
        """Test _compute_window_scores with real interaction matrix."""
        text_a = "人工智能"
        text_b = "人工智能是一项重要的技术，正在改变世界。"

        colbert_vecs = attribution._encode_colbert([text_a, text_b])
        query_vecs = torch.tensor(colbert_vecs[0]) if not isinstance(colbert_vecs[0], torch.Tensor) else colbert_vecs[0]
        doc_vecs = torch.tensor(colbert_vecs[1]) if not isinstance(colbert_vecs[1], torch.Tensor) else colbert_vecs[1]

        sim_matrix = attribution._compute_interaction_matrix(query_vecs, doc_vecs)
        result = attribution._compute_window_scores(sim_matrix, doc_vecs.shape[0])

        print(f"\nTest: Window scores")
        print(f"  Sim matrix shape: {sim_matrix.shape}")
        print(f"  Window size: {result.window_size}")
        print(f"  Stride: {result.stride}")
        print(f"  Num windows: {result.n_windows}")
        print(f"  Scores: {result.segment_scores.tolist()}")

        assert isinstance(result, WindowScoreResult)
        assert len(result.segment_scores) == result.n_windows
        assert result.n_windows > 0

    def test_resolve_spans(self, attribution):
        """Test _resolve_spans with real tokenization."""
        text_b = "人工智能技术正在改变世界。"

        top_indices = torch.tensor([0, 1])
        top_scores = torch.tensor([2.0, 1.5])

        colbert_vecs = attribution._encode_colbert([text_b])
        n_doc_tokens = colbert_vecs[0].shape[0] if isinstance(colbert_vecs[0], torch.Tensor) else len(colbert_vecs[0])

        spans, input_ids, offsets, special_mask = attribution._resolve_spans(
            text_b,
            top_indices,
            top_scores,
            window_size=attribution.window_size,
            stride=max(1, attribution.window_size - attribution.window_overlap),
            n_doc_tokens=n_doc_tokens
        )

        print(f"\nTest: Resolve spans")
        print(f"  Text B: '{text_b}'")
        print(f"  Doc tokens: {n_doc_tokens}")
        print(f"  Input IDs: {len(input_ids)}")
        print(f"  Spans resolved: {len(spans)}")

        for i, span in enumerate(spans):
            print(f"    Span {i+1}: '{span.text}' [{span.start_idx}:{span.end_idx}] score={span.score:.4f}")
            assert span.text == text_b[span.start_idx:span.end_idx]

        assert isinstance(spans, list)
        assert isinstance(input_ids, list)
        assert isinstance(offsets, list)
        assert isinstance(special_mask, list)
