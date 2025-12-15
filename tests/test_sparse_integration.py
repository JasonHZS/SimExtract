"""Integration tests for SparseAttribution with real BGE-M3 model.

This test validates that our understanding of BGE-M3's behavior is correct
by running tests against the actual model. Unlike the mock tests in
test_sparse_attribution.py, these tests verify:

1. Tokenizer behavior (WordPiece tokenization, offset mapping)
2. Lexical weights format and structure
3. Core function accuracy with real model outputs
4. Character position mapping accuracy

Requirements:
- FlagEmbedding library installed
- Local BGE-M3 model (configured in config/attribution.yaml)

Run with:
    CUDA_VISIBLE_DEVICES=1 python -m pytest tests/test_sparse_integration.py -v -s
"""

import pytest
import numpy as np

from src.attribution.token_wise.sparse import SparseAttribution


# ============== Fixtures ==============

@pytest.fixture(scope="module")
def sparse_config():
    """Configuration for sparse integration tests."""
    return {
        "model_name": "models/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
        "use_fp16": True,
        "device": "cuda:0",
        "window_size": 10,
        "window_overlap": 5,
        "top_k_spans": 3,
        "top_k_tokens": 5,
    }


@pytest.fixture(scope="module")
def attribution(sparse_config):
    """Load SparseAttribution model once for all tests in this module."""
    print("\n" + "=" * 80)
    print("Loading BGE-M3 model for integration tests...")
    print("=" * 80)

    try:
        attr = SparseAttribution(sparse_config)
        print(f"✓ Model loaded successfully")
        print(f"  Model: {sparse_config['model_name']}")
        print(f"  Device: {attr.device}")
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


class TestTokenizerBehavior:
    """Tests for tokenizer behavior."""

    def test_chinese_tokenization(self, attribution):
        """Verify tokenizer behavior on Chinese text."""
        text = "机器学习"
        tokens = attribution._tokenize_with_positions(text)

        print(f"\nTest: Chinese tokenization")
        print(f"  Input: {text}")
        print(f"  Tokens: {len(tokens)}")
        for i, t in enumerate(tokens[:5]):
            print(f"    [{i}] token_id={t['token_id']}, "
                  f"text='{t['token']}', pos=[{t['start']}:{t['end']}]")

        assert len(tokens) > 0, "Should produce tokens"

        for token in tokens:
            assert token['end'] >= token['start']
            assert token['end'] <= len(text)

    def test_english_tokenization(self, attribution):
        """Verify tokenizer behavior on English text."""
        text = "machine learning"
        tokens = attribution._tokenize_with_positions(text)

        print(f"\nTest: English tokenization")
        print(f"  Input: {text}")
        print(f"  Tokens: {len(tokens)}")
        for i, t in enumerate(tokens[:5]):
            print(f"    [{i}] token_id={t['token_id']}, "
                  f"text='{t['token']}', pos=[{t['start']}:{t['end']}]")

        assert len(tokens) > 0, "Should produce tokens"

        for token in tokens:
            assert token['end'] >= token['start']
            assert token['end'] <= len(text)

    def test_mixed_language_tokenization(self, attribution):
        """Verify tokenizer behavior on mixed Chinese-English text."""
        text = "AI人工智能 machine learning 机器学习"
        tokens = attribution._tokenize_with_positions(text)

        print(f"\nTest: Mixed language tokenization")
        print(f"  Input: {text}")
        print(f"  Tokens: {len(tokens)}")
        for i, t in enumerate(tokens[:10]):
            print(f"    [{i}] token_id={t['token_id']}, "
                  f"text='{t['token']}', pos=[{t['start']}:{t['end']}]")

        assert len(tokens) > 0, "Should produce tokens"


class TestLexicalWeights:
    """Tests for lexical weights format."""

    def test_lexical_weights_format(self, attribution):
        """Verify lexical_weights returns correct format."""
        text_a = "人工智能技术"
        text_b = "AI technology"

        weights_list = attribution._encode_sparse([text_a, text_b])

        print(f"\nTest: Lexical weights format")
        print(f"  Input A: {text_a}")
        print(f"  Input B: {text_b}")
        print(f"  Weights list length: {len(weights_list)}")

        assert len(weights_list) == 2, "Should return weights for both texts"

        for idx, weights in enumerate(weights_list):
            print(f"\n  Weights[{idx}]:")
            print(f"    Type: {type(weights)}")
            print(f"    Num tokens: {len(weights)}")

            assert isinstance(weights, dict)

            sample_items = list(weights.items())[:5]
            for token_id, weight in sample_items:
                try:
                    token_id_int = int(token_id)
                    token_str = attribution.tokenizer.decode([token_id_int])
                    print(f"      token_id={token_id} -> weight={weight:.6f}, text='{token_str}'")
                except (ValueError, TypeError) as e:
                    print(f"      token_id={token_id} -> weight={weight:.6f}, decode_error={e}")

            # Verify structure
            for token_id, weight in weights.items():
                assert isinstance(token_id, (int, str)), f"Token ID should be int or str"
                if isinstance(token_id, str):
                    assert token_id.isdigit(), f"String token_id should be numeric"
                assert isinstance(weight, (float, np.floating)), f"Weight should be float"
                assert float(weight) >= 0.0, "Weight should be non-negative"


class TestOffsetMapping:
    """Tests for offset mapping accuracy."""

    @pytest.mark.parametrize("text", [
        "机器学习",
        "machine learning",
        "AI人工智能",
    ])
    def test_offset_mapping_accuracy(self, attribution, text):
        """Verify offset_mapping produces accurate character positions."""
        tokens = attribution._tokenize_with_positions(text)

        print(f"\nTest: Offset mapping for '{text}'")
        print(f"  Tokens: {len(tokens)}")

        for token in tokens[:5]:
            start, end = token['start'], token['end']
            extracted = text[start:end]

            print(f"    [{start}:{end}] '{extracted}' (token: '{token['token']}')")

            assert start >= 0
            assert end <= len(text)
            assert start < end
            assert len(extracted) > 0


class TestTokenContributions:
    """Tests for token contribution computation."""

    def test_get_token_contributions(self, attribution):
        """Verify _get_token_contributions with real model."""
        text_a = "机器学习 machine learning"
        text_b = "机器学习技术 machine learning technology"

        contributions, total_score = attribution._get_token_contributions(text_a, text_b)

        print(f"\nTest: Token contributions")
        print(f"  Text A: {text_a}")
        print(f"  Text B: {text_b}")
        print(f"  Contributing tokens: {len(contributions)}")
        print(f"  Total score: {total_score:.6f}")

        sorted_tokens = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Top contributing tokens:")
        for token, score in sorted_tokens[:10]:
            print(f"    '{token}': {score:.6f}")

        assert len(contributions) > 0, "Should have common tokens"
        assert total_score > 0.0, "Total score should be positive"

        for token, score in contributions.items():
            assert score > 0.0, f"Token '{token}' has non-positive score"


class TestWindowScores:
    """Tests for window score computation."""

    def test_compute_window_scores(self, attribution):
        """Verify _compute_window_scores produces valid windows."""
        text_a = "人工智能"
        text_b = "人工智能是一项重要的技术，机器学习是其核心方法。"

        contributions, _ = attribution._get_token_contributions(text_a, text_b)
        windows = attribution._compute_window_scores(text_b, contributions)

        print(f"\nTest: Window scores")
        print(f"  Text A: {text_a}")
        print(f"  Text B: {text_b}")
        print(f"  Windows: {len(windows)}")
        print(f"  Window size: {attribution.window_size}")
        print(f"  Window overlap: {attribution.window_overlap}")

        assert len(windows) > 0, "Should produce windows"

        sorted_windows = sorted(windows, key=lambda w: w['score'], reverse=True)
        for i, window in enumerate(sorted_windows[:3]):
            print(f"\n  Window {i+1}:")
            print(f"    Score: {window['score']:.6f}")
            print(f"    Position: [{window['start_idx']}:{window['end_idx']}]")
            print(f"    Text: '{window['text']}'")
            print(f"    Token count: {window['token_count']}")
            print(f"    Contributing tokens: {window['contributing_tokens']}")

            assert window['start_idx'] >= 0
            assert window['end_idx'] <= len(text_b)
            assert window['start_idx'] < window['end_idx']
            assert text_b[window['start_idx']:window['end_idx']] == window['text']


class TestExtractEndToEnd:
    """Tests for full extract() pipeline."""

    def test_extract_returns_valid_result(self, attribution):
        """Test full extract() pipeline with real model."""
        text_a = "人工智能和机器学习"
        text_b = "人工智能技术的发展推动了机器学习的进步，深度学习是重要方向。"

        result = attribution.extract(text_a, text_b)

        print(f"\nTest: Extract end-to-end")
        print(f"  Text A: {text_a}")
        print(f"  Text B: {text_b}")
        print(f"  Method: {result.method_name}")
        print(f"  Spans: {len(result.spans)}")
        print(f"  Metadata:")
        for key, value in result.metadata.items():
            if key != 'top_contributing_tokens':
                print(f"    {key}: {value}")

        # Validate result structure
        assert result.text_a == text_a
        assert result.text_b == text_b
        assert result.method_name == "SparseAttribution"
        assert len(result.spans) > 0, "Should extract spans"
        assert len(result.spans) <= attribution.top_k_spans

        # Show extracted spans
        print(f"\n  Extracted spans:")
        for i, span in enumerate(result.spans):
            print(f"\n    Span {i+1}:")
            print(f"      Score: {span.score:.4f}")
            print(f"      Position: [{span.start_idx}:{span.end_idx}]")
            print(f"      Text: '{span.text}'")

            assert 0.0 <= span.score <= 1.0
            assert span.start_idx >= 0
            assert span.end_idx <= len(text_b)
            assert text_b[span.start_idx:span.end_idx] == span.text

        # Validate metadata
        assert 'total_lexical_score' in result.metadata
        assert 'num_contributing_tokens' in result.metadata
        assert 'top_contributing_tokens' in result.metadata

        # Show top tokens
        print(f"\n  Top contributing tokens:")
        for token_info in result.metadata['top_contributing_tokens'][:5]:
            print(f"    '{token_info['token']}': "
                  f"score={token_info['score']:.6f}, "
                  f"normalized={token_info['normalized_score']:.4f}")


class TestEdgeCases:
    """Tests for edge cases."""

    def test_no_common_tokens(self, attribution):
        """Test behavior when texts have no common tokens."""
        text_a = "完全不同的内容"
        text_b = "totally different content"

        result = attribution.extract(text_a, text_b)

        print(f"\nTest: No common tokens")
        print(f"  Text A: {text_a}")
        print(f"  Text B: {text_b}")
        print(f"  Spans: {len(result.spans)}")
        print(f"  Total lexical score: {result.metadata.get('total_lexical_score', 0):.6f}")

        assert result.text_a == text_a
        assert result.text_b == text_b

        if result.spans:
            for span in result.spans:
                assert span.score >= 0.0
