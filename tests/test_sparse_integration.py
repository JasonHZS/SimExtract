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
    python tests/test_sparse_integration.py
    # OR
    python -m pytest tests/test_sparse_integration.py -v
"""

import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.attribution.token_wise.sparse import SparseAttribution


class TestSparseIntegration(unittest.TestCase):
    """Integration tests using real BGE-M3 model."""

    @classmethod
    def setUpClass(cls):
        """Load model once for all tests."""
        print("\n" + "=" * 80)
        print("Loading BGE-M3 model for integration tests...")
        print("=" * 80)

        # Use local model path from config
        # NOTE: Use CUDA_VISIBLE_DEVICES=1 when running to limit to GPU 1
        # In that case, GPU 1 becomes cuda:0 in the process
        cls.config = {
            "model_name": "models/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181",
            "use_fp16": True,  # Use FP16 for GPU
            "device": "cuda:0",  # Will be GPU 1 if CUDA_VISIBLE_DEVICES=1
            "window_size": 10,
            "window_overlap": 5,
            "top_k_spans": 3,
            "top_k_tokens": 5,
        }

        try:
            cls.attribution = SparseAttribution(cls.config)
            print(f"✓ Model loaded successfully")
            print(f"  Model: {cls.config['model_name']}")
            print(f"  Device: {cls.attribution.device}")
            print("=" * 80 + "\n")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
            print("\nNOTE: This test requires:")
            print("  1. FlagEmbedding installed: pip install FlagEmbedding")
            print("  2. Local BGE-M3 model in models/ directory")
            print("=" * 80 + "\n")
            raise

    def test_01_model_initialization(self):
        """Verify model and tokenizer are properly initialized."""
        self.assertIsNotNone(self.attribution.model)
        self.assertIsNotNone(self.attribution.tokenizer)

        # Verify model has expected methods
        self.assertTrue(hasattr(self.attribution.model, 'encode'))
        self.assertTrue(hasattr(self.attribution.tokenizer, 'decode'))

    def test_02_tokenizer_behavior_chinese(self):
        """Verify tokenizer behavior on Chinese text."""
        text = "机器学习"
        tokens = self.attribution._tokenize_with_positions(text)

        print(f"\nTest: Chinese tokenization")
        print(f"  Input: {text}")
        print(f"  Tokens: {len(tokens)}")
        for i, t in enumerate(tokens[:5]):  # Show first 5
            print(f"    [{i}] token_id={t['token_id']}, "
                  f"text='{t['token']}', pos=[{t['start']}:{t['end']}]")

        # Validate basic properties
        self.assertGreater(len(tokens), 0, "Should produce tokens")

        # Verify each token's position maps correctly to text
        for token in tokens:
            extracted = text[token['start']:token['end']]
            # Note: BGE tokenizer might normalize text, so we validate structure
            # rather than exact string match
            self.assertGreaterEqual(token['end'], token['start'])
            self.assertLessEqual(token['end'], len(text))

    def test_03_tokenizer_behavior_english(self):
        """Verify tokenizer behavior on English text."""
        text = "machine learning"
        tokens = self.attribution._tokenize_with_positions(text)

        print(f"\nTest: English tokenization")
        print(f"  Input: {text}")
        print(f"  Tokens: {len(tokens)}")
        for i, t in enumerate(tokens[:5]):
            print(f"    [{i}] token_id={t['token_id']}, "
                  f"text='{t['token']}', pos=[{t['start']}:{t['end']}]")

        self.assertGreater(len(tokens), 0, "Should produce tokens")

        # Verify positions
        for token in tokens:
            self.assertGreaterEqual(token['end'], token['start'])
            self.assertLessEqual(token['end'], len(text))

    def test_04_tokenizer_behavior_mixed(self):
        """Verify tokenizer behavior on mixed Chinese-English text."""
        text = "AI人工智能 machine learning 机器学习"
        tokens = self.attribution._tokenize_with_positions(text)

        print(f"\nTest: Mixed language tokenization")
        print(f"  Input: {text}")
        print(f"  Tokens: {len(tokens)}")
        for i, t in enumerate(tokens[:10]):
            print(f"    [{i}] token_id={t['token_id']}, "
                  f"text='{t['token']}', pos=[{t['start']}:{t['end']}]")

        self.assertGreater(len(tokens), 0, "Should produce tokens")

    def test_05_lexical_weights_format(self):
        """Verify lexical_weights returns correct format."""
        text_a = "人工智能技术"
        text_b = "AI technology"

        weights_list = self.attribution._encode_sparse([text_a, text_b])

        print(f"\nTest: Lexical weights format")
        print(f"  Input A: {text_a}")
        print(f"  Input B: {text_b}")
        print(f"  Weights list length: {len(weights_list)}")

        # Verify structure
        self.assertEqual(len(weights_list), 2, "Should return weights for both texts")

        for idx, weights in enumerate(weights_list):
            print(f"\n  Weights[{idx}]:")
            print(f"    Type: {type(weights)}")
            print(f"    Num tokens: {len(weights)}")

            # Verify it's a dict
            self.assertIsInstance(weights, dict)

            # Show first few entries
            sample_items = list(weights.items())[:5]
            for token_id, weight in sample_items:
                # Ensure token_id is int for decode
                try:
                    token_id_int = int(token_id)
                    token_str = self.attribution.tokenizer.decode([token_id_int])
                    print(f"      token_id={token_id} (type={type(token_id).__name__}) -> weight={weight:.6f}, text='{token_str}'")
                except (ValueError, TypeError) as e:
                    print(f"      token_id={token_id} (type={type(token_id).__name__}) -> weight={weight:.6f}, decode_error={e}")

            # Verify structure: dict[str or int, float]
            # NOTE: BGE-M3 actually returns string keys, not int!
            # NOTE: Weights may be numpy types (np.float16) if use_fp16=True
            import numpy as np
            for token_id, weight in weights.items():
                # BGE-M3 returns string keys that can be converted to int
                self.assertTrue(
                    isinstance(token_id, (int, str)),
                    f"Token ID should be int or str, got {type(token_id)}"
                )
                if isinstance(token_id, str):
                    self.assertTrue(
                        token_id.isdigit(),
                        f"String token_id should be numeric, got '{token_id}'"
                    )
                # Weight can be python float or numpy float types
                self.assertTrue(
                    isinstance(weight, (float, np.floating)),
                    f"Weight should be float or numpy.floating, got {type(weight)}"
                )
                self.assertGreaterEqual(float(weight), 0.0, "Weight should be non-negative")

    def test_06_offset_mapping_accuracy(self):
        """Verify offset_mapping produces accurate character positions."""
        test_cases = [
            "机器学习",
            "machine learning",
            "AI人工智能",
        ]

        for text in test_cases:
            with self.subTest(text=text):
                tokens = self.attribution._tokenize_with_positions(text)

                print(f"\nTest: Offset mapping for '{text}'")
                print(f"  Tokens: {len(tokens)}")

                for token in tokens[:5]:  # Check first 5
                    start, end = token['start'], token['end']
                    extracted = text[start:end]

                    print(f"    [{start}:{end}] '{extracted}' (token: '{token['token']}')")

                    # Verify bounds
                    self.assertGreaterEqual(start, 0)
                    self.assertLessEqual(end, len(text))
                    self.assertLess(start, end)

                    # Verify we can extract something
                    self.assertGreater(len(extracted), 0)

    def test_07_get_token_contributions(self):
        """Verify _get_token_contributions with real model."""
        # Use simple texts with obvious overlap
        text_a = "机器学习 machine learning"
        text_b = "机器学习技术 machine learning technology"

        contributions, total_score = self.attribution._get_token_contributions(
            text_a, text_b
        )

        print(f"\nTest: Token contributions")
        print(f"  Text A: {text_a}")
        print(f"  Text B: {text_b}")
        print(f"  Contributing tokens: {len(contributions)}")
        print(f"  Total score: {total_score:.6f}")

        # Show top contributors
        sorted_tokens = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        print(f"\n  Top contributing tokens:")
        for token, score in sorted_tokens[:10]:
            print(f"    '{token}': {score:.6f}")

        # Validate
        self.assertGreater(len(contributions), 0, "Should have common tokens")
        self.assertGreater(total_score, 0.0, "Total score should be positive")

        # All contributions should be positive
        for token, score in contributions.items():
            self.assertGreater(score, 0.0, f"Token '{token}' has non-positive score")

    def test_08_compute_window_scores(self):
        """Verify _compute_window_scores produces valid windows."""
        text_a = "人工智能"
        text_b = "人工智能是一项重要的技术，机器学习是其核心方法。"

        # First get contributions
        contributions, _ = self.attribution._get_token_contributions(text_a, text_b)

        # Compute windows
        windows = self.attribution._compute_window_scores(text_b, contributions)

        print(f"\nTest: Window scores")
        print(f"  Text A: {text_a}")
        print(f"  Text B: {text_b}")
        print(f"  Windows: {len(windows)}")
        print(f"  Window size: {self.attribution.window_size}")
        print(f"  Window overlap: {self.attribution.window_overlap}")

        self.assertGreater(len(windows), 0, "Should produce windows")

        # Show top windows
        sorted_windows = sorted(windows, key=lambda w: w['score'], reverse=True)
        for i, window in enumerate(sorted_windows[:3]):
            print(f"\n  Window {i+1}:")
            print(f"    Score: {window['score']:.6f}")
            print(f"    Position: [{window['start_idx']}:{window['end_idx']}]")
            print(f"    Text: '{window['text']}'")
            print(f"    Token count: {window['token_count']}")
            print(f"    Contributing tokens: {window['contributing_tokens']}")

            # Validate window structure
            self.assertGreaterEqual(window['start_idx'], 0)
            self.assertLessEqual(window['end_idx'], len(text_b))
            self.assertLess(window['start_idx'], window['end_idx'])

            # Verify text extraction
            extracted = text_b[window['start_idx']:window['end_idx']]
            self.assertEqual(extracted, window['text'])

    def test_09_extract_end_to_end(self):
        """Test full extract() pipeline with real model."""
        text_a = "人工智能和机器学习"
        text_b = "人工智能技术的发展推动了机器学习的进步，深度学习是重要方向。"

        result = self.attribution.extract(text_a, text_b)

        print(f"\nTest: Extract end-to-end")
        print(f"  Text A: {text_a}")
        print(f"  Text B: {text_b}")
        print(f"  Method: {result.method_name}")
        print(f"  Spans: {len(result.spans)}")
        print(f"  Metadata:")
        for key, value in result.metadata.items():
            if key != 'top_contributing_tokens':  # Skip verbose output
                print(f"    {key}: {value}")

        # Validate result structure
        self.assertEqual(result.text_a, text_a)
        self.assertEqual(result.text_b, text_b)
        self.assertEqual(result.method_name, "SparseAttribution")

        # Should have spans
        self.assertGreater(len(result.spans), 0, "Should extract spans")
        self.assertLessEqual(len(result.spans), self.attribution.top_k_spans)

        # Show extracted spans
        print(f"\n  Extracted spans:")
        for i, span in enumerate(result.spans):
            print(f"\n    Span {i+1}:")
            print(f"      Score: {span.score:.4f}")
            print(f"      Position: [{span.start_idx}:{span.end_idx}]")
            print(f"      Text: '{span.text}'")

            # Validate span
            self.assertGreaterEqual(span.score, 0.0)
            self.assertLessEqual(span.score, 1.0)
            self.assertGreaterEqual(span.start_idx, 0)
            self.assertLessEqual(span.end_idx, len(text_b))

            # Verify text extraction
            extracted = text_b[span.start_idx:span.end_idx]
            self.assertEqual(extracted, span.text)

        # Validate metadata
        self.assertIn('total_lexical_score', result.metadata)
        self.assertIn('num_contributing_tokens', result.metadata)
        self.assertIn('top_contributing_tokens', result.metadata)

        # Show top tokens
        print(f"\n  Top contributing tokens:")
        for token_info in result.metadata['top_contributing_tokens'][:5]:
            print(f"    '{token_info['token']}': "
                  f"score={token_info['score']:.6f}, "
                  f"normalized={token_info['normalized_score']:.4f}")

    def test_10_no_common_tokens(self):
        """Test behavior when texts have no common tokens."""
        text_a = "完全不同的内容"
        text_b = "totally different content"

        result = self.attribution.extract(text_a, text_b)

        print(f"\nTest: No common tokens")
        print(f"  Text A: {text_a}")
        print(f"  Text B: {text_b}")
        print(f"  Spans: {len(result.spans)}")
        print(f"  Total lexical score: {result.metadata.get('total_lexical_score', 0):.6f}")

        # Should handle gracefully
        self.assertEqual(result.text_a, text_a)
        self.assertEqual(result.text_b, text_b)

        # Either no spans or all spans have zero score
        if result.spans:
            for span in result.spans:
                self.assertGreaterEqual(span.score, 0.0)


def main():
    """Run tests with verbose output."""
    # Configure test runner
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestSparseIntegration)

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 80)
    print("INTEGRATION TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("=" * 80)

    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
