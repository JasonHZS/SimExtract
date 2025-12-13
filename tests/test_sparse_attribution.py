import unittest
from unittest.mock import patch

from src.attribution.token_wise.sparse import SparseAttribution


class MockTokenizer:
    """Minimal tokenizer stub that mimics HuggingFace behavior for tests."""

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.next_id = 1

    def token_id_for(self, token: str) -> int:
        normalized = token.strip().lower()
        if normalized not in self.token_to_id:
            self.token_to_id[normalized] = self.next_id
            self.id_to_token[self.next_id] = token.strip()
            self.next_id += 1
        return self.token_to_id[normalized]

    def decode(self, token_ids, clean_up_tokenization_spaces=False):
        if isinstance(token_ids, list):
            tokens = [self.id_to_token.get(tid, "") for tid in token_ids]
        else:
            tokens = [self.id_to_token.get(token_ids, "")]
        return " ".join(filter(None, tokens)).strip()

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=False):
        input_ids = []
        offsets = []
        length = len(text)
        idx = 0

        while idx < length:
            if text[idx].isspace():
                idx += 1
                continue

            start = idx
            if self._is_cjk(text[idx]):
                idx += 1
            elif text[idx].isalnum():
                while idx < length and text[idx].isalnum():
                    idx += 1
            else:
                while (
                    idx < length
                    and not text[idx].isspace()
                    and not text[idx].isalnum()
                    and not self._is_cjk(text[idx])
                ):
                    idx += 1

            token = text[start:idx]
            token_id = self.token_id_for(token)
            input_ids.append(token_id)
            offsets.append((start, idx))

        return {"input_ids": input_ids, "offset_mapping": offsets}

    @staticmethod
    def _is_cjk(char: str) -> bool:
        return "\u4e00" <= char <= "\u9fff"


class MockBGEM3Model:
    """Mock BGE-M3 model that returns queued lexical weight outputs."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._queued_outputs = []

    def queue_lexical_weights(self, weights_batch):
        """Queue a list of lexical-weight dictionaries (one per text)."""
        self._queued_outputs.append(weights_batch)

    def encode(self, texts, **kwargs):
        if not self._queued_outputs:
            raise RuntimeError("No queued outputs for encode() call")

        weights = self._queued_outputs.pop(0)
        if len(weights) != len(texts):
            raise ValueError("Queued weights do not match number of texts")

        return {"lexical_weights": weights}


class TestSparseAttribution(unittest.TestCase):
    def setUp(self):
        self.tokenizer = MockTokenizer()
        self.mock_model = MockBGEM3Model(self.tokenizer)
        self.patcher = patch(
            "src.attribution.token_wise.sparse._get_cached_bge_m3_model",
            return_value=self.mock_model,
        )
        self.patcher.start()

        self.base_config = {
            "model_name": "mock-model",
            "window_size": 3,
            "window_overlap": 1,
            "top_k_spans": 2,
            "top_k_tokens": 3,
            "use_fp16": False,
        }
        self.method = SparseAttribution(self.base_config)

    def tearDown(self):
        self.patcher.stop()

    def _weights_from_tokens(self, token_weights):
        return {
            self.tokenizer.token_id_for(token): weight
            for token, weight in token_weights.items()
        }

    def _queue_pair(self, weights_a, weights_b):
        self.mock_model.queue_lexical_weights(
            [self._weights_from_tokens(weights_a), self._weights_from_tokens(weights_b)]
        )

    def test_initialization_uses_config_values(self):
        config = {
            "model_name": "custom-model",
            "window_size": 4,
            "window_overlap": 2,
            "top_k_spans": 5,
            "top_k_tokens": 7,
            "use_fp16": True,
        }
        method = SparseAttribution(config)

        self.assertEqual(method.model, self.mock_model)
        self.assertEqual(method.window_size, 4)
        self.assertEqual(method.window_overlap, 2)
        self.assertEqual(method.top_k_spans, 5)
        self.assertEqual(method.top_k_tokens, 7)
        self.assertTrue(method.use_fp16)

    def test_get_token_contributions_computes_intersection_scores(self):
        self._queue_pair(
            {"AI": 0.8, "medical": 0.5},
            {"AI": 0.4, "medical": 0.2, "weather": 0.9},
        )

        contributions, total = self.method._get_token_contributions(
            "AI text", "medical text"
        )

        self.assertAlmostEqual(contributions["AI"], 0.32)
        self.assertAlmostEqual(contributions["medical"], 0.1)
        self.assertAlmostEqual(total, 0.42)

    def test_compute_window_scores_finds_highest_scoring_span(self):
        text_b = "AI improves medical diagnosis accuracy quickly."
        contributions = {"AI": 0.5, "medical": 1.5, "diagnosis": 1.2}

        windows = self.method._compute_window_scores(text_b, contributions)
        best_window = max(windows, key=lambda w: w["score"])

        self.assertIn("medical", best_window["text"])
        self.assertIn("diagnosis", best_window["text"])
        self.assertGreater(best_window["score"], 0)
        self.assertGreaterEqual(best_window["contributing_tokens"], 2)

    def test_extract_returns_spans_and_metadata(self):
        weights = {"AI": 0.6, "medical": 0.4, "diagnosis": 0.3}
        self._queue_pair({"AI": 0.7, "medical": 0.5}, weights)
        self._queue_pair({"AI": 0.7, "medical": 0.5}, weights)

        text_a = "AI improves medical diagnosis"
        text_b = "AI systems enhance medical diagnosis accuracy."
        result = self.method.extract(text_a, text_b)

        self.assertTrue(result.spans)
        self.assertGreaterEqual(result.metadata["num_contributing_tokens"], 2)
        top_tokens = result.metadata["top_contributing_tokens"]
        self.assertTrue(top_tokens)
        self.assertEqual(top_tokens[0]["token"], "AI")

    def test_extract_handles_no_common_tokens(self):
        self._queue_pair({"cat": 1.0}, {"dog": 1.0})
        self._queue_pair({"cat": 1.0}, {"dog": 1.0})

        result = self.method.extract("cat story", "dog story")

        self.assertFalse(result.metadata["top_contributing_tokens"])
        self.assertTrue(all(span.score == 0 for span in result.spans))

    def test_extract_validates_inputs(self):
        with self.assertRaises(ValueError):
            self.method.extract("", "text")

        with self.assertRaises(ValueError):
            self.method.extract("text", "   ")

    def test_tokenize_with_positions_accuracy(self):
        # Mock tokenizer to return deterministic offsets
        # "AI improves" -> [("AI", 0, 2), ("improves", 3, 11)]
        # This relies on the MockTokenizer implementation details
        text = "AI improves"
        tokens = self.method._tokenize_with_positions(text)

        self.assertEqual(len(tokens), 2)
        
        token_0 = tokens[0]
        self.assertEqual(token_0["token"], "AI")
        self.assertEqual(token_0["start"], 0)
        self.assertEqual(token_0["end"], 2)
        self.assertEqual(text[token_0["start"]:token_0["end"]], "AI")

        token_1 = tokens[1]
        self.assertEqual(token_1["token"], "improves")
        self.assertEqual(token_1["start"], 3)
        self.assertEqual(token_1["end"], 11)
        self.assertEqual(text[token_1["start"]:token_1["end"]], "improves")


if __name__ == "__main__":
    unittest.main()

