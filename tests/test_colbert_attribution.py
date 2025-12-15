"""Unit tests for ColBERTAttribution method.

These tests use mocks to isolate the ColBERTAttribution logic from the actual
BGE-M3 model. For integration tests with the real model, see test_colbert_integration.py.

Run with:
    python -m pytest tests/test_colbert_attribution.py -v
"""

import pytest
from unittest.mock import patch

import numpy as np
import torch

from src.attribution.token_wise.colbert import ColBERTAttribution, WindowScoreResult


# ============== Mock Classes ==============

class MockTokenizer:
    """Minimal tokenizer stub that mimics HuggingFace behavior for tests."""

    def __init__(self):
        self.token_to_id = {}
        self.id_to_token = {}
        self.next_id = 1
        # Special tokens: 0 = [PAD], 101 = [CLS], 102 = [SEP]
        self.all_special_ids = {0, 101, 102}

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

    def __call__(
        self,
        text,
        return_offsets_mapping=True,
        add_special_tokens=True,
        truncation=True,
        max_length=1024
    ):
        input_ids = []
        offsets = []
        length = len(text)
        idx = 0

        # Add [CLS] token at start if add_special_tokens
        if add_special_tokens:
            input_ids.append(101)
            offsets.append((0, 0))  # Special tokens have (0, 0) offset

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

        # Add [SEP] token at end if add_special_tokens
        if add_special_tokens:
            input_ids.append(102)
            offsets.append((0, 0))

        return {"input_ids": input_ids, "offset_mapping": offsets}

    @staticmethod
    def _is_cjk(char: str) -> bool:
        return "\u4e00" <= char <= "\u9fff"


class MockBGEM3Model:
    """Mock BGE-M3 model that returns queued ColBERT vector outputs."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._queued_outputs = []

    def queue_colbert_vecs(self, vecs_batch):
        """Queue a list of ColBERT vectors (one tensor per text).

        Args:
            vecs_batch: List of tensors, each with shape [num_tokens, embedding_dim]
        """
        self._queued_outputs.append(vecs_batch)

    def encode(self, texts, **kwargs):
        if not self._queued_outputs:
            raise RuntimeError("No queued outputs for encode() call")

        vecs = self._queued_outputs.pop(0)
        if len(vecs) != len(texts):
            raise ValueError("Queued vectors do not match number of texts")

        return {"colbert_vecs": vecs}


# ============== Fixtures ==============

@pytest.fixture
def mock_tokenizer():
    """Create a fresh MockTokenizer instance."""
    return MockTokenizer()


@pytest.fixture
def mock_model(mock_tokenizer):
    """Create a MockBGEM3Model with the mock tokenizer."""
    return MockBGEM3Model(mock_tokenizer)


@pytest.fixture
def colbert_method(mock_model):
    """Create a ColBERTAttribution instance with mocked model."""
    with patch(
        "src.attribution.token_wise.colbert._get_cached_bge_m3_model",
        return_value=mock_model,
    ):
        config = {
            "model_name": "mock-model",
            "window_size": 3,
            "window_overlap": 1,
            "top_k_spans": 2,
            "top_k_topic_keywords": 3,
            "use_fp16": False,
        }
        yield ColBERTAttribution(config)


@pytest.fixture
def colbert_method_larger_window(mock_model):
    """Create a ColBERTAttribution with larger window for edge case tests."""
    with patch(
        "src.attribution.token_wise.colbert._get_cached_bge_m3_model",
        return_value=mock_model,
    ):
        config = {
            "model_name": "mock-model",
            "window_size": 5,
            "window_overlap": 2,
            "top_k_spans": 3,
            "top_k_topic_keywords": 5,
            "use_fp16": False,
        }
        yield ColBERTAttribution(config)


# ============== Helper Functions ==============

def make_vectors(num_tokens: int, dim: int = 64) -> torch.Tensor:
    """Create random normalized vectors for testing."""
    vecs = torch.randn(num_tokens, dim)
    return torch.nn.functional.normalize(vecs, p=2, dim=-1)


def make_similar_vectors(base_vecs: torch.Tensor, similarity: float = 0.9) -> torch.Tensor:
    """Create vectors similar to base with controlled similarity."""
    noise = torch.randn_like(base_vecs) * (1 - similarity)
    similar = base_vecs * similarity + noise
    return torch.nn.functional.normalize(similar, p=2, dim=-1)


# ============== Tests ==============

class TestColBERTAttributionInit:
    """Tests for ColBERTAttribution initialization."""

    def test_initialization_uses_config_values(self, mock_model):
        """Test that config values are properly applied."""
        with patch(
            "src.attribution.token_wise.colbert._get_cached_bge_m3_model",
            return_value=mock_model,
        ):
            config = {
                "model_name": "custom-model",
                "window_size": 10,
                "window_overlap": 3,
                "top_k_spans": 5,
                "top_k_topic_keywords": 7,
                "use_fp16": True,
            }
            method = ColBERTAttribution(config)

            assert method.model == mock_model
            assert method.window_size == 10
            assert method.window_overlap == 3
            assert method.top_k_spans == 5
            assert method.top_k_topic_keywords == 7
            assert method.use_fp16 is True

    def test_initialization_validates_window_size_positive(self, mock_model):
        """Test that window_size must be positive."""
        with patch(
            "src.attribution.token_wise.colbert._get_cached_bge_m3_model",
            return_value=mock_model,
        ):
            with pytest.raises(ValueError):
                ColBERTAttribution({"model_name": "m", "window_size": 0})

            with pytest.raises(ValueError):
                ColBERTAttribution({"model_name": "m", "window_size": -1})

    def test_initialization_validates_window_overlap_non_negative(self, mock_model):
        """Test that window_overlap must be non-negative."""
        with patch(
            "src.attribution.token_wise.colbert._get_cached_bge_m3_model",
            return_value=mock_model,
        ):
            with pytest.raises(ValueError):
                ColBERTAttribution({"model_name": "m", "window_overlap": -1})

    def test_initialization_validates_window_overlap_less_than_size(self, mock_model):
        """Test that window_overlap must be less than window_size."""
        with patch(
            "src.attribution.token_wise.colbert._get_cached_bge_m3_model",
            return_value=mock_model,
        ):
            # overlap == window_size
            with pytest.raises(ValueError):
                ColBERTAttribution({"model_name": "m", "window_size": 3, "window_overlap": 3})

            # overlap > window_size
            with pytest.raises(ValueError):
                ColBERTAttribution({"model_name": "m", "window_size": 3, "window_overlap": 5})

    def test_initialization_validates_top_k_topic_keywords(self, mock_model):
        """Test that top_k_topic_keywords must be positive."""
        with patch(
            "src.attribution.token_wise.colbert._get_cached_bge_m3_model",
            return_value=mock_model,
        ):
            with pytest.raises(ValueError):
                ColBERTAttribution({"model_name": "m", "top_k_topic_keywords": 0})


class TestColBERTExtract:
    """Tests for the main extract method."""

    def test_extract_validates_empty_text_a(self, colbert_method):
        """Test that extract raises ValueError for empty text_a."""
        with pytest.raises(ValueError):
            colbert_method.extract("", "text")

    def test_extract_validates_empty_text_b(self, colbert_method):
        """Test that extract raises ValueError for empty text_b."""
        with pytest.raises(ValueError):
            colbert_method.extract("text", "")

    def test_extract_validates_whitespace_text_a(self, colbert_method):
        """Test that extract raises ValueError for whitespace-only text_a."""
        with pytest.raises(ValueError):
            colbert_method.extract("   ", "text")

    def test_extract_validates_whitespace_text_b(self, colbert_method):
        """Test that extract raises ValueError for whitespace-only text_b."""
        with pytest.raises(ValueError):
            colbert_method.extract("text", "   ")

    def test_extract_returns_valid_result(self, colbert_method, mock_model):
        """Test that extract returns properly structured AttributionResult."""
        text_a = "AI technology"
        text_b = "AI improves medical diagnosis"

        # Create mock vectors
        query_vecs = make_vectors(4, dim=64)
        doc_vecs = make_vectors(6, dim=64)

        # Make first doc token (AI) similar to query
        doc_vecs[1] = make_similar_vectors(query_vecs[1:2], 0.95).squeeze(0)

        mock_model.queue_colbert_vecs([query_vecs, doc_vecs])

        result = colbert_method.extract(text_a, text_b)

        # Validate result structure
        assert result.text_a == text_a
        assert result.text_b == text_b
        assert result.method_name == "ColBERTAttribution"
        assert isinstance(result.spans, list)
        assert len(result.spans) <= colbert_method.top_k_spans

        # Validate metadata
        assert "colbert_score" in result.metadata
        assert "num_query_tokens" in result.metadata
        assert "num_doc_tokens" in result.metadata
        assert "window_size" in result.metadata
        assert "topic_keywords" in result.metadata

        # Validate spans
        for span in result.spans:
            assert 0.0 <= span.score <= 1.0
            assert span.start_idx >= 0
            assert span.end_idx <= len(text_b)
            assert span.text == text_b[span.start_idx:span.end_idx]

    def test_extract_identifies_similar_spans(self, colbert_method, mock_model):
        """Test that extract correctly identifies spans with high similarity."""
        text_a = "machine learning"
        text_b = "deep learning is great"

        # Create vectors where "learning" tokens are highly similar
        query_vecs = make_vectors(4, dim=64)
        doc_vecs = make_vectors(6, dim=64)

        # Make "learning" (index 2 in both) highly similar
        doc_vecs[2] = query_vecs[2].clone()

        mock_model.queue_colbert_vecs([query_vecs, doc_vecs])

        result = colbert_method.extract(text_a, text_b)

        assert len(result.spans) > 0
        # Top span should contain "learning"
        top_span = result.spans[0]
        assert "learning" in top_span.text.lower()

    def test_extract_handles_short_document(self, colbert_method, mock_model):
        """Test extract handles documents shorter than window size."""
        text_a = "query"
        text_b = "short"

        query_vecs = make_vectors(3, dim=64)
        doc_vecs = make_vectors(3, dim=64)

        mock_model.queue_colbert_vecs([query_vecs, doc_vecs])

        # Should not raise, window size is clamped
        result = colbert_method.extract(text_a, text_b)

        assert result.text_a == text_a
        assert result.text_b == text_b


class TestComputeColBERTScore:
    """Tests for compute_colbert_score method."""

    def test_identical_vectors_high_score(self, colbert_method, mock_model):
        """Test that identical vectors produce high score."""
        text_a = "test query"
        text_b = "test document"

        vecs = make_vectors(4, dim=64)
        mock_model.queue_colbert_vecs([vecs.clone(), vecs.clone()])

        score = colbert_method.compute_colbert_score(text_a, text_b)

        assert score > 0.9
        assert score < 1.0 + 1e-5

    def test_orthogonal_vectors_low_score(self, colbert_method, mock_model):
        """Test that orthogonal vectors produce low score."""
        text_a = "test"
        text_b = "doc"

        # Create orthogonal vectors
        query_vecs = torch.zeros(3, 64)
        query_vecs[:, :32] = 1.0
        query_vecs = torch.nn.functional.normalize(query_vecs, p=2, dim=-1)

        doc_vecs = torch.zeros(3, 64)
        doc_vecs[:, 32:] = 1.0
        doc_vecs = torch.nn.functional.normalize(doc_vecs, p=2, dim=-1)

        mock_model.queue_colbert_vecs([query_vecs, doc_vecs])

        score = colbert_method.compute_colbert_score(text_a, text_b)

        assert score < 0.1


class TestEmptyResult:
    """Tests for _empty_result method."""

    def test_returns_properly_structured_empty_result(self, colbert_method):
        """Test _empty_result returns properly structured empty result."""
        text_a = "query"
        text_b = "document"

        result = colbert_method._empty_result(text_a, text_b)

        assert result.text_a == text_a
        assert result.text_b == text_b
        assert result.method_name == "ColBERTAttribution"
        assert result.spans == []
        assert result.metadata["colbert_score"] == 0.0
        assert result.metadata["num_query_tokens"] == 0
        assert result.metadata["num_doc_tokens"] == 0
        assert result.metadata["topic_keywords"] == []


class TestModelProperty:
    """Tests for model property."""

    def test_raises_when_not_initialized(self):
        """Test that model property raises when model is None."""
        method = ColBERTAttribution.__new__(ColBERTAttribution)
        method._model = None

        with pytest.raises(RuntimeError):
            _ = method.model


class TestTopicKeywords:
    """Tests for topic keyword extraction."""

    def test_returns_valid_structure(self, colbert_method, mock_model):
        """Test _extract_topic_keywords returns properly formatted keywords."""
        text_a = "AI technology"
        text_b = "AI improves medical diagnosis accuracy"

        query_vecs = make_vectors(4, dim=64)
        doc_vecs = make_vectors(8, dim=64)

        # Make some doc tokens similar to query
        doc_vecs[1] = query_vecs[1].clone()
        doc_vecs[3] = make_similar_vectors(query_vecs[2:3], 0.8).squeeze(0)

        mock_model.queue_colbert_vecs([query_vecs, doc_vecs])

        result = colbert_method.extract(text_a, text_b)

        keywords = result.metadata.get("topic_keywords", [])
        assert isinstance(keywords, list)

        for kw in keywords:
            assert "text" in kw
            assert "score" in kw
            assert "start_idx" in kw
            assert "end_idx" in kw
            assert 0.0 <= kw["score"] <= 1.0


class TestSpanMetadata:
    """Tests for span metadata."""

    def test_contains_window_info(self, colbert_method, mock_model):
        """Test that span metadata contains window information."""
        text_a = "test query"
        text_b = "document with multiple words here"

        query_vecs = make_vectors(4, dim=64)
        doc_vecs = make_vectors(8, dim=64)

        mock_model.queue_colbert_vecs([query_vecs, doc_vecs])

        result = colbert_method.extract(text_a, text_b)

        for span in result.spans:
            assert "raw_score" in span.metadata
            assert "window_idx" in span.metadata
            assert "token_start" in span.metadata
            assert "token_end" in span.metadata
            assert "window_rank" in span.metadata


class TestColBERTAttributionEdgeCases:
    """Edge case tests for ColBERTAttribution."""

    def test_chinese_text_handling(self, colbert_method_larger_window, mock_model):
        """Test handling of Chinese text."""
        text_a = "机器学习"
        text_b = "人工智能和机器学习技术"

        query_vecs = make_vectors(6, dim=64)
        doc_vecs = make_vectors(12, dim=64)

        mock_model.queue_colbert_vecs([query_vecs, doc_vecs])

        result = colbert_method_larger_window.extract(text_a, text_b)

        assert result.text_a == text_a
        assert result.text_b == text_b
        for span in result.spans:
            assert span.start_idx >= 0
            assert span.end_idx <= len(text_b)

    def test_mixed_language_text(self, colbert_method_larger_window, mock_model):
        """Test handling of mixed Chinese-English text."""
        text_a = "AI人工智能"
        text_b = "AI技术与machine learning机器学习"

        query_vecs = make_vectors(7, dim=64)
        doc_vecs = make_vectors(15, dim=64)

        mock_model.queue_colbert_vecs([query_vecs, doc_vecs])

        result = colbert_method_larger_window.extract(text_a, text_b)

        assert result.text_a == text_a
        assert result.text_b == text_b

    def test_punctuation_in_text(self, colbert_method_larger_window, mock_model):
        """Test handling of text with punctuation."""
        text_a = "test, query!"
        text_b = "test document, with punctuation!"

        query_vecs = make_vectors(6, dim=64)
        doc_vecs = make_vectors(8, dim=64)

        mock_model.queue_colbert_vecs([query_vecs, doc_vecs])

        result = colbert_method_larger_window.extract(text_a, text_b)

        assert result.text_a == text_a
        assert result.text_b == text_b

    def test_single_token_query(self, colbert_method_larger_window, mock_model):
        """Test with single token query."""
        text_a = "AI"
        text_b = "AI improves diagnosis"

        query_vecs = make_vectors(3, dim=64)
        doc_vecs = make_vectors(5, dim=64)

        mock_model.queue_colbert_vecs([query_vecs, doc_vecs])

        result = colbert_method_larger_window.extract(text_a, text_b)

        assert result.text_a == text_a
        assert len(result.spans) > 0

    def test_numpy_array_input_handling(self, colbert_method_larger_window, mock_model):
        """Test that numpy arrays from model are handled correctly."""
        text_a = "test"
        text_b = "test doc"

        # Simulate numpy array output from model
        query_vecs = np.random.randn(3, 64).astype(np.float32)
        doc_vecs = np.random.randn(4, 64).astype(np.float32)

        mock_model.queue_colbert_vecs([query_vecs, doc_vecs])

        result = colbert_method_larger_window.extract(text_a, text_b)

        assert result.text_a == text_a
        assert result.text_b == text_b


class TestComputeInteractionMatrix:
    """Unit tests for _compute_interaction_matrix method."""

    def test_output_shape(self, colbert_method):
        """Test that interaction matrix has correct shape [N_q, N_d]."""
        n_q, n_d, dim = 4, 6, 64
        query_vecs = torch.randn(n_q, dim)
        doc_vecs = torch.randn(n_d, dim)

        sim_matrix = colbert_method._compute_interaction_matrix(query_vecs, doc_vecs)

        assert sim_matrix.shape == (n_q, n_d)

    def test_identical_vectors_max_similarity(self, colbert_method):
        """Test that identical normalized vectors produce similarity of 1.0."""
        dim = 64
        vec = torch.randn(1, dim)
        vec = torch.nn.functional.normalize(vec, p=2, dim=-1)

        sim_matrix = colbert_method._compute_interaction_matrix(vec, vec)

        assert sim_matrix[0, 0].item() == pytest.approx(1.0, abs=1e-5)

    def test_orthogonal_vectors_zero_similarity(self, colbert_method):
        """Test that orthogonal vectors produce similarity near 0."""
        dim = 64
        query_vecs = torch.zeros(1, dim)
        query_vecs[0, :32] = 1.0

        doc_vecs = torch.zeros(1, dim)
        doc_vecs[0, 32:] = 1.0

        sim_matrix = colbert_method._compute_interaction_matrix(query_vecs, doc_vecs)

        assert sim_matrix[0, 0].item() == pytest.approx(0.0, abs=1e-5)

    def test_cosine_similarity_range(self, colbert_method):
        """Test that all similarities are in [-1, 1] range."""
        query_vecs = torch.randn(5, 64)
        doc_vecs = torch.randn(10, 64)

        sim_matrix = colbert_method._compute_interaction_matrix(query_vecs, doc_vecs)

        assert torch.all(sim_matrix >= -1.0 - 1e-5)
        assert torch.all(sim_matrix <= 1.0 + 1e-5)

    def test_symmetry_for_same_input(self, colbert_method):
        """Test that matrix is symmetric when query == doc."""
        vecs = torch.randn(5, 64)

        sim_matrix = colbert_method._compute_interaction_matrix(vecs, vecs)

        diff = torch.abs(sim_matrix - sim_matrix.T)
        assert torch.all(diff < 1e-5)


class TestComputeWindowScores:
    """Unit tests for _compute_window_scores method."""

    @pytest.fixture
    def window_method(self, mock_model):
        """Create method with specific window settings for testing."""
        with patch(
            "src.attribution.token_wise.colbert._get_cached_bge_m3_model",
            return_value=mock_model,
        ):
            config = {
                "model_name": "mock-model",
                "window_size": 4,
                "window_overlap": 2,
                "top_k_spans": 3,
                "use_fp16": False,
            }
            yield ColBERTAttribution(config)

    def test_returns_window_score_result(self, window_method):
        """Test that method returns WindowScoreResult dataclass."""
        sim_matrix = torch.randn(3, 10)

        result = window_method._compute_window_scores(sim_matrix, 10)

        assert isinstance(result, WindowScoreResult)
        assert isinstance(result.segment_scores, torch.Tensor)
        assert isinstance(result.window_size, int)
        assert isinstance(result.stride, int)
        assert isinstance(result.n_windows, int)

    def test_returns_none_for_empty_doc(self, window_method):
        """Test that method returns None for doc_length < 1."""
        sim_matrix = torch.randn(3, 0)

        result = window_method._compute_window_scores(sim_matrix, 0)

        assert result is None

    def test_window_count_calculation(self, window_method):
        """Test that correct number of windows is computed."""
        sim_matrix = torch.randn(3, 10)

        result = window_method._compute_window_scores(sim_matrix, 10)

        # (10 - 4) / 2 + 1 = 4 windows
        assert result.n_windows == 4

    def test_window_size_clamped_to_doc_length(self, window_method):
        """Test that window_size is clamped when doc is shorter."""
        sim_matrix = torch.randn(3, 2)

        result = window_method._compute_window_scores(sim_matrix, 2)

        assert result.window_size == 2
        assert result.n_windows == 1

    def test_max_pooling_selects_highest_in_window(self, mock_model):
        """Test that max pooling correctly selects highest value per window."""
        with patch(
            "src.attribution.token_wise.colbert._get_cached_bge_m3_model",
            return_value=mock_model,
        ):
            config = {
                "model_name": "mock-model",
                "window_size": 2,
                "window_overlap": 0,
                "top_k_spans": 3,
                "use_fp16": False,
            }
            method = ColBERTAttribution(config)

            sim_matrix = torch.tensor([[0.1, 0.9, 0.2, 0.3]])

            result = method._compute_window_scores(sim_matrix, 4)

            # Window 0: max(0.1, 0.9) = 0.9
            # Window 1: max(0.2, 0.3) = 0.3
            expected = torch.tensor([0.9, 0.3])
            assert torch.allclose(result.segment_scores, expected, atol=1e-5)

    def test_multiple_query_tokens_summed(self, mock_model):
        """Test that scores from multiple query tokens are summed."""
        with patch(
            "src.attribution.token_wise.colbert._get_cached_bge_m3_model",
            return_value=mock_model,
        ):
            config = {
                "model_name": "mock-model",
                "window_size": 2,
                "window_overlap": 0,
                "top_k_spans": 3,
                "use_fp16": False,
            }
            method = ColBERTAttribution(config)

            sim_matrix = torch.tensor([
                [0.1, 0.9, 0.2, 0.3],
                [0.5, 0.1, 0.8, 0.2],
            ])

            result = method._compute_window_scores(sim_matrix, 4)

            # Window 0: Q1 max=0.9, Q2 max=0.5 -> sum=1.4
            # Window 1: Q1 max=0.3, Q2 max=0.8 -> sum=1.1
            expected = torch.tensor([1.4, 1.1])
            assert torch.allclose(result.segment_scores, expected, atol=1e-5)


class TestResolveSpans:
    """Unit tests for _resolve_spans method."""

    def test_returns_correct_tuple_structure(self, colbert_method):
        """Test that method returns (spans, input_ids, offsets, special_mask)."""
        text_b = "hello world test"
        top_indices = torch.tensor([0])
        top_scores = torch.tensor([1.0])

        spans, input_ids, offsets, special_mask = colbert_method._resolve_spans(
            text_b, top_indices, top_scores, window_size=3, stride=2, n_doc_tokens=5
        )

        assert isinstance(spans, list)
        assert isinstance(input_ids, list)
        assert isinstance(offsets, list)
        assert isinstance(special_mask, list)

    def test_span_text_matches_positions(self, colbert_method):
        """Test that span.text equals text_b[start_idx:end_idx]."""
        text_b = "hello world test"
        top_indices = torch.tensor([0])
        top_scores = torch.tensor([1.0])

        spans, _, _, _ = colbert_method._resolve_spans(
            text_b, top_indices, top_scores, window_size=3, stride=2, n_doc_tokens=5
        )

        for span in spans:
            extracted = text_b[span.start_idx:span.end_idx]
            assert span.text == extracted

    def test_normalized_scores_in_valid_range(self, colbert_method):
        """Test that span scores are normalized to [0, 1]."""
        text_b = "hello world test"
        top_indices = torch.tensor([0, 1])
        top_scores = torch.tensor([2.0, 1.0])

        spans, _, _, _ = colbert_method._resolve_spans(
            text_b, top_indices, top_scores, window_size=2, stride=1, n_doc_tokens=5
        )

        for span in spans:
            assert 0.0 <= span.score <= 1.0

    def test_window_rank_metadata(self, colbert_method):
        """Test that window_rank metadata is correctly assigned."""
        text_b = "hello world test data"
        top_indices = torch.tensor([0, 1])
        top_scores = torch.tensor([2.0, 1.5])

        spans, _, _, _ = colbert_method._resolve_spans(
            text_b, top_indices, top_scores, window_size=2, stride=1, n_doc_tokens=6
        )

        if len(spans) >= 2:
            assert spans[0].metadata["window_rank"] == 1
            assert spans[1].metadata["window_rank"] == 2

    def test_special_token_mask_built_correctly(self, colbert_method):
        """Test that special token mask identifies special tokens."""
        text_b = "hello world"
        top_indices = torch.tensor([0])
        top_scores = torch.tensor([1.0])

        _, input_ids, _, special_mask = colbert_method._resolve_spans(
            text_b, top_indices, top_scores, window_size=3, stride=2, n_doc_tokens=4
        )

        # First token should be [CLS] (101) -> special
        assert special_mask[0] == 1

    def test_empty_result_for_zero_doc_tokens(self, colbert_method):
        """Test that empty spans returned when n_doc_tokens causes min_len=0."""
        text_b = "hello"
        top_indices = torch.tensor([0])
        top_scores = torch.tensor([1.0])

        spans, input_ids, offsets, special_mask = colbert_method._resolve_spans(
            text_b, top_indices, top_scores, window_size=2, stride=1, n_doc_tokens=0
        )

        assert spans == []
