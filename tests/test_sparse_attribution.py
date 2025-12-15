import pytest
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
            # CJK 检查必须放在 isalnum 之前，因为中文字符的 isalnum() 也返回 True
            if self._is_cjk(text[idx]):
                idx += 1
            elif text[idx].isascii() and text[idx].isalnum():
                # 只处理 ASCII 字母数字（英文、数字）
                while idx < length and text[idx].isascii() and text[idx].isalnum():
                    idx += 1
            else:
                while (
                    idx < length
                    and not text[idx].isspace()
                    and not (text[idx].isascii() and text[idx].isalnum())
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
def sparse_method(mock_model):
    """Create a SparseAttribution instance with mocked model."""
    with patch(
        "src.attribution.token_wise.sparse._get_cached_bge_m3_model",
        return_value=mock_model,
    ):
        config = {
            "model_name": "mock-model",
            "window_size": 3,
            "window_overlap": 1,
            "top_k_spans": 2,
            "top_k_tokens": 3,
            "use_fp16": False,
        }
        yield SparseAttribution(config)


# ============== Helper Functions ==============

def weights_from_tokens(tokenizer, token_weights):
    """Convert token->weight dict to token_id->weight dict."""
    return {
        tokenizer.token_id_for(token): weight
        for token, weight in token_weights.items()
    }


def queue_pair(mock_model, tokenizer, weights_a, weights_b):
    """Queue a pair of lexical weights for text_a and text_b."""
    mock_model.queue_lexical_weights([
        weights_from_tokens(tokenizer, weights_a),
        weights_from_tokens(tokenizer, weights_b)
    ])


# ============== Tests ==============

class TestSparseAttributionInit:
    """Tests for SparseAttribution initialization."""

    def test_initialization_uses_config_values(self, mock_model):
        """Test that config values are correctly applied."""
        with patch(
            "src.attribution.token_wise.sparse._get_cached_bge_m3_model",
            return_value=mock_model,
        ):
            config = {
                "model_name": "custom-model",
                "window_size": 4,
                "window_overlap": 2,
                "top_k_spans": 5,
                "top_k_tokens": 7,
                "use_fp16": True,
            }
            method = SparseAttribution(config)

            assert method.model == mock_model
            assert method.window_size == 4
            assert method.window_overlap == 2
            assert method.top_k_spans == 5
            assert method.top_k_tokens == 7
            assert method.use_fp16 is True


class TestTokenContributions:
    """Tests for token contribution calculation."""

    def test_computes_intersection_scores(self, sparse_method, mock_model, mock_tokenizer):
        """Test that intersection scores are computed correctly."""
        queue_pair(
            mock_model, mock_tokenizer,
            {"AI": 0.8, "医疗": 0.5},
            {"AI": 0.4, "医疗": 0.2, "天气": 0.9},
        )

        contributions, total = sparse_method._get_token_contributions(
            "AI 文本", "医疗 文本"
        )

        assert contributions["AI"] == pytest.approx(0.32)
        assert contributions["医疗"] == pytest.approx(0.1)
        assert total == pytest.approx(0.42)


class TestWindowScores:
    """Tests for sliding window score computation."""

    def test_finds_highest_scoring_span(self, sparse_method):
        """Test that the highest scoring span is correctly identified."""
        text_b = "AI技术提升了医疗诊断的准确性。"
        # 设置分数使得多 token 窗口的平均分高于单 token 窗口：
        # - 窗口 [升, 了, 医] 平均分 = 1.5/1 = 1.5（只有"医"贡献）
        # - 窗口 [医, 疗, 诊] 平均分 = (1.5+2.5+2.5)/3 = 2.17（三个 token 都贡献）
        contributions = {"医": 1.5, "疗": 2.5, "诊": 2.5, "断": 0.3, "AI": 0.2}

        windows = sparse_method._compute_window_scores(text_b, contributions)
        best_window = max(windows, key=lambda w: w["score"])

        # 最高分窗口应包含医疗相关的中文字符（医、疗、诊）
        assert best_window["score"] > 0
        assert best_window["contributing_tokens"] >= 2


class TestExtract:
    """Tests for the main extract method."""

    def test_returns_spans_and_metadata(self, sparse_method, mock_model, mock_tokenizer):
        """Test that extract returns proper spans and metadata."""
        weights = {"AI": 0.6, "医": 0.4, "疗": 0.3, "诊": 0.2, "断": 0.1}
        queue_pair(mock_model, mock_tokenizer, {"AI": 0.7, "医": 0.5, "疗": 0.4}, weights)
        queue_pair(mock_model, mock_tokenizer, {"AI": 0.7, "医": 0.5, "疗": 0.4}, weights)

        text_a = "AI改进医疗诊断"
        text_b = "AI系统提升医疗诊断准确性。"
        result = sparse_method.extract(text_a, text_b)

        assert result.spans
        assert result.metadata["num_contributing_tokens"] >= 2
        top_tokens = result.metadata["top_contributing_tokens"]
        assert top_tokens
        assert top_tokens[0]["token"] == "AI"

    def test_handles_no_common_tokens(self, sparse_method, mock_model, mock_tokenizer):
        """Test handling of texts with no common tokens."""
        queue_pair(mock_model, mock_tokenizer, {"猫": 1.0}, {"狗": 1.0})
        queue_pair(mock_model, mock_tokenizer, {"猫": 1.0}, {"狗": 1.0})

        result = sparse_method.extract("猫的故事", "狗的故事")

        assert not result.metadata["top_contributing_tokens"]
        assert all(span.score == 0 for span in result.spans)

    def test_validates_inputs(self, sparse_method):
        """Test that invalid inputs raise ValueError."""
        with pytest.raises(ValueError):
            sparse_method.extract("", "text")

        with pytest.raises(ValueError):
            sparse_method.extract("text", "   ")


class TestTokenizeWithPositions:
    """Tests for tokenization with position tracking."""

    def test_english_text(self, sparse_method):
        """Test tokenization of pure English text."""
        # "AI improves" -> [("AI", 0, 2), ("improves", 3, 11)]
        text = "AI improves"
        tokens = sparse_method._tokenize_with_positions(text)

        assert len(tokens) == 2

        token_0 = tokens[0]
        assert token_0["token"] == "AI"
        assert token_0["start"] == 0
        assert token_0["end"] == 2
        assert text[token_0["start"]:token_0["end"]] == "AI"

        token_1 = tokens[1]
        assert token_1["token"] == "improves"
        assert token_1["start"] == 3
        assert token_1["end"] == 11
        assert text[token_1["start"]:token_1["end"]] == "improves"

    def test_chinese_text(self, sparse_method):
        """Test tokenization of Chinese text mixed with English."""
        # "AI技术" -> [("AI", 0, 2), ("技", 2, 3), ("术", 3, 4)]
        text = "AI技术"
        tokens = sparse_method._tokenize_with_positions(text)

        assert len(tokens) == 3

        # "AI" 是英文，占位置 0-2
        token_0 = tokens[0]
        assert token_0["token"] == "AI"
        assert token_0["start"] == 0
        assert token_0["end"] == 2
        assert text[token_0["start"]:token_0["end"]] == "AI"

        # "技" 是单个中文字符，占位置 2-3
        token_1 = tokens[1]
        assert token_1["token"] == "技"
        assert token_1["start"] == 2
        assert token_1["end"] == 3
        assert text[token_1["start"]:token_1["end"]] == "技"

        # "术" 是单个中文字符，占位置 3-4
        token_2 = tokens[2]
        assert token_2["token"] == "术"
        assert token_2["start"] == 3
        assert token_2["end"] == 4
        assert text[token_2["start"]:token_2["end"]] == "术"

    def test_mixed_text_with_space(self, sparse_method):
        """Test tokenization of mixed Chinese-English text with spaces."""
        # "Hello 世界" -> [("Hello", 0, 5), ("世", 6, 7), ("界", 7, 8)]
        text = "Hello 世界"
        tokens = sparse_method._tokenize_with_positions(text)

        assert len(tokens) == 3

        token_0 = tokens[0]
        assert token_0["token"] == "Hello"
        assert token_0["start"] == 0
        assert token_0["end"] == 5
        assert text[token_0["start"]:token_0["end"]] == "Hello"

        token_1 = tokens[1]
        assert token_1["token"] == "世"
        assert token_1["start"] == 6
        assert token_1["end"] == 7
        assert text[token_1["start"]:token_1["end"]] == "世"

        token_2 = tokens[2]
        assert token_2["token"] == "界"
        assert token_2["start"] == 7
        assert token_2["end"] == 8
        assert text[token_2["start"]:token_2["end"]] == "界"
