"""feature_predict.py 的单元测试：覆盖解析/摘要/预测的正常、边界、失败路径。"""

import pytest
from feature_predict import (
    FeaturePredictError,
    feature_digest,
    parse_feature,
    predict_feature,
)


def test_feature_digest_does_not_leak_raw_content():
    """摘要只应包含类型/长度/哈希，不应原样出现输入内容。"""
    secret = "0.1 0.2 0.3 0.4"
    digest = feature_digest(secret)
    assert secret not in digest
    assert "type=str" in digest
    assert "len=" in digest
    assert "sha256=" in digest


def test_parse_feature_normal_path():
    """正常路径：空格分隔的浮点字符串被解析为浮点列表。"""
    assert parse_feature("0.1 0.2 0.3", "udf") == pytest.approx([0.1, 0.2, 0.3])


def test_parse_feature_rejects_non_float_token():
    """失败路径：任一元素无法转成浮点数时抛出带上下文的领域异常，而不是静默返回空值。"""
    with pytest.raises(FeaturePredictError, match="udf"):
        parse_feature("0.1 not_a_float", "udf")


def test_parse_feature_rejects_non_string_input():
    """边界路径：非字符串输入也应抛出领域异常，而不是在 .split 上抛出无上下文的 AttributeError。"""
    with pytest.raises(FeaturePredictError):
        parse_feature(None, "udf")  # type: ignore[arg-type]


class _FakeClient:
    """predict_feature 的测试替身，实现 PredictClient 协议。"""

    def __init__(self, response):
        self._response = response

    def predict(self, request: str):
        self._last_request = request
        return self._response


def test_predict_feature_normal_path():
    """正常路径：cluster serving 返回形如 '[1,2,3]' 的字符串，解析成空格分隔结果。"""
    client = _FakeClient("[1,2,3]")
    result = predict_feature(client, "0.1 0.2", "predict1")
    assert result == "1 2 3"


def test_predict_feature_raises_on_non_string_response():
    """失败路径：cluster serving 返回非字符串时抛出领域异常，而不是返回空串掩盖错误。"""
    client = _FakeClient(response=None)
    with pytest.raises(FeaturePredictError, match="predict1"):
        predict_feature(client, "0.1 0.2", "predict1")


def test_predict_feature_propagates_client_errors():
    """失败路径：客户端自身异常（网络/超时等）应原样向上传播，不被吞掉。"""

    class _RaisingClient:
        def predict(self, request: str):
            raise ConnectionError("cluster serving 不可达")

    with pytest.raises(ConnectionError):
        predict_feature(_RaisingClient(), "0.1 0.2", "predict1")
