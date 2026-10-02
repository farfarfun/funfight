"""特征向量预测的公共逻辑。

单独拆成一个模块的原因：`tianchi_executor.py` 在模块顶层就要 import
`ai_flow`/`pyflink`/`zoo.serving.client` 这些比赛当年的专用包，现代环境里装不上，
于是整个模块都无法被导入、也无法被测试。本模块只依赖标准库与 `farlog`，
因此预测路径的解析、脱敏与失败语义可以被单元测试覆盖。

与 `data_type.py` 一样，本模块按 `python_codes/` 目录被放进 `PYTHONPATH` 的方式
被扁平导入（见 `../../README.md` 第 2 步）。
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol

from farlog import getLogger

logger = getLogger("funfight.tianchi.t531800.feature_predict")

__all__ = ["FeaturePredictError", "PredictClient", "feature_digest", "predict_feature"]


class FeaturePredictError(RuntimeError):
    """特征向量预测失败时抛出的领域异常。

    用于把「输入无法解析为浮点向量」「cluster serving 返回了无法识别的结果」
    这类不可恢复的失败包装成带上下文的异常，而不是静默转换成空结果。
    异常消息里只带脱敏摘要（见 :func:`feature_digest`），不回显原始特征内容。
    """


class PredictClient(Protocol):
    """cluster serving 同步预测客户端需要满足的最小协议。

    实际实现是 `zoo.serving.client.InputQueue`，这里只声明本模块用到的方法，
    便于单元测试传入替身。
    """

    def predict(self, request: str) -> Any:
        """提交一条同步预测请求，返回 cluster serving 的原始响应。"""
        ...


def feature_digest(feature_data: object) -> str:
    """返回特征数据的脱敏摘要，供日志和异常消息使用。

    摘要只包含类型名、UTF-8 字节长度与 SHA256 前 12 位，**不含任何原始特征
    内容**。本仓库处理的是人脸特征向量，按 SPEC §8.1 不应把完整特征写进日志。

    Args:
        feature_data: 任意待摘要的输入，通常是空格分隔的特征向量字符串。

    Returns:
        形如 ``type=str len=11 sha256=0123456789ab`` 的单行摘要。
    """
    if isinstance(feature_data, str):
        raw = feature_data.encode("utf-8", errors="replace")
    elif isinstance(feature_data, (bytes, bytearray)):
        raw = bytes(feature_data)
    else:
        raw = repr(feature_data).encode("utf-8", errors="replace")
    return (
        f"type={type(feature_data).__name__} "
        f"len={len(raw)} "
        f"sha256={hashlib.sha256(raw).hexdigest()[:12]}"
    )


def parse_feature(feature_data: str, udf_name: str) -> list[float]:
    """把空格分隔的特征向量字符串解析为浮点数组。

    Args:
        feature_data: 空格分隔的特征向量字符串。
        udf_name: 注册到 Flink 的 UDF 名，只用于错误上下文。

    Returns:
        解析出的浮点数列表。

    Raises:
        FeaturePredictError: 输入不是字符串，或任一元素无法转成浮点数。
            消息里只带脱敏摘要，不含原始特征内容。
    """
    if not isinstance(feature_data, str):
        raise FeaturePredictError(
            f"{udf_name}: 特征数据不是字符串（{feature_digest(feature_data)}）"
        )
    try:
        return [float(element) for element in feature_data.split(" ")]
    except ValueError as e:
        raise FeaturePredictError(
            f"{udf_name}: 特征数据无法解析为浮点向量（{feature_digest(feature_data)}）"
        ) from e


def predict_feature(input_api: PredictClient, feature_data: str, udf_name: str) -> str:
    """调用 cluster serving 对单条特征向量做预测。

    失败时**不返回空串**：解析失败和响应不可识别都抛 :class:`FeaturePredictError`，
    cluster serving 客户端自身抛出的异常（网络、超时等）原样向上传播，由 Flink
    任务失败，而不是把错误伪装成一条空结果继续往下游写。

    Args:
        input_api: 已在 UDF ``open()`` 里建好的 cluster serving 同步客户端。
        feature_data: 空格分隔的特征向量字符串。
        udf_name: 注册到 Flink 的 UDF 名，只用于错误上下文。

    Returns:
        空格分隔的预测结果字符串。

    Raises:
        FeaturePredictError: 输入无法解析，或 cluster serving 返回了非字符串结果。
    """
    feature_samples = parse_feature(feature_data, udf_name)
    request_instances = {"instances": [{"ids": feature_samples}]}
    response = input_api.predict(json.dumps(request_instances))
    if not isinstance(response, str):
        raise FeaturePredictError(
            f"{udf_name}: cluster serving 返回了非字符串结果 "
            f"(type={type(response).__name__})，输入 {feature_digest(feature_data)}"
        )
    return " ".join(response.replace("[", "").replace("]", "").split(","))
