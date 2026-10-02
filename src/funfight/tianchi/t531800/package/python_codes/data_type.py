from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pyflink.table import DataTypes
from pyproxima2 import IndexMeta


class DataType(ABC):
    """Proxima 向量类型与 Flink/numpy 类型之间的转换接口。

    三个方法分别给出同一种向量元素类型在三套类型系统里的表示，供
    `proxima_executor.py` 里的建索引/检索 UDF 统一取用。
    """

    @abstractmethod
    def to_proxima_type(self) -> Any:
        """返回对应的 Proxima `IndexMeta` 特征类型（`pyproxima2` 的 C 扩展枚举值）。"""

    @abstractmethod
    def to_flink_type(self) -> Any:
        """返回对应的 Flink `DataTypes` 类型（`pyflink.table.types.DataType`）。"""

    @abstractmethod
    def to_numpy_type(self) -> str:
        """返回对应的 numpy dtype 字符码，宽度必须与 Proxima 特征类型一致。"""


class FloatDataType(DataType):
    """单精度（32 位）浮点向量类型。"""

    def to_proxima_type(self) -> Any:
        """返回 Proxima 的 FP32 类型。"""
        return IndexMeta.FT_FP32

    def to_flink_type(self) -> Any:
        """返回 Flink 的 FLOAT 类型。"""
        return DataTypes.FLOAT()

    def to_numpy_type(self) -> str:
        """返回 numpy 的 float32 类型码 ``'f'``。"""
        return "f"


class DoubleDataType(DataType):
    """双精度（64 位）浮点向量类型。"""

    def to_proxima_type(self) -> Any:
        """返回 Proxima 的 FP64 类型。"""
        return IndexMeta.FT_FP64

    def to_flink_type(self) -> Any:
        """返回 Flink 的 DOUBLE 类型。"""
        return DataTypes.DOUBLE()

    def to_numpy_type(self) -> str:
        """返回 numpy 的 float64 类型码 ``'d'``。"""
        return "d"
