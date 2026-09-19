from abc import abstractmethod
from enum import Enum

from pyflink.table import DataTypes
from pyproxima2 import *


class DataType(object):
    """Proxima 向量类型与 Flink/numpy 类型之间的转换接口。"""

    @abstractmethod
    def to_proxima_type(self):
        """返回对应的 Proxima IndexMeta 类型。"""
        pass

    @abstractmethod
    def to_flink_type(self):
        """返回对应的 Flink DataTypes 类型。"""
        pass

    @abstractmethod
    def to_numpy_type(self) -> str:
        """返回对应的 numpy dtype 字符码。"""
        pass


class FloatDataType(DataType):
    """单精度浮点向量类型。"""

    def to_proxima_type(self):
        """返回 Proxima 的 FP32 类型。"""
        return IndexMeta.FT_FP32

    def to_flink_type(self):
        """返回 Flink 的 FLOAT 类型。"""
        return DataTypes.FLOAT()

    def to_numpy_type(self) -> str:
        """返回 numpy 的 float 类型码 'f'。"""
        return 'f'


class DoubleDataType(DataType):
    """双精度浮点向量类型。"""

    def to_proxima_type(self):
        """返回 Proxima 的 FP64 类型。"""
        return IndexMeta.FT_FP64

    def to_flink_type(self):
        """返回 Flink 的 DOUBLE 类型。"""
        return DataTypes.DOUBLE()

    def to_numpy_type(self) -> str:
        """返回 numpy 的 float 类型码 'f'（与原实现保持一致）。"""
        return 'f'
