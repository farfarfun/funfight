from __future__ import annotations

import os
import shutil

import numpy as np
from data_type import DataType
from farlog import getLogger
from feature_predict import feature_digest
from flink_ai_flow.pyflink import FlinkFunctionContext
from flink_ai_flow.pyflink.user_define_executor import Executor
from pyflink.table import DataTypes, ScalarFunction, Table
from pyflink.table.descriptors import FileSystem, OldCsv, Schema
from pyflink.table.udf import FunctionContext, udf
from pyproxima2 import *

logger = getLogger("funfight.tianchi.t531800.proxima_executor")


class SearchUDF(ScalarFunction):
    """单近邻检索 UDF：返回与输入向量最接近的一个 key。"""

    def __init__(self, index_path: str, element_type: DataType) -> None:
        """记录索引路径与向量元素类型，索引本身延迟到 ``open()`` 里加载。"""
        self.path = index_path
        self.topk = 1
        self.element_type = element_type
        self.ctx = None

    def open(self, function_context: FunctionContext) -> None:
        """加载 Proxima 索引文件并建立检索上下文。"""
        container = IndexContainer(name='MMapFileContainer', params={})
        container.load(self.path)
        searcher = IndexSearcher("ClusteringSearcher")
        self.ctx = searcher.load(container).create_context(topk=self.topk)

    def eval(self, vec: str) -> str | None:
        """检索空格分隔的特征向量 ``vec``，返回最近邻的 key（无结果时为 ``None``）。

        Raises:
            RuntimeError: ``open()`` 未被调用导致检索上下文缺失。
        """
        if self.ctx is None:
            raise RuntimeError(f"{type(self).__name__}: 检索上下文未初始化（open() 未被调用）")
        if len(vec) != 0 and not vec.isspace():
            vector = np.array([float(v) for v in vec.split(' ')]).astype(self.element_type.to_numpy_type())
            results = self.ctx.search(query=vector)
            return results[0][0].key()
        return None


class SearchUDTF3(ScalarFunction):
    """在线链路的检索 UDF：把近邻结果聚类成「可能是同一人」的分组编号。"""

    def __init__(self, index_path: str, element_type: DataType) -> None:
        """记录索引路径、向量元素类型，并初始化分组状态。"""
        self.path = index_path
        self.topk = 1
        self.element_type = element_type
        self.ctx = None
        self.map = {0: []}
        self.may_be_person_num = 0

    def open(self, function_context: FunctionContext) -> None:
        """加载 Proxima 索引文件并建立检索上下文。"""
        container = IndexContainer(name='MMapFileContainer', params={})
        container.load(self.path)
        searcher = IndexSearcher("ClusteringSearcher")
        self.ctx = searcher.load(container).create_context(topk=self.topk)

    def eval(self, vec: str) -> int | None:
        """检索空格分隔的特征向量 ``vec``，返回其所属的分组编号（无结果时为 ``None``）。

        Raises:
            RuntimeError: ``open()`` 未被调用导致检索上下文缺失。
        """
        if self.ctx is None:
            raise RuntimeError(f"{type(self).__name__}: 检索上下文未初始化（open() 未被调用）")
        if len(vec) != 0 and not vec.isspace():
            logger.debug("SearchUDTF3 收到向量: {}", feature_digest(vec))
            vector = np.array([float(v) for v in vec.split(' ')]).astype(self.element_type.to_numpy_type())
            results = self.ctx.search(query=vector)
            near_key = results[0][0].key
            for k, v in self.map.items():
                if near_key not in v:
                    self.map[self.may_be_person_num] = []
                    self.map[self.may_be_person_num].append(near_key)
                    self.may_be_person_num += 1
                    return self.may_be_person_num - 1
                else:
                    self.map[k].append(near_key)
                    return k
        return None


class SearchUDTF(ScalarFunction):
    """批 / 离线链路的检索 UDF：返回与输入向量最接近的一个 key（字符串形式）。"""

    def __init__(self, index_path: str, element_type: DataType) -> None:
        """记录索引路径与向量元素类型，索引本身延迟到 ``open()`` 里加载。"""
        self.path = index_path
        self.topk = 1
        self.element_type = element_type
        self.ctx = None

    def open(self, function_context: FunctionContext) -> None:
        """加载 Proxima 索引文件并建立检索上下文。"""
        container = IndexContainer(name='MMapFileContainer', params={})
        container.load(self.path)
        searcher = IndexSearcher("ClusteringSearcher")
        self.ctx = searcher.load(container).create_context(topk=self.topk)

    def eval(self, vec: str) -> str | None:
        """检索空格分隔的特征向量 ``vec``，返回最近邻的 key（无结果时为 ``None``）。

        Raises:
            RuntimeError: ``open()`` 未被调用导致检索上下文缺失。
        """
        if self.ctx is None:
            raise RuntimeError(f"{type(self).__name__}: 检索上下文未初始化（open() 未被调用）")
        if len(vec) != 0 and not vec.isspace():
            vector = np.array([float(v) for v in vec.split(' ')]).astype(self.element_type.to_numpy_type())
            results = self.ctx.search(query=vector)
            for i in results[0]:
                return str(i.key())
        return None


class SearchExecutor(Executor):
    def __init__(self, index_path: str, element_type: DataType, dimension: int):
        super().__init__()
        self.path = index_path
        self.element_type = element_type
        self.dimension = dimension

    def execute(self, function_context: FlinkFunctionContext, input_list: list[Table]) -> list[Table]:
        t_env = function_context.get_table_env()
        table = input_list[0]
        t_env.register_function("search", udf(SearchUDTF(self.path, self.element_type),
                                              DataTypes.STRING(), DataTypes.STRING()))
        return [table.select("face_id, search(feature_data) as near_id")]


class SearchExecutor3(Executor):
    def __init__(self, index_path: str, element_type: DataType, dimension: int):
        super().__init__()
        self.path = index_path
        self.element_type = element_type
        self.dimension = dimension

    def execute(self, function_context: FlinkFunctionContext, input_list: list[Table]) -> list[Table]:
        t_env = function_context.get_table_env()
        table = input_list[0]
        t_env.register_function("search", udf(SearchUDTF3(self.path, self.element_type),
                                              DataTypes.STRING(), DataTypes.INT()))
        return [table.select("face_id, device_id, search(feature_data) as near_id")]


class BuildIndexUDF(ScalarFunction):
    """建索引 UDF：把每条特征向量写入 Proxima ``IndexHolder``，任务结束时统一构建并落盘。"""

    def __init__(self, index_path: str, element_type: DataType, dimension: int) -> None:
        """记录索引输出路径、向量元素类型与维度；holder/builder 延迟到 ``open()`` 里创建。"""
        self.element_type = element_type
        self.dimension = dimension
        self.path = index_path
        self._docs = 100000
        self.holder = None
        self.builder = None

    def open(self, function_context: FunctionContext) -> None:
        """创建 Proxima ``IndexHolder``/``IndexBuilder``。"""
        self.holder = IndexHolder(type=self.element_type.to_proxima_type(), dimension=self.dimension)
        self.builder = IndexBuilder(
            name="ClusteringBuilder",
            meta=IndexMeta(type=self.element_type.to_proxima_type(), dimension=self.dimension),
            params={'proxima.hc.builder.max_document_count': self._docs})

    def eval(self, key: str, vec: str) -> str | None:
        """把空格分隔的特征向量 ``vec`` 以 ``key`` 写入 holder，返回写入的 ``key``。

        未写入（``vec`` 为空）时返回 ``None``。
        """
        if len(vec) != 0 and not vec.isspace():
            vector = [float(v) for v in vec.split(' ')]
            self.holder.emplace(int(key), np.array(vector).astype(self.element_type.to_numpy_type()))
            logger.debug("BuildIndexUDF 写入向量: key={}, vec={}", key, feature_digest(vec))
            return key
        return None

    def close(self) -> None:
        """训练并构建索引，落盘到 ``self.path``。"""
        self.builder.train_and_build(self.holder).dump(IndexDumper(path=self.path))


class BuildIndexExecutor(Executor):
    def __init__(self, index_path: str, element_type: DataType, dimension: int):
        self.element_type = element_type
        self.dimension = dimension
        self.path = index_path
        self._docs = 100000

    def execute(self, function_context: FlinkFunctionContext, input_list: list[Table]) -> list[Table]:
        t_env = function_context.get_table_env()
        statement_set = function_context.get_statement_set()
        table = input_list[0]
        t_env.register_function("build_index", udf(BuildIndexUDF(self.path, self.element_type, self.dimension),
                                                   [DataTypes.STRING(), DataTypes.STRING()], DataTypes.STRING()))
        dummy_output_path = '/tmp/indexed_key'
        if os.path.exists(dummy_output_path):
            if os.path.isdir(dummy_output_path):
                shutil.rmtree(dummy_output_path)
            else:
                os.remove(dummy_output_path)
        t_env.connect(FileSystem().path(dummy_output_path)) \
            .with_format(OldCsv()
                         .field('key', DataTypes.STRING())) \
            .with_schema(Schema()
                         .field('key', DataTypes.STRING())) \
            .create_temporary_table('train_sink')
        statement_set.add_insert("train_sink", table.select("build_index(uuid, feature_data)"))
        return []
