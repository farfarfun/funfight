import shutil
from subprocess import Popen
from typing import List

import numpy as np
from flink_ai_flow.pyflink import FlinkFunctionContext
from flink_ai_flow.pyflink.user_define_executor import Executor
from pyflink.table import Table, ScalarFunction, DataTypes
from pyflink.table.descriptors import FileSystem, OldCsv, Schema
from pyflink.table.udf import udf
from pyproxima2 import *

from data_type import DataType


class SearchUDF(ScalarFunction):
    def __init__(self, index_path: str, element_type: DataType):
        self.path = index_path
        self.topk = 1
        self.element_type = element_type
        self.ctx = None

    def open(self, function_context):
        container = IndexContainer(name='MMapFileContainer', params={})
        container.load(self.path)
        searcher = IndexSearcher("ClusteringSearcher")
        self.ctx = searcher.load(container).create_context(topk=self.topk)

    def eval(self, vec):
        if self.ctx is None:
            raise RuntimeError()
        if len(vec) != 0 and not vec.isspace():
            vec = np.array([float(v) for v in vec.split(' ')]).astype(self.element_type.to_numpy_type())
            results = self.ctx.search(query=vec)
            return results[0][0].key()
        return None


class SearchUDTF3(ScalarFunction):
    def __init__(self, index_path: str, element_type: DataType):
        self.path = index_path
        self.topk = 1
        self.element_type = element_type
        self.ctx = None
        self.map = {0: []}
        self.may_be_person_num = 0

    def open(self, function_context):
        container = IndexContainer(name='MMapFileContainer', params={})
        container.load(self.path)
        searcher = IndexSearcher("ClusteringSearcher")
        self.ctx = searcher.load(container).create_context(topk=self.topk)

    def eval(self, vec):
        if self.ctx is None:
            raise RuntimeError()
        with open('/root/test', 'a') as f:
            if len(vec) != 0 and not vec.isspace():
                f.write(str(vec))
                f.write('\n')
                vec = np.array([float(v) for v in vec.split(' ')]).astype(self.element_type.to_numpy_type())
                results = self.ctx.search(query=vec)
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
    def __init__(self, index_path: str, element_type: DataType):
        self.path = index_path
        self.topk = 1
        self.element_type = element_type
        self.ctx = None

    def open(self, function_context):
        container = IndexContainer(name='MMapFileContainer', params={})
        container.load(self.path)
        searcher = IndexSearcher("ClusteringSearcher")
        self.ctx = searcher.load(container).create_context(topk=self.topk)

    def eval(self, vec):
        if self.ctx is None:
            raise RuntimeError()
        if len(vec) != 0 and not vec.isspace():
            vec = np.array([float(v) for v in vec.split(' ')]).astype(self.element_type.to_numpy_type())
            results = self.ctx.search(query=vec)
            for i in results[0]:
                return str(i.key())
        return None


class SearchExecutor(Executor):
    def __init__(self, index_path: str, element_type: DataType, dimension: int):
        super().__init__()
        self.path = index_path
        self.element_type = element_type
        self.dimension = dimension

    def execute(self, function_context: FlinkFunctionContext, input_list: List[Table]) -> List[Table]:
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

    def execute(self, function_context: FlinkFunctionContext, input_list: List[Table]) -> List[Table]:
        t_env = function_context.get_table_env()
        table = input_list[0]
        Popen('rm -rf /root/test', shell=True)
        t_env.register_function("search", udf(SearchUDTF3(self.path, self.element_type),
                                              DataTypes.STRING(), DataTypes.INT()))
        return [table.select("face_id, device_id, search(feature_data) as near_id")]


class BuildIndexUDF(ScalarFunction):
    def __init__(self, index_path: str, element_type: DataType, dimension: int):
        self.element_type = element_type
        self.dimension = dimension
        self.path = index_path
        self._docs = 100000
        self.holder = None
        self.builder = None

    def open(self, function_context):
        self.holder = IndexHolder(type=self.element_type.to_proxima_type(), dimension=self.dimension)
        self.builder = IndexBuilder(
            name="ClusteringBuilder",
            meta=IndexMeta(type=self.element_type.to_proxima_type(), dimension=self.dimension),
            params={'proxima.hc.builder.max_document_count': self._docs})

    def eval(self, key, vec):
        with open('/root/debug', 'a') as f:
            if len(vec) != 0 and not vec.isspace():
                vector = [float(v) for v in vec.split(' ')]
                self.holder.emplace(int(key), np.array(vector).astype(self.element_type.to_numpy_type()))
                f.write(str(key) + ' ' + str(vec))
                f.write('\n')
                return key
            return None

    def close(self):
        self.builder.train_and_build(self.holder).dump(IndexDumper(path=self.path))


class BuildIndexExecutor(Executor):
    def __init__(self, index_path: str, element_type: DataType, dimension: int):
        self.element_type = element_type
        self.dimension = dimension
        self.path = index_path
        self._docs = 100000

    def execute(self, function_context: FlinkFunctionContext, input_list: List[Table]) -> List[Table]:
        t_env = function_context.get_table_env()
        statement_set = function_context.get_statement_set()
        table = input_list[0]
        Popen('rm -rf /root/debug', shell=True)
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
