from __future__ import annotations

import os
import shutil
from typing import Any

from ai_flow import ExampleMeta, update_notification
from feature_predict import predict_feature
from flink_ai_flow.pyflink.user_define_executor import (
    Executor,
    FlinkFunctionContext,
    SinkExecutor,
    SourceExecutor,
    TableEnvCreator,
)
from pyflink.dataset import ExecutionEnvironment
from pyflink.datastream import StreamExecutionEnvironment
from pyflink.table import (
    BatchTableEnvironment,
    CsvTableSink,
    DataTypes,
    EnvironmentSettings,
    ScalarFunction,
    StreamTableEnvironment,
    Table,
    TableEnvironment,
)
from pyflink.table.udf import FunctionContext, udf
from zoo.serving.client import InputQueue


class ClusterServingPredict(ScalarFunction):
    """把特征向量交给 cluster serving 做同步预测的 Flink 标量函数。

    三条预测链路（批训练、离线历史、在线流）此前各自内联了一份完全相同的实现，
    这里合并为一个按 UDF 名区分的公共实现，避免同类缺陷只修一处。
    """

    def __init__(self, udf_name: str):
        """记录注册到 Flink 的 UDF 名，仅用于日志与异常上下文。"""
        super().__init__()
        self._udf_name = udf_name
        self._input_api: InputQueue | None = None

    def open(self, function_context: FunctionContext) -> None:
        """建立 cluster serving 同步客户端。"""
        self._input_api = InputQueue(
            host="localhost",
            port="6379",
            sync=True,
            frontend_url="http://127.0.0.1:10020",
        )

    def eval(self, feature_data: str) -> str:
        """对一条特征向量做预测，返回空格分隔的预测结果。

        Raises:
            RuntimeError: ``open()`` 未被调用导致客户端缺失。
            feature_predict.FeaturePredictError: 输入无法解析为浮点向量，
                或 cluster serving 返回了非字符串结果。
        """
        if self._input_api is None:
            raise RuntimeError(
                f"{self._udf_name}: cluster serving 客户端未初始化（open() 未被调用）"
            )
        return predict_feature(self._input_api, feature_data, self._udf_name)


class StreamTableEnvCreatorBuildIndex(TableEnvCreator):
    """建索引作业用的流式 TableEnvironment 工厂（并行度 100）。"""

    def create_table_env(self) -> tuple[Any, Any, Any]:
        """返回 (StreamExecutionEnvironment, StreamTableEnvironment, StatementSet)。"""
        stream_env = StreamExecutionEnvironment.get_execution_environment()
        stream_env.set_parallelism(100)
        t_env = StreamTableEnvironment.create(
            stream_env,
            environment_settings=EnvironmentSettings.new_instance()
                .in_streaming_mode().use_blink_planner().build())
        statement_set = t_env.create_statement_set()
        t_env.get_config().set_python_executable('/usr/bin/python3')
        t_env.get_config().get_configuration().set_boolean("python.fn-execution.memory.managed", True)
        return stream_env, t_env, statement_set


class StreamTableEnvCreator(TableEnvCreator):
    """检索/在线作业用的流式 TableEnvironment 工厂（并行度 1）。"""

    def create_table_env(self) -> tuple[Any, Any, Any]:
        """返回 (StreamExecutionEnvironment, StreamTableEnvironment, StatementSet)。"""
        stream_env = StreamExecutionEnvironment.get_execution_environment()
        stream_env.set_parallelism(1)
        t_env = StreamTableEnvironment.create(
            stream_env,
            environment_settings=EnvironmentSettings.new_instance()
                .in_streaming_mode().use_blink_planner().build())
        statement_set = t_env.create_statement_set()
        t_env.get_config().set_python_executable('/usr/bin/python3')
        t_env.get_config().get_configuration().set_boolean("python.fn-execution.memory.managed", True)
        return stream_env, t_env, statement_set


class BatchTableEnvCreator(TableEnvCreator):
    """批作业用的 TableEnvironment 工厂（并行度 1）。"""

    def create_table_env(self) -> tuple[Any, Any, Any]:
        """返回 (ExecutionEnvironment, BatchTableEnvironment, StatementSet)。"""
        exec_env = ExecutionEnvironment.get_execution_environment()
        t_env = BatchTableEnvironment.create(
            environment_settings=EnvironmentSettings.new_instance().in_batch_mode().use_blink_planner().build())
        t_env._j_tenv.getPlanner().getExecEnv().setParallelism(1)
        statement_set = t_env.create_statement_set()
        t_env.get_config().set_python_executable('/usr/bin/python3')
        t_env.get_config().get_configuration().set_boolean("python.fn-execution.memory.managed", True)
        return exec_env, t_env, statement_set


class ReadTrainExample(SourceExecutor):
    """以 filesystem+csv connector 读入训练集，建表 ``training_table``。"""

    def execute(self, function_context: FlinkFunctionContext) -> Table:
        """按 Example 元数据里的 batch_uri 建表并返回对应 Table。"""
        table_env: TableEnvironment = function_context.get_table_env()
        path = function_context.get_example_meta().batch_uri
        ddl = f"""create table training_table(
                                uuid varchar,
                                face_id varchar,
                                device_id varchar,
                                feature_data varchar
                    ) with (
                        'connector.type' = 'filesystem',
                        'format.type' = 'csv',
                        'connector.path' = '{path}',
                        'format.ignore-first-line' = 'false',
                        'format.field-delimiter' = ';'
                    )"""
        table_env.execute_sql(ddl)
        return table_env.from_path('training_table')


class FindHistory(Executor):
    """把检索结果与训练集 join，拿到每个近邻 uuid 对应的历史 face_id。"""

    def execute(self, function_context: FlinkFunctionContext, input_list: list[Table]) -> list[Table]:
        """返回只含一个元素的列表：join 后的 Table。"""
        t_env = function_context.get_table_env()
        table_0 = input_list[0]
        t_env.create_temporary_view('near_table', table_0)
        join_query = """select
        near_table.face_id, training_table.face_id
        from training_table
        inner join near_table
        on training_table.uuid=near_table.near_id"""
        return [t_env.sql_query(join_query)]


class ReadPredictExample(SourceExecutor):
    """以 filesystem+csv connector 读入历史测试集，建表 ``test_table``。"""

    def execute(self, function_context: FlinkFunctionContext) -> Table:
        """按 Example 元数据里的 batch_uri 建表并返回对应 Table。"""
        table_env: TableEnvironment = function_context.get_table_env()
        batch_uri = function_context.get_example_meta().batch_uri
        ddl = f"""create table test_table (
                face_id varchar,
                feature_data varchar
                )with (
                        'connector.type' = 'filesystem',
                        'format.type' = 'csv',
                        'connector.path' = '{batch_uri}',
                        'format.field-delimiter' = ';'
                    )"""
        table_env.execute_sql(ddl)
        return table_env.from_path('test_table')


class ReadOnlinePredictExample(SourceExecutor):
    """以 kafka connector 读入在线推理输入流，并通知 AIFlow 开始发消息。"""

    def execute(self, function_context: FlinkFunctionContext) -> Table:
        """建表 ``online_example``、发出 source 通知，返回对应 Table。"""
        table_env: TableEnvironment = function_context.get_table_env()
        table_env.execute_sql("""
            create table online_example (
                face_id varchar,
                device_id varchar,
                feature_data varchar
            ) with (
                'connector' = 'kafka',
                'topic' = 'tianchi_read_example',
                'properties.bootstrap.servers' = 'localhost:9092',
                'properties.group.id' = 'read_example',
                'format' = 'csv',
                'scan.startup.mode' = 'earliest-offset'
            )
        """)
        table = table_env.from_path('online_example')
        # 通知 AIFlow 发送在线示例消息。
        update_notification('source', function_context.node_spec.instance_id)
        return table


class TransformTrainExample(Executor):
    """训练集透传算子：原样返回输入表，仅用于串联 workflow 节点。"""

    def execute(self, function_context: FlinkFunctionContext, input_list: list[Table]) -> list[Table]:
        """原样返回输入表。"""
        input_table = input_list[0]
        return [input_table]


class PredictAutoEncoderWithTrain(Executor):
    """批训练链路：对训练集特征做预测，输出 uuid/face_id/预测特征。"""

    def execute(self, function_context: FlinkFunctionContext, input_list: list[Table]) -> list[Table]:
        """注册 ``predict1`` UDF 并对输入表做投影。"""
        function_context.t_env.register_function(
            "predict1",
            udf(
                f=ClusterServingPredict("predict1"),
                input_types=[DataTypes.STRING()],
                result_type=DataTypes.STRING(),
            ),
        )
        return [input_list[0].select('uuid, face_id, predict1(feature_data) as feature_data')]


class PredictAutoEncoder(Executor):
    """离线历史链路：对历史测试集特征做预测，输出 face_id/预测特征。"""

    def execute(self, function_context: FlinkFunctionContext, input_list: list[Table]) -> list[Table]:
        """注册 ``predict1`` UDF 并对输入表做投影。"""
        function_context.t_env.register_function(
            "predict1",
            udf(
                f=ClusterServingPredict("predict1(offline)"),
                input_types=[DataTypes.STRING()],
                result_type=DataTypes.STRING(),
            ),
        )
        return [input_list[0].select('face_id, predict1(feature_data) as feature_data')]


class OnlinePredictAutoEncoder(Executor):
    """在线流链路：对 Kafka 流入的特征做预测，输出 face_id/device_id/预测特征。"""

    def execute(self, function_context: FlinkFunctionContext, input_list: list[Table]) -> list[Table]:
        """注册 ``predict2`` UDF 并对输入表做投影。"""
        function_context.t_env.register_function(
            "predict2",
            udf(
                f=ClusterServingPredict("predict2(online)"),
                input_types=[DataTypes.STRING()],
                result_type=DataTypes.STRING(),
            ),
        )
        return [input_list[0].select('face_id, device_id, predict2(feature_data) as feature_data')]


class SearchSink(SinkExecutor):
    """把第一阶段检索结果写成 CSV 文件，写前清理同名输出。"""

    def execute(self, function_context: FlinkFunctionContext, input_table: Table) -> None:
        """注册 CsvTableSink 并把输入表插入其中。"""
        example_meta: ExampleMeta = function_context.get_example_meta()
        output_file = example_meta.batch_uri
        if os.path.exists(output_file):
            if os.path.isdir(output_file):
                shutil.rmtree(output_file)
            else:
                os.remove(output_file)
        t_env = function_context.get_table_env()
        statement_set = function_context.get_statement_set()
        sink = CsvTableSink(['a', 'b'],
                            [DataTypes.STRING(), DataTypes.STRING()],
                            output_file,
                            ';')

        t_env.register_table_sink('mySink', sink)
        statement_set.add_insert('mySink', input_table)


class WriteSecondResult(SinkExecutor):
    """把第二阶段（在线流）检索结果写回 Kafka topic。"""

    def execute(self, function_context: FlinkFunctionContext, input_table: Table) -> None:
        """建表 ``write_example`` 并把输入表插入其中。"""
        table_env: TableEnvironment = function_context.get_table_env()
        statement_set = function_context.get_statement_set()
        table_env.execute_sql("""
               create table write_example (
                    face_id varchar,
                    device_id varchar,
                    near_id int
                ) with (
                    'connector' = 'kafka',
                    'topic' = 'tianchi_write_example',
                    'properties.bootstrap.servers' = 'localhost:9092',
                    'properties.group.id' = 'write_example',
                    'format' = 'csv',
                    'scan.startup.mode' = 'earliest-offset',
                    'csv.disable-quote-character' = 'true'
                )
                """)
        statement_set.add_insert('write_example', input_table)
