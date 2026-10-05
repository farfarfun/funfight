"""Python 作业节点：读取训练集并训练自编码器模型，训练完注册模型版本。"""

from __future__ import annotations

import os
import shutil

import numpy as np
import pandas as pd
import tensorflow as tf
from ai_flow import ExampleMeta, FunctionContext, ModelMeta, register_model_version
from farlog import getLogger
from python_ai_flow.user_define_funcs import Executor
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

logger = getLogger("funfight.tianchi.t531800.python_job_executor")


class ReadCsvExample(Executor):
    """读取训练用 CSV 文件，把第 4 列（下标 3）的特征字符串解析为浮点数组。"""

    def execute(self, function_context: FunctionContext, input_list: list) -> list:
        """从 node_spec 指定的 batch_uri 读 CSV，返回一个 (N, 512) 的浮点矩阵。

        Args:
            function_context: AIFlow 注入的上下文，``node_spec.example_meta.batch_uri``
                是 ``;`` 分隔的训练集路径。
            input_list: 上游输出，本节点是数据源，不使用。

        Returns:
            单元素列表，元素是 ``numpy.ndarray`` 形式的特征矩阵。
        """
        example_meta: ExampleMeta = function_context.node_spec.example_meta
        data = pd.read_csv(example_meta.batch_uri, sep=";", header=None, usecols=[3])
        n = data.values.tolist()
        rows = len(n)
        xx = []
        for i in range(rows):
            yy = n[i][0].split(" ")
            x = []
            for y in yy:
                x.append(float(y))
            xx.append(np.array(x))
        xx = np.array(xx)
        return [xx]


class TrainAutoEncoder(Executor):
    """训练一个简单的 Dense 自编码器模型，并注册模型版本。"""

    def execute(self, function_context: FunctionContext, input_list: list) -> list:
        """训练 512→2→512 的自编码器，导出 encoder 并注册模型版本。

        模型以 SavedModel 格式写到本文件同级的 ``model/`` 目录（已存在会先删除），
        然后调用 ``register_model_version`` 通知 cluster serving 加载。

        Args:
            function_context: AIFlow 注入的上下文，``node_spec.output_model`` 是待注册的模型元信息。
            input_list: 上游 :class:`ReadCsvExample` 的输出，第 0 个元素是训练矩阵。

        Returns:
            空列表——本节点只有训练与注册副作用，没有下游数据。
        """
        x_train = input_list[0]
        input_dim = 512
        encoding_dim = 2
        x_test = np.random.rand(30, input_dim)
        model_input = Input(shape=(input_dim,))
        encoder = Dense(encoding_dim)(model_input)
        decoder = Dense(input_dim)(encoder)
        model = Model(model_input, decoder)
        model.compile(loss="binary_crossentropy", optimizer=Adam())
        model.fit(x_train, x_train, validation_data=(x_test, x_test), epochs=1)
        encoder = Model(model_input, encoder)
        model_path = os.path.dirname(os.path.abspath(__file__)) + "/model"
        logger.info("保存训练好的模型到 {}", model_path)
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
        tf.saved_model.simple_save(
            tf.keras.backend.get_session(),
            model_path,
            inputs={"aaa_input": encoder.input},
            outputs={"bbb": encoder.output},
        )
        model_meta: ModelMeta = function_context.node_spec.output_model
        # 注册模型版本，通知 cluster serving 可以开始加载该模型版本。
        register_model_version(model=model_meta, model_path=model_path)
        return []
