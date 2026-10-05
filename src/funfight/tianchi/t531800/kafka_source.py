"""在线推理链路的 Kafka Source：监听 AIFlow 通知，重建 topic 并灌入测试数据。

收到 source 通知后先删除并重建读写两个 Kafka topic，再把 source.yaml 里
``dataset_uri`` 指向的测试集逐行发到读 topic，模拟在线推理输入流。
"""

from __future__ import annotations

import hashlib
import os
import shlex
import time

import pandas as pd
import yaml
from ai_flow import Watcher
from ai_flow.rest_endpoint.service.client.aiflow_client import AIFlowClient
from farlog import getLogger
from funshell import run_shell
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic

logger = getLogger("funfight.tianchi.t531800.kafka_source")


def delete_topic(bootstrap_servers: str, topic: str) -> None:
    """调用 ``kafka-topics.sh`` 删除一个 Kafka topic，失败即抛异常。

    按 SPEC §6 统一用 ``funshell.run_shell`` 执行外部命令，不再直接使用
    ``subprocess``；拼进命令行的参数都先经 :func:`shlex.quote` 转义。

    Args:
        bootstrap_servers: Kafka bootstrap server 地址。
        topic: 要删除的 topic 名。

    Raises:
        RuntimeError: ``kafka-topics.sh`` 返回非 0 退出码，或 ``run_shell`` 自身报错。
            原实现只把退出码写进 info 日志就继续往下建 topic，删除失败会被静默吞掉，
            导致新建 topic 失败或沿用脏数据。
    """
    command = (
        f"kafka-topics.sh --bootstrap-server {shlex.quote(bootstrap_servers)} "
        f"--delete --topic {shlex.quote(topic)}"
    )
    exit_code = run_shell(command)
    if exit_code != "0":
        logger.error("删除 kafka topic 失败: topic={}, exit_code={}", topic, exit_code)
        raise RuntimeError(f"删除 kafka topic 失败: {topic} (exit_code={exit_code})")
    logger.info("已删除 kafka topic {}", topic)


def _value_digest(value: str) -> str:
    """返回消息 value 的脱敏摘要（长度 + 哈希前 12 位），不回显原始内容。

    本模块发送的消息 value 是 ``face_id,device_id,feature_data`` 拼接串，
    包含人脸特征向量，按 SPEC §8.1 不应完整写入日志。
    """
    raw = value.encode("utf8")
    return f"len={len(raw)} sha256={hashlib.sha256(raw).hexdigest()[:12]}"


class Source:
    """
    监听 source 通知，生成在线推理读取示例消息。
    """

    def __init__(self):
        """读取同目录下的 source.yaml 并连接 AIFlow Server。

        配置用 :func:`yaml.safe_load` 解析：source.yaml 里只有字符串与数字，
        不需要构造任意 Python 对象，``yaml.load`` 不带 ``Loader`` 既会在新版
        PyYAML 里报 TypeError，也等同于不安全的 ``FullLoader`` 之前的行为。
        """
        super().__init__()
        self._yaml_config = None
        with open(os.path.dirname(os.path.abspath(__file__)) + '/source.yaml', 'r') as yaml_file:
            self._yaml_config = yaml.safe_load(yaml_file)
        self._aiflow_client = AIFlowClient(server_uri=self._yaml_config.get('master_uri'))

    def listen_notification(self):
        """向 AIFlow Server 注册 source 监听器，收到通知后重建 topic 并灌入示例数据。

        内部定义的 ``SourceWatcher`` 在每次通知到达时删除并重建读写两个 Kafka
        topic，然后把测试集逐行发到读 topic。本方法只注册监听器，不阻塞。
        """

        class SourceWatcher(Watcher):
            """source 通知的处理器：重建 Kafka topic 并生成在线推理读取示例。"""

            def __init__(self, yaml_config: dict):
                """记录 source.yaml 解析结果，供后续重建 topic / 发消息使用。"""
                super().__init__()
                self._yaml_config = yaml_config

            def process(self, listener_name, notifications):
                """AIFlow 通知回调入口，转交给 :meth:`process_notification`。"""
                self.process_notification()

            def process_notification(self):
                """删除并重建读写两个 Kafka topic，然后生成在线推理读取示例消息。

                Raises:
                    RuntimeError: 删除已存在的 topic 失败（见 :func:`delete_topic`）。
                """
                bootstrap_servers = self._yaml_config.get('bootstrap_servers')
                read_example_topic = self._yaml_config.get('read_example_topic')
                write_example_topic = self._yaml_config.get('write_example_topic')
                admin_client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
                topics = admin_client.list_topics()
                if read_example_topic in topics:
                    delete_topic(bootstrap_servers, read_example_topic)
                if write_example_topic in topics:
                    delete_topic(bootstrap_servers, write_example_topic)
                # 创建在线推理读取示例 topic。
                admin_client.create_topics(
                    new_topics=[NewTopic(name=read_example_topic, num_partitions=1, replication_factor=1)])
                # 创建向量检索结果写出 topic。
                admin_client.create_topics(
                    new_topics=[NewTopic(name=write_example_topic, num_partitions=1, replication_factor=1)])
                self.generate_read_example()

            def generate_read_example(self):
                """
                生成在线推理读取示例消息并发送到 Kafka。
                """
                bootstrap_servers = self._yaml_config.get('bootstrap_servers')
                read_example_topic = self._yaml_config.get('read_example_topic')
                # 读取在线推理示例数据集。
                df = pd.read_csv(filepath_or_buffer=self._yaml_config.get('dataset_uri'), delimiter=';', header=None)
                producer = KafkaProducer(bootstrap_servers=[bootstrap_servers])
                for index, row in df.iterrows():
                    value = f"{row.get(1)},{row.get(2)},{row.get(3)}"
                    logger.info("发送在线推理读取示例消息: topic={}, key={}, value_digest={}",
                                read_example_topic, row.get(1), _value_digest(value))
                    # 发送在线推理读取示例消息。value 含人脸特征向量，不写入日志。
                    producer.send(read_example_topic,
                                  key=bytes(row.get(1), encoding='utf8'),
                                  value=bytes(value, encoding='utf8'))
                    time.sleep(self._yaml_config.get('time_interval') / 1000)

        self._aiflow_client.start_listen_notification(listener_name='source_listener',
                                                      key=self._yaml_config.get('notification_key'),
                                                      watcher=SourceWatcher(self._yaml_config))


def main() -> None:
    """启动 Kafka Source：连接 AIFlow 并监听 source 通知。

    连接网络、启动长期运行的监听是有副作用的操作，因此放在脚本入口而不是
    模块导入时执行，使本模块可以被安全导入（例如被测试 import 而不触发连接）。
    """
    source = Source()
    source.listen_notification()


if __name__ == '__main__':
    main()
