# 天池 531800 赛题运行说明

本目录是比赛代码存档，依赖比赛当年的 `ai_flow` / `flink_ai_flow` / `pyflink 1.11`
/ `pyproxima2` / `zoo.serving.client` 等专用包（均未发布到公开 PyPI），
**现代环境无法按这份说明直接跑通**。下面还原的是当年的运行步骤与代码实际读取的
路径、文件名与环境变量，供阅读代码时对照。

## 1. 准备数据集

从[赛题页面](https://tianchi.aliyun.com/competition/entrance/531800/information)下载数据集，
解压到 `$ENV_HOME/data_set/` 目录。代码实际读取的文件只有这三个：

| 文件 | 读取位置 | 用途 |
| --- | --- | --- |
| `train_data.csv` | `tianchi_main.py` 的 `collect_data_file()` | 训练自编码器 + 构建 Proxima 索引的底库 |
| `first_test_data.csv` | `tianchi_main.py` 的 `collect_data_file()` | 离线批量检索（`find_sick` 作业）的输入 |
| `second_test_data.csv` | `source.yaml` 的 `dataset_uri` | 由 `kafka_source.py` 逐行发到 Kafka，作为在线链路输入 |

三个 CSV 都是 `;` 分隔、无表头，第 4 列（下标 3）是空格分隔的人脸特征向量。

产物由代码自动写出：离线结果 `$ENV_HOME/codes/$TASK_ID/output/first_result.csv`，
Proxima 索引 `$ENV_HOME/codes/$TASK_ID/test.index`，在线结果写回 Kafka 的
`write_example_topic`。

## 2. 设置 PYTHONPATH

`package/python_codes/` 下的模块互相之间用顶层导入（`from data_type import ...`），
必须把该目录本身加进 `PYTHONPATH`：

```bash
export PYTHONPATH=/绝对路径/src/funfight/tianchi/t531800/package/python_codes
```

## 3. 配置环境变量

代码里直接 `os.environ[...]` 读取、缺失会 `KeyError` 的三个变量：

| 变量 | 读取位置 | 说明 |
| --- | --- | --- |
| `ENV_HOME` | `tianchi_main.py:51,53,115` | 数据与产物的根目录，下面要有 `data_set/` 与 `codes/$TASK_ID/` |
| `TASK_ID` | `tianchi_main.py:53,115` | 任意整数，用于区分多次运行的产物目录 |
| `FLINK_HOME` | `tianchi_main.py:126,133,140` | Flink 1.11.0 安装目录，三个 Flink 作业都要用 |

由 `ai_flow` / Cluster Serving 框架读取（本仓库代码不直接引用）：

| 变量 | 说明 |
| --- | --- |
| `SERVING_HTTP_PATH` | Cluster Serving HTTP Jar 包路径 |
| `REST_HOST` | Flink Rest Host，默认 `localhost` |
| `REST_PORT` | Flink Rest Port，默认 `8081` |
| `CLUSTER_SERVING_PATH` | Cluster Serving 运行目录，默认 `/tmp/cluster-serving` |

AIFlow Master 与 Kafka 的地址写在配置文件里，不走环境变量：`master.yaml`
（`localhost:50052` 与 sqlite 元数据库）、`package/project.yaml`（项目名与 master 地址）、
`source.yaml`（`bootstrap_servers`、读写 topic 名、发送间隔）。

## 4. 按顺序启动

```bash
# 4.1 启动 AIFlow Master（读取同目录 master.yaml，阻塞运行）
python src/funfight/tianchi/t531800/ai_flow_master.py

# 4.2 把 source.yaml 的 dataset_uri 改成 second_test_data.csv 的实际路径
#     （默认值是相对路径 data_set/second_test_data.csv），再启动 Kafka Source；
#     它注册通知监听器，收到 source 通知后重建 topic 并把数据灌进 Kafka
python src/funfight/tianchi/t531800/kafka_source.py

# 4.3 提交工作流：训练 -> cluster serving 上线 -> 建索引 -> 离线检索 -> 在线检索
python src/funfight/tianchi/t531800/package/python_codes/tianchi_main.py
```

`tianchi_main.py` 会阻塞等待工作流结束，并以工作流的结束状态作为进程退出码；
运行日志看 AIFlow Master 所在终端。
