# funfight

一次性的比赛代码存档，**不是通用工具/库，目前已废弃、未维护**。

仓库里保存的是作者参加阿里云天池比赛（比赛编号 `531800`）时提交的方案代码：基于阿里开源的 [AI Flow](https://github.com/alibaba/flink-ai-extended) 框架编写的一套批流一体机器学习工作流，用 TensorFlow 训练自编码器（Autoencoder）、用 Proxima 构建向量索引、再用 Flink 做在线预测/检索。除了这一份比赛代码外，没有其他功能。

> 说明：目前 PyPI 上查不到 `funfight` 已发布的版本（返回 `Not Found`）。本仓库只是一次性比赛代码存档，**不建议 `pip install`**。

## 目录结构 / 代码做了什么

```
funfight/
└── tianchi/
    └── t531800/
        ├── step1.py            # 准备环境：从蓝奏云下载数据集，安装 apache-flink / kafka-python，下载 Flink/Kafka 安装包
        ├── kafka-source.py     # Source 类：监听 AIFlow 通知，创建/清理 Kafka topic，把测试集逐行发到 Kafka 作为在线推理输入
        ├── ai_flow_master.py   # 启动 AIFlowMaster（读取同目录 master.yaml）
        └── package/python_codes/
            ├── tianchi_main.py       # 定义完整 workflow：register_example/register_model 注册数据与模型 →
            │                         #   训练自编码器(TrainAutoEncoder) → cluster_serving 上线 →
            │                         #   Flink 任务构建向量索引(BuildIndexExecutor) → 预测(PredictAutoEncoder) →
            │                         #   向量检索(SearchExecutor/SearchExecutor3) → 写出结果(WriteSecondResult)
            ├── python_job_executor.py  # ReadCsvExample / TrainAutoEncoder：读 CSV、训练一个简单的 Dense 自编码器并保存
            ├── tianchi_executor.py     # Flink 侧的 Executor：读数据、预测、拼接历史、写结果等
            ├── proxima_executor.py     # BuildIndexExecutor / SearchExecutor：调用 Proxima 建索引、做近邻检索
            └── data_type.py            # FloatDataType / DoubleDataType：在 Proxima 类型和 Flink 类型之间转换
```

`example/531800.py` 是空文件，`funfight/__init__.py`、`funfight/tianchi/__init__.py`、`t531800/__init__.py` 也都是空的包占位文件，没有对外暴露任何 API。

## 依赖

代码依赖比赛当时的特定环境：`ai_flow`、`flink_ai_flow`（阿里 AI Flow 框架）、`pyflink`、`pyproxima2`（向量检索）、`kafka-python`、`tensorflow`，以及本组织的 `funtool`/`fundata`/`fundrive`。这些依赖版本较老，脚本里还有 `pip install apache-flink==1.11.0`、下载 Flink 1.11.0 / Kafka 2.3.0 安装包等步骤，无法直接在现代环境里运行，需要按天池比赛当年的环境手动搭建。

## 安装

本仓库未发布到 PyPI（`pip install funfight` 会报 `Not Found`），也不建议安装使用；获取代码请直接克隆仓库：

```bash
git clone https://github.com/farfarfun/funfight.git
```

如果确实需要把 `funfight` 当作本地包安装（例如只是为了让 `import funfight` 不报错），可以用 uv/pip 从源码安装：

```bash
uv pip install -e .
# 或
pip install -e .
```

## 最小可运行示例

除比赛方案代码外，`funfight` 包本身不提供任何公共 API，唯一能验证安装是否成功的方式是导入空包：

```bash
python3 -c "import funfight; print(funfight.__file__)"
```

## 使用（服务/流程入口）

没有命令行入口，也没有可直接调用的公共函数。这些流程依赖比赛当年的历史环境（Flink 1.11.0、Kafka 2.3.0、阿里内部的 AI Flow / Proxima 服务等），不是可以用 `scripts/setup.sh` 统一 start/stop 的常规服务，因此仍按当年的手工步骤说明，不提供伪装成「一键启停」的脚本：

1. 执行 `funfight/tianchi/t531800/step1.py` 里的 `step1()`/`step2()`/`step3()`，依次下载数据集、安装 Flink/Kafka 历史版本、准备 `ai_flow` wheel 包；
2. 参考 `funfight/tianchi/t531800/ai_flow_master.py` 启动 AIFlowMaster（读取同目录 `master.yaml`）；
3. 按 `funfight/tianchi/t531800/README.md` 配置好 `PYTHONPATH`/`ENV_HOME`/`TASK_ID` 等环境变量后，启动 `kafka-source.py`（Kafka Source）；
4. 最后运行 `package/python_codes/tianchi_main.py` 提交 workflow。

---

## 关于 farfarfun

[farfarfun](https://github.com/farfarfun) 是一个专注于实用工具库的开源组织，
涵盖云存储、数据处理、AI、多媒体与开发工具链等方向。

- 🏠 组织主页：<https://github.com/farfarfun>
- 📦 PyPI：<https://pypi.org/user/niuliangtao/>
- 📧 联系：farfarfun@qq.com

本项目基于 [MIT](LICENSE) 协议开源。
