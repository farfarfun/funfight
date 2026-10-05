# Changelog

本文件记录 funfight 的版本变更，按版本倒序排列。

## [未发布]

### 新增

- 新增 `tests/test_repo_conventions.py`：仓库级规范回归测试（禁用裸 `subprocess`、
  禁用 `yaml.load`、公开 API 必须有中文 docstring、注释必须是中文、
  `SearchUDTF3` 必须调 `.key()`、三个 `Executor` 必须调 `super().__init__()`、
  `pyproject` description 不是占位、`t531800/README.md` 与代码实际读取的数据文件
  及 `os.environ` 变量一致、不允许带连字符的模块名）。存档代码 import 不了
  （依赖 `ai_flow`/`pyflink`/`pyproxima2`），因此全部用 AST/文本静态断言。
- 所有模块补充模块级中文 docstring。
- 新增 `feature_predict.py`：把三条预测链路（批训练/离线历史/在线流）重复内联的
  cluster serving 预测逻辑合并成一份可被单元测试覆盖的实现（新增
  `tests/test_feature_predict.py`，覆盖正常解析、非法输入、非字符串响应、
  客户端异常透传等路径）。
- 为 `tianchi_main.py`、`proxima_executor.py` 的公开函数/类/UDF 生命周期方法
  补充类型标注与中文 docstring。

### 修复

- `proxima_executor.py` `SearchUDTF3.eval` 的 `results[0][0].key` 漏了括号，拿到的是
  绑定方法对象而不是 key 值，导致 `near_key not in v` 恒为真、每条记录都被判成一个
  新的人（同文件 `SearchUDF`/`SearchUDTF` 用的都是 `.key()`）。
- `BuildIndexExecutor.__init__` 漏调 `super().__init__()`（同文件另外两个 Executor
  都调了），补上。
- `kafka_source.py` 两处用 `subprocess.Popen` 执行 `kafka-topics.sh` 删除 topic，
  改为 `funshell.run_shell`（抽成 `delete_topic()`，参数经 `shlex.quote` 转义）；
  原实现只把退出码写进 info 日志就继续建 topic，删除失败被静默吞掉，现在非 0
  退出码直接抛 `RuntimeError`。
- `kafka_source.py` 的 `yaml.load(yaml_file)` 未传 `Loader`，新版 PyYAML 会直接
  `TypeError`，改为 `yaml.safe_load`。
- 补齐 `SearchExecutor`/`SearchExecutor3`/`BuildIndexExecutor` 及其 `__init__`/
  `execute`、`ReadCsvExample.execute`、`TrainAutoEncoder.execute`、
  `Source.listen_notification` 与内部 `SourceWatcher` 的中文 docstring。
- `pyproject.toml` 的 `description` 从占位的 `"funfight"` 改为与 GitHub description
  一致的真实功能描述。
- `tianchi_main.py` 里残留的 5 处英文注释改为中文。
- 重写 `src/funfight/tianchi/t531800/README.md`：原说明让下载代码里根本不存在的
  `label_file.csv`，改为列出实际读取的 `train_data.csv`/`first_test_data.csv`/
  `second_test_data.csv` 及其读取位置；补上原先漏掉的必填环境变量 `FLINK_HOME`，
  并把其实本仓库代码不读、由 ai_flow/Cluster Serving 框架读取的
  `SERVING_HTTP_PATH` 等四个变量分到单独一张表。
- 更正失实的外部资源说明：`step1()` 用的两个 `wws.lanzous.com` 蓝奏云链接已失效
  （域名早已废弃，无有效证书、不返回有效响应），docstring 加 warning 指向赛题页面；
  README 把 AI Flow 链到 `alibaba/flink-ai-extended`（已重定向到另一个项目
  `flink-extended/dl-on-flink`），改为 `flink-extended/ai-flow`；README 与
  `pyproject.toml` 注释称 `ai_flow`/`zoo.serving.client`「从未发布到公开 PyPI」，
  实际 `ai-flow` 0.1.0 与 `analytics-zoo` 都在 PyPI 上，真正装不了的原因是
  `ai-flow` 声明 `requires-python >=3.7,<3.8`，已逐个导入名改成真实现状表格。
- `kafka-source.py` 重命名为 `kafka_source.py`（带连字符的模块名无法 import，
  ruff N999，logger 名早已写成 `kafka_source`），同步更新两处 README 引用。
- `src/funfight/__init__.py` 里的 shebang 与编码声明（ruff EXE001/UP009）换成模块
  docstring；全仓库跑 `ruff format`。

- `tianchi_executor.py` 中三处 `except Exception: ... return ''`（预测失败被静默
  转换为空结果）改为：解析/响应错误抛出带上下文的 `FeaturePredictError`，
  客户端自身异常（网络、超时等）原样向上传播，不再吞掉失败语义。
- `kafka-source.py` 模块导入时直接执行 `Source()` 与 `listen_notification()`
  （导入即连接 AIFlow 并启动监听）改为放进 `main()`，通过
  `if __name__ == '__main__'` 调用。
- 日志不再输出完整人脸特征向量/消息体：`tianchi_executor.py` 的预测失败日志、
  `proxima_executor.py` 的 `SearchUDTF3`/`BuildIndexUDF` 调试日志、
  `kafka-source.py` 的发送日志，均改为只记录类型/长度/哈希摘要。
- `data_type.py` 的 `DoubleDataType.to_numpy_type()` 原来返回单精度类型码
  `'f'`（与 `FloatDataType` 重复），已改为双精度对应的 `'d'`。
- README 依赖列表与实际路径修正：删除不存在的 `funtool`/`fundata`，按
  `pyproject.toml` 列出实际依赖 `fundrive`/`farlog`/`funshell`；「使用」一节
  的 `funfight/tianchi/...` 路径统一改为仓库中的实际路径
  `src/funfight/tianchi/...`。
- `pyproject.toml` 的 `pytest.pythonpath` 补充 `python_codes` 目录，使按
  README 约定扁平导入的 `feature_predict.py` 可以被 `pytest` 直接测试。
- `kafka-source.py`、`proxima_executor.py`、`python_job_executor.py`、`step1.py` 的
  `logger.info`/`logger.error`/`logger.debug` 调用误用 stdlib logging 的 `%s` 占位符，
  farlog（loguru）不支持该语法，参数会被静默丢弃、日志只剩字面量 `%s`；统一改为
  loguru 的 `{}` 占位符，位置参数保持惰性求值。

## [0.0.5] - 2026-09-19

### 新增

- README 补充安装命令、最小可运行示例，并在文末追加组织介绍区块。
- 为 `step1.py`（`download`/`step1`/`step2`/`step3`）、`data_type.py`（`DataType`/`FloatDataType`/`DoubleDataType`）
  的公开函数/方法补充类型标注与中文 docstring。

### 修复

- `pyproject.toml` 中 `fundrive[lanzou]` 依赖补充版本下限（`>=2.0.86`），避免装到远古版本。
- `step1.py` 的 `step2()` 中一处没有传入命令的 `os.system()` 空调用（等价于无意义的运行时错误）已删除；
  其余 `os.system(...)` 改为组织自有的 `funshell.run_shell`，逐条检查返回码，失败即抛出
  `RuntimeError` 中止后续步骤。
- `tianchi_executor.py` 中三处 `except Exception: return ''`（`PredictAutoEncoderWithTrain`/
  `PredictAutoEncoder`/`OnlinePredictAutoEncoder`）不再静默吞异常，改为用 `farlog` 记录带
  `feature_data` 上下文的错误日志后再返回空串。
- `script/build.sh` 引用的 `python setup.py ...` 全套命令在仓库里根本不存在对应的 `setup.py`，
  本身已经是必然失败的死代码；其 `if [ "push" = "push" ]` 恒真分支还会在任何一次调用时都
  无条件 `git add -A && git commit -a -m "add" && git push`。连同会强制改写历史并
  `push -f` 的 `script/clear_history.sh` 一并删除，避免误执行造成数据丢失或误推送
  （本仓库未接入 `funbuild`/未发布 PyPI，不需要这两个脚本）。

### 变更

- 统一日志入口为 `farlog.getLogger`：`kafka-source.py`、`python_job_executor.py`、
  `tianchi_executor.py` 中原本的 `print(...)` 诊断输出改为 `logger.info`/`logger.error`。
- 移除 `tianchi_executor.py`（`/root/predict`、`/root/offline`、`/root/inference`）和
  `proxima_executor.py`（`/root/test`、`/root/debug`）中把运行数据直接写入 `/root/*` 的调试
  文件逻辑，改为按需 `logger.debug`；随之移除对应的 `Popen('rm -rf /root/...', shell=True)`
  清理调用。
- `kafka-source.py`、`tianchi_main.py` 中原本的英文 docstring/注释改为中文。
- `.gitignore` 补充 `*.db`、`*.rar`、`.venv/`、`.run/`、`logs/`、`.idea/`、`.vscode/`、`node_modules/`。
- `proxima_executor.py` 的 `BuildIndexExecutor.execute` 原来使用了未导入的 `os` 模块
  （`os.path.exists`/`os.remove`），顺带补上 `import os`，修复潜在的 `NameError`。

### 废弃

- 无。

## [0.0.4] - 2026-09-03

### 变更

- **破坏性变更**：仓库/包名从 `notefight` 改为 `funfight`，与仓库名保持一致。原来
  `import notefight` / `pip install notefight` 的用法需要改为 `import funfight` /
  `pip install funfight`。截至该版本，`notefight`、`funfight` 均未实际发布到 PyPI。
- 代码中失效的 `notedrive`/`note*` 包引用替换为对应的 `fundrive`/`fun*` 包。
