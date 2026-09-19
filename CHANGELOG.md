# Changelog

本文件记录 funfight 的版本变更，按版本倒序排列。

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
