"""仓库级规范回归测试（来自 farfarfun/todo-list#829 的审计发现）。

t531800 下的比赛存档代码依赖 ai_flow / pyflink / pyproxima2 等无法安装的包，
整模块 import 不了，因此这里统一用 AST / 文本层面做静态断言，不触发任何导入。
"""

from __future__ import annotations

import ast
import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
PY_FILES = sorted(SRC.rglob("*.py"))


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT))


def test_python_files_found() -> None:
    """守住下面几个参数化测试：源码目录不为空，别让它们全部空跑过关。"""
    assert len(PY_FILES) >= 10


@pytest.mark.parametrize("path", PY_FILES, ids=_rel)
def test_no_bare_subprocess(path: Path) -> None:
    """SPEC §2/§6.3：执行外部命令统一走 funshell，不直接用 subprocess。"""
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] != "subprocess", (
                    f"{_rel(path)}:{node.lineno} 直接 import subprocess"
                )
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            assert root != "subprocess", (
                f"{_rel(path)}:{node.lineno} 从 subprocess 导入 {node.names[0].name}"
            )


@pytest.mark.parametrize("path", PY_FILES, ids=_rel)
def test_yaml_load_is_safe(path: Path) -> None:
    """yaml.load 不传 Loader 在新版 PyYAML 会 TypeError，一律用 safe_load。"""
    for node in ast.walk(_tree(path)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr == "load"
            and isinstance(func.value, ast.Name)
            and func.value.id == "yaml"
        ):
            pytest.fail(
                f"{_rel(path)}:{node.lineno} 用了 yaml.load，应改为 yaml.safe_load"
            )


@pytest.mark.parametrize("path", PY_FILES, ids=_rel)
def test_public_api_has_docstring(path: Path) -> None:
    """SPEC §7：公开模块/类/函数都要有中文 docstring（下划线开头的内部实现除外）。"""
    tree = _tree(path)
    missing: list[str] = []
    if ast.get_docstring(tree) is None:
        missing.append(f"{_rel(path)}:1 模块")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_") and not node.name.startswith("__"):
            continue
        if ast.get_docstring(node) is None:
            missing.append(f"{_rel(path)}:{node.lineno} {node.name}")
    assert missing == [], f"缺少 docstring: {missing}"


@pytest.mark.parametrize("path", PY_FILES, ids=_rel)
def test_comments_are_chinese(path: Path) -> None:
    """SPEC §7：注释用中文。只检查整行注释，行尾 ``# noqa`` 这类指令不算。"""
    bad: list[str] = []
    for lineno, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped.startswith("#"):
            continue
        body = stripped.lstrip("#").strip()
        if not body or body.startswith(("noqa", "type:", "ruff:", "!")):
            continue
        if re.search(r"[一-鿿]", body):
            continue
        if re.search(r"[A-Za-z]{3,}\s+[A-Za-z]{3,}", body):
            bad.append(f"{_rel(path)}:{lineno} {body}")
    assert bad == [], f"英文注释: {bad}"


def test_proxima_search_udtf3_calls_key() -> None:
    """回归：SearchUDTF3.eval 必须调用 ``.key()``，漏括号会把绑定方法当成 key。"""
    path = SRC / "funfight/tianchi/t531800/package/python_codes/proxima_executor.py"
    assigns = [
        node
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "near_key" for t in node.targets)
    ]
    assert assigns, "没找到 near_key 赋值，测试需要跟着代码更新"
    for node in assigns:
        assert isinstance(node.value, ast.Call), (
            f"proxima_executor.py:{node.lineno} near_key 没有调用 key()"
        )
        assert (
            isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "key"
        )


def test_executors_call_super_init() -> None:
    """三个 Executor 的 __init__ 都要调 super().__init__()。"""
    path = SRC / "funfight/tianchi/t531800/package/python_codes/proxima_executor.py"
    tree = _tree(path)
    targets = {"SearchExecutor", "SearchExecutor3", "BuildIndexExecutor"}
    seen = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name not in targets:
            continue
        seen.add(node.name)
        init = next(
            (
                n
                for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name == "__init__"
            ),
            None,
        )
        assert init is not None, f"{node.name} 没有 __init__"
        calls = [
            n
            for n in ast.walk(init)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "__init__"
            and isinstance(n.func.value, ast.Call)
            and isinstance(n.func.value.func, ast.Name)
            and n.func.value.func.id == "super"
        ]
        assert calls, f"{node.name}.__init__ 没有调用 super().__init__()"
    assert seen == targets, f"没找到这些类: {targets - seen}"


def test_pyproject_description_is_not_placeholder() -> None:
    """SPEC §11：description 不能是仓库名占位。"""
    meta = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    description = meta["project"]["description"]
    assert description.strip().lower() != meta["project"]["name"].lower()
    assert len(description) > 20


def test_t531800_readme_matches_code() -> None:
    """子项目 README 必须和代码实际读取的数据文件、环境变量一致。"""
    readme = (SRC / "funfight/tianchi/t531800/README.md").read_text(encoding="utf-8")
    assert "label_file.csv" not in readme, "label_file.csv 在代码里根本不存在"
    for name in ("train_data.csv", "first_test_data.csv", "second_test_data.csv"):
        assert name in readme, f"README 没有提到代码实际读取的 {name}"
    for env in ("ENV_HOME", "TASK_ID", "FLINK_HOME"):
        assert env in readme, f"README 没有说明必填环境变量 {env}"


def test_main_env_vars_are_all_documented() -> None:
    """tianchi_main.py 里 os.environ[...] 读的变量都要在子项目 README 里有说明。"""
    path = SRC / "funfight/tianchi/t531800/package/python_codes/tianchi_main.py"
    readme = (SRC / "funfight/tianchi/t531800/README.md").read_text(encoding="utf-8")
    used = {
        node.slice.value
        for node in ast.walk(_tree(path))
        if isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Attribute)
        and node.value.attr == "environ"
        and isinstance(node.slice, ast.Constant)
        and isinstance(node.slice.value, str)
    }
    assert used, "没解析到 os.environ 用法，测试需要跟着代码更新"
    assert sorted(name for name in used if name not in readme) == []


def test_no_hyphenated_module_names() -> None:
    """带连字符的模块名无法 import（ruff N999），不允许出现。"""
    bad = [_rel(p) for p in PY_FILES if "-" in p.stem]
    assert bad == [], f"模块名不合法: {bad}"
