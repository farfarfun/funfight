"""比赛环境准备脚本：下载数据集、安装当年版本的 Flink/Kafka。"""

from farlog import getLogger
from fundrive.drives.lanzou.drive import Task
from fundrives.lanzou import LanZouCloud
from funshell import run_shell

logger = getLogger("funfight.tianchi.t531800.step1")


def download(url: str, dir_pwd: str = "./data") -> None:
    """兼容旧版 notedrive.lanzou.download(url, dir_pwd=...) 的简易封装。

    新版 fundrive 把蓝奏云操作封装成了 LanZouDrive 类，但其
    download_file() 只接受已知的 fid，不支持直接传分享链接下载，
    因此这里直接调用底层 fundrives.lanzou.LanZouCloud.down_file_by_url
    来保持和旧脚本一致的“传一个分享链接就下载”的行为。

    Args:
        url: 蓝奏云分享链接。
        dir_pwd: 下载文件保存的本地目录，默认为 ./data。
    """
    cloud = LanZouCloud()
    task = Task(url=url, path=dir_pwd)
    cloud.down_file_by_url(share_url=url, task=task, callback=lambda: None)


def step1() -> None:
    """下载当年转存到蓝奏云的数据集副本到 ./data 目录。

    .. warning::
       **这两个链接已失效。** 蓝奏云早年废弃了 ``lanzous.com`` 域名，
       ``wws.lanzous.com`` 现在既不返回有效 HTTP 响应也没有可用证书，
       本函数必然失败。请改为从赛题页面
       https://tianchi.aliyun.com/competition/entrance/531800/information
       下载数据集，并按同目录 README.md 放到 ``$ENV_HOME/data_set/``
       （注意不是本函数写入的 ``./data``）。
    """
    download("https://wws.lanzous.com/b01hlgi2b", dir_pwd="./data")
    download("https://wws.lanzous.com/izZmlfjulvg", dir_pwd="./data")


def step2() -> None:
    """安装比赛当年使用的 Flink/Kafka 历史版本环境。

    依次安装 apache-flink==1.11.0、kafka-python，并下载解压
    Flink 1.11.0、Kafka 2.3.0 安装包。任一步失败即抛出
    RuntimeError 并中止，不再继续执行后续步骤。
    """
    commands = [
        "pip install apache-flink==1.11.0",
        "pip install kafka-python",
        "wget https://archive.apache.org/dist/flink/flink-1.11.0/flink-1.11.0-bin-scala_2.11.tgz",
        "tar xzf flink-1.11.0-bin-scala_2.11.tgz",
        "wget https://archive.apache.org/dist/kafka/2.3.0/kafka_2.11-2.3.0.tgz",
        "tar xzf kafka_2.11-2.3.0.tgz",
    ]
    for command in commands:
        exit_code = run_shell(command)
        if exit_code != "0":
            logger.error(
                "step2 命令执行失败: command={}, exit_code={}", command, exit_code
            )
            raise RuntimeError(f"step2 命令执行失败: {command} (exit_code={exit_code})")


def step3() -> None:
    """占位步骤：ai_flow 的 wheel 包需手动下载安装。

    比赛提供的 OSS 地址（2026-10 复核仍可下载）：
    https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531800/ai_flow/ai_flow-0.1-py3-none-any.whl

    同一个包后来也发布到了 PyPI（``pip install ai-flow==0.1.0``），但它声明
    ``requires-python >=3.7,<3.8``，装不进本仓库要求的 Python >=3.10 环境；
    ``flink_ai_flow`` / ``python_ai_flow`` 这两个子包与 ``pyproxima2`` 则从未
    上过公开 PyPI。
    """
