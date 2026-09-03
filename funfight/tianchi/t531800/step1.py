import os

from fundrive.drives.lanzou.drive import Task
from fundrives.lanzou import LanZouCloud


def download(url, dir_pwd="./data"):
    """兼容旧版 notedrive.lanzou.download(url, dir_pwd=...) 的简易封装。

    新版 fundrive 把蓝奏云操作封装成了 LanZouDrive 类，但其
    download_file() 只接受已知的 fid，不支持直接传分享链接下载，
    因此这里直接调用底层 fundrives.lanzou.LanZouCloud.down_file_by_url
    来保持和旧脚本一致的“传一个分享链接就下载”的行为。
    """
    cloud = LanZouCloud()
    task = Task(url=url, path=dir_pwd)
    cloud.down_file_by_url(share_url=url, task=task, callback=lambda: None)


def step1():
    download('https://wws.lanzous.com/b01hlgi2b', dir_pwd='./data')
    download('https://wws.lanzous.com/izZmlfjulvg', dir_pwd='./data')
    pass


def step2():
    os.system("pip install apache-flink==1.11.0")
    os.system("pip install kafka-python")

    os.system("wget https://archive.apache.org/dist/flink/flink-1.11.0/flink-1.11.0-bin-scala_2.11.tgz")
    os.system("tar xzf flink-1.11.0-bin-scala_2.11.tgz")

    os.system(
        "wget [https://archive.apache.org/dist/kafka/2.3.0/kafka_2.11-2.3.0.tgz](https://archive.apache.org/dist/kafka/2.3.0/kafka_2.11-2.3.0.tgz)")

    os.system("tar xzf kafka_2.11-2.3.0.tgz")
    os.system()


def step3():
    # "https://tianchi-competition.oss-cn-hangzhou.aliyuncs.com/531800/ai_flow/ai_flow-0.1-py3-none-any.whl?spm=5176.12281978.0.0.239550c1InpiYD&file=ai_flow-0.1-py3-none-any.whl"
    pass
