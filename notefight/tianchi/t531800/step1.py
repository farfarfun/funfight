import os

from notedrive.lanzou import download


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
