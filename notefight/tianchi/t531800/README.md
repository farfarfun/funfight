1.下载https://tianchi.aliyun.com/competition/entrance/531800/information 数据集train_data.csv和label_file.csv文件到data_set目录;
2.设置PYTHONPATH：export PYTHONPATH=[python_codes目录路径]
3.配置环境变量：
必填：
ENV_HOME=[tianchi_ai_flow目录路径]
TASK_ID=[任意整数]
SERVING_HTTP_PATH=[Cluster Serving HTTP Jar包路径]
可选：
REST_HOST=[Flink Rest Host，默认为localhost]
REST_PORT=[Flink Rest Port，默认为8081]
CLUSTER_SERVING_PATH=[Cluster Serving运行目录，默认为/tmp/cluster-serving]
4.启动AIFlow Master：python [ai_flow_master.py绝对路径]；
5.修改source.yaml的dataset_uri配置项值为data_set目录的路径，启动Kafka Source：python [kafka-source.py绝对路径]；
6.运行Tianchi Workflow：python [tianchi_main.py绝对路径]；
7.查看AIFlow Master终端输出日志。