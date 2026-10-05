"""启动 AIFlow Master 的入口脚本。

读取同目录下的 master.yaml 并以阻塞方式启动 AIFlow Master，是跑整条流程的第一步。
"""

import os

from ai_flow import AIFlowMaster

if __name__ == '__main__':
    master = AIFlowMaster(os.path.dirname(os.path.abspath(__file__)) + '/master.yaml')
    master.start(is_block=True)
