#!/bin/bash

# 从命令行参数获取 PROJECT_ROOT，如果未提供，则使用现有方式
if [ -n "$1" ]; then
    PROJECT_ROOT="$1"
else
    CURRENT_ROOT=$(dirname "$(readlink -f "$0")")
    PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$CURRENT_ROOT")")")
fi

check_and_kill_port() {
    local port=$1
    local pid=$(sudo lsof -t -i :$port)

    if [ -n "$pid" ]; then
        echo "端口 $port 已被占用，正在终止进程 $pid"
        sudo kill $pid
    else
        echo "端口 $port 未被占用"
    fi
}

check_and_kill_port 8000
check_and_kill_port 8888

# 激活虚拟环境
CONDA_ENV='/home/weiyutao/miniconda3/bin/'
export PATH=$CONDA_ENV:$PATH
eval "$(conda shell.bash hook)"
conda init bash
conda activate whoami

timestamp=$(date +"%Y%m%d%H%M%S")
LOG_PATH=$PROJECT_ROOT/whoami/logs/real_time_streamlit
LOG_FILE="$LOG_PATH/${timestamp}.log"

if [ ! -d "$LOG_PATH" ]; then
    # 目录不存在，创建它
    mkdir -p "$LOG_PATH"
fi

echo "日志文件路径: $LOG_FILE"
cd "$PROJECT_ROOT" || { echo "无法切换到项目目录: $PROJECT_ROOT"; exit 1; }
nohup streamlit run whoami/tool/streamlit/health_report/main.py \
  --server.address 0.0.0.0 \
  --server.port 8000 \
  --server.sslCertFile /work/ai/WHOAMI/tests/shunxikj.com.crt \
  --server.sslKeyFile /work/ai/WHOAMI/tests/shunxikj.com.key > "$LOG_FILE" 2>&1 &
echo "检测脚本已在后台运行，输出日志位于: $LOG_FILE"
