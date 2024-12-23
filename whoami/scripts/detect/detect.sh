#!/bin/bash
CURRENT_ROOT=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$CURRENT_ROOT")")")
url_str_flag="${1:-new}" # Get the URL from the first argument

check_and_kill_port() {
    local port=$1
    local pid=$(lsof -t -i :$port)

    if [ -n "$pid" ]; then
        echo "端口 $port 已被占用，正在终止进程 $pid"
        kill $pid
    else
        echo "端口 $port 未被占用"
    fi
}

if [ -z "$url_str_flag" ] || [ "$url_str_flag" == "new" ]; then
    # 如果参数为 "new"，检查 9999 端口
    check_and_kill_port 9999
else
    # 否则检查 8888 端口
    check_and_kill_port 8888
fi

# 激活虚拟环境
CONDA_ENV='/work/soft/miniconda3/bin/'
export PATH=$CONDA_ENV:$PATH
eval "$(conda shell.bash hook)"
conda init bash
conda activate detect

timestamp=$(date +"%Y%m%d%H%M%S")
LOG_PATH=$PROJECT_ROOT/whoami/logs/detect
LOG_FILE="$LOG_PATH/${url_str_flag}_${timestamp}.log"

if [ ! -d "$LOG_PATH" ]; then
    # 目录不存在，创建它
    mkdir -p "$LOG_PATH"
fi

echo "日志文件路径: $LOG_FILE"
cd "$PROJECT_ROOT" || { echo "无法切换到项目目录: $PROJECT_ROOT"; exit 1; }
nohup python -m whoami.scripts.detect.scripts --url "$url_str_flag" > "$LOG_FILE" 2>&1 &
echo "检测脚本已在后台运行，输出日志位于: $LOG_FILE"

