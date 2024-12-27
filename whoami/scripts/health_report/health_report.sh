#!/bin/bash
CURRENT_ROOT=$(dirname "$(readlink -f "$0")")
PROJECT_ROOT=$(dirname "$(dirname "$(dirname "$CURRENT_ROOT")")")

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

check_and_kill_port 8000

# 激活虚拟环境
CONDA_ENV='/work/soft/miniconda3/bin/'
export PATH=$CONDA_ENV:$PATH
eval "$(conda shell.bash hook)"
conda init bash
conda activate health_report

timestamp=$(date +"%Y%m%d%H%M%S")
LOG_PATH=$PROJECT_ROOT/whoami/logs/health_report
LOG_FILE="$LOG_PATH/${timestamp}.log"

if [ ! -d "$LOG_PATH" ]; then
    # 目录不存在，创建它
    mkdir -p "$LOG_PATH"
fi

echo "日志文件路径: $LOG_FILE"
cd "$PROJECT_ROOT" || { echo "无法切换到项目目录: $PROJECT_ROOT"; exit 1; }
nohup python -m whoami.scripts.health_report.scripts > "$LOG_FILE" 2>&1 &
echo "检测脚本已在后台运行，输出日志位于: $LOG_FILE"
