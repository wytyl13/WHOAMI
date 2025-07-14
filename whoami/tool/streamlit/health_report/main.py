import subprocess
import threading
import time

def start_api_server():
    """启动API服务器"""
    try:
        subprocess.Popen(["python", "api_server.py"])
        print("API服务器已启动")
    except Exception as e:
        print(f"启动API服务器失败: {e}")

# 在main函数开始时启动API服务
def main():
    # 启动API服务器（如果尚未运行）
    if 'api_server_started' not in st.session_state:
        threading.Thread(target=start_api_server).start()
        st.session_state.api_server_started = True
        time.sleep(1)  # 给API服务器一点启动时间