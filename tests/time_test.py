#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单的系统时间显示脚本
每秒更新一次，显示当前系统时间
"""

import time
import os

def clear_screen():
    """清屏函数"""
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    """主函数"""
    print("系统时间显示 (按 Ctrl+C 退出)")
    print("=" * 40)
    
    try:
        while True:
            # 获取当前时间
            current_time = time.time()
            
            # 格式化时间显示
            formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(current_time))
            
            # 显示时间戳和格式化时间
            print(f"\r当前时间: {formatted_time} | 时间戳: {current_time:.3f}", end="", flush=True)
            
            # 等待1秒
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n\n程序已退出")

if __name__ == "__main__":
    main()
