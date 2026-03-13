#!/usr/bin/env python3
"""
PharmaAI Web 后台启动脚本
"""

import subprocess
import os
import sys
import time

def start_server():
    """启动Streamlit服务器"""
    os.chdir('/home/lutao/.openclaw/workspace')
    
    # 启动Streamlit
    process = subprocess.Popen(
        [sys.executable, '-m', 'streamlit', 'run', 'pharma_web_app.py',
         '--server.port', '8501',
         '--server.headless', 'true'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True
    )
    
    print(f"🚀 PharmaAI Web 服务器已启动!")
    print(f"   PID: {process.pid}")
    print(f"")
    print(f"📱 访问地址:")
    print(f"   本地: http://localhost:8501")
    print(f"")
    print(f"⏹️  停止服务器: kill {process.pid}")
    
    return process.pid

if __name__ == "__main__":
    pid = start_server()
    
    # 等待几秒让服务器启动
    time.sleep(3)
    
    print(f"\n✅ 服务器正在运行，PID: {pid}")
