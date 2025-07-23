import streamlit as st
import sys
import os
import hashlib
from datetime import datetime

# 添加项目路径
project_root = "/work/ai/WHOAMI"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 导入数据库相关模块
from whoami.tool.streamlit.health_report.table.user_data import UserData
from whoami.provider.sql_provider import SqlProvider

# 页面配置
st.set_page_config(
    page_title="登录 - 社区智能体",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 数据库配置
SQL_CONFIG_PATH = '/work/ai/WHOAMI/whoami/scripts/health_report/sql_config.yaml'

@st.cache_resource
def get_sql_provider():
    """获取SQL提供器实例（使用缓存避免重复创建）"""
    return SqlProvider(model=UserData, sql_config_path=SQL_CONFIG_PATH)

def hash_password(password):
    """密码哈希"""
    return hashlib.sha256(password.encode()).hexdigest()

def verify_user(username, password):
    """验证用户"""
    try:
        sql_provider = get_sql_provider()
        # 查询用户
        result = sql_provider.get_record_by_condition(
            condition={
                "username": username,
            },
            fields=["id", "username", "password", "full_name", "gender", "age", "address", "phone", "email", "status", "create_time", "tenant_id"]
        )
        
        if result and len(result) > 0:
            user = result[0]
            # 验证密码
            return user["password"] == hash_password(password)
        return False
    except Exception as e:
        st.error(f"数据库连接错误: {str(e)}")
        return False


def get_user_data(username):
    """获取用户数据"""
    try:
        sql_provider = get_sql_provider()
        result = sql_provider.get_record_by_condition(
            condition={
                "username": username,
            },
            fields=["id", "username", "full_name", "gender", "age", "address", "phone", "email", "status", "create_time", "tenant_id"]
        )
        
        if result and len(result) > 0:
            user = result[0]
            # 将SQLAlchemy对象转换为字典
            return user
        return {}
    except Exception as e:
        st.error(f"获取用户数据错误: {str(e)}")
        return {}

def check_database_connection():
    """检查数据库连接"""
    try:
        sql_provider = get_sql_provider()
        # 尝试执行一个简单的查询来测试连接
        sql_provider.get_record_by_condition(condition={"username": "test_connection_check"})
        return True
    except Exception as e:
        st.error(f"数据库连接失败: {str(e)}")
        return False

# CSS样式（保持原有样式不变）
st.markdown("""
<style>

/* 隐藏侧边栏 */
section[data-testid="stSidebar"] {
    display: none !important;
}

.css-1d391kg, .css-17lntkn, .css-1rs6os, .css-10trblm,
.css-12oz5g7, .css-1outpf7, .css-1y4p8pa, .css-1lcbmhc,
.css-1v0mbdj, .css-1cypcdb, .css-17eq0hr, .css-zt5igj {
    display: none !important;
}

button[kind="header"] {
    display: none !important;
}
/* 隐藏侧边栏 */


/* 隐藏Streamlit默认元素 */
.stApp > header {display: none;}
.main .block-container {
    padding-top: 0 !important;
    padding-bottom: 0 !important;
}

/* 全屏背景 */
.stApp {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

/* 浮动粒子动画 */
.particles {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
    z-index: -1;
    pointer-events: none;
}

.particle {
    position: absolute;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 50%;
    animation: float 6s ease-in-out infinite;
}

@keyframes float {
    0%, 100% { transform: translateY(0px) rotate(0deg); }
    50% { transform: translateY(-20px) rotate(180deg); }
}

/* 主标题样式 */
.main-title {
    text-align: center;
    color: white;
    margin: 60px 0 40px 0;
}

.main-title h1 {
    font-size: 3rem !important;
    margin: 0 0 15px 0 !important;
    font-weight: bold !important;
    text-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
}

.main-title p {
    font-size: 1.3rem !important;
    margin: 0 !important;
    opacity: 0.9 !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
}

/* 紧凑的输入框样式 */
.stTextInput > div > div > input {
    height: 42px !important;
    font-size: 18px !important;
    padding: 10px 15px !important;
    border-radius: 6px !important;
    border: 1.5px solid rgba(255, 255, 255, 0.3) !important;
    background: rgba(255, 255, 255, 0.95) !important;
    transition: all 0.3s ease !important;
}

.stTextInput > div > div > input:focus {
    border-color: rgba(255, 255, 255, 0.8) !important;
    background: white !important;
    box-shadow: 0 0 15px rgba(255, 255, 255, 0.2) !important;
}

.stTextInput label {
    color: white !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    margin-bottom: 6px !important;
}

/* 紧凑的按钮样式 */
.stButton > button {
    width: 100% !important;
    height: 42px !important;
    background: rgba(255, 255, 255, 0.2) !important;
    color: white !important;
    border: 1.5px solid rgba(255, 255, 255, 0.5) !important;
    border-radius: 6px !important;
    font-size: 17px !important;
    font-weight: bold !important;
    transition: all 0.3s ease !important;
    backdrop-filter: blur(5px) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 20px rgba(255, 255, 255, 0.2) !important;
    background: rgba(255, 255, 255, 0.3) !important;
}

/* 主要按钮样式 */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%) !important;
    border-color: rgba(255, 107, 107, 0.5) !important;
}

.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #ff5252 0%, #ff3838 100%) !important;
}

/* 紧凑的登录卡片 */
.login-card {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(15px) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 12px !important;
    padding: 25px !important;
    box-shadow: 0 20px 40px rgba(0,0,0,0.3) !important;
    margin: 0 auto !important;
    max-width: 320px !important;
}

.login-card h3 {
    text-align: center !important;
    color: white !important;
    margin: 0 0 20px 0 !important;
    font-weight: 600 !important;
    font-size: 1.4rem !important;
}

/* 消息样式 */
.success-msg {
    background: rgba(40, 167, 69, 0.9);
    color: white;
    padding: 10px 15px;
    border-radius: 6px;
    text-align: center;
    margin: 10px 0;
    font-size: 16px;
    font-weight: 600;
}

.error-msg {
    background: rgba(220, 53, 69, 0.9);
    color: white;
    padding: 10px 15px;
    border-radius: 6px;
    text-align: center;
    margin: 10px 0;
    font-size: 16px;
    font-weight: 600;
}

/* 注册链接 */
.register-link {
    text-align: center;
    margin-top: 15px;
    color: rgba(255, 255, 255, 0.9);
    font-size: 15px;
}

.register-link a {
    color: white;
    text-decoration: none;
    font-weight: bold;
    padding: 3px 8px;
    border-radius: 5px;
    transition: all 0.3s ease;
}

.register-link a:hover {
    background: rgba(255, 255, 255, 0.2);
}

/* 表单间距优化 */
.stForm {
    border: none !important;
    background: transparent !important;
}

.stForm > div {
    gap: 12px !important;
}

/* 数据库状态指示器 */
.db-status {
    position: fixed;
    top: 10px;
    right: 10px;
    padding: 5px 10px;
    border-radius: 15px;
    font-size: 12px;
    font-weight: bold;
    z-index: 1000;
}

.db-status.connected {
    background: rgba(40, 167, 69, 0.8);
    color: white;
}

.db-status.disconnected {
    background: rgba(220, 53, 69, 0.8);
    color: white;
}
</style>
""", unsafe_allow_html=True)

def show_success(msg):
    st.markdown(f'<div class="success-msg">✅ {msg}</div>', unsafe_allow_html=True)

def show_error(msg):
    st.markdown(f'<div class="error-msg">❌ {msg}</div>', unsafe_allow_html=True)

def main():
    """主函数"""
    
    # 检查数据库连接状态
    db_connected = check_database_connection()
    
    # 显示数据库连接状态
    if db_connected:
        st.markdown('<div class="db-status connected">🟢 数据库已连接</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="db-status disconnected">🔴 数据库连接失败</div>', unsafe_allow_html=True)
        st.error("⚠️ 数据库连接失败，请检查配置文件和网络连接")
        return
    
    # 背景粒子效果
    st.markdown("""
    <div class="particles">
        <div class="particle" style="left: 10%; top: 20%; width: 8px; height: 8px; animation-delay: 0s;"></div>
        <div class="particle" style="left: 20%; top: 80%; width: 6px; height: 6px; animation-delay: 1s;"></div>
        <div class="particle" style="left: 80%; top: 30%; width: 10px; height: 10px; animation-delay: 2s;"></div>
        <div class="particle" style="left: 70%; top: 70%; width: 5px; height: 5px; animation-delay: 3s;"></div>
        <div class="particle" style="left: 30%; top: 40%; width: 12px; height: 12px; animation-delay: 4s;"></div>
        <div class="particle" style="left: 90%; top: 60%; width: 7px; height: 7px; animation-delay: 5s;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # 主标题
    st.markdown("""
    <div class="main-title">
        <h1>🤖 社区智能体</h1>
        <p>专业的社区实时交互解决方案</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 居中容器
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        # 登录卡片
        st.markdown("""
        <div class="login-card">
            <h3>🔐 用户登录</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # 登录表单
        with st.form("login_form"):
            username = st.text_input("👤 用户名", placeholder="请输入用户名")
            password = st.text_input("🔒 密码", type="password", placeholder="请输入密码")
            
            col_a, col_b = st.columns(2)
            with col_a:
                login_btn = st.form_submit_button("🚀 登录", type="primary")
            with col_b:
                register_btn = st.form_submit_button("📝 注册")
            
            if register_btn:
                st.switch_page("pages/register.py")
            
            if login_btn:
                if username and password:
                    if verify_user(username, password):
                        # 保存登录状态到session
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.user_data = get_user_data(username)
                        show_success("登录成功！正在跳转...")
                        st.balloons()
                        # 跳转到用户主页
                        st.switch_page("pages/user_dashboard.py")
                    else:
                        show_error("用户名或密码错误！")
                else:
                    show_error("请输入用户名和密码")
        
        # 注册链接
        st.markdown("""
        <div class="register-link">
            还没有账号？<a href="/register">立即注册</a>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()