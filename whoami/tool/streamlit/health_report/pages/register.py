import streamlit as st
import sys
import os
import hashlib
import re
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
    page_title="注册 - 睡眠健康管理系统",
    page_icon="📝",
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

def validate_email(email):
    """验证邮箱格式"""
    if not email:  # 非必填
        return True
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_phone(phone):
    """验证手机号格式"""
    if not phone:  # 非必填
        return True
    pattern = r'^1[3-9]\d{9}$'
    return re.match(pattern, phone) is not None

def check_username_exists(username):
    """检查用户名是否已存在"""
    try:
        sql_provider = get_sql_provider()
        result = sql_provider.get_record_by_condition(
            condition={"username": username},
            fields=["id"]
        )
        return len(result) > 0
    except Exception as e:
        st.error(f"检查用户名失败: {str(e)}")
        return True  # 出错时返回True，避免重复注册

def check_email_exists(email):
    """检查邮箱是否已被使用"""
    if not email:
        return False
    try:
        sql_provider = get_sql_provider()
        result = sql_provider.get_record_by_condition(
            condition={"email": email},
            fields=["id"]
        )
        return len(result) > 0
    except Exception as e:
        st.error(f"检查邮箱失败: {str(e)}")
        return True  # 出错时返回True，避免重复注册

def check_phone_exists(phone):
    """检查手机号是否已被使用"""
    if not phone:
        return False
    try:
        sql_provider = get_sql_provider()
        result = sql_provider.get_record_by_condition(
            condition={"phone": phone},
            fields=["id"]
        )
        return len(result) > 0
    except Exception as e:
        st.error(f"检查手机号失败: {str(e)}")
        return True  # 出错时返回True，避免重复注册

def register_user(username, password, name=None, gender=None, age=None, email=None, phone=None, address=None):
    """注册用户"""
    try:
        # 检查用户名是否已存在
        if check_username_exists(username):
            return False, "用户名已存在"
        
        # 检查邮箱是否已被使用（如果提供了邮箱）
        if email and check_email_exists(email):
            return False, "邮箱已被使用"
        
        # 检查手机号是否已被使用（如果提供了手机号）
        if phone and check_phone_exists(phone):
            return False, "手机号已被使用"
        
        # 准备用户数据
        user_data = {
            "username": username,
            "password": hash_password(password),
            "full_name": name or "",
            "gender": gender or "",
            "age": age,
            "email": email or "",
            "phone": phone or "",
            "address": address or "",
            "status": "active",
            "creator": "system",
            "tenant_id": 0
        }
        
        # 插入用户数据到数据库
        sql_provider = get_sql_provider()
        record_id = sql_provider.add_record(user_data)
        
        if record_id:
            return True, "注册成功"
        else:
            return False, "注册失败，请重试"
            
    except Exception as e:
        st.error(f"注册过程中发生错误: {str(e)}")
        return False, "注册失败，系统错误"

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

# CSS样式 - 和登录页面保持一致
st.markdown("""
<style>
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
    margin: 40px 0 30px 0;
}

.main-title h1 {
    font-size: 2.8rem !important;
    margin: 0 0 10px 0 !important;
    font-weight: bold !important;
    text-shadow: 0 4px 8px rgba(0,0,0,0.3) !important;
}

.main-title p {
    font-size: 1.2rem !important;
    margin: 0 !important;
    opacity: 0.9 !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3) !important;
}

/* 输入框样式 */
.stTextInput > div > div > input, .stSelectbox > div > div > input {
    height: 42px !important;
    font-size: 16px !important;
    padding: 10px 15px !important;
    border-radius: 8px !important;
    border: 1.5px solid rgba(255, 255, 255, 0.3) !important;
    background: rgba(255, 255, 255, 0.95) !important;
    transition: all 0.3s ease !important;
}

.stTextInput > div > div > input:focus, .stSelectbox > div > div > input:focus {
    border-color: rgba(255, 255, 255, 0.8) !important;
    background: white !important;
    box-shadow: 0 0 15px rgba(255, 255, 255, 0.2) !important;
}

.stTextInput label, .stSelectbox label {
    color: white !important;
    font-weight: 600 !important;
    font-size: 15px !important;
    margin-bottom: 6px !important;
}

/* 选择框样式 */
.stSelectbox > div > div {
    background: rgba(255, 255, 255, 0.95) !important;
    border-radius: 8px !important;
    border: 1.5px solid rgba(255, 255, 255, 0.3) !important;
}

.stSelectbox > div > div > div {
    color: #333 !important;
    font-size: 16px !important;
}

/* 按钮样式 */
.stButton > button {
    width: 100% !important;
    height: 42px !important;
    background: rgba(255, 255, 255, 0.2) !important;
    color: white !important;
    border: 1.5px solid rgba(255, 255, 255, 0.5) !important;
    border-radius: 8px !important;
    font-size: 16px !important;
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

/* 注册卡片 */
.register-card {
    background: rgba(255, 255, 255, 0.1) !important;
    backdrop-filter: blur(15px) !important;
    border: 1.5px solid rgba(255, 255, 255, 0.2) !important;
    border-radius: 15px !important;
    padding: 25px !important;
    box-shadow: 0 20px 40px rgba(0,0,0,0.3) !important;
    margin: 0 auto !important;
    max-width: 450px !important;
}

.register-card h3 {
    text-align: center !important;
    color: white !important;
    margin: 0 0 20px 0 !important;
    font-weight: 600 !important;
    font-size: 1.3rem !important;
}

/* 消息样式 */
.success-msg {
    background: rgba(40, 167, 69, 0.9);
    color: white;
    padding: 10px 15px;
    border-radius: 8px;
    text-align: center;
    margin: 10px 0;
    font-size: 15px;
    font-weight: 600;
}

.error-msg {
    background: rgba(220, 53, 69, 0.9);
    color: white;
    padding: 10px 15px;
    border-radius: 8px;
    text-align: center;
    margin: 10px 0;
    font-size: 15px;
    font-weight: 600;
}

/* 提示信息 */
.hint-text {
    color: rgba(255, 255, 255, 0.8);
    font-size: 13px;
    margin-top: 4px;
    margin-bottom: 8px;
}

.required {
    color: #ff6b6b;
    font-weight: bold;
}

/* 表单间距优化 */
.stForm {
    border: none !important;
    background: transparent !important;
}

.stForm > div {
    gap: 10px !important;
}

/* 两列布局 */
.col-container {
    display: flex;
    gap: 15px;
    margin-bottom: 10px;
}

.col-container > div {
    flex: 1;
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
        <h1>🌙 睡眠健康管理系统</h1>
        <p>专业的睡眠监测与健康管理平台</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 居中容器
    col1, col2, col3 = st.columns([0.5, 1, 0.5])
    with col2:
        # 注册卡片
        st.markdown("""
        <div class="register-card">
            <h3>📝 用户注册</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # 注册表单
        with st.form("register_form"):
            # 必填信息
            st.markdown('<p style="color: white; font-size: 16px; font-weight: bold; margin: 15px 0 10px 0;">必填信息 <span class="required">*</span></p>', unsafe_allow_html=True)
            
            username = st.text_input("👤 用户名 *", placeholder="请输入用户名")
            st.markdown('<div class="hint-text">用户名长度为3-20个字符</div>', unsafe_allow_html=True)
            
            password = st.text_input("🔒 密码 *", type="password", placeholder="请输入密码")
            st.markdown('<div class="hint-text">密码长度至少6位</div>', unsafe_allow_html=True)
            
            confirm_pwd = st.text_input("🔒 确认密码 *", type="password", placeholder="请再次输入密码")
            st.markdown('<div class="hint-text">请确保两次密码一致</div>', unsafe_allow_html=True)
            
            # 可选信息
            st.markdown('<p style="color: white; font-size: 16px; font-weight: bold; margin: 20px 0 10px 0;">可选信息（可跳过）</p>', unsafe_allow_html=True)
            
            name = st.text_input("👨‍💼 姓名", placeholder="请输入真实姓名（可选）")
            
            # 性别和年龄一行显示
            col_gender, col_age = st.columns(2)
            with col_gender:
                gender = st.selectbox("⚧ 性别", ["", "男", "女", "其他"], index=0)
            with col_age:
                age = st.text_input("📅 年龄", placeholder="年龄（可选）")
            
            email = st.text_input("📧 邮箱", placeholder="请输入邮箱地址（可选）")
            st.markdown('<div class="hint-text">用于接收系统通知</div>', unsafe_allow_html=True)
            
            phone = st.text_input("📱 手机号", placeholder="请输入手机号（可选）")
            st.markdown('<div class="hint-text">用于账户安全验证</div>', unsafe_allow_html=True)
            
            address = st.text_input("🏠 地址", placeholder="请输入居住地址（可选）")
            st.markdown('<div class="hint-text">用于个性化服务推荐</div>', unsafe_allow_html=True)
            
            # 按钮
            col_a, col_b = st.columns(2)
            with col_a:
                register_btn = st.form_submit_button("🚀 注册", type="primary")
            with col_b:
                back_btn = st.form_submit_button("🔙 返回登录")
            
            if back_btn:
                st.switch_page("pages/login.py")
            
            if register_btn:
                # 验证必填项
                if not username or not password or not confirm_pwd:
                    show_error("请填写所有必填信息（用户名、密码、确认密码）")
                elif len(username) < 3 or len(username) > 20:
                    show_error("用户名长度应为3-20个字符")
                elif len(password) < 6:
                    show_error("密码长度至少6位")
                elif password != confirm_pwd:
                    show_error("两次密码不一致")
                elif age and not age.isdigit():
                    show_error("年龄必须为数字")
                elif age and (int(age) < 1 or int(age) > 150):
                    show_error("年龄必须在1-150之间")
                elif not validate_email(email):
                    show_error("请输入有效的邮箱地址")
                elif not validate_phone(phone):
                    show_error("请输入有效的手机号码")
                else:
                    # 注册用户
                    success, message = register_user(
                        username=username,
                        password=password,
                        name=name if name else None,
                        gender=gender if gender else None,
                        age=int(age) if age and age.isdigit() else None,
                        email=email if email else None,
                        phone=phone if phone else None,
                        address=address if address else None
                    )
                    if success:
                        show_success(message + "！正在跳转到登录页面...")
                        st.balloons()
                        # 延迟跳转到登录页面
                        st.switch_page("pages/login.py")
                    else:
                        show_error(message)

if __name__ == "__main__":
    main()