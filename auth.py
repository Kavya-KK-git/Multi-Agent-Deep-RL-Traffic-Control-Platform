import streamlit as st
from supabase import create_client, Client
import time

@st.cache_resource
def init_supabase() -> Client:
    url: str = st.secrets["SUPABASE_URL"]
    key: str = st.secrets["SUPABASE_KEY"]
    return create_client(url, key)

supabase = init_supabase()

def login_user(email, password):
    try:
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if response.user:
            st.session_state.user = response.user
            st.session_state.authenticated = True
            return True
    except (TypeError, AttributeError):
        try:
            if hasattr(supabase.auth, 'sign_in'):
                response = supabase.auth.sign_in(email=email, password=password)
            else:
                response = supabase.auth.sign_in_with_password(email=email, password=password)
                
            if response.user:
                st.session_state.user = response.user
                st.session_state.authenticated = True
                return True
        except Exception as e:
            return handle_auth_error(e, "Login")
    except Exception as e:
        return handle_auth_error(e, "Login")

def handle_auth_error(e, action="Action"):
    error_msg = getattr(e, 'message', str(e))
    if not error_msg:
        error_msg = repr(e)
    
    if "Invalid login credentials" in error_msg:
        st.error("Invalid email or password.")
    else:
        st.error(f"{action} failed: {error_msg}")
    return False

def sign_up_user(email, password):
    try:
        response = supabase.auth.sign_up({"email": email, "password": password})
        if response.user:
            st.success("Registration successful! You can now log in.")
            return True
    except TypeError:
        try:
            response = supabase.auth.sign_up(email=email, password=password)
            if response.user:
                st.success("Registration successful! You can now log in.")
                return True
        except Exception as e:
            return handle_auth_error(e, "Sign up")
    except Exception as e:
        return handle_auth_error(e, "Sign up")

def show_auth_page():
    st.markdown("""
    <style>
        #MainMenu {visibility: hidden;}
        header {visibility: hidden;}
        footer {visibility: hidden;}
        .viewerBadge_container__1QSob {display: none;}
        
        .stApp { 
            background: radial-gradient(circle at 10% 20%, rgb(14, 26, 40) 0%, rgb(8, 15, 23) 90%);
            color: #e2e8f0; 
        }
        h1 { color: #38bdf8 !important; font-family: 'Inter', sans-serif; font-weight: 800 !important; letter-spacing: -1px; text-align: center; text-shadow: 0 0 20px rgba(56, 189, 248, 0.3);}
        
        .stTabs [data-baseweb="tab-list"] {
            gap: 20px;
            background-color: transparent;
        }
        .stTabs [data-baseweb="tab"] {
            color: #94a3b8 !important;
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0px 0px;
            font-size: 16px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            color: #38bdf8 !important;
            border-bottom: 2px solid #38bdf8 !important;
        }
        .stTabs [data-baseweb="tab-highlight"] {
            background-color: transparent !important;
        }

        .stTextInput>div>div>input {
            background-color: rgba(15, 23, 42, 0.6) !important;
            color: #e2e8f0 !important;
            border: 1px solid rgba(56, 189, 248, 0.3) !important;
            border-radius: 10px !important;
            padding: 12px 15px !important;
        }
        .stTextInput>div>div>input:focus {
            border-color: #38bdf8 !important;
            box-shadow: 0 0 10px rgba(56, 189, 248, 0.3) !important;
        }
        
        .stButton>button { 
            width: 100%; border-radius: 12px; height: 3em; 
            background: linear-gradient(135deg, #0ea5e9, #0284c7); 
            color: white; font-weight: 700; border: none; font-size: 16px; 
            letter-spacing: 0.5px;
            box-shadow: 0 4px 15px rgba(2, 132, 199, 0.4);
            transition: all 0.3s ease;
            margin-top: 15px;
        }
        .stButton>button:hover { 
            background: linear-gradient(135deg, #38bdf8, #0ea5e9); 
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(56, 189, 248, 0.6); 
            color: white;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<br><br><h1>🚥 AI Traffic Network</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #bae6fd; font-size: 18px; margin-bottom: 30px; font-weight: 500;'>Welcome!</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tab_login, tab_signup = st.tabs([" Sign In", " Sign Up"])
        
        with tab_login:
            st.markdown("<br>", unsafe_allow_html=True)
            email_login = st.text_input("Email Address", placeholder="admin@example.com", key="login_email")
            password_login = st.text_input("Password", type="password", placeholder="••••••••", key="login_pass")
            
            if st.button(" Login", key="btn_login"):
                if email_login and password_login:
                    with st.spinner("Authenticating..."):
                        if login_user(email_login, password_login):
                            st.rerun()
                else:
                    st.warning("Please enter email and password")
                    
        with tab_signup:
            st.markdown("<br>", unsafe_allow_html=True)
            email_signup = st.text_input("Email Address", placeholder="admin@example.com", key="signup_email")
            password_signup = st.text_input("Password", type="password", placeholder="••••••••", key="signup_pass")
            
            if st.button(" Register User", key="btn_signup"):
                if email_signup and password_signup:
                    with st.spinner("Registering..."):
                        if sign_up_user(email_signup, password_signup):
                            pass
                else:
                    st.warning("Please enter email and password")


def init_auth():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
        
    if not st.session_state.authenticated:
        show_auth_page()
        st.stop()

def logout_button():
    if st.button(" Logout", key="logout_btn"):
        try:
            supabase.auth.sign_out()
        except Exception:
            pass
        st.session_state.authenticated = False
        if 'user' in st.session_state:
            del st.session_state.user
        st.rerun()
