import streamlit as st
import pandas as pd
import altair as alt
import os
import time
import subprocess
import sys

st.set_page_config(
    page_title="Multi-Agent RL Dashboard",
    page_icon="🚥",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    /* Hide top right Streamlit menus */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    .viewerBadge_container__1QSob {display: none;} /* Hide deploy button if exists */
    
    /* Dark Premium Glassmorphism Theme */
    .stApp { 
        background: radial-gradient(circle at 10% 20%, rgb(14, 26, 40) 0%, rgb(8, 15, 23) 90%);
        color: #e2e8f0; 
    }
    h1 { color: #38bdf8 !important; font-family: 'Inter', sans-serif; font-weight: 800 !important; letter-spacing: -1px; text-shadow: 0 0 20px rgba(56, 189, 248, 0.3);}
    h2, h3 { color: #7dd3fc !important; font-family: 'Inter', sans-serif;}
    
    /* Glassmorphism Cards */
    .premium-card { 
        background: rgba(30, 41, 59, 0.4); 
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px; 
        padding: 30px; 
        margin-bottom: 25px; 
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5); 
    }
    .stat-box { 
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.6), rgba(30, 41, 59, 0.4));
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 15px; 
        padding: 25px; 
        text-align: center; 
        border-left: 4px solid #0ea5e9; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.3s ease;
    }
    .stat-box:hover {
        transform: translateY(-5px);
        border-left: 4px solid #38bdf8;
        box-shadow: 0 8px 25px rgba(14, 165, 233, 0.4);
    }
    .stat-value { font-size: 38px; font-weight: 900; color: #bae6fd; text-shadow: 0 0 10px rgba(186, 230, 253, 0.3);}
    .stat-label { font-size: 13px; color: #94a3b8; text-transform: uppercase; font-weight: 700; letter-spacing: 1px;}
    
    /* Buttons */
    .stButton>button { 
        width: 100%; border-radius: 12px; height: 3.5em; 
        background: linear-gradient(135deg, #0ea5e9, #0284c7); 
        color: white; font-weight: 700; border: none; font-size: 16px; 
        letter-spacing: 0.5px;
        box-shadow: 0 4px 15px rgba(2, 132, 199, 0.4);
        transition: all 0.3s ease;
    }
    .stButton>button:hover { 
        background: linear-gradient(135deg, #38bdf8, #0ea5e9); 
        color: white; 
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(56, 189, 248, 0.6); 
    }
    
    /* Center the main image / CSS Header */
    .hero-header {
        background: linear-gradient(135deg, rgba(2, 132, 199, 0.8), rgba(15, 23, 42, 0.9)), url("https://images.unsplash.com/photo-1449824913935-59a10b8d2000?auto=format&fit=crop&w=1200&q=80");
        background-size: cover;
        background-position: center;
        border-radius: 20px;
        padding: 40px;
        margin-bottom: 30px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Target second column button for Stop */
    div[data-testid="column"]:nth-of-type(2) button {
        background: linear-gradient(135deg, #ef4444, #b91c1c);
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.4);
    }
    div[data-testid="column"]:nth-of-type(2) button:hover {
        background: linear-gradient(135deg, #f87171, #ef4444);
        box-shadow: 0 8px 25px rgba(248, 113, 113, 0.6);
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-header">
    <h1 style="color: white !important; font-size: 45px; margin-bottom: 5px;">🚥 Advanced AI Traffic Network</h1>
    <p style='font-size: 18px; color: #bae6fd; font-weight: 600;'>Multi-Agent Dynamic Signal Optimization</p>
</div>
""", unsafe_allow_html=True)

colA, colB, colC = st.columns([1, 1, 2])
with colA:
    if st.button("🚀 INITIATE AI SIMULATION"):
        st.session_state.train_process = subprocess.Popen([sys.executable, "training/train.py"])
        st.toast("Neural Network Initialized...", icon="✅")
with colB:
    if st.button("⏹️ HALT SIMULATION"):
        if 'train_process' in st.session_state:
            st.session_state.train_process.kill()
            st.toast("Simulation manually stopped.", icon="🛑")
            del st.session_state.train_process
        else:
            st.warning("No tracking process found.")

st.markdown("<hr style='border-color: rgba(255,255,255,0.1);'>", unsafe_allow_html=True)

log_file = "training_log.csv"
placeholder = st.empty()

if os.path.exists(log_file):
    try:
        df = pd.read_csv(log_file)
        if not df.empty:
            
            st.markdown("### ⚡ Live City Analytics")
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"<div class='stat-box'><div class='stat-value'>{int(df['step'].iloc[-1])}</div><div class='stat-label'>Neural Step Count</div></div>", unsafe_allow_html=True)
            with m2:
                q_val = df['queue_length'].iloc[-1]
                q_color = "#34d399" if q_val < 5 else ("#fbbf24" if q_val < 15 else "#f87171")
                st.markdown(f"<div class='stat-box'><div class='stat-value' style='color: {q_color};'>{q_val:.0f}</div><div class='stat-label'>Cars Waiting (Queue)</div></div>", unsafe_allow_html=True)
            with m3:
                spd = df['avg_speed'].iloc[-1] if 'avg_speed' in df.columns else 0.0
                st.markdown(f"<div class='stat-box'><div class='stat-value'>{spd:.1f} m/s</div><div class='stat-label'>Network Average Speed</div></div>", unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            n1, n2 = st.columns(2)
            with n1:
                pg = df['passed_green'].iloc[-1] if 'passed_green' in df.columns else 0
                st.markdown(f"<div class='stat-box' style='border-left-color: #10b981;'><div class='stat-value' style='color: #10b981;'>{pg}</div><div class='stat-label'>Vehicles Passed (Green)</div></div>", unsafe_allow_html=True)
            with n2:
                py = df['passed_yellow'].iloc[-1] if 'passed_yellow' in df.columns else 0
                st.markdown(f"<div class='stat-box' style='border-left-color: #facc15;'><div class='stat-value' style='color: #facc15;'>{py}</div><div class='stat-label'>Vehicles Passed (Yellow)</div></div>", unsafe_allow_html=True)

            st.markdown("<br><h3 style='color: #38bdf8 !important;'>📊 AI Optimization Telemetry</h3>", unsafe_allow_html=True)
            
           
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown("**Traffic Queue Reduction Engine (Lower is Better)**")
                chart_q = alt.Chart(df.tail(200)).mark_area(
                    line={'color':'#38bdf8'},
                    color=alt.Gradient(
                        gradient='linear',
                        stops=[alt.GradientStop(color='#0284c7', offset=0),
                               alt.GradientStop(color='rgba(2,132,199,0)', offset=1)],
                        x1=1, x2=1, y1=1, y2=0
                    )
                ).encode(
                    x=alt.X('step:Q', title="Processing Tick", axis=alt.Axis(grid=False, labelColor='#94a3b8', titleColor='#cbd5e1')),
                    y=alt.Y('queue_length:Q', title="Cars in Queue", axis=alt.Axis(gridColor='rgba(255,255,255,0.1)', labelColor='#94a3b8', titleColor='#cbd5e1'))
                ).properties(height=300).configure_view(strokeWidth=0).configure_axis(domain=False)
                
                st.altair_chart(chart_q, use_container_width=True)
                
            with c2:
                st.markdown("**AI Reward / Learning Progression (Higher is Better)**")
                chart_r = alt.Chart(df.tail(200)).mark_line(color="#10b981", strokeWidth=3).encode(
                    x=alt.X('step:Q', title="Processing Tick", axis=alt.Axis(grid=False, labelColor='#94a3b8', titleColor='#cbd5e1')),
                    y=alt.Y('reward:Q', title="Reward Points", axis=alt.Axis(gridColor='rgba(255,255,255,0.1)', labelColor='#94a3b8', titleColor='#cbd5e1'))
                ).properties(height=300).configure_view(strokeWidth=0).configure_axis(domain=False)
                
                st.altair_chart(chart_r, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("** Signal Throughput Analytics (Passed on Green vs Yellow)**")
            
            df_melted = df.tail(200).melt(id_vars=['step'], value_vars=['passed_green', 'passed_yellow'], var_name='Metric', value_name='Count')
            
            chart_t = alt.Chart(df_melted).mark_line(strokeWidth=3).encode(
                x=alt.X('step:Q', title="Processing Tick", axis=alt.Axis(grid=False, labelColor='#94a3b8', titleColor='#cbd5e1')),
                y=alt.Y('Count:Q', title="Cumulative Vehicles", axis=alt.Axis(gridColor='rgba(255,255,255,0.1)', labelColor='#94a3b8', titleColor='#cbd5e1')),
                color=alt.Color('Metric:N', scale=alt.Scale(domain=['passed_green', 'passed_yellow'], range=['#10b981', '#facc15']))
            ).properties(height=300).configure_view(strokeWidth=0).configure_axis(domain=False)
            
            st.altair_chart(chart_t, use_container_width=True)
            
            st.markdown("<br><h3 style='color: #10b981 !important;'>🚦 Live Signal Intelligent Switches</h3>", unsafe_allow_html=True)
            st.markdown("<p style='color:#cbd5e1; font-size:15px;'>Real-time log of the AI Engine deciding precisely how many seconds to hold the Green Light based on the exact number of cars waiting.</p>", unsafe_allow_html=True)
            
            if os.path.exists("signal_changes.txt"):
                try:
                    with open("signal_changes.txt", "r") as f:
                        lines = f.readlines()
                        recent_logs = lines[-15:] if len(lines) > 15 else lines
                        log_lines = []
                        for line in recent_logs[::-1]:
                            if line.strip():
                                log_lines.append(f"<div style='padding: 4px 0; border-bottom: 1px dashed rgba(16, 185, 129, 0.2); margin-bottom: 2px;'>{line.strip()}</div>")
                        
                        log_content = "".join(log_lines)
                        if not log_content:
                            log_content = "<div>AI compiling... Waiting for first signal switch decision...</div>"
                        
                        st.markdown(f"""
                        <div style="background-color: #0f172a; padding: 20px; border-radius: 15px; border-left: 5px solid #10b981; font-family: monospace; font-size: 15px; box-shadow: inset 0 0 10px rgba(0,0,0,0.5); overflow-y: auto; max-height: 300px; color: #34d399;">
                            {log_content}
                        </div>
                        """, unsafe_allow_html=True)
                except Exception:
                    pass
            else:
                st.info("Signal log not yet generated.")
                
        else:
            placeholder.info("Loading Telemetry...")
    except BaseException as e:
        pass
else:
    placeholder.markdown("""
    <div class="premium-card" style="text-align: center; margin-top: 50px;">
        <h2 style="color: #64748b !important;">System Idle</h2>
        <p>Click 'INITIATE AI SIMULATION' to boot the neural network and connect to the Salem Traffic Grid.</p>
    </div>
    """, unsafe_allow_html=True)

time.sleep(2)
try:
    st.rerun()
except AttributeError:
    st.experimental_rerun()
