"""
Goal Probability Simulator - Streamlit Web Application
A working demo that analyzes football/soccer clips to predict goal probability
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import sys
from pathlib import Path

# Import our modules
from xg_calculator import xGCalculator
from video_analyzer import VideoAnalyzer

# Page configuration
st.set_page_config(
    page_title="Goal Probability Simulator",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .xg-high {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    }
    .xg-medium {
        background: linear-gradient(135deg, #f12711 0%, #f5af19 100%);
    }
    .xg-low {
        background: linear-gradient(135deg, #4a00e0 0%, #8e2de2 100%);
    }
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #1E88E5 0%, #1565C0 100%);
        color: white;
        border: none;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        border-radius: 8px;
    }
    .info-box {
        background: #E3F2FD;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1E88E5;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def initialize_session_state():
    """Initialize session state variables."""
    if 'analyzer' not in st.session_state:
        with st.spinner("Loading AI models..."):
            try:
                st.session_state.analyzer = VideoAnalyzer()
            except Exception as e:
                st.session_state.analyzer = None
                st.warning(f"Using simplified mode (YOLO not available): {e}")
    
    if 'calculator' not in st.session_state:
        st.session_state.calculator = xGCalculator()


def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary location."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name


def get_xg_color(xg: float) -> str:
    """Return CSS class based on xG value."""
    if xg >= 0.5:
        return "xg-high"
    elif xg >= 0.2:
        return "xg-medium"
    else:
        return "xg-low"


def display_xg_gauge(xg: float, width: int = 400):
    """Display an xG gauge visualization."""
    percentage = xg * 100
    
    # Color based on value
    if xg >= 0.5:
        color = "#38ef7d"
    elif xg >= 0.2:
        color = "#f5af19"
    else:
        color = "#8e2de2"
    
    html = f"""
    <div style="text-align: center; margin: 1rem 0;">
        <div style="font-size: 3rem; font-weight: bold; color: {color};">
            {percentage:.1f}%
        </div>
        <div style="background: #eee; border-radius: 20px; height: 20px; width: {width}px; margin: 0 auto; overflow: hidden;">
            <div style="background: {color}; width: {percentage}%; height: 100%; border-radius: 20px; transition: width 0.5s;"></div>
        </div>
        <div style="margin-top: 0.5rem; color: #666;">
            Expected Goals: {xg:.3f}
        </div>
    </div>
    """
    return html


def main():
    # Initialize
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">⚽ Goal Probability Simulator</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload a football highlight clip and predict goal probability using AI</p>', unsafe_allow_html=True)
    
    # Sidebar - Controls and Info
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Shot parameters
        st.subheader("Shot Details")
        
        shot_distance = st.slider(
            "Distance to Goal (meters)",
            min_value=3,
            max_value=35,
            value=16,
            help="Estimated distance from ball to goal center"
        )
        
        shot_angle = st.slider(
            "Shot Angle (degrees)",
            min_value=10,
            max_value=60,
            value=35,
            help="Angle of shot relative to goal (wider = harder)"
        )
        
        defenders = st.slider(
            "Defenders Blocking",
            min_value=0,
            max_value=5,
            value=1,
            help="Number of defenders between ball and goal"
        )
        
        body_part = st.selectbox(
            "Body Part",
            options=['right_foot', 'left_foot', 'head', 'other'],
            index=0,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        shot_type = st.selectbox(
            "Shot Type",
            options=['open_play', 'counter', 'set_piece', 'corner', 'free_kick'],
            index=0,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        is_counter = st.checkbox("Counter Attack", value=shot_type == 'counter')
        
        st.divider()
        
        # Info section
        st.subheader("ℹ️ About xG")
        st.info(
            "xG (Expected Goals) measures the quality of a chance. "
            "A 0.30 xG means a typical player scores 30% of the time from that position."
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Video")
        
        uploaded_file = st.file_uploader(
            "Choose a football/soccer highlight clip",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a short clip (max 30 seconds recommended)"
        )
        
        if uploaded_file:
            st.success(f"✅ Uploaded: {uploaded_file.name}")
            
            # Video preview
            video_path = save_uploaded_file(uploaded_file)
            
            # Display video
            st.video(uploaded_file)
            
            # Analyze button
            if st.button("🔍 Analyze Shot", type="primary"):
                with st.spinner("Analyzing video..."):
                    # Use video analyzer if available
                    if st.session_state.analyzer:
                        try:
                            video_results = st.session_state.analyzer.analyze_video(video_path, max_frames=50)
                            st.session_state['video_results'] = video_results
                            st.session_state['shot_distance'] = shot_distance
                            st.session_state['shot_angle'] = shot_angle
                            st.session_state['defenders'] = defenders
                            st.session_state['body_part'] = body_part
                            st.session_state['shot_type'] = shot_type
                            st.session_state['is_counter'] = is_counter
                            st.success("✅ Analysis complete!")
                        except Exception as e:
                            st.error(f"Error analyzing video: {e}")
                            st.info("Using manual parameters instead.")
                    else:
                        st.info("Using manual parameters from sidebar.")
        else:
            st.info("👆 Upload a video clip to get started, or use the sidebar controls to manually adjust parameters.")
    
    with col2:
        st.subheader("⚽ Goal Probability Analysis")
        
        # Calculate xG
        calc = st.session_state.calculator
        
        # Use session state values if available, otherwise use sliders
        dist = st.session_state.get('shot_distance', shot_distance)
        ang = st.session_state.get('shot_angle', shot_angle)
        defs = st.session_state.get('defenders', defenders)
        body = st.session_state.get('body_part', body_part)
        stype = st.session_state.get('shot_type', shot_type)
        counter = st.session_state.get('is_counter', is_counter)
        
        # Calculate xG
        result = calc.calculate_xg(
            shot_x=105 - dist,  # Convert distance to x coordinate
            shot_y=34,  # Center of pitch
            defenders_in_path=defs,
            body_part=body,
            shot_type=stype if stype != 'counter' else 'through_ball',
            is_counter=counter
        )
        
        # Display results
        xg_value = result['xG']
        
        # Main xG display
        st.markdown(display_xg_gauge(xg_value), unsafe_allow_html=True)
        
        # Interpretation
        st.markdown(f"""
        <div class="info-box">
            <strong>📊 Interpretation:</strong> {result['interpretation']}
        </div>
        """, unsafe_allow_html=True)
        
        # Detailed breakdown
        with st.expander("📋 Detailed Breakdown", expanded=True):
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("Distance", f"{result['distance_meters']} m")
                st.metric("Angle", f"{result['angle_degrees']}°")
                st.metric("Body Part", body.replace('_', ' ').title())
            
            with col_b:
                st.metric("Defenders", defs)
                st.metric("Shot Type", stype.replace('_', ' ').title())
                st.metric("Counter", "Yes" if counter else "No")
        
        # Factor analysis
        st.subheader("📈 Probability Factors")
        
        factors = result['factors']
        
        # Create factor bars
        st.write("**Shot Quality Multiplier:**")
        sq_mult = factors['shot_quality_mult']
        st.progress(sq_mult, text=f"{sq_mult:.2f}x")
        
        st.write("**Pressure Factor:**")
        p_factor = factors['pressure_factor']
        st.progress(p_factor, text=f"{p_factor:.2f} ({p_factor*100:.0f}% of chance retained)")
        
        # Video analysis summary if available
        if 'video_results' in st.session_state:
            st.divider()
            st.subheader("🎥 Video Analysis Summary")
            
            vr = st.session_state['video_results']
            
            metric_cols = st.columns(3)
            with metric_cols[0]:
                st.metric("Frames Analyzed", vr['total_frames_analyzed'])
            with metric_cols[1]:
                st.metric("Avg Players", f"{vr['avg_players']:.1f}")
            with metric_cols[2]:
                st.metric("Ball Detection", f"{vr['ball_detection_rate']*100:.0f}%")
            
            st.info(f"Most common shot zone detected: **{vr['most_common_zone'].replace('_', ' ').title()}**")
    
    # Footer
    st.divider()
    
    # Demo scenarios
    st.subheader("🎮 Try These Example Scenarios")
    
    demo_cols = st.columns(3)
    
    demo_scenarios = [
        ("⚽ Penalty Kick", {"dist": 11, "ang": 45, "defs": 0, "body": "right_foot", "stype": "penalty"}),
        ("🏃 Counter Attack", {"dist": 14, "ang": 40, "defs": 1, "body": "right_foot", "stype": "through_ball"}),
        ("🤕 Long Range", {"dist": 28, "ang": 25, "defs": 2, "body": "right_foot", "stype": "open_play"}),
        ("🗣️ Header from Corner", {"dist": 12, "ang": 50, "defs": 2, "body": "head", "stype": "corner"}),
        ("✋ Free Kick", {"dist": 22, "ang": 30, "defs": 0, "body": "right_foot", "stype": "free_kick"}),
        ("🎯 Tap In", {"dist": 5, "ang": 55, "defs": 0, "body": "right_foot", "stype": "open_play"}),
    ]
    
    for idx, (name, params) in enumerate(demo_scenarios):
        with demo_cols[idx % 3]:
            if st.button(name, key=f"demo_{idx}"):
                # Update session state
                st.session_state['shot_distance'] = params['dist']
                st.session_state['shot_angle'] = params['ang']
                st.session_state['defenders'] = params['defs']
                st.session_state['body_part'] = params['body']
                st.session_state['shot_type'] = params['stype']
                st.session_state['is_counter'] = params['stype'] == 'through_ball'
                st.rerun()


if __name__ == "__main__":
    main()
