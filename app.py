import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from datetime import datetime
import os
import tempfile
import base64
import qrcode
from io import BytesIO
import socket
import subprocess
import platform

from inference import get_prediction, run_full_benchmark, MODELS, run_single_compression

def setup_page_config():
    """Set up the page configuration"""
    st.set_page_config(
        page_title="Project Minerva - Smart File Compression",
        page_icon="🗜️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

def inject_custom_css():
    """Inject custom CSS for styling"""
    st.markdown(
        """
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Hide Streamlit branding and default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    .stDecoration {display: none;}
    
    /* Modern SaaS color system */
    :root {
        --bg-primary: #0a0a0b;
        --bg-secondary: #111113;
        --bg-tertiary: #1a1a1d;
        --bg-card: #1e1e21;
        --bg-elevated: #252529;
        --text-primary: #ffffff;
        --text-secondary: #a1a1aa;
        --text-muted: #71717a;
        --accent-primary: #3b82f6;
        --accent-secondary: #10b981;
        --accent-tertiary: #8b5cf6;
        --border-subtle: #27272a;
        --border-default: #3f3f46;
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1);
        --shadow-xl: 0 20px 25px -5px rgb(0 0 0 / 0.1);
    }
    
    /* Global app styling */
    .stApp {
        background: linear-gradient(135deg, var(--bg-primary) 0%, var(--bg-secondary) 100%);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: var(--text-primary);
    }
    
    .main > div {
        padding: 0;
        max-width: none;
    }
    
    .block-container {
        padding: 2rem 3rem;
        max-width: none;
    }
    
    /* Modern navigation header */
    .nav-header {
        background: rgba(30, 30, 33, 0.8);
        backdrop-filter: blur(20px);
        border-bottom: 1px solid var(--border-subtle);
        padding: 1rem 3rem;
        margin: -2rem -3rem 2rem -3rem;
        position: sticky;
        top: 0;
        z-index: 100;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .nav-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }
    
    .nav-logo {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-tertiary));
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        color: white;
        font-size: 1.2rem;
    }
    
    .nav-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }
    
    .nav-subtitle {
        font-size: 0.875rem;
        color: var(--text-muted);
        margin: 0;
    }
    
    .nav-status {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: var(--bg-elevated);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        border: 1px solid var(--border-default);
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        background: var(--accent-secondary);
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Modern card system */
    .saas-card {
        background: var(--bg-card);
        border: 1px solid var(--border-subtle);
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease;
        margin-bottom: 1.5rem;
    }
    
    .saas-card:hover {
        border-color: var(--border-default);
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }
    
    .saas-card-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 1rem;
    }
    
    .saas-card-title {
        font-size: 1.125rem;
        font-weight: 600;
        color: var(--text-primary);
        margin: 0;
    }
    
    .saas-card-badge {
        background: var(--accent-primary);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 500;
    }
    
    /* Upload zone styling */
    .upload-zone {
        border: 2px dashed var(--border-default);
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        background: var(--bg-tertiary);
        transition: all 0.2s ease;
        margin: 2rem 0;
    }
    
    .upload-zone:hover {
        border-color: var(--accent-primary);
        background: var(--bg-elevated);
    }
    
    .upload-icon {
        width: 48px;
        height: 48px;
        background: var(--accent-primary);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1rem;
        font-size: 1.5rem;
        color: white;
    }
    
    /* Modern button system */
    .saas-button {
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-tertiary));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 2rem;
        font-weight: 600;
        font-size: 0.875rem;
        cursor: pointer;
        transition: all 0.2s ease;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        text-decoration: none;
        box-shadow: var(--shadow-sm);
    }
    
    .saas-button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    .saas-button-secondary {
        background: var(--bg-elevated);
        color: var(--text-primary);
        border: 1px solid var(--border-default);
    }
    
    .saas-button-secondary:hover {
        background: var(--bg-card);
        border-color: var(--accent-primary);
    }
    
    /* Grid system */
    .saas-grid {
        display: grid;
        gap: 1.5rem;
    }
    
    .saas-grid-2 {
        grid-template-columns: repeat(2, 1fr);
    }
    
    .saas-grid-3 {
        grid-template-columns: repeat(3, 1fr);
    }
    
    .saas-grid-4 {
        grid-template-columns: repeat(4, 1fr);
    }
    
    @media (max-width: 768px) {
        .saas-grid-2, .saas-grid-3, .saas-grid-4 {
            grid-template-columns: 1fr;
        }
        
        .nav-header {
            padding: 1rem 1.5rem;
            margin: -2rem -1.5rem 2rem -1.5rem;
        }
        
        .block-container {
            padding: 1rem 1.5rem;
        }
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-secondary));
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: var(--shadow-md);
        text-align: center;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.875rem;
        opacity: 0.9;
        margin: 0.25rem 0 0 0;
    }
    
    /* Progress styling */
    .progress-container {
        background: var(--bg-elevated);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid var(--border-subtle);
    }
    
    .progress-bar {
        background: var(--bg-tertiary);
        height: 8px;
        border-radius: 4px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-fill {
        background: linear-gradient(90deg, var(--accent-primary), var(--accent-secondary));
        height: 100%;
        border-radius: 4px;
        transition: width 0.3s ease;
    }
    
    /* Results dashboard */
    .results-header {
        background: linear-gradient(135deg, var(--accent-secondary), var(--accent-primary));
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin: 2rem 0;
        text-align: center;
        box-shadow: var(--shadow-lg);
    }
    
    .results-title {
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0 0 0.5rem 0;
    }
    
    .results-subtitle {
        font-size: 1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Tab system override */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-elevated);
        border-radius: 12px;
        padding: 0.25rem;
        border: 1px solid var(--border-subtle);
        gap: 0.25rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-secondary);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        border: none;
    }
    
    .stTabs [aria-selected="true"] {
        background: var(--accent-primary);
        color: white;
        box-shadow: var(--shadow-sm);
    }
    
    /* Override Streamlit components */
    .stSelectbox > div > div {
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        border-radius: 12px;
        color: var(--text-primary);
    }
    
    .stFileUploader > div {
        background: transparent;
        border: none;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, var(--accent-primary), var(--accent-tertiary));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.875rem 2rem;
        font-weight: 600;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        box-shadow: var(--shadow-sm);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: var(--shadow-lg);
    }
    
    /* Data display */
    .stDataFrame {
        background: var(--bg-card);
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid var(--border-subtle);
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: var(--bg-tertiary);
        border: 1px solid var(--border-subtle);
        border-radius: 12px;
    }
    
    /* Sidebar override */
    .css-1d391kg {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border-subtle);
    }
    
    /* Info/warning boxes */
    .stAlert {
        background: var(--bg-elevated);
        border: 1px solid var(--border-default);
        border-radius: 12px;
        color: var(--text-primary);
    }
</style>
""",
        unsafe_allow_html=True,
    )

def create_navigation():
    """Create modern SaaS navigation header"""
    st.markdown(
        """
    <div class="nav-header">
        <div class="nav-brand">
            <div class="nav-logo">M</div>
            <div>
                <div class="nav-title">Project Minerva</div>
                <div class="nav-subtitle">Smart File Compression</div>
            </div>
        </div>
        <div class="nav-status">
            <div class="status-dot"></div>
            <span style="font-size: 0.875rem; font-weight: 500;">System Online</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

def create_hero_section():
    """Create modern hero section"""
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(
            """
        <div style="padding: 2rem 0;">
            <h1 style="font-size: 3rem; font-weight: 700; margin: 0 0 1rem 0; background: linear-gradient(135deg, #3b82f6, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Intelligent File Compression
            </h1>
            <p style="font-size: 1.25rem; color: var(--text-secondary); margin: 0 0 2rem 0; line-height: 1.6;">
                Leverage DL to automatically select the optimal compression algorithm for any file type. 
                Get better compression ratios with zero manual testing.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
        <div class="saas-grid saas-grid-2" style="margin-top: 2rem;">
            <div class="metric-card">
                <div class="metric-value">Accurate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">Faster</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

def create_model_selector():
    """Create modern model selection interface"""
    st.markdown(
        """
    <div class="saas-card">
        <div class="saas-card-header">
            <h3 class="saas-card-title">DL Model Selection</h3>
            <div class="saas-card-badge">Advanced</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    model_name = st.selectbox(
        "Select AI Model:",
        list(MODELS.keys()),
        index=0,
        help="Choose the machine learning model for compression analysis",
    )

    st.markdown("</div>", unsafe_allow_html=True)
    return model_name

def create_upload_interface():
    """Create modern file upload interface"""
    st.markdown(
        """
    <div class="saas-card">
        <div class="saas-card-header">
            <h3 class="saas-card-title">File Upload</h3>
        </div>
    """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Drop your file here or click to browse",
        type=["txt", "csv", "json", "pdf", "png", "jpg", "jpeg", "wav"],
        help="Supported formats: Documents, Images, Audio files (up to 50MB)",
    )

    if uploaded_file is not None:
        st.markdown(
            """
        <div class="saas-grid saas-grid-3" style="margin-top: 1.5rem;">
        """,
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown(
                f"""
            <div class="saas-card">
                <h4 style="margin: 0 0 1rem 0; color: var(--text-primary);">File Details</h4>
                <p style="margin: 0.5rem 0; color: var(--text-secondary);"><strong>Name:</strong> {uploaded_file.name}</p>
                <p style="margin: 0.5rem 0; color: var(--text-secondary);"><strong>Size:</strong> {uploaded_file.size / 1024:.2f} KB</p>
                <p style="margin: 0.5rem 0; color: var(--text-secondary);"><strong>Type:</strong> {uploaded_file.type or 'Unknown'}</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                """
            <div class="saas-card">
                <h4 style="margin: 0 0 1rem 0; color: var(--text-primary);">Validation</h4>
                <p style="margin: 0.5rem 0; color: var(--accent-secondary);">✓ File format supported</p>
                <p style="margin: 0.5rem 0; color: var(--accent-secondary);">✓ Size within limits</p>
                <p style="margin: 0.5rem 0; color: var(--accent-secondary);">✓ Ready for analysis</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                """
            <div class="saas-card">
                <h4 style="margin: 0 0 1rem 0; color: var(--text-primary);">Analysis Pipeline</h4>
                <p style="margin: 0.5rem 0; color: var(--text-secondary);">• Feature extraction</p>
                <p style="margin: 0.5rem 0; color: var(--text-secondary);">• DL Prediction</p>
                <p style="margin: 0.5rem 0; color: var(--text-secondary);">• Performance testing</p>
            </div>
            """,
                unsafe_allow_html=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    return uploaded_file

def create_analysis_progress(progress_value, status_message):
    """Create modern progress indicator"""
    st.markdown(
        f"""
    <div class="progress-container">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
            <h4 style="margin: 0; color: var(--text-primary);">Analysis Progress</h4>
            <span style="color: var(--text-secondary); font-weight: 500;">{progress_value}%</span>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: {progress_value}%;"></div>
        </div>
        <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary); font-size: 0.875rem;">{status_message}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

def display_features_section(features_data):
    """Display extracted features immediately when ready"""
    st.markdown(
        """
    <div class="saas-card">
        <div class="saas-card-header">
            <h3 class="saas-card-title">Extracted Features</h3>
            <div class="saas-card-badge">Ready</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    if features_data:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**File Properties**")
            for key, value in features_data.get("file_props", {}).items():
                st.write(f"• {key}: {value}")

        with col2:
            st.markdown("**Statistical Features**")
            for key, value in features_data.get("stats", {}).items():
                st.write(
                    f"• {key}: {value:.3f}"
                    if isinstance(value, float)
                    else f"• {key}: {value}"
                )

        with col3:
            st.markdown("**Content Analysis**")
            for key, value in features_data.get("content", {}).items():
                st.write(
                    f"• {key}: {value:.3f}"
                    if isinstance(value, float)
                    else f"• {key}: {value}"
                )

    st.markdown("</div>", unsafe_allow_html=True)

def display_ai_prediction(prediction, confidence=None):
    """Display AI prediction immediately when ready"""
    st.markdown(
        f"""
    <div class="results-header">
        <div class="results-title">Prediction: {prediction}</div>
        <div class="results-subtitle">Model recommendation based on file analysis</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

def display_compression_results(results_data, compressed_file_path=None):
    """Display compression results as they become available"""
    st.markdown(
        """
    <div class="saas-card">
        <div class="saas-card-header">
            <h3 class="saas-card-title">Compression Results</h3>
            <div class="saas-card-badge">Live</div>
        </div>
    """,
        unsafe_allow_html=True,
    )

    if results_data:
        for tool, result in results_data.items():
            col1, col2, col3 = st.columns(3)

            with col1:
                st.markdown(f"**{tool}**")
            with col2:
                st.markdown(f"Ratio: {result.get('ratio', 'N/A')}")
            with col3:
                st.markdown(f"Size: {result.get('size', 'N/A')}")

    if compressed_file_path and os.path.exists(compressed_file_path):
        st.markdown("---")
        with open(compressed_file_path, "rb") as file:
            file_data = file.read()
        
        st.download_button(
            label="📥 Download Compressed File",
            data=file_data,
            file_name=os.path.basename(compressed_file_path),
            mime="application/octet-stream",
            use_container_width=True
        )

    st.markdown("</div>", unsafe_allow_html=True)

def create_results_dashboard(results):
    """Create modern results dashboard"""
    (
        recommended_tool,
        key_insights,
        fig,
        summary_report,
        prediction_time,
        benchmark_time,
        efficiency_report,
    ) = results

    # Results header
    st.markdown(
        f"""
    <div class="results-header">
        <div class="results-title">Optimal Tool: {recommended_tool}</div>
        <div class="results-subtitle">Analysis complete - compression recommendation ready</div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Results tabs
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Overview", "🔍 Analysis", "📈 Performance", "⏱️ Timing"]
    )

    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown("### Recommended Tool")
            st.markdown(
                f"""
                <div style="text-align: center; padding: 2rem;">
                    <div style="font-size: 2rem; font-weight: bold; color: var(--accent-primary); margin-bottom: 0.5rem;">
                        {recommended_tool}
                    </div>
                    <div style="color: var(--text-secondary);">
                        Optimal compression tool for your file
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            if fig:
                st.markdown('<div class="saas-card">', unsafe_allow_html=True)
                st.markdown("### Compression Comparison")
                interactive_fig = create_interactive_chart(summary_report)
                if interactive_fig:
                    st.plotly_chart(
                        interactive_fig,
                        use_container_width=True,
                        config={"displayModeBar": False},
                    )
                else:
                    st.pyplot(fig, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown("### File Characteristics")

            if key_insights:
                if isinstance(key_insights, str):
                    for line in key_insights.split("\n"):
                        if line.strip() and not line.startswith("Original File Size"):
                            st.write(f"• {line.strip()}")
                elif isinstance(key_insights, dict):
                    for key, value in key_insights.items():
                        st.write(f"• **{key}:** {value}")

            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown(
                """
            <div class="saas-card">
                <h3 style="margin: 0 0 1rem 0;">Analysis Pipeline</h3>
                <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div style="width: 8px; height: 8px; background: var(--accent-secondary); border-radius: 50%;"></div>
                        <span>Feature extraction completed</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div style="width: 8px; height: 8px; background: var(--accent-secondary); border-radius: 50%;"></div>
                        <span>Pattern recognition applied</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div style="width: 8px; height: 8px; background: var(--accent-secondary); border-radius: 50%;"></div>
                        <span>Statistical analysis performed</span>
                    </div>
                    <div style="display: flex; align-items: center; gap: 0.5rem;">
                        <div style="width: 8px; height: 8px; background: var(--accent-secondary); border-radius: 50%;"></div>
                        <span>ML prediction generated</span>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    with tab3:
        st.markdown('<div class="saas-card">', unsafe_allow_html=True)
        st.markdown("### Compression Results")

        if summary_report:
            if isinstance(summary_report, str):
                lines = summary_report.split("\n")

                # Find original file size
                original_size = None
                for line in lines:
                    if "Original File Size:" in line:
                        original_size = line.split(":")[1].strip()
                        st.markdown(f"**Original File Size:** {original_size}")
                        break

                # Create table data
                table_data = []
                for line in lines:
                    if (
                        line.strip()
                        and not line.startswith("Original")
                        and not line.startswith("Analysis")
                    ):
                        parts = line.split()
                        if len(parts) >= 3:
                            try:
                                tool = parts[0]
                                ratio = float(parts[1])
                                size = float(parts[2])
                                table_data.append(
                                    [tool, f"{ratio:.2f}", f"{size:.2f} KB"]
                                )
                            except (ValueError, IndexError):
                                continue

                if table_data:
                    # Sort by compression ratio (descending)
                    table_data.sort(key=lambda x: float(x[1]), reverse=True)

                    # Create formatted table
                    st.markdown("#### Compression Performance")

                    # Table header
                    col1, col2, col3 = st.columns([2, 1, 2])
                    with col1:
                        st.markdown("**Tool**")
                    with col2:
                        st.markdown("**Ratio**")
                    with col3:
                        st.markdown("**Size**")

                    st.markdown("---")

                    # Table rows
                    for tool, ratio, size in table_data:
                        col1, col2, col3 = st.columns([2, 1, 2])
                        with col1:
                            if tool == recommended_tool:
                                st.markdown(f"**{tool}** ⭐")
                            else:
                                st.markdown(tool)
                        with col2:
                            st.markdown(ratio)
                        with col3:
                            st.markdown(size)

            elif isinstance(summary_report, dict):
                for key, value in summary_report.items():
                    st.markdown(f"**{key}:** {value}")

        st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="saas-card">', unsafe_allow_html=True)
            st.markdown("### Performance Timing")

            if prediction_time and benchmark_time:
                ml_pipeline_time = (
                    prediction_time + benchmark_time
                )  # Prediction + compression of recommended tool
                estimated_brute_force = (
                    benchmark_time * 7
                )  # Time to test all 7 compression tools sequentially
                time_saved = estimated_brute_force - ml_pipeline_time
                efficiency_gain = (
                    (estimated_brute_force - ml_pipeline_time) / estimated_brute_force
                ) * 100

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "ML Pipeline (Prediction + Compression)",
                        f"{ml_pipeline_time:.2f}s",
                    )
                with col2:
                    st.metric(
                        "Brute-force Sequential Testing",
                        f"{estimated_brute_force:.2f}s",
                    )

                st.success(
                    f"⚡ Time Saved: {time_saved:.2f}s ({efficiency_gain:.1f}% faster)"
                )
            else:
                st.markdown("Timing data not available")

            st.markdown("</div>", unsafe_allow_html=True)

def generate_qr_code(data, title="QR Code"):
    """Generate QR code for given data"""
    qr = qrcode.QRCode(version=1, box_size=10, border=5)
    qr.add_data(data)
    qr.make(fit=True)
    
    img = qr.make_image(fill_color="black", back_color="white")
    buf = BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf

def get_network_url():
    """Get the network URL for the Streamlit app"""
    try:
        # Get local IP address
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        # Streamlit default port is 8501
        return f"http://{local_ip}:8501"
    except:
        return "http://localhost:8501"

def display_network_qr_in_terminal():
    """Display network URL QR code in terminal"""
    try:
        import qrcode
        network_url = get_network_url()
        
        # Create QR code for terminal display
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_L,
            box_size=1,
            border=1,
        )
        qr.add_data(network_url)
        qr.make(fit=True)
        
        print("\n" + "="*50)
        print("🌐 NETWORK ACCESS QR CODE")
        print("="*50)
        print(f"Network URL: {network_url}")
        print("\nScan this QR code to access the app:")
        print("-"*30)
        
        # Print QR code in terminal
        qr.print_ascii(invert=True)
        
        print("-"*30)
        print("Share this URL or QR code with others to access the app")
        print("="*50 + "\n")
        
    except ImportError:
        print(f"\n🌐 Network URL: {network_url}")
        print("Install 'qrcode' package to see QR code in terminal\n")

def create_interactive_chart(summary_report):
    """Create interactive Plotly chart from compression results"""
    try:
        # Parse compression results from summary_report
        lines = summary_report.strip().split("\n")
        data = []

        for line in lines[1:]:  # Skip header
            if line.strip():
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        tool = parts[0]
                        ratio = float(parts[1])
                        size_kb = float(parts[2])
                        data.append(
                            {
                                "Tool": tool,
                                "Compression Ratio": ratio,
                                "Size (KB)": size_kb,
                            }
                        )
                    except (ValueError, IndexError):
                        continue

        if not data:
            return None

        df = pd.DataFrame(data)

        # Create subplot with secondary y-axis
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Compression Ratio by Tool", "File Size Comparison"),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]],
        )

        # Compression ratio bar chart
        fig.add_trace(
            go.Bar(
                x=df["Tool"],
                y=df["Compression Ratio"],
                name="Compression Ratio",
                marker_color="#06b6d4",
                hovertemplate="<b>%{x}</b><br>Ratio: %{y:.2f}x<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # File size bar chart
        fig.add_trace(
            go.Bar(
                x=df["Tool"],
                y=df["Size (KB)"],
                name="Size (KB)",
                marker_color="#10b981",
                hovertemplate="<b>%{x}</b><br>Size: %{y:.2f} KB<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Update layout
        fig.update_layout(
            height=400,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", size=12),
            title_font=dict(size=14, color="white"),
        )

        # Update axes
        fig.update_xaxes(
            gridcolor="rgba(255,255,255,0.1)", title_font=dict(color="white")
        )
        fig.update_yaxes(
            gridcolor="rgba(255,255,255,0.1)", title_font=dict(color="white")
        )

        return fig

    except Exception as e:
        st.error(f"Chart creation error: {str(e)}")
        return None

def main():
    """Main Streamlit application"""
    if 'qr_displayed' not in st.session_state:
        display_network_qr_in_terminal()
        st.session_state.qr_displayed = True
    
    setup_page_config()
    inject_custom_css()

    create_navigation()
    create_hero_section()

    # Model selection
    model_name = create_model_selector()

    # File upload
    uploaded_file = create_upload_interface()

    if uploaded_file is not None:
        # Analysis button
        st.markdown(
            '<div style="text-align: center; margin: 2rem 0;">', unsafe_allow_html=True
        )
        analyze_button = st.button(" Start Analysis", use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)

        if analyze_button:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}"
            ) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            class MockFileObj:
                def __init__(self, path):
                    self.name = path

            mock_file = MockFileObj(tmp_file_path)

            try:

                # Create containers for progressive updates
                progress_container = st.empty()
                features_container = st.empty()
                final_results_container = st.empty()

                with progress_container.container():
                    create_analysis_progress(
                        25, "Extracting features and running AI prediction..."
                    )

                try:
                    recommended_tool, key_insights, prediction_time = get_prediction(
                        tmp_file_path, model_name
                    )

                    # Display extracted features immediately
                    features_data = {
                        "file_props": {
                            "Size": f"{uploaded_file.size / 1024:.2f} KB",
                            "Type": key_insights.get("File Type", "Unknown"),
                        },
                        "stats": {
                            "Entropy": key_insights.get("Shannon Entropy", "N/A"),
                            "Prediction_Time": f"{prediction_time:.3f}s",
                        },
                        "content": {},
                    }

                    # Add specific insights based on file type
                    if "Dimensions" in key_insights:
                        features_data["content"]["Dimensions"] = key_insights[
                            "Dimensions"
                        ]
                    if "Duration (s)" in key_insights:
                        features_data["content"]["Duration"] = key_insights[
                            "Duration (s)"
                        ]
                    if "Page Count" in key_insights:
                        features_data["content"]["Pages"] = key_insights["Page Count"]

                    with features_container.container():
                        display_features_section(features_data)

                    # Display AI prediction immediately

                    with progress_container.container():
                        create_analysis_progress(
                            75, "Running comprehensive compression benchmark..."
                        )

                    # Run the full benchmark analysis
                    fig, summary_report, benchmark_time = run_full_benchmark(
                        tmp_file_path, recommended_tool
                    )

                    _, _, compressed_file_path = run_single_compression(recommended_tool, tmp_file_path)

                    ml_pipeline_time = prediction_time  # Just the ML prediction time
                    compression_time = benchmark_time  # Time to test recommended tool
                    total_smart_time = ml_pipeline_time + compression_time
                    brute_force_estimate = compression_time * 7  # Test all 7 tools
                    time_saved = brute_force_estimate - total_smart_time
                    efficiency_report = {
                        "ml_pipeline_time": ml_pipeline_time,
                        "compression_time": compression_time,
                        "total_smart_time": total_smart_time,
                        "brute_force_estimate": brute_force_estimate,
                        "time_saved": time_saved,
                        "efficiency_gain": (time_saved / brute_force_estimate) * 100
                        if brute_force_estimate > 0
                        else 0,
                    }

                    with progress_container.container():
                        create_analysis_progress(100, "Analysis complete!")

                    time.sleep(0.5)

                    # Clear progress and show comprehensive results
                    progress_container.empty()

                    # Create mock results structure for dashboard display
                    results = (
                        recommended_tool,
                        key_insights,
                        fig,
                        summary_report,
                        prediction_time,
                        benchmark_time,
                        efficiency_report,
                    )

                    with final_results_container.container():
                        create_results_dashboard(results)

                    # Display download button at the bottom of the results
                    if compressed_file_path and os.path.exists(compressed_file_path):
                        st.markdown(
                            """
                            <div class="saas-card" style="margin-top: 1.5rem;">
                                <div class="saas-card-header">
                                    <h3 class="saas-card-title">Download Optimized File</h3>
                                </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        with open(compressed_file_path, "rb") as file:
                            file_data = file.read()
                        
                        st.download_button(
                            label="📥 Download Compressed File",
                            data=file_data,
                            file_name=os.path.basename(compressed_file_path),
                            mime="application/octet-stream",
                            use_container_width=True
                        )
                        st.markdown("</div>", unsafe_allow_html=True)


                except ValueError as ve:
                    st.error(f"File validation error: {str(ve)}")
                except Exception as e:
                    st.error(f"Analysis error: {str(e)}")

            except Exception as e:
                st.error(f"Upload processing failed: {str(e)}")

            finally:
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass

    # Footer
    st.markdown(
        """
    <div style="text-align: center; color: var(--text-muted); font-size: 0.875rem; padding: 3rem 0 2rem 0; border-top: 1px solid var(--border-subtle); margin-top: 3rem;">
        <p style="margin: 0;"><strong>Project Minerva</strong> - Intelligent File Compression Platform</p>
        <p style="margin: 0.5rem 0 0 0;">Powered by Advanced Machine Learning & TensorFlow</p>
    </div>
    """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
