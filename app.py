import streamlit as st
import tempfile
import os
import sys

# Ensure src modules can be imported
sys.path.append(os.path.abspath("src"))
from src.orchestrator import MultiAgentFusionOrchestrator
from src import ui_theme

st.set_page_config(
    page_title="ReadOrSpeak | AI-Assisted Interview Detection",
    layout="wide",
    page_icon=":material/mic:",
)

ui_theme.inject_theme()

if "view" not in st.session_state:
    st.session_state.view = "empty"
if "report" not in st.session_state:
    st.session_state.report = None
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "video_name" not in st.session_state:
    st.session_state.video_name = None
if "video_size_mb" not in st.session_state:
    st.session_state.video_size_mb = 0.0

# ---- Header row: brand (left) + Analysis/Upload tabs (right), one row ----
header_col, tabs_col = st.columns([3, 2])
with header_col:
    ui_theme.render_brand()
with tabs_col:
    t1, t2 = st.columns(2)
    with t1:
        if st.button(
            "Analysis",
            icon=":material/analytics:",
            type="primary" if st.session_state.view == "results" else "secondary",
            use_container_width=True,
            disabled=st.session_state.report is None,
            help=None if st.session_state.report is not None else "Run an analysis from the Upload tab first",
        ):
            st.session_state.view = "results"
            st.rerun()
    with t2:
        if st.button(
            "Upload",
            icon=":material/upload:",
            type="primary" if st.session_state.view == "empty" else "secondary",
            use_container_width=True,
        ):
            st.session_state.view = "empty"
            st.rerun()

ui_theme.render_divider()

# CSS-only visibility toggle: both view containers below are ALWAYS
# constructed by Python on every run (so their widgets, e.g. the file
# uploader, keep their identity/state across tab switches) — only which
# one is visible changes, driven by session_state.view.
view = st.session_state.view
st.html(
    f"""
    <style>
    .st-key-ros_view_upload {{ display: {"block" if view == "empty" else "none"}; }}
    .st-key-ros_view_results {{ display: {"block" if view == "results" else "none"}; }}
    </style>
    """
)

# ============ UPLOAD VIEW (always mounted; CSS-hidden when not active) ============
upload_box = st.container(key="ros_view_upload")
with upload_box:
    st.markdown(
        """
        <div style="max-width:900px;margin:0 auto;padding-top:12px;text-align:center;">
          <div class="ros-kicker">New Evaluation</div>
          <h1 style="font-size:32px;letter-spacing:-0.02em;margin:0 0 10px;">Upload a candidate response</h1>
          <p style="font-size:14px;color:var(--color-neutral-400);max-width:560px;margin:0 auto;line-height:1.5;">Drop an interview clip and the vision, audio and linguistic agents will each score it before the orchestrator fuses a final verdict.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_l, col_mid, col_r = st.columns([1, 2, 1])
    with col_mid:
        uploaded_file = st.file_uploader(
            "Choose a video file", type=["mp4", "mov", "avi"], label_visibility="collapsed",
            key="ros_file_uploader",
        )
        analyze_btn = st.button(
            "Run Multi-Agent Analysis", icon=":material/play_arrow:",
            type="primary", use_container_width=True,
            disabled=uploaded_file is None, key="ros_run_btn",
        )

    if uploaded_file is not None and analyze_btn:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            file_bytes = uploaded_file.read()
            tfile.write(file_bytes)
            temp_video_path = tfile.name

        with st.spinner("Executing Vision, Audio, and Linguistic Agents..."):
            orchestrator = MultiAgentFusionOrchestrator()
            report = orchestrator.process_video_interview(temp_video_path)

        st.session_state.report = report
        st.session_state.video_path = temp_video_path
        st.session_state.video_name = uploaded_file.name
        st.session_state.video_size_mb = len(file_bytes) / (1024 * 1024)
        st.session_state.view = "results"
        st.rerun()

    ui_theme.render_idle_agent_cards()

# ============ RESULTS VIEW (always mounted; CSS-hidden when not active) ============
results_box = st.container(key="ros_view_results")
with results_box:
    report = st.session_state.report

    if report is None:
        st.markdown(
            f"""
            <div class="ros-card" style="text-align:center;padding:48px 24px;">
              {ui_theme.icon('donut_large', 40, 'var(--color-neutral-600)')}
              <div style="font-size:16px;font-weight:600;margin-top:14px;">No analysis yet</div>
              <div style="font-size:13px;color:var(--color-neutral-500);margin-top:6px;">Switch to the Upload tab, choose a video, and run the multi-agent analysis.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="ros-kicker">Candidate Report</div>
            <h1 style="font-size:34px;letter-spacing:-0.02em;margin:0 0 8px;">Multi-Agent Interview Analysis</h1>
            <p style="font-size:14px;color:var(--color-neutral-400);margin:0 0 22px;line-height:1.5;">Detect whether a candidate's video response is <strong style="color:var(--color-text);">spontaneous</strong> or <strong style="color:var(--color-text);">read from an AI-generated script</strong> using three cooperative perceptual agents fused by a weighted orchestrator.</p>
            """,
            unsafe_allow_html=True,
        )

        left, right = st.columns([1, 1.55], gap="medium")

        with left:
            ui_theme.render_source_card_open(
                st.session_state.video_name, st.session_state.video_size_mb
            )
            if st.session_state.video_path and os.path.exists(st.session_state.video_path):
                st.video(st.session_state.video_path)
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            ui_theme.render_risk_meter(int(report["confidence_score"].rstrip("%")))

        with right:
            ui_theme.render_verdict_banner(
                report["final_verdict"], int(report["confidence_score"].rstrip("%"))
            )
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            ui_theme.render_agent_gauges(
                report["agent_findings"]["vision"],
                report["agent_findings"]["audio"],
                report["agent_findings"]["linguistics"],
            )
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            tcol, ccol = st.columns(2, gap="medium")
            with tcol:
                ui_theme.render_timeline(report)
            with ccol:
                ui_theme.render_transcript(report["agent_findings"]["linguistics"])
