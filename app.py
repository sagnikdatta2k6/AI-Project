import streamlit as st
import tempfile
import os
import sys

# Ensure src modules can be imported
sys.path.append(os.path.abspath("src"))
from src import ui_theme, analysis_worker

st.set_page_config(
    page_title="ReadOrSpeak | AI-Assisted Interview Detection",
    layout="wide",
    page_icon=":material/mic:",
)

ui_theme.inject_theme()


@st.cache_resource
def get_analysis_worker() -> analysis_worker.AnalysisWorker:
    """One AnalysisWorker (and its one background thread) for the whole
    app's lifetime, shared across every session — see src/analysis_worker.py
    for why this has to be a single persistent thread rather than one
    spawned per request."""
    return analysis_worker.AnalysisWorker()


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
header_col, tabs_col = st.columns([3, 2], gap="large")
with header_col:
    ui_theme.render_brand()
with tabs_col:
    t1, t2 = st.columns(2, gap="medium")
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

        flow_placeholder = st.empty()
        pipeline_state = {"completed": set(), "current": None}

        def _render_flow():
            flow_placeholder.html(
                ui_theme.render_pipeline_html(pipeline_state["completed"], pipeline_state["current"])
            )

        def on_progress(message):
            pipeline_state["completed"], pipeline_state["current"] = ui_theme.pipeline_stage_update(
                message, pipeline_state["completed"], pipeline_state["current"]
            )
            _render_flow()
            status.write(message)

        _render_flow()
        with st.status("Running Multi-Agent Analysis...", expanded=True) as status:
            # Analysis always runs on the single shared background worker
            # thread (see src/analysis_worker.py) — never directly on this
            # Streamlit session's own thread. All the st.* calls above and
            # below still happen here, on the session thread, which is
            # required for them to reach the right browser tab; only the
            # actual ML work is handed off.
            job = get_analysis_worker().submit(temp_video_path)
            analysis_worker.drain_progress(job, on_progress)
            if job.error:
                raise job.error
            report = job.result
            status.update(
                label=f"Analysis complete — {report['final_verdict']}",
                state="complete",
                expanded=False,
            )

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
            <div class="ros-card" style="text-align:center;padding:var(--ros-space-4) var(--ros-space-3);">
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

        left, right = st.columns([1, 1.55], gap="large")

        with left:
            ui_theme.render_source_card_open(
                st.session_state.video_name, st.session_state.video_size_mb
            )
            if st.session_state.video_path and os.path.exists(st.session_state.video_path):
                st.video(st.session_state.video_path)
            st.markdown("<div style='height:var(--ros-space-4)'></div>", unsafe_allow_html=True)
            ui_theme.render_risk_meter(int(report["confidence_score"].rstrip("%")))

        with right:
            ui_theme.render_verdict_banner(
                report["final_verdict"], int(report["confidence_score"].rstrip("%"))
            )
            st.markdown("<div style='height:var(--ros-space-4)'></div>", unsafe_allow_html=True)
            ui_theme.render_agent_gauges(
                report["agent_findings"]["vision"],
                report["agent_findings"]["audio"],
                report["agent_findings"]["linguistics"],
            )
            st.markdown("<div style='height:var(--ros-space-4)'></div>", unsafe_allow_html=True)
            tcol, ccol = st.columns(2, gap="large")
            with tcol:
                ui_theme.render_timeline(report)
            with ccol:
                ui_theme.render_transcript(report["agent_findings"]["linguistics"])
