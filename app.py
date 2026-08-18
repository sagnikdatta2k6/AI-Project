import streamlit as st
import tempfile
import os
import sys

# Ensure src modules can be imported
sys.path.append(os.path.abspath("src"))
from src.orchestrator import MultiAgentFusionOrchestrator

st.set_page_config(
    page_title="ReadOrSpeak | AI-Assisted Interview Detection",
    layout="wide",
    page_icon="🎙️"
)

st.title("🎙️ ReadOrSpeak: Multimodal AI Interview Analysis")
st.markdown(
    "Detect whether a candidate's video response is **spontaneous** or **read from an AI-generated script** "
    "using cooperative perceptual AI agents."
)

st.divider()

# Sidebar: File Uploader & Controls
with st.sidebar:
    st.header("Upload Response")
    uploaded_file = st.file_uploader("Choose a video file (.mp4, .mov)", type=["mp4", "mov", "avi"])
    analyze_btn = st.button("Run Multi-Agent Analysis", type="primary", use_container_width=True)

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📹 Video Playback")
        st.video(temp_video_path)

    with col_right:
        st.subheader("🧠 Multi-Agent Diagnostics")

        if analyze_btn:
            with st.spinner("Executing Vision, Audio, and Linguistic Agents..."):
                orchestrator = MultiAgentFusionOrchestrator()
                report = orchestrator.process_video_interview(temp_video_path)

            # --- Verdict Banner ---
            is_scripted = "Scripted" in report["final_verdict"]
            if is_scripted:
                st.error(f"### Verdict: {report['final_verdict']} (Risk Score: {report['confidence_score']})")
            else:
                st.success(f"### Verdict: {report['final_verdict']} (Risk Score: {report['confidence_score']})")

            # --- Individual Agent Cards ---
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Vision Agent", report["agent_findings"]["vision"]["visual_verdict"])
                st.caption(f"Gaze Var: `{report['agent_findings']['vision']['gaze_horizontal_variance']}`")
            with c2:
                st.metric("Audio Agent", report["agent_findings"]["audio"]["audio_verdict"])
                st.caption(f"Pause Ratio: `{report['agent_findings']['audio']['pause_ratio']}`")
            with c3:
                st.metric("Linguistic Agent", report["agent_findings"]["linguistics"]["text_verdict"])
                st.caption(f"Filler Ratio: `{report['agent_findings']['linguistics']['filler_ratio']}`")

            # --- Orchestrator Explainability Synthesis ---
            st.markdown("#### 📝 Orchestrator Findings Breakdown")
            for item in report["orchestrator_synthesis"]:
                st.write(f"- {item}")

            # --- Transcript Section ---
            with st.expander("📄 View Whisper Audio Transcript"):
                st.write(report["agent_findings"]["linguistics"]["transcript"])

            # Clean up temp file
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
        else:
            st.info("Click **'Run Multi-Agent Analysis'** in the sidebar to start evaluation.")
else:
    st.info("Upload a video response file from the sidebar to begin.")