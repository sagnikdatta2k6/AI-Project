import os
import json
import tempfile
import threading
from moviepy.editor import VideoFileClip
from agents import VisionAgent, AudioAgent, LinguisticAgent

class MultiAgentFusionOrchestrator:
    # Class-level (shared across every instance/thread) lock that serializes
    # the whole vision+audio+linguistic pipeline. Streamlit runs each
    # browser session's script in its own thread within one shared process,
    # and the underlying native ML libraries here (mediapipe's C++ backend,
    # PyTorch/whisper, ffmpeg via moviepy) aren't proven safe for concurrent
    # use — observed in testing as anything from wrong-user results to a
    # hard segfault. Rather than chase individual narrow locks around each
    # library, this makes the whole analysis a queue: only one video is
    # ever being processed at a time, and everyone else's request simply
    # waits its turn instead of racing.
    _process_lock = threading.Lock()

    def process_video_interview(self, video_path, on_progress=None):
        """on_progress: optional callable(str) invoked with a human-readable
        status message at each stage — e.g. wire it to a Streamlit
        st.status() so the UI shows the same progress the terminal does,
        instead of the caller only finding out once everything is done."""

        def emit(message: str):
            print(message)
            if on_progress:
                on_progress(message)

        if MultiAgentFusionOrchestrator._process_lock.locked():
            emit("Another analysis is currently running — waiting for your turn...")

        with MultiAgentFusionOrchestrator._process_lock:
            # Built fresh here, inside the lock, in the calling thread —
            # not shared/cached across sessions. VisionAgent's MediaPipe
            # FaceMesh (via a TFLite/XNNPACK delegate) is documented as
            # bound to whichever OS thread constructs it; invoking it from
            # a different thread than that — even serialized in time by
            # the lock above — was enough to segfault the whole process in
            # testing. Constructing it here guarantees construction and use
            # always happen on the same thread.
            emit("Initializing AI Sub-Agents (Vision, Audio, Linguistic)...")
            vision_agent = VisionAgent()
            audio_agent = AudioAgent()
            linguistic_agent = LinguisticAgent()

            emit(f"Orchestrating analysis for: {os.path.basename(video_path)}")

            # 1. Split video track into temporary audio for audio/text agents.
            # Each call gets its own uniquely-named file (tempfile guarantees
            # this) — a shared hardcoded filename here would let two runs
            # silently overwrite each other's audio mid-analysis. Belt and
            # braces alongside the lock above, not a substitute for it.
            emit("Extracting audio track from video...")
            temp_audio_fd, temp_audio = tempfile.mkstemp(suffix=".wav", prefix="ros_audio_")
            os.close(temp_audio_fd)
            video = VideoFileClip(video_path)
            if video.audio:
                video.audio.write_audiofile(temp_audio, verbose=False, logger=None)
            video.close()

            try:
                # 2. Sequential Sub-Agent Execution
                emit("[1/3] Vision Agent evaluating eye saccades & gaze sweep...")
                vision_report = vision_agent.analyze(video_path)
                emit(f"[1/3] Vision Agent done — {vision_report['visual_verdict']}")

                emit("[2/3] Audio Agent evaluating prosody & pause ratios...")
                audio_report = audio_agent.analyze(temp_audio)
                emit(f"[2/3] Audio Agent done — {audio_report['audio_verdict']}")

                emit("[3/3] Linguistic Agent running Whisper transcription & disfluency analysis...")
                linguistic_report = linguistic_agent.analyze(temp_audio)
                emit(f"[3/3] Linguistic Agent done — {linguistic_report['text_verdict']}")
            finally:
                # Always clean up this call's own temp file, even if an agent
                # raised — otherwise a failed run leaks a .wav file instead
                # of just the one that failed.
                if os.path.exists(temp_audio):
                    os.remove(temp_audio)

            emit("Fusing agent verdicts into a final decision...")

        # 3. Decision Fusion (Weighted Multi-Agent Voting)
        suspicion_score = 0
        signals = []

        if "Suspicious" in vision_report["visual_verdict"]:
            suspicion_score += 40
            signals.append("Vision Agent detected repetitive horizontal gaze sweep consistent with reading off-screen.")
        
        if "Suspicious" in audio_report["audio_verdict"]:
            suspicion_score += 30
            signals.append("Audio Agent detected lack of natural thinking pauses and low pitch variance.")

        if "Suspicious" in linguistic_report["text_verdict"]:
            suspicion_score += 30
            signals.append("Linguistic Agent detected hyper-polished syntax with zero natural filler hesitations.")

        # 4. Final Aggregated Report
        final_classification = "Likely Scripted / AI-Read" if suspicion_score >= 50 else "Likely Spontaneous"

        final_report = {
            "file": os.path.basename(video_path),
            "final_verdict": final_classification,
            "confidence_score": f"{suspicion_score}%",
            "agent_findings": {
                "vision": vision_report,
                "audio": audio_report,
                "linguistics": linguistic_report
            },
            "orchestrator_synthesis": signals if signals else ["All modalities reflect natural, unscripted responses."]
        }

        emit(f"Analysis complete — {final_classification} ({suspicion_score}% risk).")
        return final_report

if __name__ == "__main__":
    orchestrator = MultiAgentFusionOrchestrator()
    print("\nMulti-Agent Orchestrator is ready to run on interview recordings.")