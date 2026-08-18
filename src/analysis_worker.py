"""
Single dedicated worker thread that runs every video analysis for the whole
app's lifetime.

Why this exists: testing showed that MultiAgentFusionOrchestrator segfaults
the entire process when a second video is analyzed while one is already
running — not just on genuinely overlapping native calls, but seemingly
whenever a second OS thread is active at all while MediaPipe's FaceMesh
(a TFLite/XNNPACK delegate) is in use. A per-call threading.Lock in the
orchestrator serializes execution in time but does not fix this, because
Streamlit still spawns a fresh thread per session/rerun to run the caller's
own code, and simply having that second thread alive was enough to crash.

The fix: never let a Streamlit session thread call into the ML pipeline
directly. Instead, everything routes through one persistent background
thread (started once, reused for the app's whole life) that is the only
thread ever touching MediaPipe/Whisper/spaCy. Session threads submit a job
and poll a queue for progress messages + the final result — all actual
st.* UI calls still happen on the calling (session) thread, which is
required for Streamlit's per-session context to work correctly.
"""

import queue
import threading

from orchestrator import MultiAgentFusionOrchestrator

# Sentinel put on a job's progress queue to signal "no more messages coming".
_DONE = object()


class AnalysisJob:
    """Handle returned by AnalysisWorker.submit(). Poll `progress` for
    status strings (ending with a _DONE sentinel), then read `result`/
    `error` once `finished` is set."""

    def __init__(self):
        self.progress: "queue.Queue[object]" = queue.Queue()
        self.finished = threading.Event()
        self.result = None
        self.error = None


class AnalysisWorker:
    def __init__(self):
        self._jobs: "queue.Queue[tuple[str, AnalysisJob]]" = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True, name="ros-analysis-worker")
        self._thread.start()

    def _run(self):
        # Everything below — for every job, for the life of the process —
        # executes on this one thread. Nothing else may call into
        # MultiAgentFusionOrchestrator/VisionAgent/AudioAgent/LinguisticAgent.
        orchestrator = MultiAgentFusionOrchestrator()
        while True:
            video_path, job = self._jobs.get()
            try:
                job.result = orchestrator.process_video_interview(
                    video_path, on_progress=job.progress.put
                )
            except Exception as e:  # noqa: BLE001 - surface any failure to the caller
                job.error = e
            finally:
                job.progress.put(_DONE)
                job.finished.set()

    def submit(self, video_path: str) -> AnalysisJob:
        """Queue a video for analysis. Returns immediately; the analysis
        itself (and any queued analyses ahead of it) run on the worker
        thread. FIFO — first submitted, first processed."""
        job = AnalysisJob()
        self._jobs.put((video_path, job))
        return job


def drain_progress(job: AnalysisJob, on_message):
    """Block the calling thread until the job finishes, calling
    on_message(str) for each progress update as it arrives (in order).
    Safe to call from a Streamlit session thread — this function does no
    ML work itself, it only reads from the job's queue."""
    while True:
        message = job.progress.get()
        if message is _DONE:
            break
        on_message(message)
    job.finished.wait()
