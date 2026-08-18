"""
Nocturne-themed UI layer for the ReadOrSpeak Streamlit app.

Adapted from the "ReadOrSpeak Dashboard" mockup built in Claude Design
(project 0d316ad2-2fa9-4b70-8769-b0236f506b52), which itself sits on the
Nocturne design system (project 712f5455-d089-4977-86fa-3b52c348c885).
The vendored stylesheet lives at static/nocturne/styles.css — see
static/nocturne/SOURCE.md for provenance.

Streamlit can't host the mockup's original React-ish `.dc.html` component
tree directly, so this module re-expresses the same layout/visual language
(dark ground, single accent, gauge cards, reasoning timeline, transcript
panel) as plain HTML/CSS blocks rendered via st.markdown(unsafe_allow_html),
wired to the real MultiAgentFusionOrchestrator output instead of the
mockup's static demo numbers.
"""

import os
import html

import streamlit as st


def _md(html_str: str):
    """Render a raw HTML block, bypassing Streamlit's markdown parser.

    st.markdown(..., unsafe_allow_html=True) routes content through a
    CommonMark parser first: lines indented >=4 spaces get treated as code
    blocks (rendered as literal text), and mixing <link>/<style> tags in
    one call can trip up raw-HTML-block detection so the CSS leaks onto
    the page as visible text. st.html() skips markdown parsing entirely
    and injects the string as-is, which is what every block in this module
    actually wants.
    """
    st.html(html_str)


def icon(name: str, size: int = 16, color: str = "currentColor") -> str:
    """Render a Google Material Icons glyph (classic ligature font).

    `name` is a Material Icons name in snake_case, e.g. "mic", "visibility",
    "graphic_eq" — see https://fonts.google.com/icons?icon.set=Material+Icons
    """
    return (
        f'<span class="material-icons" '
        f'style="font-size:{size}px;color:{color};line-height:1;vertical-align:middle;">'
        f"{name}</span>"
    )

_STYLES_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "static", "nocturne", "styles.css",
)

# Gauge calibration — each agent's raw metric is drawn as a fraction of a
# plausible max value for visualization only. These are NOT the suspicion
# thresholds (those live in src/agents.py); they just set how "full" the
# ring looks. threshold sits at roughly the ring's midpoint by design,
# mirroring the original ReadOrSpeak Dashboard mockup's gauge proportions.
VISION_GAUGE_MAX = 0.10       # gaze_horizontal_variance threshold is 0.05
AUDIO_GAUGE_MAX = 0.50        # pause_ratio threshold is 0.15
LINGUISTIC_GAUGE_MAX = 0.10   # filler_ratio threshold is 0.02

_EXTRA_CSS = """
:root{
  --ok:#63d6ad; --ok-dim:rgba(99,214,173,.13); --ok-line:rgba(99,214,173,.4);
  --amber:#e2c06a; --amber-dim:rgba(226,192,106,.13); --amber-line:rgba(226,192,106,.4);
  --danger:#e5736a; --danger-dim:rgba(229,115,106,.13); --danger-line:rgba(229,115,106,.4);

  /* Golden-ratio (phi ~= 1.618) spacing scale — each step is the previous
     one times phi, rounded to whole pixels (a Fibonacci-style progression).
     Used for the "space between items" gaps: card grids, section margins,
     stacked panels. Fine icon-to-label gaps (6-9px) intentionally stay off
     this scale since those aren't separate items. */
  --ros-space-1: 8px;
  --ros-space-2: 13px;
  --ros-space-3: 21px;
  --ros-space-4: 34px;
  --ros-space-5: 55px;
}
/* Same golden-ratio progression applied to corner radii, overriding the
   design system's own --radius-* tokens (cascades over styles.css since
   this block loads after it) so every card/button/tag/dialog curve — not
   just our own .ros-* blocks — follows the same rhythm. */
:root{
  --radius-sm: 8px;
  --radius-md: 13px;
  --radius-lg: 21px;
}
@keyframes rosPulse{0%,100%{opacity:.55}50%{opacity:1}}
@keyframes rosSpin{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}
.ros-scroll::-webkit-scrollbar{width:8px;height:8px;}
.ros-scroll::-webkit-scrollbar-thumb{background:var(--color-neutral-800);border-radius:8px;}
.ros-scroll::-webkit-scrollbar-track{background:transparent;}

/* ---- Streamlit chrome ---- */
#MainMenu, header[data-testid="stHeader"], footer { visibility: hidden; height: 0; }
.stApp {
  background: radial-gradient(1200px 600px at 78% -10%, #1c1f31 0%, var(--color-bg) 60%);
  color: var(--color-text);
  font-family: var(--font-body);
}
.block-container { max-width: 1440px; padding-top: 1.2rem; padding-bottom: 2.5rem; }
[data-testid="stFileUploader"] section {
  background: var(--color-surface); border: 1.5px dashed var(--color-neutral-700);
  border-radius: var(--radius-lg);
}
[data-testid="stFileUploader"] label, [data-testid="stFileUploaderDropzoneInstructions"] span {
  color: var(--color-text) !important;
}
.stAlert { background: var(--color-surface); border-radius: var(--radius-md); }

/* Reskin native buttons as Nocturne pill/outline buttons */
.stButton button {
  font-family: var(--font-heading); font-weight: var(--font-heading-weight);
  border-radius: var(--radius-md) !important; border: 1px solid var(--color-divider) !important;
  background: transparent !important; color: var(--color-text) !important;
}
.stButton button[kind="primary"] {
  color: var(--color-accent) !important; border-color: var(--color-accent) !important;
  background: color-mix(in srgb, var(--color-accent) 14%, transparent) !important;
}
.stButton button:hover { background: color-mix(in srgb, var(--color-text) 7%, transparent) !important; }

/* ---- ReadOrSpeak dashboard blocks ---- */
.ros-logo{width:40px;height:40px;border-radius:11px;display:grid;place-items:center;background:linear-gradient(145deg, var(--color-accent-700), var(--color-accent-900));box-shadow:inset 0 0 0 1px color-mix(in srgb, var(--color-accent) 45%, transparent), 0 0 24px color-mix(in srgb, var(--color-accent) 22%, transparent);flex:none;}
.ros-title{font-family:var(--font-heading);font-weight:600;font-size:18px;letter-spacing:-0.02em;line-height:1.1;}
.ros-subtitle{font-size:12px;color:var(--color-neutral-500);letter-spacing:.01em;}
.ros-kicker{font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:var(--color-accent);margin-bottom:8px;}
.ros-card{background:var(--color-surface);border-radius:var(--radius-lg);box-shadow:var(--shadow-sm);padding:var(--ros-space-3);}
.ros-card h4{display:flex;align-items:center;gap:9px;margin:0 0 14px;font-size:14px;letter-spacing:.02em;}
.ros-verdict{position:relative;border-radius:var(--radius-lg);padding:var(--ros-space-3) var(--ros-space-4);overflow:hidden;}
.ros-gauge{position:relative;width:104px;height:104px;margin:2px auto 0;border-radius:50%;}
.ros-gauge-inner{position:absolute;inset:11px;border-radius:50%;background:var(--color-surface);display:flex;flex-direction:column;align-items:center;justify-content:center;}
.ros-idle-dot{width:8px;height:8px;border-radius:50%;background:var(--color-neutral-600);animation:rosPulse 1.8s ease-in-out infinite;}
.ros-timeline-dot{width:11px;height:11px;border-radius:50%;margin-top:4px;}
.ros-timeline-rule{width:2px;flex:1;background:var(--color-neutral-800);margin:4px 0;}
"""


def inject_theme():
    """Load the Nocturne stylesheet + ReadOrSpeak overrides into the page, once per run."""
    with open(_STYLES_PATH, "r", encoding="utf-8") as f:
        nocturne_css = f.read()
    _md(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/icon?family=Material+Icons');
        {nocturne_css}
        {_EXTRA_CSS}
        </style>
        """
    )


def render_brand():
    """The logo + title/subtitle block, meant to sit in the header row's left column."""
    _md(
        f"""
        <div style="display:flex;align-items:center;gap:var(--ros-space-3);padding:4px 4px 4px 0;">
          <div class="ros-logo">{icon('mic', 22, 'var(--color-accent-200)')}</div>
          <div>
            <div class="ros-title">ReadOrSpeak</div>
            <div class="ros-subtitle">Multimodal AI Interview Analysis</div>
          </div>
        </div>
        """
    )


def render_divider():
    """A full-width rule under the header row (brand + tabs)."""
    _md('<div class="hr" style="margin:4px 0 22px;"></div>')


def render_header():
    """Deprecated: combined brand+divider in one block. Kept for compatibility;
    prefer render_brand() + render_divider() so tabs can sit in the same row."""
    render_brand()
    render_divider()


def render_verdict_banner(final_verdict: str, score: int):
    scripted = "Scripted" in final_verdict
    color = "var(--danger)" if scripted else "var(--ok)"
    dim = "var(--danger-dim)" if scripted else "var(--ok-dim)"
    line = "var(--danger-line)" if scripted else "var(--ok-line)"
    icon_name = "warning" if scripted else "verified"
    _md(
        f"""
        <section class="ros-verdict" style="background:linear-gradient(135deg,{dim},transparent 70%);box-shadow:inset 0 0 0 1px {line};">
          <div style="display:flex;align-items:center;gap:var(--ros-space-3);">
            <div style="width:52px;height:52px;flex:none;border-radius:14px;display:grid;place-items:center;background:{dim};box-shadow:inset 0 0 0 1px {line};">
              {icon(icon_name, 27, color)}
            </div>
            <div style="flex:1;">
              <div style="font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:{color};margin-bottom:4px;">Final Verdict</div>
              <div style="font-family:var(--font-heading);font-weight:600;font-size:26px;letter-spacing:-0.02em;line-height:1.1;">{html.escape(final_verdict)}</div>
            </div>
            <div style="text-align:right;">
              <div style="font-family:var(--font-heading);font-weight:600;font-size:30px;color:{color};line-height:1;">{score}%</div>
              <div style="font-size:11px;color:var(--color-neutral-400);margin-top:3px;">risk score</div>
            </div>
          </div>
        </section>
        """
    )


def render_risk_meter(score: int):
    pct = max(0, min(score, 100))
    color = "var(--danger)" if pct >= 50 else "var(--ok)"
    _md(
        f"""
        <section class="ros-card">
          <h4>{icon('speed', 17, 'var(--color-accent)')}Fused Risk Score
            <span style="margin-left:auto;font-size:11px;color:var(--color-neutral-500);">threshold 50%</span>
          </h4>
          <div style="display:flex;align-items:baseline;gap:10px;margin-bottom:var(--ros-space-3);">
            <span style="font-family:var(--font-heading);font-weight:600;font-size:46px;line-height:1;color:{color};letter-spacing:-0.03em;">{pct}<span style="font-size:22px;">%</span></span>
            <span style="font-size:13px;color:var(--color-neutral-400);">scripted-risk · {'above' if pct>=50 else 'below'} decision line</span>
          </div>
          <div style="position:relative;height:12px;border-radius:8px;background:var(--color-neutral-900);overflow:hidden;">
            <div style="position:absolute;inset:0;width:{pct}%;background:linear-gradient(90deg,{color},color-mix(in srgb,{color} 70%,var(--amber)));border-radius:8px;"></div>
          </div>
          <div style="position:relative;height:20px;margin-top:2px;">
            <div style="position:absolute;left:50%;top:-14px;bottom:0;width:2px;background:var(--color-neutral-500);transform:translateX(-50%);"></div>
            <div style="position:absolute;left:50%;top:2px;transform:translateX(-50%);font-size:10px;color:var(--color-neutral-500);white-space:nowrap;">DECISION 50%</div>
          </div>
          <div style="display:flex;justify-content:space-between;font-size:11px;color:var(--color-neutral-500);margin-top:4px;">
            <span style="color:var(--ok);">&#9666; Spontaneous</span>
            <span style="color:var(--danger);">Scripted &#9656;</span>
          </div>
        </section>
        """
    )


def _gauge_html(icon_name, label, pct, value_str, unit_label, suspicious, verdict_text, sub_text, pts):
    color = "var(--amber)" if suspicious else "var(--ok)"
    dim = "var(--amber-dim)" if suspicious else "var(--ok-dim)"
    line = "var(--amber-line)" if suspicious else "var(--ok-line)"
    deg = round(max(0.0, min(pct, 1.0)) * 360)
    return f"""
        <div class="ros-card" style="display:flex;flex-direction:column;gap:var(--ros-space-2);">
          <div style="display:flex;align-items:center;gap:8px;">
            {icon(icon_name, 16, color)}
            <span style="font-size:12px;font-weight:600;letter-spacing:.02em;">{label}</span>
            <span style="margin-left:auto;font-size:9px;font-weight:600;letter-spacing:.06em;color:{color};background:{dim};border:1px solid {line};border-radius:20px;padding:3px 8px;">{pts} pts</span>
          </div>
          <div class="ros-gauge" style="background:conic-gradient({color} 0deg {deg}deg, var(--color-neutral-900) {deg}deg 360deg);">
            <div class="ros-gauge-inner">
              <span style="font-family:ui-monospace,monospace;font-size:20px;font-weight:600;color:{color};">{value_str}</span>
              <span style="font-size:9px;color:var(--color-neutral-500);letter-spacing:.04em;">{unit_label}</span>
            </div>
          </div>
          <div style="text-align:center;">
            <div style="font-size:12px;font-weight:600;color:{color};">{verdict_text}</div>
            <div style="font-size:10px;color:var(--color-neutral-500);margin-top:2px;">{html.escape(sub_text)}</div>
          </div>
        </div>
    """


def render_agent_gauges(vision: dict, audio: dict, linguistics: dict):
    v_susp = "Suspicious" in vision["visual_verdict"]
    a_susp = "Suspicious" in audio["audio_verdict"]
    l_susp = "Suspicious" in linguistics["text_verdict"]

    v_html = _gauge_html(
        "visibility", "Vision",
        vision["gaze_horizontal_variance"] / VISION_GAUGE_MAX,
        f"{vision['gaze_horizontal_variance']:.3f}", "gaze var",
        v_susp, "Suspicious" if v_susp else "Normal",
        "Reading gaze sweep · thr > 0.05" if v_susp else "Natural fixation",
        40 if v_susp else 0,
    )
    a_html = _gauge_html(
        "graphic_eq", "Audio",
        audio["pause_ratio"] / AUDIO_GAUGE_MAX,
        f"{audio['pause_ratio']:.3f}", "pause ratio",
        a_susp, "Suspicious" if a_susp else "Normal",
        f"pitch σ {audio.get('pitch_std_hz', 0):.0f} Hz",
        30 if a_susp else 0,
    )
    l_html = _gauge_html(
        "chat", "Linguistic",
        linguistics["filler_ratio"] / LINGUISTIC_GAUGE_MAX,
        f"{linguistics['filler_ratio']:.3f}", "filler ratio",
        l_susp, "Suspicious" if l_susp else "Normal",
        f"{linguistics.get('word_count', 0)} words",
        30 if l_susp else 0,
    )

    _md(
        f"""
        <div style="display:flex;align-items:center;gap:9px;margin-bottom:var(--ros-space-3);">
          {icon('widgets', 17, 'var(--color-accent)')}
          <h4 style="margin:0;font-size:14px;letter-spacing:.02em;">Multi-Agent Diagnostics</h4>
          <span style="margin-left:auto;font-size:11px;color:var(--color-neutral-500);">late-fusion · 40 / 30 / 30 weights</span>
        </div>
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:var(--ros-space-3);">{v_html}{a_html}{l_html}</div>
        """
    )


def render_timeline(report: dict):
    vision = report["agent_findings"]["vision"]
    audio = report["agent_findings"]["audio"]
    linguistics = report["agent_findings"]["linguistics"]
    v_susp = "Suspicious" in vision["visual_verdict"]
    a_susp = "Suspicious" in audio["audio_verdict"]
    l_susp = "Suspicious" in linguistics["text_verdict"]
    score = report["confidence_score"]
    scripted = "Scripted" in report["final_verdict"]

    steps = [
        ("Vision Agent", "+40" if v_susp else "+0", "var(--amber)" if v_susp else "var(--ok)",
         f"Gaze variance {vision['gaze_horizontal_variance']:.3f} — " + vision["visual_verdict"]),
        ("Audio Agent", "+30" if a_susp else "+0", "var(--amber)" if a_susp else "var(--ok)",
         f"Pause ratio {audio['pause_ratio']:.3f} — " + audio["audio_verdict"]),
        ("Linguistic Agent", "+30" if l_susp else "+0", "var(--amber)" if l_susp else "var(--ok)",
         f"Filler ratio {linguistics['filler_ratio']:.3f} — " + linguistics["text_verdict"]),
        ("Fusion Decision", f"{score} vs 50%", "var(--danger)" if scripted else "var(--color-accent)",
         f"Weighted vote totals {score} scripted-risk → {report['final_verdict']}."),
    ]

    rows = ""
    for i, (title, pts, dot, body) in enumerate(steps):
        rule = '<div class="ros-timeline-rule"></div>' if i < len(steps) - 1 else ""
        rows += f"""
          <div style="display:flex;gap:var(--ros-space-2);">
            <div style="display:flex;flex-direction:column;align-items:center;flex:none;">
              <div class="ros-timeline-dot" style="background:{dot};box-shadow:0 0 0 4px color-mix(in srgb, {dot} 20%, transparent);"></div>
              {rule}
            </div>
            <div style="padding-bottom:var(--ros-space-3);">
              <div style="display:flex;align-items:center;gap:8px;">
                <span style="font-size:13px;font-weight:600;">{html.escape(title)}</span>
                <span style="font-size:10px;font-weight:600;color:{dot};">{html.escape(pts)}</span>
              </div>
              <div style="font-size:12px;color:var(--color-neutral-400);line-height:1.5;margin-top:3px;">{html.escape(body)}</div>
            </div>
          </div>
        """

    _md(
        f"""
        <section class="ros-card">
          <h4>{icon('account_tree', 17, 'var(--color-accent)')}Orchestrator Reasoning</h4>
          <div style="display:flex;flex-direction:column;">{rows}</div>
        </section>
        """
    )


def render_transcript(linguistics: dict):
    transcript = linguistics.get("transcript", "") or "(empty transcript)"
    paragraphs = "".join(
        f'<p style="margin:0 0 10px;">{html.escape(p)}</p>'
        for p in transcript.split("\n") if p.strip()
    ) or f'<p style="margin:0;">{html.escape(transcript)}</p>'

    susp = "Suspicious" in linguistics["text_verdict"]
    color = "var(--amber)" if susp else "var(--ok)"
    filler_count = round(linguistics["filler_ratio"] * max(linguistics.get("word_count", 1), 1))

    _md(
        f"""
        <section class="ros-card" style="display:flex;flex-direction:column;">
          <div style="display:flex;align-items:center;gap:9px;margin-bottom:var(--ros-space-3);">
            {icon('description', 17, 'var(--color-accent)')}
            <h4 style="margin:0;font-size:14px;letter-spacing:.02em;">Whisper Transcript</h4>
            <span style="margin-left:auto;font-size:10px;color:var(--color-neutral-500);">tiny · {linguistics.get('word_count', 0)} tokens</span>
          </div>
          <div class="ros-scroll" style="font-size:13px;line-height:1.65;color:var(--color-neutral-300);max-height:236px;overflow-y:auto;padding-right:6px;">
            {paragraphs}
          </div>
          <div style="display:flex;gap:var(--ros-space-3);margin-top:var(--ros-space-3);padding-top:12px;border-top:1px solid var(--color-divider);">
            <div><div style="font-size:10px;color:var(--color-neutral-500);">Fillers</div><div style="font-family:ui-monospace,monospace;font-size:13px;color:{color};">{filler_count} / {linguistics.get('word_count', 0)}</div></div>
            <div><div style="font-size:10px;color:var(--color-neutral-500);">Filler ratio</div><div style="font-family:ui-monospace,monospace;font-size:13px;color:{color};">{linguistics['filler_ratio']:.4f}</div></div>
            <div><div style="font-size:10px;color:var(--color-neutral-500);">Verdict</div><div style="font-family:ui-monospace,monospace;font-size:13px;color:{color};">{'Scripted' if susp else 'Spontaneous'}</div></div>
          </div>
        </section>
        """
    )


def render_source_card_open(filename: str, size_mb: float):
    _md(
        f"""
        <section class="ros-card" style="padding:var(--ros-space-3);">
          <div style="display:flex;align-items:center;gap:9px;margin-bottom:var(--ros-space-2);">
            {icon('movie', 17, 'var(--color-accent)')}
            <h4 style="margin:0;font-size:14px;letter-spacing:.02em;">Source Response</h4>
          </div>
          <div style="display:flex;align-items:center;gap:var(--ros-space-2);padding:var(--ros-space-2);border:1px solid var(--color-divider);border-radius:var(--radius-md);background:var(--color-bg);">
            <div style="width:38px;height:38px;flex:none;border-radius:9px;display:grid;place-items:center;background:var(--color-neutral-900);">{icon('videocam', 19, 'var(--color-neutral-300)')}</div>
            <div style="min-width:0;flex:1;">
              <div style="font-size:13px;font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{html.escape(filename)}</div>
              <div style="font-size:11px;color:var(--color-neutral-500);margin-top:2px;">{size_mb:.1f} MB</div>
            </div>
            <span class="tag tag-neutral" style="flex:none;">{html.escape(os.path.splitext(filename)[1].lstrip('.') or 'video')}</span>
          </div>
        </section>
        """
    )


def render_idle_agent_cards():
    agents = [
        ("visibility", "Vision", "MediaPipe Face Mesh — horizontal gaze-sweep variance."),
        ("graphic_eq", "Audio", "Librosa DSP — pause ratio and pitch (F0) variance."),
        ("chat", "Linguistic", "Whisper + spaCy — disfluency / filler-word ratio."),
    ]
    cards = "".join(
        f"""
        <div class="ros-card" style="opacity:.85;">
          <div style="display:flex;align-items:center;gap:9px;margin-bottom:10px;">
            {icon(icon_name, 18, 'var(--color-neutral-500)')}
            <span style="font-size:13px;font-weight:600;">{name}</span>
            <span class="ros-idle-dot" style="margin-left:auto;"></span>
          </div>
          <div style="font-size:12px;color:var(--color-neutral-500);line-height:1.5;">{desc}</div>
          <div style="font-size:11px;color:var(--color-neutral-600);margin-top:10px;font-family:ui-monospace,monospace;">awaiting input</div>
        </div>
        """
        for icon_name, name, desc in agents
    )
    _md(
        f'<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:var(--ros-space-3);margin-top:var(--ros-space-4);">{cards}</div>'
    )


# ---- Live pipeline flowchart ----------------------------------------------
# Tracks which stage of MultiAgentFusionOrchestrator's run is active/done by
# pattern-matching the exact progress strings it emits (see
# src/orchestrator.py's _emit calls). Kept as a pure state-update function so
# app.py can hold the (completed, current) state itself between callback
# invocations and just re-render the diagram each time.

PIPELINE_STAGES = [
    ("models", "Load Models"),
    ("audio_extract", "Extract Audio"),
    ("vision", "Vision Agent"),
    ("audio", "Audio Agent"),
    ("linguistic", "Linguistic Agent"),
    ("fusion", "Fusion"),
    ("done", "Verdict"),
]


def pipeline_stage_update(message: str, completed: set, current):
    """Given one orchestrator progress message, return the updated
    (completed, current) pipeline state. `completed` is a set of stage keys;
    `current` is the in-progress stage key or None."""
    completed = set(completed)
    if message.startswith("Initializing AI Sub-Agents"):
        current = "models"
    elif message.startswith("Extracting audio track"):
        completed.add("models")
        current = "audio_extract"
    elif message.startswith("[1/3] Vision Agent evaluating"):
        completed.add("audio_extract")
        current = "vision"
    elif message.startswith("[1/3] Vision Agent done"):
        completed.add("vision")
        current = None
    elif message.startswith("[2/3] Audio Agent evaluating"):
        current = "audio"
    elif message.startswith("[2/3] Audio Agent done"):
        completed.add("audio")
        current = None
    elif message.startswith("[3/3] Linguistic Agent running"):
        current = "linguistic"
    elif message.startswith("[3/3] Linguistic Agent done"):
        completed.add("linguistic")
        current = None
    elif message.startswith("Fusing agent verdicts"):
        current = "fusion"
    elif message.startswith("Analysis complete"):
        completed.add("fusion")
        completed.add("done")
        current = "done"
    return completed, current


def _pipeline_node_html(label: str, state: str) -> str:
    if state == "done":
        bg, border, color, glyph, extra = "var(--ok-dim)", "var(--ok-line)", "var(--ok)", "check_circle", ""
    elif state == "active":
        bg = "color-mix(in srgb, var(--color-accent) 16%, transparent)"
        border, color, glyph = "var(--color-accent)", "var(--color-accent)", "autorenew"
        extra = "animation:rosSpin 1.1s linear infinite;"
    else:
        bg, border, color, glyph, extra = "var(--color-neutral-900)", "var(--color-neutral-800)", "var(--color-neutral-600)", "radio_button_unchecked", ""
    return f"""
        <div style="display:flex;flex-direction:column;align-items:center;gap:6px;flex:none;width:84px;">
          <div style="width:40px;height:40px;border-radius:50%;display:grid;place-items:center;background:{bg};box-shadow:inset 0 0 0 1.5px {border};">
            <span class="material-icons" style="font-size:20px;color:{color};{extra}">{glyph}</span>
          </div>
          <div style="font-size:10.5px;text-align:center;color:{color};font-weight:600;line-height:1.25;">{html.escape(label)}</div>
        </div>
    """


def render_pipeline_html(completed: set, current) -> str:
    """Return the flowchart HTML string for the given pipeline state.
    Caller is responsible for pushing it to a placeholder, e.g.:
        placeholder.html(ui_theme.render_pipeline_html(completed, current))
    """
    parts = []
    for i, (key, label) in enumerate(PIPELINE_STAGES):
        state = "done" if key in completed else ("active" if key == current else "pending")
        parts.append(_pipeline_node_html(label, state))
        if i < len(PIPELINE_STAGES) - 1:
            line_color = "var(--ok)" if key in completed else "var(--color-neutral-800)"
            parts.append(f'<div style="flex:1;min-width:12px;height:2px;background:{line_color};margin-top:19px;"></div>')
    return (
        '<div class="ros-card" style="display:flex;align-items:flex-start;gap:4px;'
        'margin:var(--ros-space-3) 0;'
        f'padding:var(--ros-space-3) var(--ros-space-3);overflow-x:auto;">{"".join(parts)}</div>'
    )
