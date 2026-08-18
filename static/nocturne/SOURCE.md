# Source

`styles.css` in this folder is vendored from the **Nocturne** design system
(Claude Design project `712f5455-d089-4977-86fa-3b52c348c885`), imported via
the `ReadOrSpeak Dashboard` project
(`claude.ai/design/p/0d316ad2-2fa9-4b70-8769-b0236f506b52`).

It is used unmodified as the token/component base for the Streamlit app's
custom theme — see [`src/ui_theme.py`](../../src/ui_theme.py) for the
ReadOrSpeak-specific styling layered on top (semantic verdict colors, agent
gauges, timeline, transcript panel, etc.), adapted from the design's
`ReadOrSpeak Dashboard.dc.html` mockup.

The design project's own JS runtime files (`support.js`, `_ds_bundle.js`)
and lint config (`_adherence.oxlintrc.json`) are internal to the Claude
Design preview tool and are not used here — they have no function outside
that environment.
