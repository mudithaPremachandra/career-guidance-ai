"""
Full-screen loading overlay: greys out the page, shows which pipeline step is running,
and cycles light-hearted messages so a slow step (usually the Gemini call) never looks frozen.
"""

import html
import random
from contextlib import contextmanager
from typing import Callable, Iterator, Optional

import streamlit as st

QUIRKY_MESSAGES = [
    "Consulting the career crystal ball…",
    "Asking 80 decision trees for their honest opinion…",
    "Fuzzifying your strengths (it's a real thing, promise)…",
    "Minding the skill gap…",
    "Polishing your future job title…",
    "Cross-checking your GPA with the oracle…",
    "Counting hackathons per AI Engineer…",
    "Sharpening the learning pathway…",
]
SECONDS_PER_MESSAGE = 2.5

OVERLAY_CSS = """
<style>
.pf-loading-overlay {
    position: fixed; inset: 0; z-index: 1000000;
    display: flex; align-items: center; justify-content: center;
    background: rgba(7, 11, 20, 0.78);
    backdrop-filter: blur(6px);
}
.pf-loading-box {
    width: min(460px, 90vw); padding: 2rem 2.2rem; text-align: center;
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(13, 20, 36, 0.98) 100%);
    border: 1px solid rgba(0, 240, 255, 0.45); border-radius: 16px;
    box-shadow: 0 0 35px rgba(0, 240, 255, 0.18);
    font-family: 'Inter', sans-serif;
}
.pf-loading-spinner {
    width: 46px; height: 46px; margin: 0 auto 1.2rem auto; border-radius: 50%;
    border: 4px solid rgba(56, 189, 248, 0.18); border-top-color: #00F0FF;
    animation: pf-spin 0.9s linear infinite;
}
.pf-loading-step {
    font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase; color: #38BDF8;
}
.pf-loading-stage { font-size: 1.05rem; font-weight: 700; color: #F8FAFC; margin: 0.35rem 0 1rem 0; }
.pf-loading-subtitle { font-size: 0.85rem; color: #CBD5E1; line-height: 1.45; margin: -0.6rem 0 1rem 0; }
.pf-loading-quips { position: relative; height: 1.4rem; font-size: 0.88rem; color: #94A3B8; font-style: italic; }
.pf-loading-quips span {
    position: absolute; left: 0; right: 0; opacity: 0;
    animation-name: pf-quip; animation-timing-function: ease-in-out; animation-iteration-count: infinite;
}
@keyframes pf-spin { to { transform: rotate(360deg); } }
@keyframes pf-quip {
    0% { opacity: 0; transform: translateY(6px); }
    __FADE_IN__% { opacity: 1; transform: translateY(0); }
    __HOLD__% { opacity: 1; transform: translateY(0); }
    __SLOT__% { opacity: 0; transform: translateY(-6px); }
    100% { opacity: 0; }
}
</style>
"""
# Each message owns 1/N of the animation cycle, so the keyframes follow the message count
_SLOT_PCT = 100 / len(QUIRKY_MESSAGES)
OVERLAY_CSS = (
    OVERLAY_CSS.replace("__FADE_IN__", f"{_SLOT_PCT * 0.25:.2f}")
    .replace("__HOLD__", f"{_SLOT_PCT * 0.8:.2f}")
    .replace("__SLOT__", f"{_SLOT_PCT:.2f}")
)


def overlay_html(stage: str, step: Optional[int] = None, total: Optional[int] = None, subtitle: str = "") -> str:
    """Overlay markup. Messages are shuffled each render so a stage change doesn't always restart on the same one."""
    quips = random.sample(QUIRKY_MESSAGES, len(QUIRKY_MESSAGES))
    cycle = SECONDS_PER_MESSAGE * len(quips)
    spans = "".join(
        f'<span style="animation-duration: {cycle}s; animation-delay: {i * SECONDS_PER_MESSAGE}s;">{html.escape(q)}</span>'
        for i, q in enumerate(quips)
    )
    step_html = f'<div class="pf-loading-step">Step {step} of {total}</div>' if step else ""
    subtitle_html = f'<div class="pf-loading-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    return (
        f'{OVERLAY_CSS}<div class="pf-loading-overlay"><div class="pf-loading-box">'
        f'<div class="pf-loading-spinner"></div>'
        f'{step_html}'
        f'<div class="pf-loading-stage">{html.escape(stage)}</div>'
        f'{subtitle_html}'
        f'<div class="pf-loading-quips">{spans}</div>'
        f"</div></div>"
    )


class LoadingOverlay:
    """Create once at the top level of the page (so no transformed ancestor traps the fixed overlay), then use `show()`."""

    def __init__(self) -> None:
        self.slot = st.empty()

    @contextmanager
    def show(self, total_steps: int) -> Iterator[Callable[[int, str], None]]:
        """Yields `stage(step, text)`; the overlay is removed when the block exits, even on an error."""
        def stage(step: int, text: str) -> None:
            self.slot.markdown(overlay_html(text, step, total_steps), unsafe_allow_html=True)

        try:
            yield stage
        finally:
            self.slot.empty()

    @contextmanager
    def message(self, title: str, subtitle: str = "") -> Iterator[None]:
        """One title for the whole block, without a step counter (e.g. app startup)."""
        self.slot.markdown(overlay_html(title, subtitle=subtitle), unsafe_allow_html=True)
        try:
            yield
        finally:
            self.slot.empty()
