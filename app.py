"""
Streamlit chat UI.

Beyond a chat box, this deliberately surfaces the three things the challenge
spec asks to "show" explicitly: the ingestion pipeline/corpus stats, the
retrieved chunks before generation (retrieval visualization), and a per-query
latency breakdown against the 2-5s target -- so the judges can see the
pipeline working, not just trust a final answer.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from src import config, pipeline

st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="wide")

CSS = """
<style>
#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 1.2rem; max-width: 1200px;}
html, body, [class*="css"] {font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;}

.hero {background: linear-gradient(120deg, #4338ca 0%, #6d28d9 55%, #0891b2 100%); border-radius: 16px; padding: 22px 28px; margin-bottom: 1.4rem; box-shadow: 0 4px 18px rgba(67,56,202,0.25);}
.app-title {font-size: 1.7rem; font-weight: 800; color: #ffffff; margin-bottom: 0.25rem; letter-spacing: -.3px;}
.app-subtitle {color: #e0e7ff; font-size: 0.92rem;}

.chip {display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: 0.78rem; font-weight: 700; letter-spacing: .2px;}
.chip-green {background: #d1fae5; color: #065f46;}
.chip-orange {background: #ffedd5; color: #9a3412;}
.chip-gray {background: #f1f3f5; color: #495057;}
.chip-blue {background: #dbeafe; color: #1e40af;}

.section-label {font-size: 0.72rem; font-weight: 800; text-transform: uppercase; letter-spacing: 1px; color: #7c3aed; margin: 1.1rem 0 0.5rem 0; border-left: 3px solid #7c3aed; padding-left: 8px;}

.source-card {border: 1px solid #ede9fe; border-left: 4px solid #7c3aed; border-radius: 10px; padding: 12px 16px; margin-bottom: 10px; background: linear-gradient(180deg, #faf8ff 0%, #fdfdff 100%);}
.source-card .fname {font-weight: 700; color: #312e81; font-size: 0.92rem;}
.source-card .snippet {color: #5f6368; font-size: 0.85rem; font-style: italic; margin-top: 6px; line-height: 1.4;}
.score-row {display: flex; align-items: center; gap: 8px; margin-top: 6px;}
.score-label {font-size: 0.72rem; color: #80868b; width: 64px;}
.score-track {flex: 1; background: #ece9fe; border-radius: 4px; height: 6px; overflow: hidden;}
.score-fill-sim {background: linear-gradient(90deg, #0891b2, #22d3ee); height: 6px;}
.score-fill-rerank {background: linear-gradient(90deg, #7c3aed, #c084fc); height: 6px;}
.ocr-badge {background: #fef08a; color: #854d0e; padding: 1px 9px; border-radius: 8px; font-size: 0.68rem; font-weight: 700; margin-left: 8px;}
.citation-chip {background: #ede9fe; color: #5b21b6; padding: 1px 8px; border-radius: 6px; font-size: 0.82rem; font-weight: 700; white-space: nowrap;}

.pipeline-flow {display: flex; align-items: center; flex-wrap: wrap; gap: 4px; margin-bottom: 0.6rem;}
.pipeline-step {border-radius: 8px; padding: 6px 10px; font-size: 0.76rem; font-weight: 600; white-space: nowrap;}
.pipeline-step-ingest {background: #dbeafe; color: #1e40af;}
.pipeline-step-query {background: #ede9fe; color: #5b21b6;}
.pipeline-arrow {color: #c1c7cd; font-size: 0.9rem;}

.metric-row {display: flex; gap: 10px; margin-bottom: 0.8rem;}
.metric-box {flex: 1; border-radius: 10px; padding: 10px 14px; text-align: center;}
.metric-box .v {font-size: 1.25rem; font-weight: 800;}
.metric-box .l {font-size: 0.68rem; text-transform: uppercase; letter-spacing: .5px; opacity: 0.75;}
.metric-blue {background: #dbeafe; color: #1e3a8a;}
.metric-violet {background: #ede9fe; color: #4c1d95;}
.metric-teal {background: #cffafe; color: #164e63;}

[data-testid="chatAvatarIcon-user"], [data-testid="stChatMessageAvatarUser"] {
    background: linear-gradient(135deg, #0891b2, #22d3ee) !important; color: #fff !important;
}
[data-testid="chatAvatarIcon-assistant"], [data-testid="stChatMessageAvatarAssistant"] {
    background: linear-gradient(135deg, #7c3aed, #4338ca) !important; color: #fff !important;
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_data
def load_manifest():
    if config.MANIFEST_PATH.exists():
        return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    return []


@st.cache_data
def load_ingest_report():
    if config.INGEST_REPORT_PATH.exists():
        return json.loads(config.INGEST_REPORT_PATH.read_text(encoding="utf-8"))
    return None


def pipeline_flow(steps: list[str], variant: str) -> str:
    parts = []
    for i, step in enumerate(steps):
        if i:
            parts.append('<span class="pipeline-arrow">&#8594;</span>')
        parts.append(f'<span class="pipeline-step pipeline-step-{variant}">{step}</span>')
    return f'<div class="pipeline-flow">{"".join(parts)}</div>'


def latency_chip(total_ms: float) -> str:
    _, hi = config.LATENCY_TARGET_MS
    cls = "chip-green" if total_ms <= hi else "chip-orange"
    return f'<span class="chip {cls}">{total_ms:.0f} ms</span>'


_DASH = "‐‑‒–—−-"
CITATION_RE = re.compile(rf"\[([^,\]]+),\s*(pp?\.\s*\d+(?:[{_DASH}]\d+)?)\]")


def highlight_citations(answer: str) -> str:
    """Renders inline [filename, p.N] citations as styled chips instead of
    plain bracketed text, so provenance is visually scannable, not just
    technically present -- markdown (**bold**, paragraphs) still renders
    normally around the injected HTML since st.markdown handles both."""

    def _sub(m: re.Match) -> str:
        filename, page = m.group(1).strip(), m.group(2).strip()
        return f'<span class="citation-chip">{filename}, {page}</span>'

    return CITATION_RE.sub(_sub, answer)


def score_bar(label: str, value: float, variant: str) -> str:
    pct = max(0, min(100, value * 100))
    return (
        f'<div class="score-row"><span class="score-label">{label}</span>'
        f'<div class="score-track"><div class="score-fill-{variant}" style="width:{pct:.0f}%"></div></div>'
        f'<span style="font-size:0.72rem;color:#80868b;width:40px;text-align:right">{value:.2f}</span></div>'
    )


def render_sources(sources: list[dict]) -> None:
    st.markdown('<div class="section-label">Retrieval visualization &middot; top retrieved chunks</div>', unsafe_allow_html=True)
    for s in sources:
        page = f"p.{s['page_start']}" if s["page_end"] == s["page_start"] else f"pp.{s['page_start']}-{s['page_end']}"
        ocr_badge = '<span class="ocr-badge">OCR</span>' if s["ocr"] else ""
        card = (
            f'<div class="source-card">'
            f'<span class="fname">{s["filename"]}</span>, {page}{ocr_badge}'
            f'{score_bar("similarity", s["similarity"], "sim")}'
            f'{score_bar("rerank", s["rerank_score"], "rerank")}'
            f'<div class="snippet">&ldquo;{s["snippet"]}&hellip;&rdquo;</div>'
            f'</div>'
        )
        st.markdown(card, unsafe_allow_html=True)


def render_timing(t: dict) -> None:
    st.markdown(
        f'<div style="font-size:0.78rem;color:#9aa0a6;margin-top:4px">'
        f'embed {t["embed_query_ms"]:.0f}ms &middot; search {t["vector_search_ms"]:.0f}ms &middot; '
        f'rerank {t["rerank_ms"]:.0f}ms &middot; generate {t["generate_ms"]:.0f}ms</div>',
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown('<div class="section-label">Document ingestion pipeline</div>', unsafe_allow_html=True)
    manifest = load_manifest()
    report = load_ingest_report()

    if report:
        st.markdown(
            f'<div class="metric-row">'
            f'<div class="metric-box metric-blue"><div class="v">{len(report["documents"])}</div><div class="l">Documents</div></div>'
            f'<div class="metric-box metric-violet"><div class="v">{report["total_chunks"]:,}</div><div class="l">Chunks</div></div>'
            f'<div class="metric-box metric-teal"><div class="v">{report["total_seconds"]:.0f}s</div><div class="l">Ingest time</div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        with st.expander("Per-document detail (pages, OCR pages, chunks)"):
            st.dataframe(report["documents"], use_container_width=True, hide_index=True)
    else:
        st.info("No ingest report found yet. Run `python scripts/ingest.py` first.")

    if manifest:
        with st.expander(f"Corpus manifest — {len(manifest)} sources"):
            st.dataframe(manifest, use_container_width=True, hide_index=True)

    st.markdown(pipeline_flow(["PDF", "extract (native/OCR)", "clean + chunk", "embed", "LanceDB (IVF_PQ)"], "ingest"), unsafe_allow_html=True)
    st.markdown(pipeline_flow(["query", "embed", "ANN top-20", "rerank", "Groq LLM", "cited answer"], "query"), unsafe_allow_html=True)

    st.markdown('<div class="section-label">Stack</div>', unsafe_allow_html=True)
    st.caption(
        f"Embeddings: `{config.EMBEDDING_MODEL}`  \n"
        f"Reranker: `{config.RERANKER_MODEL}`  \n"
        f"Vector DB: LanceDB (IVF_PQ)  \n"
        f"LLM: Groq `{config.GROQ_MODEL}`"
    )

st.markdown(
    '<div class="hero">'
    '<div class="app-title">RAG Chatbot — Indian Education &amp; Skilling Policy</div>'
    '<div class="app-subtitle">Open-source embeddings + vector DB, Groq-served open-weight LLM, '
    '2–5s target latency, source-cited answers.</div>'
    '</div>',
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []


def render_turn(query: str, result_like: dict) -> None:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(query)
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown('<div class="section-label">Generated answer with citations</div>', unsafe_allow_html=True)
        st.markdown(highlight_citations(result_like["answer"]), unsafe_allow_html=True)
        st.markdown(latency_chip(result_like["timings_ms"]["total_ms"]), unsafe_allow_html=True)
        with st.expander(f"Sources ({len(result_like['sources'])}) & retrieval detail", expanded=False):
            render_sources(result_like["sources"])
        render_timing(result_like["timings_ms"])


if not st.session_state.history:
    with st.chat_message("assistant", avatar="🤖"):
        st.write("Hi! Ask me anything about Indian education & skilling policy — every answer is cited.")

for turn in st.session_state.history:
    render_turn(turn["query"], turn)

query = st.chat_input("Ask a question about the ingested policy documents...")
if query:
    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(query)
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Retrieving + generating..."):
            result = pipeline.answer(query)
        st.markdown('<div class="section-label">Generated answer with citations</div>', unsafe_allow_html=True)
        st.markdown(highlight_citations(result.answer), unsafe_allow_html=True)
        st.markdown(latency_chip(result.timings_ms["total_ms"]), unsafe_allow_html=True)
        with st.expander(f"Sources ({len(result.sources)}) & retrieval detail", expanded=False):
            render_sources(result.sources)
        render_timing(result.timings_ms)
    st.session_state.history.append(
        {
            "query": query,
            "answer": result.answer,
            "sources": result.sources,
            "timings_ms": result.timings_ms,
        }
    )
