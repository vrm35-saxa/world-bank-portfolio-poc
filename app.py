import os
import re
from pathlib import Path

import gdown
import numpy as np
import pandas as pd
import streamlit as st
from anthropic import Anthropic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="World Bank Health Infrastructure Portfolio Explorer",
    page_icon="🌐",
    layout="wide",
)

ARCHETYPE_FILE = "world_bank_project_archetypes_k3.csv"
LOCAL_RAG_CACHE = "/tmp/world_bank_rag_serving.csv"

st.markdown(
    """
    <style>
    .stApp { background-color: #0b1117; color: #e8eef5; }
    [data-testid="stSidebar"] { background-color: #0d141c; border-right: 1px solid #243241; }
    .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] { background: #111a23; border: 1px solid #243241; border-radius: 8px; padding: 12px; }
    .evidence-card { background: #111a23; border: 1px solid #243241; border-radius: 8px; padding: 14px; margin-bottom: 10px; }
    .muted { color: #8ea0b2; font-size: 0.86rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


def extract_gdrive_file_id(value: str) -> str:
    value = value.strip()
    if not value:
        return ""
    if "/" not in value and "?" not in value:
        return value
    patterns = [r"/file/d/([a-zA-Z0-9_-]+)", r"[?&]id=([a-zA-Z0-9_-]+)"]
    for pattern in patterns:
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    raise ValueError("Could not extract a Google Drive file ID from that value.")


@st.cache_data(show_spinner=False)
def download_and_load_rag(gdrive_value: str):
    file_id = extract_gdrive_file_id(gdrive_value)
    if not file_id:
        raise ValueError("Google Drive file ID or share URL is required.")
    gdown.download(id=file_id, output=LOCAL_RAG_CACHE, quiet=True)
    rag = pd.read_csv(LOCAL_RAG_CACHE)
    required = {
        "chunk_id", "project_id", "document_id", "doc_type",
        "page_number", "section_label", "heading_text", "text_clean",
    }
    missing = required - set(rag.columns)
    if missing:
        raise ValueError(f"RAG file is missing columns: {sorted(missing)}")
    rag["project_id"] = rag["project_id"].astype(str)
    rag["doc_type"] = rag["doc_type"].astype(str).str.upper().str.strip()
    return rag


@st.cache_data(show_spinner=False)
def load_archetypes():
    path = Path(ARCHETYPE_FILE)
    if not path.exists():
        raise FileNotFoundError(f"`{ARCHETYPE_FILE}` was not found in the repository.")
    archetypes = pd.read_csv(path)
    required = {"project_id", "cluster", "archetype"}
    missing = required - set(archetypes.columns)
    if missing:
        raise ValueError(f"Archetype file is missing columns: {sorted(missing)}")
    archetypes["project_id"] = archetypes["project_id"].astype(str)
    return archetypes


def filter_corpus(df, project, doc_type, archetype, topic, archetypes_df):
    out = df.copy()
    if project != "All projects":
        out = out[out["project_id"] == project]
    if doc_type != "PAD + ICR":
        out = out[out["doc_type"] == doc_type.replace(" only", "").upper()]
    if archetype != "All archetypes":
        project_ids = archetypes_df.loc[
            archetypes_df["archetype"] == archetype, "project_id"
        ].astype(str).tolist()
        out = out[out["project_id"].isin(project_ids)]

    topic_map = {
        "Physical Infrastructure": ["facility", "facilities", "hospital", "clinic", "laboratory", "construction", "rehabilitation", "renovation", "retrofit", "upgrade", "rebuild"],
        "DRM / Facility Resilience": ["resilience", "resilient", "disaster risk management", "withstand", "recover", "shock"],
        "Health-System Resilience": ["health system", "preparedness", "capacity building", "response capacity", "maintain core functions"],
        "Hazard Specificity": ["flood", "cyclone", "hurricane", "typhoon", "earthquake", "drought", "heat", "landslide", "tsunami", "storm surge"],
        "Climate Adaptation": ["climate", "temperature", "precipitation", "sea level"],
        "Lifelines / Utilities": ["water", "sanitation", "electricity", "solar", "generator", "drainage", "power"],
    }
    if topic != "All sponsor dimensions":
        terms = topic_map.get(topic, [])
        pattern = "|".join(re.escape(term) for term in terms)
        if pattern:
            out = out[out["text_clean"].fillna("").str.contains(pattern, case=False, regex=True)]
    return out


def score_corpus(query, df):
    if df.empty:
        return df.assign(retrieval_score=pd.Series(dtype=float))
    scored = df.copy()
    texts = scored["text_clean"].fillna("").astype(str)
    vectorizer = TfidfVectorizer(
        stop_words="english", ngram_range=(1, 2), min_df=1, max_df=0.98, sublinear_tf=True
    )
    matrix = vectorizer.fit_transform(texts)
    query_vec = vectorizer.transform([query])
    scored["retrieval_score"] = cosine_similarity(query_vec, matrix).ravel()
    return scored


def retrieve_global(query, df, top_k=8):
    scored = score_corpus(query, df)
    if scored.empty:
        return scored
    return scored.sort_values("retrieval_score", ascending=False).head(min(top_k, len(scored))).copy()


def retrieve_stratified(query, df, chunks_per_project=1, balance_pad_icr=True):
    """Ensure broad project coverage for portfolio-wide questions."""
    scored = score_corpus(query, df)
    if scored.empty:
        return scored
    scored = scored.sort_values("retrieval_score", ascending=False)
    available_doc_types = set(scored["doc_type"].dropna().unique())
    use_doc_balance = balance_pad_icr and "PAD" in available_doc_types and "ICR" in available_doc_types

    selected = []
    if use_doc_balance:
        for project_id, project_df in scored.groupby("project_id", sort=True):
            for dt in ["PAD", "ICR"]:
                subset = project_df[project_df["doc_type"] == dt]
                if not subset.empty:
                    selected.append(subset.head(chunks_per_project))
    else:
        for project_id, project_df in scored.groupby("project_id", sort=True):
            selected.append(project_df.head(chunks_per_project))

    if not selected:
        return scored.iloc[0:0].copy()

    return pd.concat(selected, ignore_index=False).sort_values(
        ["retrieval_score", "project_id"], ascending=[False, True]
    ).copy()


def retrieval_coverage(results, filtered):
    return {
        "eligible_projects": filtered["project_id"].nunique(),
        "represented_projects": results["project_id"].nunique(),
        "eligible_pad_projects": filtered.loc[filtered["doc_type"] == "PAD", "project_id"].nunique(),
        "eligible_icr_projects": filtered.loc[filtered["doc_type"] == "ICR", "project_id"].nunique(),
        "retrieved_pad_projects": results.loc[results["doc_type"] == "PAD", "project_id"].nunique(),
        "retrieved_icr_projects": results.loc[results["doc_type"] == "ICR", "project_id"].nunique(),
    }


def build_context(results):
    blocks = []
    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        page = "" if pd.isna(row.get("page_number")) else row.get("page_number")
        section = "" if pd.isna(row.get("section_label")) else row.get("section_label")
        heading = "" if pd.isna(row.get("heading_text")) else row.get("heading_text")
        blocks.append(
            f"""SOURCE {rank}
Project: {row['project_id']}
Document: {row['doc_type']}
Document ID: {row['document_id']}
Page: {page}
Section: {section}
Heading: {heading}
Retrieval score: {row['retrieval_score']:.4f}
Text:
{row['text_clean']}
"""
        )
    return "\n\n".join(blocks)


def build_archetype_context(project_ids, archetypes):
    subset = archetypes[archetypes["project_id"].isin(project_ids)].copy()
    if subset.empty:
        return "No archetype metadata available."
    return subset.to_string(index=False)


def build_coverage_context(coverage):
    return f"""Retrieval coverage:
- Projects eligible after filters: {coverage['eligible_projects']}
- Projects represented in retrieved evidence: {coverage['represented_projects']}
- Projects with PAD evidence retrieved: {coverage['retrieved_pad_projects']} of {coverage['eligible_pad_projects']} eligible PAD projects
- Projects with ICR evidence retrieved: {coverage['retrieved_icr_projects']} of {coverage['eligible_icr_projects']} eligible ICR projects
"""


def ask_anthropic(api_key, model, query, results, archetypes, coverage):
    client = Anthropic(api_key=api_key)
    evidence = build_context(results)
    archetype_context = build_archetype_context(
        results["project_id"].dropna().unique().tolist(), archetypes
    )
    coverage_context = build_coverage_context(coverage)

    system_prompt = """You are an analytical assistant for a World Bank health infrastructure capstone project.

Use only the retrieved PAD/ICR evidence and supplied project metadata.

Rules:
- Do not invent facts.
- Distinguish PAD appraisal-stage evidence from ICR completion-stage evidence.
- Treat clusters and taxonomy scores as exploratory analytical metadata, not causal evidence.
- Never say a project "lacks documentation" merely because it was not retrieved.
- Describe retrieval gaps precisely as "not represented in the retrieved evidence for this query."
- Do not generalize to the full portfolio unless retrieval coverage supports that claim.
- If PAD evidence is present but ICR evidence is absent or incomplete, say that outcome conclusions are limited for this query.
- If both PAD and ICR are available, distinguish planned design from implementation/completion evidence.
- Cite substantive claims inline as [Project ID, PAD/ICR, p. X].
- Keep the response concise, analytical, and sponsor-facing.
"""

    user_prompt = f"""QUESTION
{query}

RETRIEVAL COVERAGE
{coverage_context}

PROJECT ARCHETYPE METADATA
{archetype_context}

RETRIEVED EVIDENCE
{evidence}

Provide:
1. A direct answer.
2. The main evidence-backed patterns.
3. Any important caveat or limitation.

When discussing limitations, distinguish:
- limitations of the underlying corpus, and
- limitations of what was retrieved for this specific query.
"""

    response = client.messages.create(
        model=model,
        max_tokens=1600,
        temperature=0.1,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return "".join(block.text for block in response.content if hasattr(block, "text"))


st.sidebar.markdown("## Health Infrastructure Intelligence")
st.sidebar.caption("World Bank Capstone II")
st.sidebar.markdown("### Data")

default_drive_value = ""
try:
    default_drive_value = st.secrets.get("RAG_GDRIVE_FILE_ID", "")
except Exception:
    pass

gdrive_value = st.sidebar.text_input(
    "Google Drive RAG file ID or share URL",
    value=default_drive_value,
    placeholder="Paste file ID or Drive share URL",
    help="The file must be accessible to the deployed app. For a prototype, set Google Drive sharing to anyone with the link.",
)

st.sidebar.markdown("### Anthropic")
api_key = st.sidebar.text_input(
    "Anthropic API key", type="password", placeholder="sk-ant-...",
    help="Used only for this browser/app session unless you configure Streamlit secrets.",
)
model = st.sidebar.selectbox(
    "Claude model", ["claude-sonnet-4-5", "claude-haiku-4-5"], index=0
)

try:
    archetypes = load_archetypes()
except Exception as exc:
    st.error(f"Could not load archetype data: {exc}")
    st.stop()

if not gdrive_value:
    st.title("World Bank Health Infrastructure Portfolio Explorer")
    st.info("Enter the Google Drive file ID or share URL for `world_bank_rag_serving.csv` in the sidebar.")
    st.stop()

try:
    with st.spinner("Loading RAG corpus from Google Drive..."):
        rag = download_and_load_rag(gdrive_value)
except Exception as exc:
    st.error(
        "Could not load the RAG corpus from Google Drive.\n\n"
        f"Error: {exc}\n\n"
        "Check that the file ID/share URL is correct and that the file is accessible."
    )
    st.stop()

st.sidebar.markdown("### Filters")
project = st.sidebar.selectbox(
    "Project", ["All projects"] + sorted(rag["project_id"].dropna().unique().tolist())
)
doc_type = st.sidebar.selectbox("Document type", ["PAD + ICR", "PAD only", "ICR only"])
archetype = st.sidebar.selectbox(
    "Project archetype", ["All archetypes"] + sorted(archetypes["archetype"].dropna().unique().tolist())
)
topic = st.sidebar.selectbox(
    "Sponsor dimension",
    [
        "All sponsor dimensions", "Physical Infrastructure", "DRM / Facility Resilience",
        "Health-System Resilience", "Hazard Specificity", "Climate Adaptation", "Lifelines / Utilities",
    ],
)

st.sidebar.markdown("### Retrieval")
if project == "All projects":
    retrieval_mode = st.sidebar.radio(
        "Portfolio retrieval mode", ["Project-stratified", "Global top-k"], index=0,
        help="Project-stratified retrieval ensures broad project coverage. Global top-k may be dominated by a few projects.",
    )
else:
    retrieval_mode = "Global top-k"

if retrieval_mode == "Project-stratified":
    chunks_per_project = st.sidebar.slider(
        "Chunks per project", min_value=1, max_value=3, value=1,
        help="With PAD / ICR balancing enabled, this is the number retrieved per project per document type.",
    )
    balance_pad_icr = st.sidebar.checkbox(
        "Balance PAD / ICR evidence", value=(doc_type == "PAD + ICR"), disabled=(doc_type != "PAD + ICR"),
        help="For portfolio-wide PAD + ICR queries, retrieve evidence from both document types where available.",
    )
else:
    top_k = st.sidebar.slider("Retrieved chunks", min_value=4, max_value=30, value=8)

st.sidebar.divider()
st.sidebar.caption(
    f"{rag['project_id'].nunique()} projects • {rag['document_id'].nunique()} documents • {len(rag):,} chunks"
)

st.title("World Bank Health Infrastructure Portfolio Explorer")
st.caption("Sponsor-guided portfolio analysis with traceable PAD/ICR evidence")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Projects", rag["project_id"].nunique())
m2.metric("PAD / ICR Documents", rag["document_id"].nunique())
m3.metric("RAG Chunks", f"{len(rag):,}")
m4.metric("Project Archetypes", archetypes["archetype"].nunique())

prompt_options = {
    "Resilience Drivers": "Which infrastructure investment characteristics are associated with stronger resilience to disasters and shocks?",
    "Top DRM Projects": "Which projects show the strongest DRM and facility resilience profile, and what evidence supports that classification?",
    "PAD vs ICR Change": "How do PADs and ICRs differ in their emphasis on resilience, preparedness, hazards, and climate adaptation?",
    "Compare Archetypes": "What distinguishes the Hazard and Climate Resilience Infrastructure archetype from the General and Mixed Infrastructure archetype?",
    "Integrated Resilience": "Which projects combine physical infrastructure with climate adaptation, hazard-specific design, and supporting utility systems?",
    "Investigate Specialized Subgroup": "Why are P166783 and P173800 classified as Resilience and Preparedness Intensive, and is the signal stronger in the PAD or ICR?",
    "Project Evolution": "Which projects appear to have broadened or shifted their resilience focus between appraisal and completion?",
    "Lifelines and Utilities": "What evidence suggests that utilities such as power, water, sanitation, or drainage contribute to health facility resilience?",
}

selected_prompt = st.selectbox("Pre-configured query", ["Custom question"] + list(prompt_options.keys()))
query = st.text_area(
    "Ask the portfolio",
    value=(prompt_options[selected_prompt] if selected_prompt != "Custom question" else ""),
    height=100,
    placeholder="Ask a question about the matched PAD/ICR portfolio...",
)
run = st.button("Run query", type="primary")

with st.expander("Project archetype overview", expanded=False):
    overview = (
        archetypes.groupby(["cluster", "archetype"]).size().reset_index(name="projects").sort_values("cluster")
    )
    st.dataframe(overview, use_container_width=True, hide_index=True)

if run:
    if not query.strip():
        st.warning("Enter a question or select a pre-configured prompt.")
        st.stop()

    filtered = filter_corpus(rag, project, doc_type, archetype, topic, archetypes)
    if filtered.empty:
        st.warning("No chunks match the current filters.")
        st.stop()

    if retrieval_mode == "Project-stratified":
        results = retrieve_stratified(
            query=query,
            df=filtered,
            chunks_per_project=chunks_per_project,
            balance_pad_icr=balance_pad_icr,
        )
    else:
        results = retrieve_global(query=query, df=filtered, top_k=top_k)

    coverage = retrieval_coverage(results, filtered)

    st.markdown("### Retrieval Coverage")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Projects represented", f"{coverage['represented_projects']} / {coverage['eligible_projects']}")
    c2.metric("PAD projects represented", f"{coverage['retrieved_pad_projects']} / {coverage['eligible_pad_projects']}")
    c3.metric("ICR projects represented", f"{coverage['retrieved_icr_projects']} / {coverage['eligible_icr_projects']}")
    c4.metric("Evidence chunks", len(results))

    if coverage["represented_projects"] < coverage["eligible_projects"]:
        st.warning(
            "This response does not represent every project remaining after the current filters. "
            "Projects not represented here still exist in the underlying corpus; they were not surfaced by this retrieval result."
        )

    # --------------------------------------------------------
    # Answer first
    # --------------------------------------------------------

    st.markdown("### Answer")

    if results.empty:
        st.warning("No evidence was retrieved.")
        st.stop()

    if not api_key:
        st.info(
            "Enter an Anthropic API key in the sidebar to generate a grounded answer. "
            "Retrieved evidence is available below."
        )
    else:
        try:
            with st.spinner("Generating grounded synthesis..."):
                answer = ask_anthropic(
                    api_key=api_key,
                    model=model,
                    query=query,
                    results=results,
                    archetypes=archetypes,
                    coverage=coverage,
                )
            st.markdown(answer)
        except Exception as exc:
            st.error(f"Anthropic request failed: {exc}")

    # --------------------------------------------------------
    # Supporting evidence second
    # --------------------------------------------------------

    with st.expander(
        f"Supporting Evidence ({len(results)} passages)",
        expanded=False,
    ):
        if results["retrieval_score"].max() <= 0:
            st.info(
                "The lexical retriever found no strong term overlap. "
                "Broaden the filters or rephrase the question."
            )

        for _, row in results.iterrows():
            page = "" if pd.isna(row.get("page_number")) else row.get("page_number")
            section = "" if pd.isna(row.get("section_label")) else row.get("section_label")
            heading = "" if pd.isna(row.get("heading_text")) else row.get("heading_text")
            st.markdown(
                f"""
                <div class="evidence-card">
                    <div class="muted"><b>{row['project_id']}</b> · {row['doc_type']} · Page {page} · {section}</div>
                    <div style="margin-top:6px;"><b>{heading}</b></div>
                    <div style="margin-top:8px;">{row['text_clean']}</div>
                    <div class="muted" style="margin-top:8px;">Retrieval score: {row['retrieval_score']:.3f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

st.divider()
st.caption(
    "Retrieval uses TF-IDF over cleaned chunk text. Portfolio-wide queries default to project-stratified retrieval so the answer is not dominated by a small number of projects."
)
