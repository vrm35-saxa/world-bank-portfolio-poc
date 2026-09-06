import re
from pathlib import Path

import gdown
import numpy as np
import pandas as pd
import streamlit as st
from anthropic import Anthropic
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ------------------------------------------------------------
# App configuration
# ------------------------------------------------------------

st.set_page_config(
    page_title="World Bank Health Infrastructure Portfolio Explorer",
    page_icon="🌐",
    layout="wide",
)

ARCHETYPE_FILE = "world_bank_project_archetypes_k3.csv"
LOCAL_RAG_CACHE = "/tmp/world_bank_rag_serving.csv"


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1117;
        color: #e8eef5;
    }

    [data-testid="stSidebar"] {
        background-color: #0d141c;
        border-right: 1px solid #243241;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background: #111a23;
        border: 1px solid #243241;
        border-radius: 8px;
        padding: 12px;
    }

    div[data-testid="stExpander"] {
        border: 1px solid #243241;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Google Drive helpers
# ------------------------------------------------------------

def extract_gdrive_file_id(value: str) -> str:
    """
    Accept either:
    - a raw Google Drive file ID
    - a normal Google Drive share URL
    """
    value = value.strip()

    if not value:
        return ""

    # Raw file ID
    if "/" not in value and "?" not in value:
        return value

    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"[?&]id=([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, value)

        if match:
            return match.group(1)

    raise ValueError(
        "Could not extract a Google Drive file ID from that value."
    )


@st.cache_data(show_spinner=False)
def download_and_load_rag(gdrive_value: str):
    """
    Download world_bank_rag_serving.csv from Google Drive
    and load it into a DataFrame.
    """
    file_id = extract_gdrive_file_id(
        gdrive_value
    )

    if not file_id:
        raise ValueError(
            "Google Drive file ID or share URL is required."
        )

    gdown.download(
        id=file_id,
        output=LOCAL_RAG_CACHE,
        quiet=True,
    )

    rag = pd.read_csv(
        LOCAL_RAG_CACHE
    )

    required = {
        "chunk_id",
        "project_id",
        "document_id",
        "doc_type",
        "page_number",
        "section_label",
        "heading_text",
        "text_clean",
    }

    missing = required - set(
        rag.columns
    )

    if missing:
        raise ValueError(
            f"RAG file is missing columns: {sorted(missing)}"
        )

    return rag


# ------------------------------------------------------------
# Archetype loading
# ------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_archetypes():
    path = Path(
        ARCHETYPE_FILE
    )

    if not path.exists():
        raise FileNotFoundError(
            f"`{ARCHETYPE_FILE}` was not found in the repository."
        )

    archetypes = pd.read_csv(
        path
    )

    required = {
        "project_id",
        "cluster",
        "archetype",
    }

    missing = required - set(
        archetypes.columns
    )

    if missing:
        raise ValueError(
            f"Archetype file is missing columns: {sorted(missing)}"
        )

    return archetypes


# ------------------------------------------------------------
# Filtering
# ------------------------------------------------------------

def filter_corpus(
    df,
    project,
    doc_type,
    archetype,
    topic,
    archetypes_df,
):
    out = df.copy()

    # Project filter
    if project != "All projects":
        out = out[
            out["project_id"] == project
        ]

    # Document type filter
    if doc_type != "PAD + ICR":
        target_doc_type = doc_type.replace(
            " only",
            "",
        )

        out = out[
            out["doc_type"] == target_doc_type
        ]

    # Archetype filter
    if archetype != "All archetypes":

        project_ids = (
            archetypes_df.loc[
                archetypes_df["archetype"] == archetype,
                "project_id",
            ]
            .tolist()
        )

        out = out[
            out["project_id"].isin(
                project_ids
            )
        ]

    # Sponsor taxonomy / concept filter
    topic_map = {

        "Physical Infrastructure": [
            "facility",
            "facilities",
            "hospital",
            "clinic",
            "laboratory",
            "construction",
            "rehabilitation",
            "renovation",
            "retrofit",
            "upgrade",
            "rebuild",
        ],

        "DRM / Facility Resilience": [
            "resilience",
            "resilient",
            "disaster risk management",
            "withstand",
            "recover",
            "shock",
        ],

        "Health-System Resilience": [
            "health system",
            "preparedness",
            "capacity building",
            "response capacity",
            "maintain core functions",
        ],

        "Hazard Specificity": [
            "flood",
            "cyclone",
            "hurricane",
            "typhoon",
            "earthquake",
            "drought",
            "heat",
            "landslide",
            "tsunami",
            "storm surge",
        ],

        "Climate Adaptation": [
            "climate",
            "temperature",
            "precipitation",
            "sea level",
        ],

        "Lifelines / Utilities": [
            "water",
            "sanitation",
            "electricity",
            "solar",
            "generator",
            "drainage",
            "power",
        ],
    }

    if topic != "All sponsor dimensions":

        terms = topic_map.get(
            topic,
            [],
        )

        pattern = "|".join(
            re.escape(term)
            for term in terms
        )

        if pattern:

            out = out[
                out["text_clean"]
                .fillna("")
                .str.contains(
                    pattern,
                    case=False,
                    regex=True,
                )
            ]

    return out


# ------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------

def retrieve_chunks(
    query,
    df,
    top_k=8,
):
    """
    TF-IDF retrieval over cleaned chunk text.

    Stored embeddings are not used because the original
    embedding model has not yet been identified.
    """

    if df.empty:
        return df.assign(
            retrieval_score=pd.Series(
                dtype=float
            )
        )

    texts = (
        df["text_clean"]
        .fillna("")
        .astype(str)
    )

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
        sublinear_tf=True,
    )

    matrix = vectorizer.fit_transform(
        texts
    )

    query_vec = vectorizer.transform(
        [query]
    )

    scores = cosine_similarity(
        query_vec,
        matrix,
    ).ravel()

    n = min(
        top_k,
        len(df),
    )

    top_idx = np.argsort(
        scores
    )[::-1][:n]

    results = df.iloc[
        top_idx
    ].copy()

    results[
        "retrieval_score"
    ] = scores[
        top_idx
    ]

    return results


# ------------------------------------------------------------
# Anthropic context builders
# ------------------------------------------------------------

def build_context(
    results
):
    blocks = []

    for rank, (_, row) in enumerate(
        results.iterrows(),
        start=1,
    ):

        page = (
            ""
            if pd.isna(
                row.get("page_number")
            )
            else row.get("page_number")
        )

        section = (
            ""
            if pd.isna(
                row.get("section_label")
            )
            else row.get("section_label")
        )

        heading = (
            ""
            if pd.isna(
                row.get("heading_text")
            )
            else row.get("heading_text")
        )

        blocks.append(
            f"""
SOURCE {rank}

Project: {row['project_id']}
Document: {row['doc_type']}
Document ID: {row['document_id']}
Page: {page}
Section: {section}
Heading: {heading}

Text:
{row['text_clean']}
"""
        )

    return "\n\n".join(
        blocks
    )


def build_archetype_context(
    project_ids,
    archetypes,
):
    subset = archetypes[
        archetypes["project_id"].isin(
            project_ids
        )
    ].copy()

    if subset.empty:
        return (
            "No archetype metadata available."
        )

    return subset.to_string(
        index=False
    )


# ------------------------------------------------------------
# Anthropic call
# ------------------------------------------------------------

def ask_anthropic(
    api_key,
    model,
    query,
    results,
    archetypes,
):
    client = Anthropic(
        api_key=api_key
    )

    evidence = build_context(
        results
    )

    archetype_context = (
        build_archetype_context(
            results[
                "project_id"
            ]
            .dropna()
            .unique()
            .tolist(),
            archetypes,
        )
    )

    system_prompt = """
You are an analytical assistant for a World Bank
health infrastructure capstone project.

Use only the retrieved PAD/ICR evidence and supplied
project metadata.

Rules:

- Do not invent facts.
- Distinguish PAD appraisal-stage evidence from
  ICR completion-stage evidence.
- Treat project archetypes and taxonomy scores as
  exploratory analytical metadata, not causal evidence.
- Do not claim that a project characteristic caused
  a project outcome unless the evidence explicitly supports it.
- If the retrieved evidence is insufficient, say so.
- Cite substantive claims inline as:
  [Project ID, PAD/ICR, p. X]
- Keep the response concise, analytical, and sponsor-facing.
"""

    user_prompt = f"""
QUESTION

{query}


PROJECT ARCHETYPE METADATA

{archetype_context}


RETRIEVED PAD/ICR EVIDENCE

{evidence}


Provide:

1. A direct answer to the question.
2. The main evidence-backed patterns.
3. Any important caveat or limitation.
"""

    response = client.messages.create(
        model=model,
        max_tokens=1200,
        temperature=0.1,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    )

    return "".join(
        block.text
        for block in response.content
        if hasattr(
            block,
            "text",
        )
    )


# ------------------------------------------------------------
# Evidence renderer
# ------------------------------------------------------------

def render_evidence(
    results
):
    """
    Render evidence using native Streamlit components
    instead of raw HTML.
    """

    if results.empty:
        st.info(
            "No evidence was retrieved."
        )
        return

    if (
        results[
            "retrieval_score"
        ].max()
        <= 0
    ):
        st.info(
            "The lexical retriever found no strong term overlap. "
            "Broaden the filters or rephrase the question."
        )

    for _, row in results.iterrows():

        page = (
            ""
            if pd.isna(
                row.get("page_number")
            )
            else row.get("page_number")
        )

        section = (
            ""
            if pd.isna(
                row.get("section_label")
            )
            else row.get("section_label")
        )

        heading = (
            ""
            if pd.isna(
                row.get("heading_text")
            )
            else row.get("heading_text")
        )

        with st.container(
            border=True
        ):

            st.caption(
                f"{row['project_id']} · "
                f"{row['doc_type']} · "
                f"Page {page} · "
                f"{section}"
            )

            if heading:
                st.markdown(
                    f"**{heading}**"
                )

            st.write(
                row["text_clean"]
            )

            st.caption(
                "Retrieval score: "
                f"{row['retrieval_score']:.3f}"
            )


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.markdown(
    "## Health Infrastructure Intelligence"
)

st.sidebar.caption(
    "World Bank Capstone II"
)


# ------------------------------------------------------------
# Sidebar: Data
# ------------------------------------------------------------

st.sidebar.markdown(
    "### Data"
)

default_drive_value = ""

try:
    default_drive_value = (
        st.secrets.get(
            "RAG_GDRIVE_FILE_ID",
            "",
        )
    )
except Exception:
    pass


gdrive_value = (
    st.sidebar.text_input(
        "Google Drive RAG file ID or share URL",
        value=default_drive_value,
        placeholder=(
            "Paste file ID or Drive share URL"
        ),
        help=(
            "Google Drive source for "
            "world_bank_rag_serving.csv"
        ),
    )
)


# ------------------------------------------------------------
# Sidebar: Anthropic
# ------------------------------------------------------------

st.sidebar.markdown(
    "### Anthropic"
)

api_key = (
    st.sidebar.text_input(
        "Anthropic API key",
        type="password",
        placeholder="sk-ant-...",
        help=(
            "Used only for the active app session. "
            "The key is not stored in the repository."
        ),
    )
)


model = (
    st.sidebar.selectbox(
        "Claude model",
        [
            "claude-sonnet-4-5",
            "claude-haiku-4-5",
        ],
        index=0,
    )
)


# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

try:

    archetypes = (
        load_archetypes()
    )

except Exception as exc:

    st.error(
        f"Could not load archetype data: {exc}"
    )

    st.stop()


if not gdrive_value:

    st.title(
        "World Bank Health Infrastructure Portfolio Explorer"
    )

    st.info(
        "Enter the Google Drive file ID or share URL "
        "for `world_bank_rag_serving.csv` in the sidebar."
    )

    st.stop()


try:

    with st.spinner(
        "Loading RAG corpus from Google Drive..."
    ):

        rag = (
            download_and_load_rag(
                gdrive_value
            )
        )

except Exception as exc:

    st.error(
        "Could not load the RAG corpus from Google Drive.\n\n"
        f"Error: {exc}\n\n"
        "Check that the file ID/share URL is correct "
        "and that the file is accessible."
    )

    st.stop()


# ------------------------------------------------------------
# Sidebar: Filters
# ------------------------------------------------------------

st.sidebar.markdown(
    "### Filters"
)


project = (
    st.sidebar.selectbox(
        "Project",
        [
            "All projects"
        ]
        + sorted(
            rag[
                "project_id"
            ]
            .dropna()
            .unique()
            .tolist()
        ),
    )
)


doc_type = (
    st.sidebar.selectbox(
        "Document type",
        [
            "PAD + ICR",
            "PAD only",
            "ICR only",
        ],
    )
)


archetype = (
    st.sidebar.selectbox(
        "Project archetype",
        [
            "All archetypes"
        ]
        + sorted(
            archetypes[
                "archetype"
            ]
            .dropna()
            .unique()
            .tolist()
        ),
    )
)


topic = (
    st.sidebar.selectbox(
        "Sponsor dimension",
        [
            "All sponsor dimensions",
            "Physical Infrastructure",
            "DRM / Facility Resilience",
            "Health-System Resilience",
            "Hazard Specificity",
            "Climate Adaptation",
            "Lifelines / Utilities",
        ],
    )
)


top_k = (
    st.sidebar.slider(
        "Retrieved chunks",
        min_value=4,
        max_value=15,
        value=8,
    )
)


st.sidebar.divider()


st.sidebar.caption(
    f"{rag['project_id'].nunique()} projects • "
    f"{rag['document_id'].nunique()} documents • "
    f"{len(rag):,} chunks"
)


# ------------------------------------------------------------
# Main UI
# ------------------------------------------------------------

st.title(
    "World Bank Health Infrastructure Portfolio Explorer"
)

st.caption(
    "Sponsor-guided portfolio analysis "
    "with traceable PAD/ICR evidence"
)


# ------------------------------------------------------------
# KPI row
# ------------------------------------------------------------

m1, m2, m3, m4 = st.columns(
    4
)


m1.metric(
    "Projects",
    rag[
        "project_id"
    ].nunique(),
)


m2.metric(
    "PAD / ICR Documents",
    rag[
        "document_id"
    ].nunique(),
)


m3.metric(
    "RAG Chunks",
    f"{len(rag):,}",
)


m4.metric(
    "Project Archetypes",
    archetypes[
        "archetype"
    ].nunique(),
)


# ------------------------------------------------------------
# Suggested queries
# ------------------------------------------------------------

prompt_options = {

    "Resilience Drivers":
        "Which infrastructure investment characteristics "
        "are associated with stronger resilience to "
        "disasters and shocks?",

    "Top DRM Projects":
        "Which projects show the strongest DRM and "
        "facility resilience profile, and what evidence "
        "supports that classification?",

    "PAD vs ICR Change":
        "How do PADs and ICRs differ in their emphasis "
        "on resilience, preparedness, hazards, and "
        "climate adaptation?",

    "Compare Archetypes":
        "What distinguishes the Hazard and Climate "
        "Resilience Infrastructure archetype from "
        "the General and Mixed Infrastructure archetype?",

    "Integrated Resilience":
        "Which projects combine physical infrastructure "
        "with climate adaptation, hazard-specific design, "
        "and supporting utility systems?",

    "Investigate Specialized Subgroup":
        "Why are P166783 and P173800 classified as "
        "Resilience and Preparedness Intensive, and "
        "is the signal stronger in the PAD or ICR?",

    "Project Evolution":
        "Which projects appear to have broadened or "
        "shifted their resilience focus between "
        "appraisal and completion?",

    "Lifelines and Utilities":
        "What evidence suggests that utilities such "
        "as power, water, sanitation, or drainage "
        "contribute to health facility resilience?",
}


selected_prompt = (
    st.selectbox(
        "Pre-configured query",
        [
            "Custom question"
        ]
        + list(
            prompt_options.keys()
        ),
    )
)


query = (
    st.text_area(
        "Ask the portfolio",
        value=(
            prompt_options[
                selected_prompt
            ]
            if selected_prompt
            != "Custom question"
            else ""
        ),
        height=100,
        placeholder=(
            "Ask a question about the matched "
            "PAD/ICR portfolio..."
        ),
    )
)


run = (
    st.button(
        "Run query",
        type="primary",
    )
)


# ------------------------------------------------------------
# Archetype overview
# ------------------------------------------------------------

with st.expander(
    "Project archetype overview",
    expanded=False,
):

    overview = (
        archetypes
        .groupby(
            [
                "cluster",
                "archetype",
            ]
        )
        .size()
        .reset_index(
            name="projects"
        )
        .sort_values(
            "cluster"
        )
    )

    st.dataframe(
        overview,
        use_container_width=True,
        hide_index=True,
    )


# ------------------------------------------------------------
# Query execution
# ------------------------------------------------------------

if run:

    if not query.strip():

        st.warning(
            "Enter a question or select "
            "a pre-configured prompt."
        )

        st.stop()


    # Apply filters
    filtered = (
        filter_corpus(
            rag,
            project,
            doc_type,
            archetype,
            topic,
            archetypes,
        )
    )


    if filtered.empty:

        st.warning(
            "No chunks match the current filters."
        )

        st.stop()


    # Retrieve evidence
    results = (
        retrieve_chunks(
            query=query,
            df=filtered,
            top_k=top_k,
        )
    )


    # --------------------------------------------------------
    # API KEY PRESENT:
    # Answer first
    # --------------------------------------------------------

    if api_key:

        st.markdown(
            "## Answer"
        )

        try:

            with st.spinner(
                "Analyzing retrieved PAD/ICR evidence..."
            ):

                answer = (
                    ask_anthropic(
                        api_key=api_key,
                        model=model,
                        query=query,
                        results=results,
                        archetypes=archetypes,
                    )
                )

            st.markdown(
                answer
            )

        except Exception as exc:

            st.error(
                f"Anthropic request failed: {exc}"
            )


        # Evidence underneath answer
        with st.expander(
            f"Supporting Evidence ({len(results)} passages)",
            expanded=False,
        ):

            render_evidence(
                results
            )


    # --------------------------------------------------------
    # NO API KEY:
    # Evidence-only mode
    # --------------------------------------------------------

    else:

        st.markdown(
            "## Retrieved Evidence"
        )

        st.info(
            "Retrieval-only mode. "
            "Enter an Anthropic API key in the sidebar "
            "to generate a synthesized answer."
        )

        render_evidence(
            results
        )


    # --------------------------------------------------------
    # Project / archetype context
    # --------------------------------------------------------

    relevant_project_ids = (
        results[
            "project_id"
        ]
        .dropna()
        .unique()
        .tolist()
    )


    relevant_archetypes = (
        archetypes[
            archetypes[
                "project_id"
            ].isin(
                relevant_project_ids
            )
        ]
        .copy()
    )


    with st.expander(
        "Project and Archetype Context",
        expanded=False,
    ):

        if relevant_archetypes.empty:

            st.caption(
                "No archetype metadata is available "
                "for the retrieved projects."
            )

        else:

            st.dataframe(
                relevant_archetypes,
                use_container_width=True,
                hide_index=True,
            )


# ------------------------------------------------------------
# Footer
# ------------------------------------------------------------

st.divider()

st.caption(
    "Retrieval currently uses TF-IDF over cleaned chunk text "
    "because the original model used to create the stored "
    "768-dimensional embeddings has not yet been identified."
)
