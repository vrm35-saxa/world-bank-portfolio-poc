# World Bank Health Infrastructure Portfolio Explorer

A Streamlit-based analytical prototype for exploring World Bank health infrastructure projects using matched Project Appraisal Documents (PADs), Implementation Completion and Results Reports (ICRs), sponsor-guided taxonomy features, project archetypes, and retrieval-augmented question answering.

## Project Scope

The current analytical sample contains:

- 27 World Bank projects
- 54 matched PAD/ICR documents
- 5,144 cleaned analytical text chunks
- 3 project archetypes from a K=3 clustering solution

The analysis focuses on sponsor-defined dimensions including:

- Physical health infrastructure
- DRM / facility resilience
- Health-system resilience
- Disaster / hazard specificity
- Climate impacts and adaptation
- Resilience interventions
- Supporting lifelines and utilities
- Capacity, preparedness, and accessibility

## App-Facing Data

The planned Streamlit application uses two primary data files.

### `world_bank_rag_serving.csv`

Cleaned chunk-level corpus containing:

- `chunk_id`
- `project_id`
- `document_id`
- `doc_type`
- `page_number`
- `section_label`
- `heading_text`
- `text_clean`
- `embedding`

Validated coverage:

- 5,144 chunks
- 27 projects
- 54 documents
- 768-dimensional stored embeddings
- No missing project IDs
- No missing cleaned text
- No missing embeddings in the validated file

### `world_bank_project_archetypes_k3.csv`

Project-level clustering and taxonomy table containing:

- `project_id`
- `cluster`
- `archetype`
- Eight sponsor-guided project-level taxonomy scores

Validated coverage:

- 27 rows
- 27 unique projects
- No duplicate project IDs
- No missing cluster assignments
- No missing archetype labels

## Current Project Archetypes

The K=3 solution produced three analytically distinct groups:

1. **Resilience and Preparedness Intensive**
   - 2 projects
   - Strongest emphasis on DRM/facility resilience, health-system resilience, and preparedness/accessibility

2. **General and Mixed Infrastructure**
   - 19 projects
   - Broad physical infrastructure activity without a strong specialization in the resilience dimensions

3. **Hazard and Climate Resilience Infrastructure**
   - 6 projects
   - Strongest emphasis on hazard specificity, climate adaptation, resilience interventions, and supporting lifelines/utilities

These labels are interpretive summaries of the cluster profiles, not causal classifications.

## Clustering Validation

The K=3 solution was evaluated using multiple checks:

- Silhouette score: approximately **0.253**
- Mean resampling Adjusted Rand Index (ARI): approximately **0.798**
- Median resampling ARI: approximately **0.914**
- K-Means vs. Ward hierarchical clustering ARI: approximately **0.863**

The clustering is reasonably stable overall, but the two-project resilience/preparedness cluster should be interpreted as a small specialized subgroup rather than a broad portfolio segment.

## Intended Application Features

The Streamlit interface is designed to support:

- Natural-language portfolio questions
- Pre-configured sponsor-oriented prompts
- Project filtering
- PAD vs. ICR filtering
- Archetype filtering
- Sponsor-taxonomy filtering
- Project archetype exploration
- PAD-to-ICR comparison
- Retrieval of source evidence with project, document type, page, and section metadata

Example prompts include:

- Which infrastructure investment characteristics are associated with stronger resilience to disasters and shocks?
- Which projects show the strongest DRM and facility-resilience profile?
- How do PADs and ICRs differ in their emphasis on resilience and preparedness?
- What distinguishes the Hazard and Climate Resilience Infrastructure archetype from the General and Mixed Infrastructure archetype?
- Which projects combine physical infrastructure with climate adaptation, hazard-specific design, and supporting utilities?
- Why are P166783 and P173800 classified as Resilience and Preparedness Intensive?

## Repository Structure

```text
world-bank-health-infrastructure-rag/
├── app.py
├── README.md
└── requirements.txt
```

The two app-facing data files may be stored outside GitHub, such as in Google Drive:

```text
world_bank_rag_serving.csv
world_bank_project_archetypes_k3.csv
```

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

## Important Retrieval Note

The stored chunk embeddings are 768-dimensional, but the original embedding model has not yet been identified.

That means the application should **not** embed a user query with an arbitrary model and compare it directly with the stored vectors. Valid vector similarity requires the query and corpus embeddings to be produced in the same embedding space.

Before production semantic retrieval, use one of these approaches:

1. identify and use the original embedding model, or
2. re-embed the cleaned RAG corpus and user queries with a known common model.

Until then, the stored chunk text and metadata remain usable for filtering, taxonomy-driven lookup, lexical retrieval, and interface development.

## Methodological Note

The application combines two different analytical layers:

- **Chunk-level corpus:** source evidence and retrieval
- **Project-level archetype table:** cluster metadata and sponsor-guided project comparison

Cluster labels and taxonomy scores should be used as analytical metadata, not as source evidence.

## Current Scope

The current primary analysis uses chunk-level data. A separate sentence-level analysis may be added later as a robustness and finer-grained evidence layer.

## Disclaimer

This application is an academic capstone analytical prototype. Cluster assignments, taxonomy scores, and retrieved text patterns are exploratory and should not be interpreted as causal estimates of project effectiveness, resilience, or development impact.
