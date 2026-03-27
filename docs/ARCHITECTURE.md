# Architecture Reference

## Overview

Obsidian Track Collaborator is a local-first retrieval and collaboration system built around an Obsidian vault, local ChromaDB storage, and workflow-aware prompting. The same service layer powers both the CLI and Streamlit UI.

Main path:

```text
vault -> loader -> chunker -> embeddings -> ChromaDB -> retriever -> prompt service -> chat model -> answer/save-back
```

## Core Subsystems

- `main.py`: CLI entrypoint
- `streamlit_app.py`: Streamlit UI
- `config.py`: environment loading and validation
- `model_provider.py` and `model_clients.py`: provider-aware client construction and interfaces
- `services/query_service.py`: direct ask orchestration
- `services/research_service.py`: visible multi-step research flow
- `services/prompt_service.py`: answer policy, workflow-aware prompts, citation behavior, and track-aware prompt shaping
- `services/index_service.py`: indexing and rebuild flow
- `services/track_context_service.py`: YAML Track Context persistence
- `services/arrangement_service.py`: arrangement parsing and rendering
- `services/video_ingestion_service.py` and `services/webpage_ingestion_service.py`: external knowledge import pipelines

## Retrieval and Answering

- Notes are loaded from the configured vault and chunked for retrieval.
- Embeddings are stored in local ChromaDB.
- Retrieval can be `local_only`, `auto`, or `hybrid`.
- Retrieval scope can stay on curated knowledge or expand into working notes and saved outputs.
- Prompt construction is workflow-aware and can inject Track Context, arrangement cues, and recent session state.
- Answers preserve clear separation between local, saved, imported, and web evidence.

## Track Memory and Structure

Track memory is intentionally split into separate layers:

- `Track Context`: persistent YAML memory for track identity, known issues, goals, and sections
- `Arrangement notes`: structural description of sections, bars, and energy over time
- `Saved Outputs`: generated collaborator artifacts

Track Context update proposals are reviewable before they are applied. The CLI and UI both support proposal preview and explicit user approval.

## Workflows

The main collaboration workflows are:

- `General Ask`
- `Genre Fit Review`
- `Track Concept Critique`
- `Arrangement Planner`
- `Sound Design Brainstorm`
- `Research Session`

These workflows share the same trust boundaries:

- retrieval scope controls eligible local material
- retrieval mode controls web usage
- answer mode controls how far the model can reason beyond direct evidence
- evidence labels remain explicit

## Imports and Save-Back

- webpage and YouTube/video imports are converted into markdown notes for later retrieval
- generated outputs are saved into `Saved Outputs/`
- saved notes can carry Track Context summaries and reviewable Track Context update metadata

## Compatibility Notes

- YAML Track Context is the primary editable path
- legacy markdown `track_context.md` remains tolerated for backward compatibility
- newer structured systems are layered on top of the same core retrieval pipeline
