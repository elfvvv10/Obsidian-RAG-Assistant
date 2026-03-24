importance: high
source_type: system
domain: system_design

---
type: roadmap
domain: system_design
project: obsidian_rag_assistant
version: v1
status: active
focus: ingestion_driven_intelligence
priority: high
---

# Ingestion-Driven Music Collaborator — Roadmap

## Purpose

Transform the Obsidian RAG Assistant into a knowledge-shaped collaborator where:

- The assistant reflects ingested knowledge, not model priors
- Retrieval + context drive intelligence
- The model acts as a reasoning and synthesis layer, not the source of truth

---

## Core Principle

The system should increasingly reflect what the user has ingested, not what the model already knows.

---

# Layer 1 — Ingestion as Primary Intelligence

## Goal
Make ingestion the main source of intelligence

---

## Phase 1.1 — Structured Ingestion

### Required Metadata

source_type: tutorial | reference | track_breakdown | personal_note | research | workflow
domain: bassline | arrangement | sound_design | mixing | groove | harmony
genre: progressive_house | techno | melodic_house
importance: high | medium | low

### Structural Extraction

- Steps (production workflows)
- Parameters (cutoff, envelopes, etc.)
- Comparisons (technique A vs B)
- Named techniques

---

## Phase 1.2 — Music-Aware Chunking

Replace generic token chunking with:

- semantic blocks
- preserved structure (headings, lists, workflows)

### Chunk Metadata

chunk_type: technique | workflow | explanation
section:
has_steps: true

---

# Layer 2 — Retrieval That Prioritizes User Knowledge

## Goal
Favor curated knowledge over model knowledge

---

## Phase 2.1 — Weighted Retrieval

Scoring should include:

- semantic similarity
- tag match (genre/domain)
- importance weighting
- track context relevance

---

## Phase 2.2 — Query Rewriting (Critical)

Transform user queries using:

- track context
- workflow type

Example:

“help with bassline” →
“progressive house bassline groove rhythm low-end movement techniques”

---

## Phase 2.3 — Linked Knowledge Expansion

- Expand from strong matches
- Limit depth to avoid noise

---

## Phase 2.4 — Retrieval Modes Evolution

Replace:

local_only / hybrid / auto

With:

- grounded_strict
- grounded_plus_inference
- exploratory

---

# Layer 3 — Grounded Collaborator Behavior

## Goal
Assistant behaves like a producer collaborator

---

## Phase 3.1 — Prompt Policy

Shift to:

- prioritize retrieved knowledge
- synthesize across sources
- reference track context
- distinguish grounded knowledge vs inference

---

## Phase 3.2 — Workflow-Aware Reasoning

Sound Design Mode:
- parameter suggestions
- modulation strategies

Arrangement Mode:
- structure critique
- transition analysis

Ideation Mode:
- grounded variation generation

---

## Phase 3.3 — Anti-Generic Guardrails

If retrieval is weak:

- say so
- ask for clarification
- avoid generic filler

---

# Layer 4 — Persistent Track Context

## Goal
Anchor all interactions to a track session

---

## Phase 4.1 — Track Context Object

track_name:
genre:
bpm:
key:
vibe:
references: []
current_section:
known_issues: []
goals: []

---

## Phase 4.2 — System Integration

Track context should influence:

- query rewriting
- retrieval ranking
- prompt generation
- saved outputs

---

## Phase 4.3 — Session Memory (Future)

- evolving track state
- persistent assistant awareness

---

# Layer 5 — Evaluation System

## Goal
Measure grounded usefulness, not just output quality

---

## Benchmark Setup

Corpus:
- 10–20 curated notes
- sound design, arrangement, critique, workflow

Test Prompts:
- bassline ideation
- arrangement critique
- sound design refinement

---

## Evaluation Metrics

- groundedness — uses ingested knowledge
- specificity — avoids generic answers
- usefulness — actionable output
- style fit — genre-aware
- collaboration — feels like a producer

---

## Model Comparison (Hermes vs DeepSeek)

Test with:

- same ingestion
- same retrieval
- same prompts

Compare:

- grounded reasoning
- context adherence
- production usefulness

---

# Execution Plan

## Immediate (Do First)

1. Structured ingestion
2. Music-aware chunking
3. Query rewriting

---

## Next

4. Weighted retrieval
5. Track context system

---

## Later

6. Workflow refinement
7. Evaluation harness
8. Model comparison

---

# Key Insight

You are not building a chatbot with memory.

You are building a knowledge-shaped production collaborator.

Where:

- ingestion = memory
- retrieval = recall
- model = reasoning
- track context = perspective

---

# Notes

- Do not over-invest in prompt tuning early
- Focus on ingestion + retrieval quality
- Test with real curated material
