---
type: system_direction
priority: critical
status: active
project: obsidian_track_collaborator
---

## When to Use This Document

This document should be consulted when:
- deciding what to build next
- reviewing pull requests
- evaluating new feature ideas
- debugging system behavior that feels "off direction"

If a change conflicts with this direction, this document takes precedence.

# Next Build Sequence — Plan Alignment Checkpoint

## Purpose

This note captures the current recommended build direction for Obsidian Track Collaborator after the latest repo review.

The project is progressing well, especially in:
- persistent Track Context
- collaborator-style prompting
- session-based chat flow
- workflow specialization
- reviewable Track Context updates

The main strategic risk is surface capability advancing faster than grounding capability.

In other words: the collaborator is becoming more convincing, but the system still needs to better ensure that intelligence comes from ingested knowledge + retrieval + track context, not mostly prompt sophistication.

---

## Current Strategic Assessment

### What is going well
- Track Context is now a real system feature, not just a roadmap concept
- The app is moving toward a session-aware collaborator
- Workflow modes are becoming more producer-specific
- The architecture is more serious and test-backed than an early prototype

### Main risk
The project may drift toward:

“feature-rich producer assistant with strong prompting”

instead of the intended target:

“knowledge-shaped track collaborator grounded in ingested material, retrieval, and persistent track state.”

---

## Core Direction

The next phase should focus on closing the gap between:

- collaborator surface
and
- ingestion / retrieval intelligence

This means the next major work should improve:
1. retrieval quality
2. grounding quality
3. track-linked persistence

more than UI expansion or additional prompt complexity.

---

# Priority 1 — Make Retrieval Track-Aware in Ranking, Not Just Rewriting

## Why this matters
Track Context already influences query rewriting, which is a strong start.

But that is not enough.

The roadmap direction requires retrieval to actually favor the most relevant knowledge based on the active track, not just expand the search query.

If this is not improved, the system will still rely too heavily on model priors and prompt instructions.

## Goal
Add explicit retrieval scoring that favors:
- semantic similarity
- genre match
- domain match
- importance weighting
- track context relevance
- possibly workflow relevance

## Desired outcome
When the user is working on a specific track, the system should increasingly pull back knowledge that fits:
- that genre
- that production problem
- that section/state of the track
- that creative goal

## Recommended implementation direction
- introduce weighted retrieval scoring after initial candidate search
- preserve semantic similarity, but blend with metadata-aware weighting
- use active Track Context fields such as:
  - genre
  - references
  - known issues
  - goals
  - current section
- optionally include workflow-aware weighting for critique / arrangement / sound design requests

## Success condition
The assistant’s answers should feel more specifically shaped by the user’s own ingested knowledge base and current track context, not just more cleverly phrased.

---

# Priority 2 — Consolidate Around One Primary Track Context System

## Why this matters
Track Context is now the center of the product.

That means the architecture should have a single clear source of truth.

If both YAML Track Context and legacy markdown context remain first-class, the system risks:
- duplicated pathways
- user confusion
- prompt branching complexity
- slower product clarity

## Goal
Make YAML Track Context the canonical system.

Legacy markdown context should only exist as:
- a migration bridge
- an import source
- a fallback for older projects

It should not continue as an equal long-term workflow path.

## Recommended implementation direction
- clearly designate YAML Track Context as canonical in code and UI
- reduce visibility of legacy markdown workflow context
- add migration/import helpers if needed
- update docs so the current expected workflow is unambiguous
- ensure new features only target the canonical Track Context path

## Success condition
There is one obvious answer to:
“Where does track state live?”

And that answer should be:
the YAML Track Context system

---

# Priority 3 — Convert Session Tasks into Persisted Per-Track Tasks

## Why this matters
The task system is valuable, but its real power only arrives when it becomes part of track progression over time.

Right now, session tasks are helpful as working memory.
The next step is to make them part of the track’s persistent collaborator loop.

This directly supports the roadmap direction of:
- session awareness
- assistant initiative
- project-stage understanding
- finishing tracks instead of just discussing them

## Goal
Persist tasks per track, not just per session.

## Recommended implementation direction
- store tasks alongside Track Context or in a closely linked track task file
- support:
  - open
  - done
  - deferred
  - priority
  - optional section/category
- allow the assistant to propose tasks from critiques or answers
- let users review and accept task additions
- surface current track tasks in the workspace automatically when a track is loaded

## Suggested minimum task schema
- id
- text
- status
- priority
- linked_section
- created_from
- created_at
- completed_at

## Success condition
The assistant can help move a track forward across multiple sessions, with persistent awareness of what still needs to be done.

---

# Priority 4 — Keep Prompt Work in Support Role, Not Lead Role

## Why this matters
Prompt work has clearly been useful.
It has helped create a better collaborator voice and more practical output behavior.

But prompt sophistication should now support the system, not compensate for missing grounding.

## Rule
Do not let prompt growth outpace ingestion/retrieval quality.

## Practical guidance
Before adding more major collaborator instructions, first ask:
- is the missing behavior actually a retrieval problem?
- is this a context persistence problem?
- is the answer too generic because the knowledge base is weakly structured?
- is the system failing because ranking is not strong enough?

## Success condition
Prompting becomes the final behavior-shaping layer, not the main engine of intelligence.

---

# Priority 5 — Strengthen the “Knowledge-Shaped Collaborator” Identity

## Why this matters
This is the real product identity.

The system should increasingly feel like:
- a collaborator shaped by the user’s own references, notes, workflows, and track state

not:
- a generic music assistant with a better UI and stronger tone

## Recommended direction
As new features are evaluated, prefer features that improve:
- grounding
- retrieval relevance
- track continuity
- progress toward finishing tracks

Be more skeptical of features that mainly increase:
- surface complexity
- mode proliferation
- prompt branching
- UI clutter without deeper intelligence

---

# Suggested Build Order

## Step 1
Implement weighted retrieval / ranking using Track Context + metadata.

## Step 2
Clean up Track Context architecture so YAML is the clear source of truth.

## Step 3
Persist tasks per track and connect them to the loaded track session.

## Step 4
Refactor any prompt logic that is currently compensating for weak retrieval.

## Step 5
Run a focused evaluation pass using real music-production scenarios:
- arrangement critique
- sound design help
- track stuck / next-step guidance
- section-specific improvement requests

Measure:
- specificity
- grounding
- usefulness
- track relevance
- collaborator quality

---

# What to Avoid Right Now

## Avoid
- expanding UI complexity before grounding improves
- adding too many new modes
- making legacy context paths permanent
- relying on collaborator prompt sophistication as the main solution
- building an overcomplicated task system before basic persistence works cleanly

---

# Summary

The project is moving in the right direction.

The strongest current win is that Track Context and collaborator behavior are now real.

The most important next move is to make the system’s intelligence more clearly come from:
- ingested knowledge
- retrieval quality
- persistent track-linked state

That is the key step that keeps the project aligned with the intended vision of a true knowledge-shaped track collaborator.