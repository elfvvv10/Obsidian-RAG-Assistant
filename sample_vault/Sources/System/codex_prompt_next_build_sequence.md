You are working inside the Obsidian Track Collaborator repository.

Your task is to implement the next priority build sequence in a way that keeps the project aligned with its intended identity:

> a knowledge-shaped track collaborator
> where ingestion = memory, retrieval = recall, model = reasoning, and track context = perspective.

Do not treat this as a generic feature expansion task.
Do not optimize primarily for UI polish.
Do not solve missing intelligence with prompt complexity if the deeper issue is retrieval or persistence.

---

# Strategic Goal

The current system has made strong progress in:
- persistent Track Context
- collaborator-style prompting
- multi-turn chat flow
- workflow specialization

But the next phase must close the gap between:
- collaborator surface
and
- ingestion / retrieval intelligence

Your implementation should prioritize:
1. retrieval quality
2. grounding quality
3. track-linked persistence

over:
- new UI complexity
- mode proliferation
- prompt growth for its own sake

---

# Primary Objectives

## Objective 1 — Make Retrieval Track-Aware in Ranking, Not Just Query Rewriting

The system already uses Track Context in query rewriting.
That is useful, but not sufficient.

Implement or strengthen weighted retrieval/reranking so that final retrieval ranking can incorporate:
- semantic similarity
- genre match
- domain match
- importance weighting
- track context relevance
- optional workflow relevance

The aim is that retrieved knowledge should better reflect:
- the active track’s genre
- the track’s goals
- known issues
- current section
- relevant references
- the active workflow, when applicable

### Implementation guidance
- Keep the existing retrieval pipeline intact where possible
- Add a clear weighted scoring stage after initial candidate retrieval
- Make the logic readable and inspectable
- Prefer explicit scoring helpers over hidden magic constants scattered through the code
- Use metadata already present where available
- Gracefully degrade if metadata is missing

### Desired output behavior
Answers should increasingly feel shaped by:
- the user’s ingested knowledge
- the active track state

not just by a stronger prompt voice.

---

## Objective 2 — Consolidate Around YAML Track Context as Canonical

The project now revolves around Track Context.
There must be one obvious source of truth.

Make YAML Track Context the canonical track-state system.

Legacy markdown-based workflow context may remain temporarily only as:
- a migration bridge
- an import source
- a backward-compatibility path

It should not remain an equal long-term pathway.

### Implementation guidance
- Audit where both YAML and legacy markdown contexts are supported
- Reduce architectural ambiguity
- Ensure new logic targets YAML Track Context first
- Preserve compatibility where necessary, but make canonical behavior explicit
- Add or improve migration/import helper logic if needed
- Update user-facing wording in code/comments/UI where needed so the intended path is clear

### Desired outcome
There should be one clear answer to:
“Where does track state live?”

Answer:
in the YAML Track Context system

---

## Objective 3 — Evolve Session Tasks into Persisted Per-Track Tasks

The current session task behavior is useful, but the roadmap direction now requires per-track persistence.

Implement a basic persisted task layer tied to the active track.

### Minimum requirements
Persist tasks per track with a simple schema such as:
- id
- text
- status
- priority
- linked_section
- created_from
- created_at
- completed_at

### Functional expectations
- tasks should load automatically when a track is loaded
- tasks should persist across sessions
- tasks should be associated with a specific track
- the assistant should be able to propose tasks from critiques or responses
- task creation should remain reviewable where appropriate
- keep the system simple and robust; do not overbuild

### Design preference
Use a storage approach that matches the existing Track Context philosophy.
If appropriate, store tasks:
- alongside Track Context
or
- in a clearly linked per-track file/system

Avoid creating a heavy or overly abstract task subsystem unless clearly justified.

---

# Guardrails

## Guardrail 1 — Do Not Let Prompting Lead the Architecture
Prompt logic may be adjusted only where needed to support:
- new retrieval signals
- canonical YAML Track Context usage
- persisted per-track task awareness

Do not respond to weak grounding by mainly adding more instructions.

Before adding major prompt behavior, ask:
- is this actually a retrieval problem?
- is this actually a persistence problem?
- is this actually a weak metadata problem?

---

## Guardrail 2 — Preserve the Project Identity
This project is not meant to become:
- a generic chatbot with music language
- a feature-rich assistant with shallow grounding
- a UI-heavy workspace with weak recall

It should increasingly feel like:
- a collaborator shaped by ingested knowledge
- a system aware of track state
- a tool that helps the user finish tracks

---

## Guardrail 3 — Prefer Clear Incremental Changes
Make changes that are:
- testable
- reviewable
- easy to reason about

Avoid wide, unnecessary refactors unless they materially reduce ambiguity or drift.

---

# Existing Direction You Must Respect

The implementation must remain aligned with these principles:

- The system should reflect ingested knowledge, not model priors
- Retrieval should prioritize user-curated knowledge
- Track Context should influence query rewriting, retrieval, prompting, and saved outputs
- The collaborator should be conversational, context-aware, initiative-taking, anti-generic, and execution-focused
- Tasks are moving toward persisted per-track memory
- Prompting should support grounding, not replace it

---

# Expected Deliverables

Please make a best-effort implementation and then provide a reviewable summary with:

## 1. What you changed
Be specific.
Mention files, functions, classes, and any new schemas.

## 2. Architectural decisions
Explain how you handled:
- weighted retrieval/ranking
- YAML Track Context canonicalization
- persisted per-track tasks

## 3. Any migration or compatibility behavior
Be explicit about what remains supported and what is now considered legacy.

## 4. Tests added or updated
Add or update tests wherever practical.
Favor tests that verify:
- ranking behavior
- track-context-aware retrieval effects
- canonical YAML usage
- task persistence/load behavior

## 5. Remaining gaps
Call out what still needs follow-up.

---

# Prioritized Build Order

Follow this order unless you discover a strong reason not to:

### Step 1
Implement weighted retrieval/ranking using Track Context + metadata.

### Step 2
Clarify and consolidate Track Context architecture so YAML is the canonical source of truth.

### Step 3
Persist tasks per track and connect them to track loading/session flow.

### Step 4
Adjust prompts only where needed to reflect the new grounding and persistence behavior.

### Step 5
Strengthen or add evaluation coverage for realistic scenarios such as:
- arrangement critique
- sound design help
- stuck-track guidance
- section-specific next-step advice

---

# Quality Bar

A good result should make the system feel:
- more grounded
- more track-specific
- more persistent across sessions
- less dependent on prompt cleverness

A bad result would be:
- mostly prompt edits
- mostly UI changes
- more modes with little grounding improvement
- more complexity without clearer system intelligence

Proceed with implementation.