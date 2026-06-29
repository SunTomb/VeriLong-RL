# Phase 2 Real-Paper Evidence Design

## Context

Phase 1 synthetic multi-document evidence tasks are complete and validated. They provide deterministic task generation, validation, parsing, scoring, API/open-source evaluation, SFT warmup, and an RLVR pipeline that reuses the same programmatic reward. Phase 2 is now explicitly approved as the next research direction, but this document is intentionally **design-only**: it does not add paper ingestion, new task generation, or new evaluation runs.

## Goal

Extend VeriLong-RL from synthetic documents to real scientific/technical documents while preserving the key property that made Phase 1 useful: answers are scored against verifiable evidence IDs and the fixed `Evidence:` / `Steps:` / `Answer:` output contract.

## Non-goals for this stage

- Do not scrape, download, or redistribute real papers yet.
- Do not add Phase 2 generator code in this document step.
- Do not change Phase 1 task families, parser, or reward semantics.
- Do not fabricate Phase 2 metrics.
- Do not use LLM judges as online RL rewards.
- Do not start Phase 3 repo-level tasks.

## Candidate source classes

Phase 2 should prefer sources that are legally and operationally simple:

1. **Open-access papers with permissive licenses**
   - arXiv source/PDF can be used for research, but redistribution rules still need care.
   - Prefer papers with explicit CC-BY or similarly permissive licensing where possible.

2. **Open technical reports / documentation**
   - Often easier to segment and cite than PDFs.
   - Can include model cards, benchmark reports, API docs, or standards documents.

3. **User-provided paper set**
   - Safest for an internal demo if the user controls the source files.
   - Still store provenance and license fields.

The first implementation should use a small allowlisted corpus with recorded source URLs, license notes, and hashes. Avoid broad crawling.

## Data model recommendation

Keep a separate Phase 2 source schema, then compile it into the existing `VeriLongTask` shape for evaluation.

### Source schema

A future `RealPaperSource` record should contain:

- `source_id`: stable internal ID.
- `title`, `authors`, `year`.
- `source_url` or DOI/arXiv ID.
- `license`: explicit license note or `unknown`.
- `document_hash`: hash of the normalized text used for task creation.
- `sections`: ordered sections with section title and paragraph/span records.

A future `EvidenceSpan` record should contain:

- `evidence_id`: benchmark-facing ID such as `E01`.
- `source_id`.
- `section_id`.
- `paragraph_id` or span offsets.
- `text`.
- `role`: `gold`, `distractor`, `stale`, or `neutral` where applicable.
- `provenance`: page/section/paragraph metadata for audit.

### Compiled task schema

The compiled task should remain compatible with `VeriLongTask`:

- `documents` become evidence spans rendered as document snippets.
- `gold_evidence_ids` remain the scoring target.
- `distractor_evidence_ids` and `stale_evidence_ids` stay available for penalties.
- `metadata.extra` stores real-paper provenance, license, source IDs, and span anchors.

This avoids mutating the Phase 1 schema prematurely and lets existing parser/scorer/eval code continue to work.

## Candidate Phase 2 task families

Start with a narrow subset that can be programmatically verified:

1. **paper_fact_retrieval**
   - Question asks for a stated fact from one paper section.
   - Gold evidence is one span.
   - Distractors are nearby or same-topic spans from the same or related paper.

2. **cross_section_synthesis**
   - Question requires combining two or three spans from different sections of the same paper.
   - Similar to Phase 1 multi-hop but grounded in real section structure.

3. **claim_method_alignment**
   - Question asks which method, dataset, or condition supports a reported claim.
   - Gold evidence includes both claim and method/result context.

4. **version_or_revision_update**
   - Optional later family for papers/reports with versions, errata, or updated claims.
   - Mirrors temporal update but should only be used when provenance is strong.

The first MVP should implement only `paper_fact_retrieval` and `cross_section_synthesis`.

## Task construction flow

1. **Source intake**
   - Accept only allowlisted source files or URLs.
   - Record license/provenance before processing.
   - Normalize text deterministically.

2. **Segmentation**
   - Split into sections and paragraphs.
   - Preserve section titles and paragraph IDs.
   - Drop references, boilerplate, and captions only if the rule is deterministic and documented.

3. **Candidate evidence mining**
   - Select spans that contain concrete answerable facts.
   - Prefer spans with named methods, datasets, metrics, settings, limitations, or conclusions.

4. **Question/answer authoring**
   - Use deterministic templates where possible.
   - If LLM assistance is used, it is for candidate generation/filtering only.
   - Final accepted tasks must have explicit gold answers and evidence IDs.

5. **Distractor selection**
   - Same-section distractors test local confusion.
   - Same-topic cross-paper distractors test retrieval precision.
   - Do not include distractors that make the gold answer ambiguous.

6. **Validation**
   - Verify evidence IDs are unique.
   - Verify gold answer appears in or is entailed by gold spans.
   - Verify distractor/stale spans do not independently answer the question.
   - Verify provenance fields are present.

7. **Compilation**
   - Emit `VeriLongTask` JSONL for the existing evaluation path.
   - Keep source/provenance JSONL separately for audit.

## Validation and filtering

Programmatic validation should reject:

- Missing source/license/provenance fields.
- Duplicate evidence IDs.
- Gold evidence not present in the rendered document list.
- Empty gold answer or answer absent from accepted evidence for retrieval tasks.
- Distractor evidence containing the same answer string for simple retrieval tasks unless explicitly marked as a contrastive ambiguity case.
- Tasks whose question can be answered without the listed evidence.

LLM judge use is allowed only for filtering/calibration:

- Does the gold evidence support the answer?
- Is any distractor also sufficient?
- Is the question unambiguous?
- Does the required answer depend on external knowledge?

Judge outputs are not online RL rewards.

## Evaluation contract

Phase 2 should keep the same model-facing output format:

```text
Evidence: E01, E03
Steps:
1. Short grounded reasoning step.
2. Short grounded reasoning step.
Answer: final answer only
```

The first implementation should reuse:

- `experiments/eval_api/run_api_eval.py` prompt structure with a Phase 2 prompt variant if needed.
- `benchmark/parser/output_parser.py` unchanged.
- `benchmark/reward/score.py` unchanged where possible.

If Phase 2 needs additional analysis metrics, add them as reporting fields first, not as changes to the core reward.

## Storage layout

Proposed future layout:

```text
data/phase2/
  sources/
    sources.jsonl
    normalized_text/
  tasks/
    phase2_pilot.jsonl
  audit/
    rejected_candidates.jsonl
    judge_filter_outputs.jsonl
results/phase2/
  pilot/
results/raw/phase2/
  api_cache/
```

Raw provider outputs and judge outputs should remain under `results/raw/` or `data/phase2/audit/` depending on sensitivity and size. Do not commit private or license-uncertain source documents.

## MVP acceptance criteria

A future Phase 2 MVP is complete only when:

- A small allowlisted corpus has documented license/provenance.
- At least one Phase 2 task family compiles to `VeriLongTask` JSONL.
- Validation rejects ambiguous or unsupported tasks.
- Existing parser/scorer/eval can run on Phase 2 compiled tasks.
- A small smoke eval is run with real outputs and reported honestly.
- No Phase 1 tests or metrics regress.

## Risks

- **Licensing risk:** real papers may not be redistributable. Mitigation: track license fields and commit only derived task snippets when allowed.
- **Ambiguity risk:** real text often supports multiple interpretations. Mitigation: stricter validation and judge filtering.
- **Reward mismatch:** existing citation metrics may miss span-level subtlety. Mitigation: add reporting metrics before changing reward.
- **Data leakage:** popular papers may be memorized by models. Mitigation: prefer less-memorized technical reports or require evidence citation precision.
- **Pipeline complexity:** PDF extraction can dominate the schedule. Mitigation: start with normalized text or source-provided markdown/LaTeX where possible.

## Explicit next gate

This design authorizes planning only. Before implementation, the next step should be a separate Phase 2 implementation plan that chooses:

- source corpus,
- licensing policy,
- first task family,
- exact source schema,
- validation tests,
- and pilot size.

No Phase 2 ingestion or task-generation code should be written until that implementation plan is approved.
