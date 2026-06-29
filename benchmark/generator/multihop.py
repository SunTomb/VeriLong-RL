from benchmark.generator.common import doc_id, evidence_id, make_rng, pad_neutral_documents
from benchmark.schemas.task import Difficulty, EvidenceDocument, TaskMetadata, VeriLongTask


_ENTITIES = ["Aurora", "Borealis", "Cygnus", "Draco", "Equinox", "Fornax"]
_PROTOCOLS = ["Protocol Blue", "Protocol Green", "Protocol Silver", "Protocol Amber"]
_CONDITIONS = ["low humidity", "night operations", "sealed transit", "manual inspection"]
_APPROVALS = ["Team Delta", "Team Kappa", "Team Meridian", "Team Sol"]
_DESTINATIONS = ["Route 14", "Vault K", "Channel 9", "Bay 27"]


def generate_multihop_task(
    task_id: str,
    seed: int,
    hop_count: int = 3,
    target_context_tokens: int = 8000,
    difficulty: Difficulty = "medium",
    irrelevant_rule_count: int = 1,
    conflicting_rule_count: int = 0,
) -> VeriLongTask:
    if hop_count < 2 or hop_count > 5:
        raise ValueError("hop_count must be between 2 and 5")
    if irrelevant_rule_count < 0:
        raise ValueError("irrelevant_rule_count must be non-negative")
    if conflicting_rule_count < 0:
        raise ValueError("conflicting_rule_count must be non-negative")

    rng = make_rng(seed)
    entity = rng.choice(_ENTITIES)
    protocol = rng.choice(_PROTOCOLS)
    condition = rng.choice(_CONDITIONS)
    approval = rng.choice(_APPROVALS)
    answer = rng.choice(_DESTINATIONS)

    hop_texts = _chain_texts(
        entity=entity,
        protocol=protocol,
        condition=condition,
        approval=approval,
        answer=answer,
        hop_count=hop_count,
    )
    documents = [
        EvidenceDocument(
            doc_id=doc_id(index),
            evidence_id=evidence_id(index),
            text=text,
            role="gold",
        )
        for index, text in enumerate(hop_texts, start=1)
    ]

    next_index = len(documents) + 1
    for offset in range(irrelevant_rule_count):
        documents.append(
            EvidenceDocument(
                doc_id=doc_id(next_index),
                evidence_id=evidence_id(next_index),
                text=(
                    f"Facility {entity} Annex follows reference rule {offset + 1}, "
                    f"which routes unrelated audit materials to Holding Bay {offset + 3}."
                ),
                role="distractor",
            )
        )
        next_index += 1

    # Conflicting rules must never name the true destination, or a distractor
    # would become answer-supporting while labeled role="distractor".
    other_destinations = [destination for destination in _DESTINATIONS if destination != answer]
    for offset in range(conflicting_rule_count):
        documents.append(
            EvidenceDocument(
                doc_id=doc_id(next_index),
                evidence_id=evidence_id(next_index),
                text=(
                    f"A retired mapping for {protocol} mentions {other_destinations[offset % len(other_destinations)]}, "
                    "but the note applies only to deprecated facilities and not the active chain."
                ),
                role="distractor",
            )
        )
        next_index += 1

    documents = pad_neutral_documents(documents, target_context_tokens, rng)

    return VeriLongTask(
        id=task_id,
        task_family="multi_hop_reasoning",
        difficulty=difficulty,
        question=f"Following the active chain for Facility {entity}, what is the required destination?",
        documents=documents,
        gold_answer=answer,
        gold_evidence_ids=[evidence_id(index) for index in range(1, hop_count + 1)],
        distractor_evidence_ids=[evidence_id(index) for index in range(hop_count + 1, next_index)],
        stale_evidence_ids=[],
        expected_steps=_expected_steps(entity=entity, hop_count=hop_count),
        metadata=TaskMetadata(
            target_context_tokens=target_context_tokens,
            evidence_position="distributed",
            hop_count=hop_count,
            extra={
                "irrelevant_rule_count": irrelevant_rule_count,
                "conflicting_rule_count": conflicting_rule_count,
            },
        ),
    )


def _chain_texts(
    entity: str,
    protocol: str,
    condition: str,
    approval: str,
    answer: str,
    hop_count: int,
) -> list[str]:
    """Build a gold evidence chain of exactly hop_count facts.

    The answer must appear in the LAST hop for every hop_count, otherwise the
    gold evidence is insufficient to derive the answer. Intermediate links are
    chosen so each hop introduces the term the next hop consumes, and the final
    hop always names the destination.
    """

    first = f"Facility {entity} is assigned to {protocol} for the active pilot audit."
    # Intermediate links, used only when there is room before the final hop.
    middle = [
        f"{protocol} applies when the operating condition is {condition}.",
        f"For {condition}, the approval owner is {approval}.",
        f"{approval} uses routing table R7 for this audit family.",
    ]

    if hop_count == 2:
        return [first, f"{protocol} routes the final package to {answer}."]

    middle_needed = hop_count - 2
    links = middle[:middle_needed]
    last_term = [protocol, condition, approval, "routing table R7"][middle_needed]
    final = f"{last_term} sends the final package to {answer}."
    return [first, *links, final]


def _expected_steps(entity: str, hop_count: int) -> list[str]:
    steps = [f"Find Facility {entity}'s active protocol in {evidence_id(1)}."]
    if hop_count >= 3:
        steps.append(f"Follow the chain through {evidence_id(2)} to narrow the applicable rule.")
    # Always reference the final hop, where the destination is stated.
    steps.append(f"Use {evidence_id(hop_count)} to reach the final destination.")
    return steps[:4]
