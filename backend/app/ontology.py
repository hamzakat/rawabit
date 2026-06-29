from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# POLE entity types.
# References: docs/pole.md, the UK policing POLE data model, and the
# Neo4j pole graph example (github.com/neo4j-graph-examples/pole)

CONTROLLED_ENTITY_TYPES: tuple[str, ...] = (
    "person",
    "organization",
    "object",
    "location",
    "event",
)

# Subtypes stay separate so the frontend can render better icons without
# widening the controlled POLE categories.
ENTITY_SUBTYPES: tuple[str, ...] = (
    "device",
    "document",
    "account",
    "asset",
    "vehicle",
    "weapon",
    "communication",  # phone call, message, email; modelled as Event in POLE
    "file",
    "other",
)


# POLE relationship types.

CONTROLLED_RELATION_TYPES: tuple[str, ...] = (
    # Core POLE relations.
    "PARTICIPATED_IN",   # Person -> Event (role: suspect, victim, witness, sender)
    "USED_IN",           # Object -> Event (role: weapon, vehicle, device, account)
    "OCCURRED_AT",       # Event -> Location
    "RESIDENCE_AT",      # Person -> Location
    "LOCATED_AT",        # Object -> Location
    "ASSOCIATED_WITH",   # Person -> Person; also membership, ownership, co-mention
    "WITHIN",            # Location -> Location hierarchy
    "SAME_AS",           # Object -> Object alias or dedupe

    # Identity and aliases.
    "POSSIBLE_SAME_AS",
    "ALIAS_OF",

    # Social and organisational relations.
    "MEMBER_OF",
    "EMPLOYED_BY",
    "REPORTS_TO",
    "SUPERVISES",
    "REPRESENTS",

    # Communication and interaction.
    "COMMUNICATED_WITH",
    "MET_WITH",
    "COLLABORATED_WITH",

    # Event participation and roles.
    "ORGANIZED",
    "PERPETRATED",
    "TARGET_OF",
    "WITNESS_OF",

    # Place and time.
    "LOCATED_IN",
    "PRESENT_AT",

    # Assets, ownership, and control.
    "OWNS",
    "INVESTED_IN",
    "CONTROLS",
    "USES_ASSET",
    "BENEFITS_FROM",

    # Transactions and flows.
    "TRANSFERRED_TO",
    "SUPPLIED",
    "PAID",
    "RECEIVED_FROM",

    # Provenance and analysis.
    "MENTIONED_IN",
    "EXTRACTED_FROM",
    "EVIDENCE_FOR",
    "EVIDENCE_AGAINST",

)


# Free-form relation tokens from LightRAG/LLM output mapped to the controlled
# POLE vocabulary.

_RELATION_SYNONYMS: dict[str, str] = {
    # PARTICIPATED_IN
    "communicated": "PARTICIPATED_IN",
    "contacted": "PARTICIPATED_IN",
    "emailed": "PARTICIPATED_IN",
    "called": "PARTICIPATED_IN",
    "messaged": "PARTICIPATED_IN",
    "met": "PARTICIPATED_IN",
    "meeting": "PARTICIPATED_IN",
    "talked": "PARTICIPATED_IN",
    "discussed": "PARTICIPATED_IN",
    "participated": "PARTICIPATED_IN",
    "attended": "PARTICIPATED_IN",
    "present_at": "PARTICIPATED_IN",
    "posted": "PARTICIPATED_IN",
    "posted_by": "PARTICIPATED_IN",
    "published": "PARTICIPATED_IN",
    "reposted": "PARTICIPATED_IN",
    "shared": "PARTICIPATED_IN",
    "retweeted": "PARTICIPATED_IN",
    "reported": "PARTICIPATED_IN",
    "reporting": "PARTICIPATED_IN",
    "stated_by": "PARTICIPATED_IN",
    "authored_by": "PARTICIPATED_IN",
    "quoted": "PARTICIPATED_IN",
    "travel": "PARTICIPATED_IN",
    "traveled": "PARTICIPATED_IN",
    "moved": "PARTICIPATED_IN",
    "arrived": "PARTICIPATED_IN",
    "departed": "PARTICIPATED_IN",

    # USED_IN
    "used": "USED_IN",
    "utilized": "USED_IN",
    "leveraged": "USED_IN",
    "paid": "USED_IN",
    "sent": "USED_IN",
    "wired": "USED_IN",
    "transferred": "USED_IN",
    "remitted": "USED_IN",
    "shipped": "USED_IN",
    "delivered": "USED_IN",
    "transaction": "USED_IN",
    "bought": "USED_IN",
    "sold": "USED_IN",
    "purchased": "USED_IN",
    "supplied": "USED_IN",
    "procured": "USED_IN",

    # OCCURRED_AT
    "occurred_at": "OCCURRED_AT",
    "happened_at": "OCCURRED_AT",
    "took_place_at": "OCCURRED_AT",

    # RESIDENCE_AT
    "resides": "RESIDENCE_AT",
    "lives_at": "RESIDENCE_AT",

    # LOCATED_AT
    "located": "LOCATED_AT",
    "based": "LOCATED_AT",
    "at": "LOCATED_AT",

    # ASSOCIATED_WITH (membership, ownership, co-mention, references)
    "associated": "ASSOCIATED_WITH",
    "related": "ASSOCIATED_WITH",
    "linked": "ASSOCIATED_WITH",
    "affiliated": "ASSOCIATED_WITH",
    "worked_with": "ASSOCIATED_WITH",
    "member": "ASSOCIATED_WITH",
    "member_of": "ASSOCIATED_WITH",
    "joined": "ASSOCIATED_WITH",
    "part_of": "ASSOCIATED_WITH",
    "owns": "ASSOCIATED_WITH",
    "owned": "ASSOCIATED_WITH",
    "controls": "ASSOCIATED_WITH",
    "operates": "ASSOCIATED_WITH",
    "manages": "ASSOCIATED_WITH",
    "runs": "ASSOCIATED_WITH",
    "depicts": "ASSOCIATED_WITH",
    "depicted": "ASSOCIATED_WITH",
    "shows": "ASSOCIATED_WITH",
    "references": "ASSOCIATED_WITH",
    "referenced": "ASSOCIATED_WITH",
    "mentions": "ASSOCIATED_WITH",
    "mentioned": "ASSOCIATED_WITH",
    "co_mentioned": "ASSOCIATED_WITH",
    "shows_position_of": "ASSOCIATED_WITH",
    "position_of": "ASSOCIATED_WITH",
    "involves": "ASSOCIATED_WITH",
    "involved": "ASSOCIATED_WITH",

    # WITHIN
    "within": "WITHIN",
    "inside": "WITHIN",
    "contained_in": "WITHIN",

    # SAME_AS
    "same_as": "SAME_AS",
    "alias": "SAME_AS",
    "duplicate": "SAME_AS",

    # POSSIBLE_SAME_AS
    "possible_same_as": "POSSIBLE_SAME_AS",
    "probably_same": "POSSIBLE_SAME_AS",
    "likely_same": "POSSIBLE_SAME_AS",
    "maybe_same": "POSSIBLE_SAME_AS",
    "similar": "POSSIBLE_SAME_AS",
    "similar_to": "POSSIBLE_SAME_AS",
    "similar_company_as": "POSSIBLE_SAME_AS",
    "probably_same_officer_as": "POSSIBLE_SAME_AS",

    # ALIAS_OF
    "alias_of": "ALIAS_OF",
    "also_known_as": "ALIAS_OF",
    "aka": "ALIAS_OF",

    # MEMBER_OF
    "member_of": "MEMBER_OF",
    "belongs_to": "MEMBER_OF",
    "affiliated_with": "MEMBER_OF",
    "joined": "MEMBER_OF",

    # EMPLOYED_BY
    "employed_by": "EMPLOYED_BY",
    "works_for": "EMPLOYED_BY",
    "works_at": "EMPLOYED_BY",
    "employee_of": "EMPLOYED_BY",
    "staff_of": "EMPLOYED_BY",
    "officer_of": "EMPLOYED_BY",
    "director_of": "EMPLOYED_BY",
    "secretary_of": "EMPLOYED_BY",
    "treasurer_of": "EMPLOYED_BY",
    "president_of": "EMPLOYED_BY",
    "vice_president_of": "EMPLOYED_BY",
    "ceo_of": "EMPLOYED_BY",
    "cfo_of": "EMPLOYED_BY",
    "manager_of": "EMPLOYED_BY",
    "managing_director_of": "EMPLOYED_BY",

    # REPORTS_TO
    "reports_to": "REPORTS_TO",
    "reporting_to": "REPORTS_TO",
    "subordinate_to": "REPORTS_TO",
    "answers_to": "REPORTS_TO",

    # SUPERVISES
    "supervises": "SUPERVISES",
    "oversees": "SUPERVISES",
    "manages": "SUPERVISES",
    "leads": "SUPERVISES",
    "chairman_of": "SUPERVISES",

    # REPRESENTS
    "represents": "REPRESENTS",
    "representative_of": "REPRESENTS",
    "agent_of": "REPRESENTS",
    "proxy_of": "REPRESENTS",
    "attorney_of": "REPRESENTS",
    "power_of_attorney": "REPRESENTS",
    "intermediary_of": "REPRESENTS",
    "intermediary": "REPRESENTS",
    "signatory_of": "REPRESENTS",
    "signatory": "REPRESENTS",
    "nominee_of": "REPRESENTS",
    "nominee": "REPRESENTS",

    # COMMUNICATED_WITH
    "communicated_with": "COMMUNICATED_WITH",
    "corresponded_with": "COMMUNICATED_WITH",
    "contacted": "COMMUNICATED_WITH",
    "emailed": "COMMUNICATED_WITH",
    "called": "COMMUNICATED_WITH",
    "messaged": "COMMUNICATED_WITH",

    # MET_WITH
    "met_with": "MET_WITH",
    "met": "MET_WITH",
    "meeting_with": "MET_WITH",

    # COLLABORATED_WITH
    "collaborated_with": "COLLABORATED_WITH",
    "cooperated_with": "COLLABORATED_WITH",
    "partnered_with": "COLLABORATED_WITH",

    # ORGANIZED
    "organized": "ORGANIZED",
    "organised": "ORGANIZED",
    "planned": "ORGANIZED",
    "arranged": "ORGANIZED",
    "convened": "ORGANIZED",

    # PERPETRATED
    "perpetrated": "PERPETRATED",
    "committed": "PERPETRATED",
    "carried_out": "PERPETRATED",
    "executed": "PERPETRATED",

    # TARGET_OF
    "target_of": "TARGET_OF",
    "targeted_by": "TARGET_OF",
    "victim_of": "TARGET_OF",

    # WITNESS_OF
    "witness_of": "WITNESS_OF",
    "witnessed": "WITNESS_OF",
    "observed": "WITNESS_OF",
    "saw": "WITNESS_OF",

    # LOCATED_IN
    "located_in": "LOCATED_IN",
    "resides_in": "LOCATED_IN",
    "lives_in": "LOCATED_IN",
    "based_in": "LOCATED_IN",
    "headquartered_in": "LOCATED_IN",
    "registered_in": "LOCATED_IN",
    "registered_address": "LOCATED_IN",
    "registered_office": "LOCATED_IN",
    "business_address": "LOCATED_IN",
    "mailing_address": "LOCATED_IN",
    "residential_address": "LOCATED_IN",

    # PRESENT_AT
    "present_at": "PRESENT_AT",
    "seen_at": "PRESENT_AT",
    "observed_at": "PRESENT_AT",
    "attended": "PRESENT_AT",

    # OWNS
    "owns": "OWNS",
    "owned": "OWNS",
    "owner_of": "OWNS",
    "ownership": "OWNS",
    "shareholder_of": "OWNS",
    "shareholder": "OWNS",
    "stockholder": "OWNS",
    "stockholder_of": "OWNS",
    "member_of": "OWNS",
    "partner_of": "OWNS",

    # INVESTED_IN
    "invested_in": "INVESTED_IN",
    "investor_in": "INVESTED_IN",
    "funded": "INVESTED_IN",
    "financed": "INVESTED_IN",

    # CONTROLS
    "controls": "CONTROLS",
    "controlled": "CONTROLS",
    "controller_of": "CONTROLS",
    "controlling": "CONTROLS",
    "dominates": "CONTROLS",
    "dominant": "CONTROLS",
    "ultimate_beneficial_owner": "CONTROLS",
    "ubo": "CONTROLS",
    "beneficial_owner_of": "CONTROLS",
    "beneficial_owner": "CONTROLS",

    # USES_ASSET
    "uses_asset": "USES_ASSET",
    "uses": "USES_ASSET",
    "utilizes": "USES_ASSET",
    "operates": "USES_ASSET",

    # BENEFITS_FROM
    "benefits_from": "BENEFITS_FROM",
    "beneficiary_of": "BENEFITS_FROM",
    "beneficiary": "BENEFITS_FROM",
    "receives_benefits_from": "BENEFITS_FROM",
    "profits_from": "BENEFITS_FROM",
    "gains_from": "BENEFITS_FROM",

    # TRANSFERRED_TO
    "transferred_to": "TRANSFERRED_TO",
    "transfer_to": "TRANSFERRED_TO",
    "sent_to": "TRANSFERRED_TO",
    "wired_to": "TRANSFERRED_TO",
    "moved_to": "TRANSFERRED_TO",
    "remitted_to": "TRANSFERRED_TO",
    "shipped_to": "TRANSFERRED_TO",
    "delivered_to": "TRANSFERRED_TO",

    # SUPPLIED
    "supplied": "SUPPLIED",
    "supplier_of": "SUPPLIED",
    "provider_of": "SUPPLIED",
    "furnished": "SUPPLIED",

    # PAID
    "paid": "PAID",
    "payment_to": "PAID",
    "paid_to": "PAID",
    "remitted": "PAID",

    # RECEIVED_FROM
    "received_from": "RECEIVED_FROM",
    "receives_from": "RECEIVED_FROM",
    "got_from": "RECEIVED_FROM",
    "incoming_from": "RECEIVED_FROM",

    # MENTIONED_IN
    "mentioned_in": "MENTIONED_IN",
    "cited_in": "MENTIONED_IN",
    "referenced_in": "MENTIONED_IN",
    "appears_in": "MENTIONED_IN",
    "documented_in": "MENTIONED_IN",

    # EXTRACTED_FROM
    "extracted_from": "EXTRACTED_FROM",
    "derived_from": "EXTRACTED_FROM",
    "sourced_from": "EXTRACTED_FROM",
    "taken_from": "EXTRACTED_FROM",

    # EVIDENCE_FOR
    "evidence_for": "EVIDENCE_FOR",
    "proves": "EVIDENCE_FOR",
    "supports": "EVIDENCE_FOR",
    "corroborates": "EVIDENCE_FOR",
    "confirms": "EVIDENCE_FOR",

    # EVIDENCE_AGAINST
    "evidence_against": "EVIDENCE_AGAINST",
    "contradicts": "EVIDENCE_AGAINST",
    "refutes": "EVIDENCE_AGAINST",
    "disproves": "EVIDENCE_AGAINST",
    "undermines": "EVIDENCE_AGAINST",

}


@dataclass(frozen=True)
class NormalizedRelation:
    relation_type: str
    raw_phrase: str | None
    confidence_score: float
    confidence_band: str


def _band(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.55:
        return "medium"
    return "low"


def normalize_relation(candidate_texts: Iterable[str | None]) -> NormalizedRelation:
    """
    Map free-form relation keywords / labels into the POLE relational schema.

    Handles both single-word tokens and multi-word phrases, such as
    ``"director of"`` becoming ``"director_of"``.

    Returns the canonical POLE type, the best-effort raw phrase, a numeric
    confidence score, and a band (high / medium / low).
    """
    seen_texts = [
        text.strip()
        for text in candidate_texts
        if isinstance(text, str) and text.strip()
    ]
    raw_phrase = seen_texts[0] if seen_texts else None
    tokens: list[str] = []
    for text in seen_texts:
        # Keep the phrase form before falling back to individual words.
        normalized = text.replace(",", " ").replace("|", " ").replace(";", " ")
        underscored = "_".join(normalized.split())
        if underscored:
            tokens.append(underscored.lower())
        for token in normalized.split():
            cleaned = token.strip().lower()
            if cleaned and cleaned != underscored.lower():
                tokens.append(cleaned)

    matched: list[str] = []
    for token in tokens:
        canonical = _RELATION_SYNONYMS.get(token)
        if canonical:
            matched.append(canonical)

    if matched:
        relation_type = max(set(matched), key=matched.count)
        confidence = 0.82 if matched.count(relation_type) > 1 else 0.72
    else:
        relation_type = "ASSOCIATED_WITH"
        confidence = 0.48 if seen_texts else 0.4

    return NormalizedRelation(
        relation_type=relation_type,
        raw_phrase=raw_phrase,
        confidence_score=round(confidence, 3),
        confidence_band=_band(confidence),
    )


def normalize_entity_type(entity_type: str | None) -> str:
    """
    Collapse a recognised entity label into one of the five POLE types:

        person       - individuals
        organization - companies, agencies, NGOs, trusts, institutions
        object       - physical or digital items
        location     - places at any granularity
        event        - anything that happens or is recorded as an incident

    Falls back to ``"object"`` when the input is unrecognised.
    """
    if not entity_type:
        return "object"
    normalized = entity_type.strip().lower()

    if normalized in CONTROLLED_ENTITY_TYPES:
        return normalized

    if normalized in {
        "organization", "org", "company", "corp", "ngo", "agency",
        "group", "institution", "gang", "organisation",
    }:
        return "organization"

    if normalized in {
        "asset", "artifact", "equipment", "cargo",
        "document", "report", "file", "certificate", "letter",
        "account", "bank_account", "iban",
        "device", "phone", "sim", "computer", "server", "laptop",
        "vehicle", "car", "truck", "van", "boat", "plane",
        "weapon", "firearm", "knife",
        "other", "unknown",
    }:
        return "object"

    if normalized in {
        "loc", "place", "city", "country", "region", "address",
        "building", "facility", "warehouse", "airport", "port",
    }:
        return "location"

    if normalized in {
        "incident", "transaction", "transfer", "payment",
        "communication", "email", "message", "call", "phone_call",
        "meeting_event", "crime", "arrest",
    }:
        return "event"

    return "object"


def resolve_entity_subtype(entity_type: str | None) -> str | None:
    """
    Return a subtype tag when the raw entity_type carries richer semantics
    than the POLE category alone (e.g. ``"device"``, ``"account"``,
    ``"communication"``).

    Returns ``None`` when no subtype is needed.
    """
    if not entity_type:
        return None
    normalized = entity_type.strip().lower()
    if normalized in CONTROLLED_ENTITY_TYPES:
        return None  # already a top-level POLE type
    if normalized in {
        "asset", "artifact", "equipment", "cargo",
    }:
        return "asset"
    if normalized in {
        "document", "report", "file", "certificate", "letter",
    }:
        return "document"
    if normalized in {
        "account", "bank_account", "iban",
    }:
        return "account"
    if normalized in {
        "device", "phone", "sim", "computer", "server", "laptop",
    }:
        return "device"
    if normalized in {
        "vehicle", "car", "truck", "van", "boat", "plane",
    }:
        return "vehicle"
    if normalized in {
        "weapon", "firearm", "knife",
    }:
        return "weapon"
    if normalized in {
        "communication", "email", "message", "call", "phone_call",
    }:
        return "communication"
    return None
