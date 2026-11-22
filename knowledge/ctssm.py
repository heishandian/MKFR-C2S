from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


ENTITY_TYPES: Tuple[str, ...] = (
    "Cluster", "Node", "Workload", "Container", "Task", "Resource", "Attribute",
)

VIEWS: Tuple[str, ...] = ("RV", "WV", "AV", "AAV")
VIEW_INDEX: Dict[str, int] = {v: i for i, v in enumerate(VIEWS)}


RELATIONS: Dict[str, Tuple[str, ...]] = {
    "RV": ("hasResourceID", "hasResourceType", "hasResourceStatus",
           "hasResourceQuota", "hasRemainingResource"),
    "WV": ("hasTaskID", "hasTaskType", "hasTaskStatus", "requiresResource",
           "hasInstanceNum"),
    "AV": ("dependsOn", "isLocatedAt", "isCoDeployedWith", "hasPreferenceFor",
           "isBackupFor"),
    "AAV": ("isTwinOf", "hasSimilarityWith", "hasAversionTo",
            "isSeparatedFrom", "hasExclusiveNode"),
}

ALL_RELATIONS: Tuple[str, ...] = tuple(r for v in VIEWS for r in RELATIONS[v])
REL_INDEX: Dict[str, int] = {r: i for i, r in enumerate(ALL_RELATIONS)}


REL_VIEW: Dict[str, str] = {r: v for v, rs in RELATIONS.items() for r in rs}


INSTRUCTIONS: Dict[str, Tuple[str, ...]] = {
    "RV": (
        'Extract resource-related triples from the provided "Input Text" in the format (entity1, relation, entity2).',
        'Extract triples focusing on node resource from the provided "Input Text" in the format (entity1, relation, entity2).',
        'From the provided "Input Text," extract triples that describe the attributes of nodes in the format (entity1, relation, entity2).',
    ),
    "WV": (
        'Extract workload-related triples from the provided "Input Text" in the format (entity1, relation, entity2).',
        'Extract triples focusing on tasks from the provided "Input Text" in the format (entity1, relation, entity2).',
        'From the provided "Input Text," extract triples that describe the attributes of workloads in the format (entity1, relation, entity2).',
    ),
    "AV": (
        'Extract affinity-related triples from the provided "Input Text" in the format (entity1, relation, entity2).',
        'Extract triples focusing on dependency relationships of tasks from the provided "Input Text" in the format (entity1, relation, entity2).',
        'From the provided "Input Text," extract triples that describe affinity preferences in the format (entity1, relation, entity2).',
    ),
    "AAV": (
        'Extract anti-affinity-related triples from the provided "Input Text" in the format (entity1, relation, entity2).',
        'Extract triples focusing on mutual exclusivity relationships of tasks or nodes from the provided "Input Text" in the format (entity1, relation, entity2).',
        'From the provided "Input Text," extract triples that describe inverse relationships in the format (entity1, relation, entity2).',
    ),
}


PROMPT_TEMPLATE = """Extract triples in the form of <entity1, relation, entity2> from the following
input text {input_text}. And classify every triple into the most appropriate view.
The available type of views are as follows: resource view {instruction_rv},
workload view {instruction_wv}, affinity view {instruction_av},
anti-affinity view {instruction_aav}.
Resource view focuses on balancing system load, avoiding resource overload or
idleness, such as hasResourceID, hasResourceType, hasResourceStatus,
hasResourceQuota, hasRemainingResource.
Workload view focuses on the precise matching between the resource demands of
workloads and the remaining resources of nodes, such as hasTaskID, hasTaskType,
hasTaskStatus, requiresResource.
Affinity view focuses on the compatibility relationships between workloads, and
between workloads and nodes, such as dependsOn, isLocatedAt, isCoDeployedWith,
hasPreferenceFor.
Anti-affinity view focuses on the mutual exclusion relationships between
workloads, and between workloads and nodes, such as isTwinOf,
hasSimilarityWith, hasAversionTo."""


@dataclass(frozen=True)
class Triple:
    head: str
    relation: str
    tail: str
    view: str

    def as_tuple(self) -> Tuple[str, str, str]:
        return self.head, self.relation, self.tail

    def __str__(self) -> str:
        return f"<{self.head}, {self.relation}, {self.tail}>"


def route(head: str, relation: str, tail: str) -> Triple:
    view = REL_VIEW.get(relation)
    if view is None:
        raise KeyError(f"unknown relation {relation!r}")
    return Triple(head, relation, tail, view)
