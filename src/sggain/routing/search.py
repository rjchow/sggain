from __future__ import annotations

from dataclasses import asdict, dataclass
import re

import networkx as nx

from sggain.routing.scoring import (
    edge_set_jaccard,
    gain_density,
    length_weighted_containment,
    length_weighted_jaccard,
)


@dataclass
class RouteResult:
    route_id: str
    budget_km: float
    mode: str
    rank: int
    distance_m: float
    ascent_m: float
    descent_m: float
    gain_density_m_per_km: float
    source_confidence: float
    directed_edge_ids: list[str]
    undirected_edge_ids: list[str]
    node_ids: list[str]
    warnings: list[str]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "RouteResult":
        return cls(**payload)


@dataclass(frozen=True)
class _State:
    start_node: str
    current_node: str
    distance_m: float
    ascent_m: float
    descent_m: float
    confidence_sum: float
    directed_edge_ids: tuple[str, ...]
    undirected_edge_ids: tuple[str, ...]
    node_ids: tuple[str, ...]

    @property
    def source_confidence(self) -> float:
        if not self.directed_edge_ids:
            return 0.0
        return self.confidence_sum / len(self.directed_edge_ids)


def search_routes(
    graph: nx.MultiDiGraph,
    budgets_km: list[float],
    modes: list[str],
    beam_width: int = 80,
    max_routes_per_budget: int = 25,
    max_start_nodes: int | None = 2000,
    start_node_grid_m: float | None = 1500,
    max_start_nodes_per_grid: int | None = 25,
    route_diversity_grid_m: float | None = 2500,
    max_routes_per_grid: int | None = 3,
    frontier_grid_m: float | None = 2500,
    max_frontier_states_per_grid: int | None = 20,
    jaccard_similarity_threshold: float = 0.85,
    length_jaccard_similarity_threshold: float = 0.78,
    length_containment_similarity_threshold: float = 0.88,
    landmark_path_name_patterns: list[str] | None = None,
    min_ascent_m: float = 0.0,
) -> list[RouteResult]:
    all_routes: list[RouteResult] = []
    edge_lengths = _undirected_edge_lengths(graph)
    landmark_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in (landmark_path_name_patterns or [])]
    edge_metadata = _directed_edge_metadata(graph)
    landmark_start_nodes = _landmark_start_nodes(graph, landmark_patterns)
    emitted_signatures: set[tuple[str, frozenset[str]]] = set()
    for budget_km in sorted(budgets_km, key=float):
        for mode in modes:
            candidates = _search_one(
                graph,
                float(budget_km) * 1000,
                mode,
                beam_width,
                max_start_nodes,
                start_node_grid_m,
                max_start_nodes_per_grid,
                landmark_start_nodes,
                frontier_grid_m,
                max_frontier_states_per_grid,
                edge_metadata,
                landmark_patterns,
            )
            candidates = [state for state in candidates if state.ascent_m >= min_ascent_m]
            ranked_candidates = _deduplicate(
                candidates,
                graph,
                edge_metadata,
                landmark_patterns,
                edge_lengths,
                jaccard_similarity_threshold,
                length_jaccard_similarity_threshold,
                length_containment_similarity_threshold,
                route_diversity_grid_m,
                max_routes_per_grid,
            )
            ranked = []
            for state in ranked_candidates:
                signature = _route_family_signature(mode, state)
                if signature in emitted_signatures:
                    continue
                emitted_signatures.add(signature)
                ranked.append(state)
                if len(ranked) >= max_routes_per_budget:
                    break
            for rank, state in enumerate(ranked, start=1):
                route_id = f"{mode}_{int(round(float(budget_km) * 1000)):05d}m_{rank:03d}"
                all_routes.append(_state_to_route(route_id, state, float(budget_km), mode, rank))
    return sorted(
        all_routes,
        key=lambda route: (-route.ascent_m, -route.gain_density_m_per_km, -route.source_confidence, route.route_id),
    )


def _route_family_signature(mode: str, state: _State) -> tuple[str, frozenset[str]]:
    return (mode, frozenset(state.undirected_edge_ids))


def _search_one(
    graph: nx.MultiDiGraph,
    budget_m: float,
    mode: str,
    beam_width: int,
    max_start_nodes: int | None,
    start_node_grid_m: float | None,
    max_start_nodes_per_grid: int | None,
    landmark_start_nodes: set[str],
    frontier_grid_m: float | None,
    max_frontier_states_per_grid: int | None,
    edge_metadata: dict[str, dict[str, str]],
    landmark_patterns: list[re.Pattern],
) -> list[_State]:
    if mode not in {"loop", "point_to_point"}:
        raise ValueError(f"Unsupported route mode: {mode}")
    max_edges = max(1, graph.number_of_edges() // 2)
    frontier = [
        _State(
            start_node=node,
            current_node=node,
            distance_m=0.0,
            ascent_m=0.0,
            descent_m=0.0,
            confidence_sum=0.0,
            directed_edge_ids=(),
            undirected_edge_ids=(),
            node_ids=(node,),
        )
        for node in _candidate_start_nodes(
            graph,
            max_start_nodes,
            start_node_grid_m,
            max_start_nodes_per_grid,
            landmark_start_nodes,
        )
    ]
    accepted: list[_State] = []
    while frontier:
        next_frontier: list[_State] = []
        for state in frontier:
            if len(state.undirected_edge_ids) >= max_edges:
                continue
            for _, next_node, _, data in sorted(
                graph.out_edges(state.current_node, keys=True, data=True),
                key=lambda item: (item[3]["directed_edge_id"], str(item[1])),
            ):
                undirected_id = data["undirected_edge_id"]
                if undirected_id in state.undirected_edge_ids:
                    continue
                new_distance = state.distance_m + float(data["length_m"])
                if new_distance > budget_m + 1e-9:
                    continue
                new_state = _State(
                    start_node=state.start_node,
                    current_node=next_node,
                    distance_m=new_distance,
                    ascent_m=state.ascent_m + float(data["ascent_m"]),
                    descent_m=state.descent_m + float(data["descent_m"]),
                    confidence_sum=state.confidence_sum + float(data["source_confidence"]),
                    directed_edge_ids=state.directed_edge_ids + (data["directed_edge_id"],),
                    undirected_edge_ids=state.undirected_edge_ids + (undirected_id,),
                    node_ids=state.node_ids + (next_node,),
                )
                if _is_accepted(new_state, mode):
                    accepted.append(new_state)
                next_frontier.append(new_state)
        frontier = _prune_frontier(
            next_frontier,
            graph,
            beam_width,
            frontier_grid_m,
            max_frontier_states_per_grid,
            edge_metadata,
            landmark_patterns,
        )
    return _rank_states(accepted)


def _candidate_start_nodes(
    graph: nx.MultiDiGraph,
    max_start_nodes: int | None,
    grid_m: float | None,
    max_per_grid: int | None,
    landmark_start_nodes: set[str],
) -> list[str]:
    scored = []
    for node in graph.nodes:
        outgoing = list(graph.out_edges(node, data=True))
        if not outgoing:
            continue
        best_ascent = max(float(data.get("ascent_m", 0.0)) for _, _, data in outgoing)
        total_ascent = sum(float(data.get("ascent_m", 0.0)) for _, _, data in outgoing)
        degree = len(outgoing)
        scored.append((node, best_ascent, total_ascent, degree))
    ranked = sorted(scored, key=lambda item: (-item[1], -item[2], -item[3], str(item[0])))
    if grid_m and max_per_grid:
        ranked = _cap_start_nodes_per_grid(graph, ranked, grid_m, max_per_grid)
    if max_start_nodes is not None and max_start_nodes > 0:
        ranked = ranked[:max_start_nodes]
    nodes = [str(node) for node, _, _, _ in ranked]
    for node in sorted(landmark_start_nodes):
        if node not in nodes:
            nodes.append(node)
    return nodes


def _cap_start_nodes_per_grid(graph: nx.MultiDiGraph, ranked: list[tuple], grid_m: float, max_per_grid: int) -> list[tuple]:
    counts: dict[tuple[int, int], int] = {}
    kept = []
    for item in ranked:
        node = item[0]
        cell = _node_grid_cell(graph, node, grid_m)
        if counts.get(cell, 0) >= max_per_grid:
            continue
        counts[cell] = counts.get(cell, 0) + 1
        kept.append(item)
    return kept


def _is_accepted(state: _State, mode: str) -> bool:
    if not state.directed_edge_ids:
        return False
    if mode == "point_to_point":
        return state.current_node != state.start_node
    return len(state.directed_edge_ids) >= 2 and state.current_node == state.start_node


def _rank_states(states: list[_State]) -> list[_State]:
    return sorted(
        states,
        key=lambda state: (
            -state.ascent_m,
            -gain_density(state.ascent_m, state.distance_m),
            -state.source_confidence,
            state.distance_m,
            state.directed_edge_ids,
        ),
    )


def _prune_frontier(
    states: list[_State],
    graph: nx.MultiDiGraph,
    beam_width: int,
    grid_m: float | None,
    max_per_grid: int | None,
    edge_metadata: dict[str, dict[str, str]],
    landmark_patterns: list[re.Pattern],
) -> list[_State]:
    ranked = _rank_states(states)
    if beam_width <= 0:
        return ranked

    kept: list[_State] = []
    kept_set: set[_State] = set()
    for state in ranked:
        if _matching_landmarks(state, edge_metadata, landmark_patterns, require_steps=True):
            kept.append(state)
            kept_set.add(state)
            if len(kept) >= beam_width:
                return kept

    grid_counts: dict[tuple[int, int], int] = {}
    for state in ranked:
        if state in kept_set:
            continue
        if grid_m and max_per_grid:
            cell = _route_grid_cell(graph, state, grid_m)
            if grid_counts.get(cell, 0) >= max_per_grid:
                continue
            grid_counts[cell] = grid_counts.get(cell, 0) + 1
        kept.append(state)
        if len(kept) >= beam_width:
            break
    return kept


def _deduplicate(
    states: list[_State],
    graph: nx.MultiDiGraph,
    edge_metadata: dict[str, dict[str, str]],
    landmark_patterns: list[re.Pattern],
    edge_lengths: dict[str, float],
    jaccard_threshold: float,
    length_jaccard_threshold: float,
    length_containment_threshold: float,
    route_diversity_grid_m: float | None,
    max_routes_per_grid: int | None,
) -> list[_State]:
    kept: list[_State] = []
    grid_counts: dict[tuple[int, int], int] = {}
    selected_landmarks: set[str] = set()
    _select_landmark_states(states, kept, selected_landmarks, graph, edge_metadata, landmark_patterns, grid_counts, route_diversity_grid_m, max_routes_per_grid, require_steps=True)
    _select_landmark_states(states, kept, selected_landmarks, graph, edge_metadata, landmark_patterns, grid_counts, route_diversity_grid_m, max_routes_per_grid, require_steps=False)
    for state in states:
        if state in kept:
            continue
        if route_diversity_grid_m and max_routes_per_grid:
            cell = _route_grid_cell(graph, state, route_diversity_grid_m)
            if grid_counts.get(cell, 0) >= max_routes_per_grid:
                continue
        if all(
            not _is_near_duplicate(
                list(state.undirected_edge_ids),
                list(existing.undirected_edge_ids),
                edge_lengths,
                jaccard_threshold,
                length_jaccard_threshold,
                length_containment_threshold,
            )
            for existing in kept
        ):
            kept.append(state)
            if route_diversity_grid_m and max_routes_per_grid:
                grid_counts[cell] = grid_counts.get(cell, 0) + 1
    return kept


def _is_near_duplicate(
    left: list[str],
    right: list[str],
    edge_lengths: dict[str, float],
    jaccard_threshold: float,
    length_jaccard_threshold: float,
    length_containment_threshold: float,
) -> bool:
    return (
        edge_set_jaccard(left, right) >= jaccard_threshold
        or length_weighted_jaccard(left, right, edge_lengths) >= length_jaccard_threshold
        or length_weighted_containment(left, right, edge_lengths) >= length_containment_threshold
    )


def _undirected_edge_lengths(graph: nx.MultiDiGraph) -> dict[str, float]:
    lengths: dict[str, float] = {}
    for _, _, data in graph.edges(data=True):
        lengths.setdefault(str(data["undirected_edge_id"]), float(data.get("length_m", 1.0) or 1.0))
    return lengths


def _select_landmark_states(
    states: list[_State],
    kept: list[_State],
    selected_landmarks: set[str],
    graph: nx.MultiDiGraph,
    edge_metadata: dict[str, dict[str, str]],
    landmark_patterns: list[re.Pattern],
    grid_counts: dict[tuple[int, int], int],
    route_diversity_grid_m: float | None,
    max_routes_per_grid: int | None,
    require_steps: bool,
) -> None:
    for state in states:
        if state in kept:
            continue
        for landmark in _matching_landmarks(state, edge_metadata, landmark_patterns, require_steps=require_steps):
            if landmark in selected_landmarks:
                continue
            kept.append(state)
            selected_landmarks.add(landmark)
            if route_diversity_grid_m and max_routes_per_grid:
                cell = _route_grid_cell(graph, state, route_diversity_grid_m)
                grid_counts[cell] = grid_counts.get(cell, 0) + 1
            break


def _directed_edge_metadata(graph: nx.MultiDiGraph) -> dict[str, dict[str, str]]:
    return {
        str(data["directed_edge_id"]): {
            "name": str(data.get("name", "")),
            "highway": str(data.get("highway", "")),
        }
        for _, _, data in graph.edges(data=True)
    }


def _landmark_start_nodes(graph: nx.MultiDiGraph, patterns: list[re.Pattern]) -> set[str]:
    nodes: set[str] = set()
    if not patterns:
        return nodes
    for u, v, data in graph.edges(data=True):
        name = str(data.get("name", ""))
        if any(pattern.search(name) for pattern in patterns):
            nodes.add(str(u))
            nodes.add(str(v))
    return nodes


def _matching_landmarks(
    state: _State,
    edge_metadata: dict[str, dict[str, str]],
    patterns: list[re.Pattern],
    require_steps: bool = False,
) -> list[str]:
    matches: list[str] = []
    if not patterns:
        return matches
    metadata = [edge_metadata.get(edge_id, {"name": "", "highway": ""}) for edge_id in state.directed_edge_ids]
    for pattern in patterns:
        if any(
            pattern.search(item["name"]) and (not require_steps or item["highway"] == "steps")
            for item in metadata
        ):
            matches.append(pattern.pattern)
    return matches


def _route_grid_cell(graph: nx.MultiDiGraph, state: _State, grid_m: float) -> tuple[int, int]:
    xs = [float(graph.nodes[node]["x"]) for node in state.node_ids if node in graph.nodes]
    ys = [float(graph.nodes[node]["y"]) for node in state.node_ids if node in graph.nodes]
    if not xs or not ys:
        return (0, 0)
    return (int((sum(xs) / len(xs)) // grid_m), int((sum(ys) / len(ys)) // grid_m))


def _node_grid_cell(graph: nx.MultiDiGraph, node: str, grid_m: float) -> tuple[int, int]:
    data = graph.nodes[node]
    return (int(float(data["x"]) // grid_m), int(float(data["y"]) // grid_m))


def _state_to_route(route_id: str, state: _State, budget_km: float, mode: str, rank: int) -> RouteResult:
    return RouteResult(
        route_id=route_id,
        budget_km=budget_km,
        mode=mode,
        rank=rank,
        distance_m=state.distance_m,
        ascent_m=state.ascent_m,
        descent_m=state.descent_m,
        gain_density_m_per_km=gain_density(state.ascent_m, state.distance_m),
        source_confidence=state.source_confidence,
        directed_edge_ids=list(state.directed_edge_ids),
        undirected_edge_ids=list(state.undirected_edge_ids),
        node_ids=list(state.node_ids),
        warnings=[],
    )
