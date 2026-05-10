from __future__ import annotations


def gain_density(ascent_m: float, distance_m: float) -> float:
    if distance_m <= 0:
        return 0.0
    return ascent_m / (distance_m / 1000)


def edge_set_jaccard(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    if not union:
        return 1.0
    return len(left_set & right_set) / len(union)


def length_weighted_jaccard(left: list[str], right: list[str], edge_lengths: dict[str, float]) -> float:
    left_set = set(left)
    right_set = set(right)
    union = left_set | right_set
    union_length = sum(edge_lengths.get(edge_id, 1.0) for edge_id in union)
    if union_length <= 0:
        return 1.0
    intersection_length = sum(edge_lengths.get(edge_id, 1.0) for edge_id in left_set & right_set)
    return intersection_length / union_length


def length_weighted_containment(left: list[str], right: list[str], edge_lengths: dict[str, float]) -> float:
    left_set = set(left)
    right_set = set(right)
    left_length = sum(edge_lengths.get(edge_id, 1.0) for edge_id in left_set)
    right_length = sum(edge_lengths.get(edge_id, 1.0) for edge_id in right_set)
    denominator = min(left_length, right_length)
    if denominator <= 0:
        return 1.0
    intersection_length = sum(edge_lengths.get(edge_id, 1.0) for edge_id in left_set & right_set)
    return intersection_length / denominator
