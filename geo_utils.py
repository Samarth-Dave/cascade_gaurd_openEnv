"""
geo_utils.py — Geo-aware coverage model for CascadeGuard
=========================================================
Implements real-world-inspired radius coverage physics.

Key concepts:
  - Every node has (lat, lon, service_radius_km)
  - Nodes within each other's service radius share coverage:
      if Power_Plant_A covers Hospital_B (within 50 km),
      then a failure of Power_Plant_A partially protects Hospital_B
      from cascade because backup coverage is available
  - Telecom towers reduce observation noise for nodes within range
  - Coverage quality degrades with distance: quality = 1 - (dist / radius)

Usage:
    from cascade_guard.geo_utils import (
        haversine, compute_coverage_matrix,
        compute_cascade_mitigation, compute_obs_confidence
    )

    coverage = compute_coverage_matrix(nodes_with_coords)
    mitigation = compute_cascade_mitigation("HOSP_1", "POWER_DIST_1", coverage, node_states)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance between two (lat, lon) points in kilometres.

    Args:
        lat1, lon1: Coordinates of point 1 in decimal degrees.
        lat2, lon2: Coordinates of point 2 in decimal degrees.

    Returns:
        Distance in kilometres (always >= 0.0).
    """
    R = 6371.0  # Earth radius in km

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return R * c


# ---------------------------------------------------------------------------
# Coverage matrix computation
# ---------------------------------------------------------------------------

def compute_coverage_matrix(
    node_coords: Dict[str, Tuple[float, float, float]]
    # {node_id: (lat, lon, service_radius_km)}
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Build a coverage map: for each node, which other nodes fall inside its
    service radius, and the coverage quality (0.0–1.0, decays with distance).

    Args:
        node_coords: {node_id: (lat, lon, service_radius_km)}

    Returns:
        {node_id: [(covered_node_id, coverage_quality), ...]}
        where coverage_quality = 1 - (distance / radius), clipped to [0, 1].
    """
    coverage: Dict[str, List[Tuple[str, float]]] = {nid: [] for nid in node_coords}

    node_list = list(node_coords.items())
    for i, (nid_a, (lat_a, lon_a, radius_a)) in enumerate(node_list):
        for j, (nid_b, (lat_b, lon_b, _)) in enumerate(node_list):
            if i == j:
                continue
            dist = haversine(lat_a, lon_a, lat_b, lon_b)
            if dist <= radius_a and radius_a > 0:
                quality = max(0.0, 1.0 - dist / radius_a)
                coverage[nid_a].append((nid_b, round(quality, 3)))

    return coverage


# ---------------------------------------------------------------------------
# Cascade mitigation from radius coverage
# ---------------------------------------------------------------------------

def compute_cascade_mitigation(
    target_node_id: str,
    failed_upstream_id: str,
    coverage_matrix: Dict[str, List[Tuple[str, float]]],
    operational_nodes: Dict[str, bool],
    sector_map: Dict[str, str],
) -> float:
    """
    Compute how much of the upstream failure's cascade damage is mitigated by
    alternative nodes covering the target.

    If POWER_DIST_2 is operational and covers HOSP_1 (within its radius),
    and POWER_DIST_1 (HOSP_1's primary) just failed, then the cascade to
    HOSP_1 is reduced by POWER_DIST_2's coverage quality.

    Args:
        target_node_id:     Node receiving cascade damage.
        failed_upstream_id: The node that just failed (causing the cascade).
        coverage_matrix:    Pre-computed coverage map from compute_coverage_matrix().
        operational_nodes:  {node_id: is_operational} — current runtime state.
        sector_map:         {node_id: sector_name}.

    Returns:
        Mitigation factor in [0.0, 1.0].
        0.0 = no mitigation (full cascade damage).
        1.0 = fully mitigated (no cascade damage reaches target).
    """
    failed_sector = sector_map.get(failed_upstream_id, "")

    # Collect all operational nodes covering the target that are in the same sector
    # as the failed node (sector-compatible backup coverage)
    total_backup_quality = 0.0

    for provider_id, covered_list in coverage_matrix.items():
        if not operational_nodes.get(provider_id, False):
            continue
        if provider_id == failed_upstream_id:
            continue
        if sector_map.get(provider_id, "") != failed_sector:
            continue
        # Check if this provider covers the target
        for covered_id, quality in covered_list:
            if covered_id == target_node_id:
                total_backup_quality += quality
                break

    # Mitigation saturates at 1.0 (cannot over-protect)
    return min(1.0, total_backup_quality)


# ---------------------------------------------------------------------------
# Observation confidence from telecom coverage
# ---------------------------------------------------------------------------

def compute_obs_confidence(
    node_id: str,
    telecom_operational: List[Tuple[str, float, float, float]],
    # [(telecom_node_id, lat, lon, service_radius_km)]
    node_lat: float,
    node_lon: float,
) -> float:
    """
    Compute the observation confidence for a node based on telecom coverage.

    If there is no operational telecom node within range, confidence is low
    and the observed health will be noisy.

    Args:
        node_id:             The node being observed.
        telecom_operational: List of operational telecom nodes with coordinates.
        node_lat, node_lon:  Coordinates of the node being observed.

    Returns:
        Confidence in [0.0, 1.0].
        1.0 = perfect telemetry (close to a strong telecom node).
        0.3 = degraded telemetry (no nearby telecom).
        0.0 = no coverage (all telecom nodes down or out of range).
    """
    if not telecom_operational:
        return 0.2  # Background telemetry (radio / satellite fallback)

    best_quality = 0.0
    for _tid, t_lat, t_lon, t_radius in telecom_operational:
        dist = haversine(node_lat, node_lon, t_lat, t_lon)
        if dist <= t_radius and t_radius > 0:
            quality = max(0.0, 1.0 - dist / t_radius)
            best_quality = max(best_quality, quality)

    # Even with 0 coverage quality we keep a 0.2 floor (satellite / radio backup)
    return max(0.2, best_quality)


# ---------------------------------------------------------------------------
# Coverage summary for display / diagnostics
# ---------------------------------------------------------------------------

def coverage_events_summary(
    coverage_matrix: Dict[str, List[Tuple[str, float]]],
    operational_nodes: Dict[str, bool],
    failed_this_step: List[str],
    sector_map: Dict[str, str],
) -> List[str]:
    """
    Return human-readable coverage-event strings for the current step.

    Example output:
        "POWER_GEN_2 (50km) covers HOSP_1 — protects against POWER_DIST_1 failure"

    Args:
        coverage_matrix:   Pre-computed coverage map.
        operational_nodes: Current operational status.
        failed_this_step:  Nodes that failed this step (cascade sources).
        sector_map:        {node_id: sector}.

    Returns:
        List of human-readable event strings (max 8).
    """
    events: List[str] = []
    for failed_id in failed_this_step:
        failed_sector = sector_map.get(failed_id, "")
        for provider_id, covered_list in coverage_matrix.items():
            if not operational_nodes.get(provider_id, False):
                continue
            if sector_map.get(provider_id, "") != failed_sector:
                continue
            for covered_id, quality in covered_list:
                if quality >= 0.3:
                    events.append(
                        f"{provider_id} covers {covered_id} "
                        f"(quality={quality:.2f}) — backup for {failed_id}"
                    )
        if len(events) >= 8:
            break
    return events[:8]


# ---------------------------------------------------------------------------
# Default node coordinates — Mumbai Metro City grid (anonymised)
# ---------------------------------------------------------------------------
# Based on real geographic spread of a major South-Asian metro area.
# Distances are realistic: power plants ~50km apart, hospitals closer.

DEFAULT_NODE_COORDS: Dict[str, Tuple[float, float, float]] = {
    # Power Generation (high radius — large plants serve wide areas)
    "POWER_GEN_1":      (19.076, 72.877, 50.0),   # Central Mumbai
    "POWER_GEN_2":      (19.220, 72.978, 45.0),   # North-East
    # Power Transmission (medium radius)
    "POWER_TRANS_1":    (19.120, 72.840, 30.0),   # Bandra
    "POWER_TRANS_2":    (19.180, 72.960, 30.0),   # Thane
    # Power Distribution (local radius)
    "POWER_DIST_1":     (19.090, 72.860, 20.0),   # Dadar
    "POWER_DIST_2":     (19.150, 72.900, 20.0),   # Kurla
    # Water Treatment (mid-radius — serves large zones)
    "WATER_TREAT_1":    (19.070, 72.830, 25.0),   # Worli
    # Water Pumps (local)
    "WATER_PUMP_1":     (19.060, 72.820, 15.0),   # Lower Parel
    "WATER_PUMP_2":     (19.105, 72.870, 15.0),   # Matunga
    # Water Distribution
    "WATER_DIST_1":     (19.080, 72.855, 12.0),   # Sion
    # Hospitals (small radius — local critical services)
    "HOSP_1":           (19.055, 72.835, 8.0),    # KEM Hospital area
    "HOSP_2":           (19.115, 72.875, 8.0),    # Sion hospital area
    "EMERG_1":          (19.075, 72.845, 10.0),   # Emergency response
    # Telecom (large radius — towers cover wide areas)
    "TELECOM_1":        (19.100, 72.820, 100.0),  # North tower
    "TELECOM_2":        (19.050, 72.900, 100.0),  # South tower
    "TELECOM_SWITCH_1": (19.080, 72.870, 80.0),   # Central switch

    # Metro City task nodes (extended network)
    "POWER_GEN_3":      (19.300, 73.050, 45.0),   # Eastern plant
    "POWER_TRANS_3":    (19.250, 73.000, 28.0),
    "WATER_TREAT_2":    (19.200, 72.850, 22.0),
    "WATER_PUMP_3":     (19.190, 72.840, 14.0),
    "WATER_DIST_2":     (19.210, 72.860, 11.0),
    "HOSP_3":           (19.160, 72.820, 8.0),
    "EMERG_2":          (19.230, 72.950, 9.0),
    "TELECOM_3":        (19.320, 73.080, 95.0),   # Eastern tower
}
