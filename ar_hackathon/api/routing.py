"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.
Students will implement the route_package function in this module.

*****IMPORTANT*****
Team name: cookie-monster
Email address: jess.c.zhou@gmail.com
*******************
"""

from typing import Optional
from ar_hackathon.models.game_state import GameState
from ar_hackathon.models.package import Package

# Global cache for shortest paths to avoid recomputation
_path_cache = {}
_network_hash = 0

def _get_network_hash(state):
    """Generate a hash of the network structure for cache invalidation."""
    return hash(tuple((conn.from_fc, conn.to_fc, conn.weight, conn.bandwidth)
                     for conn in state.connections))

def _build_adjacency_list(state):
    """Build adjacency list from connections. Returns {fc_id: [(neighbor_fc, weight, bandwidth)]}"""
    adj = {}
    for fc in state.fulfillment_centers:
        adj[fc.id] = []

    for conn in state.connections:
        if conn.from_fc in adj:
            bandwidth = conn.bandwidth if conn.bandwidth is not None else float('inf')
            adj[conn.from_fc].append((conn.to_fc, conn.weight, bandwidth))

    return adj

def _get_connection_usage(state):
    """Count how many packages are currently using each connection."""
    usage = {}

    # Count packages in transit on each connection
    for package in state.active_packages:
        if package.in_transit and package.transit_destination:
            key = (package.current_fc, package.transit_destination)
            usage[key] = usage.get(key, 0) + 1

    return usage

def _find_shortest_path(adj, start, end, connection_usage):
    """
    Find shortest path using BFS with bandwidth awareness.
    Uses BFS instead of Dijkstra to avoid heapq dependency.
    """
    if start == end:
        return [start]

    # BFS queue: (current_fc, path, total_cost)
    queue = [(start, [start], 0)]
    visited = set()
    best_path = None
    best_cost = float('inf')

    while queue:
        current_fc, path, cost = queue.pop(0)

        if current_fc in visited:
            continue

        visited.add(current_fc)

        if current_fc == end:
            if cost < best_cost:
                best_cost = cost
                best_path = path
            continue

        # Explore neighbors
        for neighbor_fc, weight, bandwidth in adj.get(current_fc, []):
            if neighbor_fc in visited:
                continue

            # Calculate dynamic cost based on bandwidth usage
            connection_key = (current_fc, neighbor_fc)
            usage = connection_usage.get(connection_key, 0)

            # If bandwidth is limited and connection is at capacity, skip
            if bandwidth != float('inf') and usage >= bandwidth:
                continue

            # Add penalty for congested connections
            congestion_penalty = 0
            if bandwidth != float('inf') and usage > 0:
                congestion_ratio = usage / bandwidth
                congestion_penalty = weight * congestion_ratio * 0.1

            new_cost = cost + weight + congestion_penalty
            new_path = path + [neighbor_fc]

            # Add to queue (BFS)
            queue.append((neighbor_fc, new_path, new_cost))

    return best_path if best_path else []

def _get_optimal_next_hop(state, package):
    """
    Find the optimal next hop for a package using cached shortest path algorithm.
    """
    global _path_cache, _network_hash

    current_fc = package.current_fc
    destination_fc = package.destination_fc

    # If already at destination, don't move
    if current_fc == destination_fc:
        return None

    # Check if network structure changed (invalidate cache)
    current_hash = _get_network_hash(state)
    if current_hash != _network_hash:
        _path_cache.clear()
        _network_hash = current_hash

    # Check cache first
    cache_key = (current_fc, destination_fc)
    if cache_key in _path_cache:
        path = _path_cache[cache_key]
        if len(path) > 1:
            return path[1]  # Return next hop
        return None

    # Build adjacency list and get connection usage
    adj = _build_adjacency_list(state)
    connection_usage = _get_connection_usage(state)

    # Find shortest path
    path = _find_shortest_path(adj, current_fc, destination_fc, connection_usage)

    # Cache the result
    _path_cache[cache_key] = path

    # Return next hop if path exists and has more than one node
    if len(path) > 1:
        return path[1]

    return None

def _is_valid_move(state, package, next_fc):
    """Check if moving to next_fc is a valid move considering bandwidth constraints."""
    if package.in_transit or package.current_fc == next_fc:
        return False

    # Find the connection
    connection = state.get_connection(package.current_fc, next_fc)
    if connection is None:
        return False

    # Check bandwidth constraints
    if connection.bandwidth is not None and connection.available_bandwidth <= 0:
        return False

    return True

def route_package(state: GameState, package: Package) -> Optional[str]:
    """
    Determine the next FC to route a package to.

    This is the function that students will implement. The game engine will call
    this function for each package at each time step to determine where to route it.

    Args:
        state: GameState object containing the current state of the network
        package: Package object containing information about the package

    Returns:
        next_fc_id: ID of the next FC to route the package to, or None to stay at current FC
    """
    # If package is already in transit, don't make any moves
    if package.in_transit:
        return None

    # If already at destination, stay put
    if package.current_fc == package.destination_fc:
        return None

    # Get the optimal next hop using our advanced routing algorithm
    next_fc = _get_optimal_next_hop(state, package)

    # Validate the move
    if next_fc and _is_valid_move(state, package, next_fc):
        return next_fc

    # If optimal path is blocked, try alternative paths
    # This is a fallback for when bandwidth constraints block the optimal path
    adj = _build_adjacency_list(state)
    connection_usage = _get_connection_usage(state)

    # Try to find any valid path with higher congestion tolerance
    for neighbor_fc, weight, bandwidth in adj.get(package.current_fc, []):
        if neighbor_fc == package.destination_fc:
            # Direct connection to destination - always try this first
            if _is_valid_move(state, package, neighbor_fc):
                return neighbor_fc

        # Check if this connection is available
        connection_key = (package.current_fc, neighbor_fc)
        usage = connection_usage.get(connection_key, 0)

        if bandwidth == float('inf') or usage < bandwidth:
            if _is_valid_move(state, package, neighbor_fc):
                return neighbor_fc

    # No valid moves available
    return None
