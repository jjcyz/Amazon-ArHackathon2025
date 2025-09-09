"""
Amazon Robotics Hackathon - Routing API

This module defines the routing API for the Amazon Robotics Hackathon.
Students will implement the route_package function in this module.

*****IMPORTANT*****
Team name: cookie-monster
Email address: jess.c.zhou@gmail.com
*******************
"""

from typing import Optional, Dict, List, Tuple, Set
from ar_hackathon.models.game_state import GameState
from ar_hackathon.models.package import Package

# Global cache for shortest paths to avoid recomputation
_path_cache = {}
_network_hash = 0
_adaptive_params = {
    'congestion_penalty': 0.3,
    'future_congestion_penalty': 0.6,
    'bottleneck_penalty': 2.5,
    'load_balance_factor': 0.4
}

class PriorityQueue:
    """Simple priority queue implementation without external libraries."""

    def __init__(self):
        self.items = []

    def push(self, item):
        """Add item to queue and maintain sorted order."""
        self.items.append(item)
        self.items.sort(key=lambda x: x[0])  # Sort by first element (priority)

    def pop(self):
        """Remove and return the item with highest priority (lowest value)."""
        if not self.items:
            raise IndexError("pop from empty queue")
        return self.items.pop(0)

    def empty(self):
        """Check if queue is empty."""
        return len(self.items) == 0

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

def _calculate_heuristic(current_fc: str, destination_fc: str, adj: Dict) -> float:
    """
    Calculate heuristic distance for A* algorithm.
    Uses a simple distance estimation based on network topology.
    """
    if current_fc == destination_fc:
        return 0.0

    # Simple heuristic: estimate minimum possible distance
    # This could be improved with actual geographic coordinates if available
    visited = set()
    queue = [(current_fc, 0)]
    min_distance = float('inf')

    while queue and len(visited) < 10:  # Limit search for performance
        fc, dist = queue.pop(0)
        if fc in visited:
            continue
        visited.add(fc)

        if fc == destination_fc:
            min_distance = min(min_distance, dist)
            break

        for neighbor_fc, weight, _ in adj.get(fc, []):
            if neighbor_fc not in visited:
                queue.append((neighbor_fc, dist + weight))

    return min_distance if min_distance != float('inf') else 10.0

def _adaptive_parameter_adjustment(state: GameState):
    """
    Dynamically adjust algorithm parameters based on network conditions.
    """
    global _adaptive_params

    total_connections = len(state.connections)
    congested_connections = 0
    total_bandwidth_usage = 0

    connection_usage = _get_connection_usage(state)

    for conn in state.connections:
        key = (conn.from_fc, conn.to_fc)
        usage = connection_usage.get(key, 0)

        if conn.bandwidth is not None:
            if usage >= conn.bandwidth * 0.8:  # 80% capacity threshold
                congested_connections += 1
            total_bandwidth_usage += usage / conn.bandwidth if conn.bandwidth > 0 else 0

    congestion_ratio = congested_connections / total_connections if total_connections > 0 else 0
    avg_bandwidth_usage = total_bandwidth_usage / total_connections if total_connections > 0 else 0

    # Adjust parameters based on network congestion
    if congestion_ratio > 0.3:  # High congestion
        _adaptive_params['congestion_penalty'] = min(0.5, _adaptive_params['congestion_penalty'] * 1.2)
        _adaptive_params['future_congestion_penalty'] = min(0.8, _adaptive_params['future_congestion_penalty'] * 1.3)
        _adaptive_params['bottleneck_penalty'] = min(4.0, _adaptive_params['bottleneck_penalty'] * 1.4)
    elif congestion_ratio < 0.1:  # Low congestion
        _adaptive_params['congestion_penalty'] = max(0.1, _adaptive_params['congestion_penalty'] * 0.9)
        _adaptive_params['future_congestion_penalty'] = max(0.3, _adaptive_params['future_congestion_penalty'] * 0.9)
        _adaptive_params['bottleneck_penalty'] = max(1.5, _adaptive_params['bottleneck_penalty'] * 0.9)

    # Adjust load balancing based on average bandwidth usage
    if avg_bandwidth_usage > 0.7:
        _adaptive_params['load_balance_factor'] = min(0.6, _adaptive_params['load_balance_factor'] * 1.2)
    elif avg_bandwidth_usage < 0.3:
        _adaptive_params['load_balance_factor'] = max(0.2, _adaptive_params['load_balance_factor'] * 0.9)

def _simulate_future_usage(state, current_package, proposed_path):
    """
    Simulate future network usage if this package takes the proposed path.
    Returns updated connection usage map.
    """
    # Start with current usage
    future_usage = {}
    for package in state.active_packages:
        if package.in_transit and package.transit_destination:
            key = (package.current_fc, package.transit_destination)
            future_usage[key] = future_usage.get(key, 0) + 1

    # Add our package's proposed path
    for i in range(len(proposed_path) - 1):
        key = (proposed_path[i], proposed_path[i + 1])
        future_usage[key] = future_usage.get(key, 0) + 1

    return future_usage

def _evaluate_path_quality(adj, path, connection_usage, future_usage):
    """
    Evaluate the quality of a path considering current and future congestion.
    Returns a score (lower is better).
    """
    if len(path) < 2:
        return float('inf')

    total_cost = 0
    bottleneck_penalty = 0
    load_balance_bonus = 0

    for i in range(len(path) - 1):
        from_fc = path[i]
        to_fc = path[i + 1]

        # Find the connection details
        weight = float('inf')
        bandwidth = float('inf')
        for neighbor_fc, w, b in adj.get(from_fc, []):
            if neighbor_fc == to_fc:
                weight = w
                bandwidth = b
                break

        if weight == float('inf'):
            return float('inf')  # Invalid path

        total_cost += weight

        # Calculate congestion penalties
        current_usage = connection_usage.get((from_fc, to_fc), 0)
        future_usage_count = future_usage.get((from_fc, to_fc), 0)

        # Current congestion penalty
        if bandwidth != float('inf') and current_usage > 0:
            congestion_ratio = current_usage / bandwidth
            total_cost += weight * congestion_ratio * _adaptive_params['congestion_penalty']

        # Future congestion penalty (more important)
        if bandwidth != float('inf') and future_usage_count > 0:
            future_congestion_ratio = future_usage_count / bandwidth
            total_cost += weight * future_congestion_ratio * _adaptive_params['future_congestion_penalty']

            # Extra penalty for approaching capacity
            if future_usage_count >= bandwidth * 0.8:
                bottleneck_penalty += weight * _adaptive_params['bottleneck_penalty']

            # Load balancing bonus for underutilized connections
            elif future_usage_count < bandwidth * 0.3:
                load_balance_bonus -= weight * _adaptive_params['load_balance_factor'] * 0.1

    return total_cost + bottleneck_penalty + load_balance_bonus

def _find_shortest_path_with_lookahead(adj, start, end, connection_usage, state, current_package):
    """
    Find shortest path using A* algorithm with advanced lookahead and congestion prediction.
    """
    if start == end:
        return [start]

    # A* priority queue: (f_score, g_score, current_fc, path)
    # f_score = g_score + heuristic
    heap = PriorityQueue()
    heap.push((0, 0, start, [start]))

    visited = set()
    g_scores = {start: 0}
    best_path = None
    best_score = float('inf')

    # Limit search depth to avoid infinite loops and improve performance
    max_depth = 15

    while not heap.empty():
        f_score, g_score, current_fc, path = heap.pop()

        if current_fc in visited or len(path) > max_depth:
            continue

        visited.add(current_fc)

        if current_fc == end:
            # Evaluate this complete path
            future_usage = _simulate_future_usage(state, current_package, path)
            path_score = _evaluate_path_quality(adj, path, connection_usage, future_usage)

            if path_score < best_score:
                best_score = path_score
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
                congestion_penalty = weight * congestion_ratio * _adaptive_params['congestion_penalty']

            new_g_score = g_score + weight + congestion_penalty
            new_path = path + [neighbor_fc]

            # Only add if this is a better path to this node
            if neighbor_fc not in g_scores or new_g_score < g_scores[neighbor_fc]:
                g_scores[neighbor_fc] = new_g_score
                heuristic = _calculate_heuristic(neighbor_fc, end, adj)
                f_score = new_g_score + heuristic

                heap.push((f_score, new_g_score, neighbor_fc, new_path))

    return best_path if best_path else []

def _find_shortest_path(adj, start, end, connection_usage):
    """
    Legacy function for backward compatibility - now uses lookahead version.
    """
    return _find_shortest_path_with_lookahead(adj, start, end, connection_usage, None, None)

def _get_optimal_next_hop(state, package):
    """
    Find the optimal next hop for a package using advanced lookahead algorithm.
    """
    global _path_cache, _network_hash

    current_fc = package.current_fc
    destination_fc = package.destination_fc

    # If already at destination, don't move
    if current_fc == destination_fc:
        return None

    # Adjust parameters based on current network conditions
    _adaptive_parameter_adjustment(state)

    # For lookahead algorithm, we need to consider current network state
    # Use a simplified cache key that's more likely to hit
    current_hash = _get_network_hash(state)
    connection_usage = _get_connection_usage(state)

    # Create a simplified cache key (less specific for better hit rate)
    # Only include high-level congestion info, not exact usage counts
    high_congestion_connections = sum(1 for key, usage in connection_usage.items()
                                    if usage > 5)  # Threshold for "high congestion"
    cache_key = (current_fc, destination_fc, current_hash, high_congestion_connections)

    # Check cache first
    if cache_key in _path_cache:
        path = _path_cache[cache_key]
        if len(path) > 1:
            return path[1]  # Return next hop
        return None

    # Build adjacency list
    adj = _build_adjacency_list(state)

    # Find shortest path with lookahead
    path = _find_shortest_path_with_lookahead(adj, current_fc, destination_fc,
                                            connection_usage, state, package)

    # Cache the result with improved cache management
    if len(_path_cache) < 2000:  # Increased cache size
        _path_cache[cache_key] = path
    else:
        # Simple cache eviction: remove oldest 20% of entries
        keys_to_remove = list(_path_cache.keys())[:len(_path_cache)//5]
        for key in keys_to_remove:
            del _path_cache[key]
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

def _get_package_priority(package: Package, state: GameState) -> float:
    """
    Calculate priority for a package based on various factors.
    Higher priority packages get better routing decisions.
    """
    # Base priority: packages that have been waiting longer get higher priority
    wait_time = state.current_time_step - package.entry_time

    # Distance factor: packages with longer distances get higher priority
    # (This is a heuristic - we don't know the exact distance without computing it)
    distance_factor = 1.0  # Could be improved with actual distance calculation

    # Urgency factor: packages closer to deadline get higher priority
    # Assuming packages have some implicit deadline based on network size
    network_size = len(state.fulfillment_centers)
    urgency_factor = max(1.0, network_size / 10.0)

    return wait_time * distance_factor * urgency_factor

def _coordinate_package_routing(state: GameState, packages_to_route: List[Package]) -> Dict[str, str]:
    """
    Coordinate routing decisions for multiple packages to avoid conflicts.
    Returns a mapping of package_id -> next_fc_id.
    """
    # Sort packages by priority (highest first)
    packages_by_priority = sorted(packages_to_route,
                                key=lambda p: _get_package_priority(p, state),
                                reverse=True)

    routing_decisions = {}
    connection_usage = _get_connection_usage(state)

    for package in packages_by_priority:
        current_fc = package.current_fc
        destination_fc = package.destination_fc

        if current_fc == destination_fc:
            routing_decisions[package.id] = None
            continue

        adj = _build_adjacency_list(state)
        best_next_hop = None
        best_score = float('inf')

        # Evaluate all possible next hops
        for neighbor_fc, weight, bandwidth in adj.get(current_fc, []):
            if not _is_valid_move(state, package, neighbor_fc):
                continue

            # Check if this connection is already heavily used by other packages
            connection_key = (current_fc, neighbor_fc)
            current_usage = connection_usage.get(connection_key, 0)

            # Add usage from other packages that have already been routed
            for other_pkg_id, other_next_fc in routing_decisions.items():
                if other_next_fc == neighbor_fc:
                    current_usage += 1

            # Skip if connection would be overloaded
            if bandwidth != float('inf') and current_usage >= bandwidth:
                continue

            # Create a 2-hop path to evaluate
            two_hop_path = [current_fc, neighbor_fc]

            # If this is the destination, it's always good
            if neighbor_fc == destination_fc:
                best_next_hop = neighbor_fc
                break

            # Simulate future usage for this path
            future_usage = _simulate_future_usage(state, package, two_hop_path)

            # Add usage from other routed packages
            for other_pkg_id, other_next_fc in routing_decisions.items():
                if other_next_fc == neighbor_fc:
                    future_usage[connection_key] = future_usage.get(connection_key, 0) + 1

            # Evaluate this path
            path_score = _evaluate_path_quality(adj, two_hop_path, connection_usage, future_usage)

            if path_score < best_score:
                best_score = path_score
                best_next_hop = neighbor_fc

        routing_decisions[package.id] = best_next_hop

        # Update connection usage for next package
        if best_next_hop:
            connection_key = (current_fc, best_next_hop)
            connection_usage[connection_key] = connection_usage.get(connection_key, 0) + 1

    return routing_decisions

def _evaluate_alternative_paths(state, package):
    """
    Evaluate multiple alternative paths and return the best next hop.
    This provides a more sophisticated decision-making process.
    """
    current_fc = package.current_fc
    destination_fc = package.destination_fc

    adj = _build_adjacency_list(state)
    connection_usage = _get_connection_usage(state)

    best_next_hop = None
    best_score = float('inf')

    # Evaluate all possible next hops
    for neighbor_fc, weight, bandwidth in adj.get(current_fc, []):
        if not _is_valid_move(state, package, neighbor_fc):
            continue

        # Create a 2-hop path to evaluate
        two_hop_path = [current_fc, neighbor_fc]

        # If this is the destination, it's always good
        if neighbor_fc == destination_fc:
            return neighbor_fc

        # Simulate future usage for this path
        future_usage = _simulate_future_usage(state, package, two_hop_path)

        # Evaluate this path
        path_score = _evaluate_path_quality(adj, two_hop_path, connection_usage, future_usage)

        # Add bonus for paths that get closer to destination
        # (This is a heuristic - in a real implementation you'd want more sophisticated distance calculation)
        if neighbor_fc != current_fc:
            # Simple heuristic: prefer moves that might lead to shorter overall paths
            remaining_path = _find_shortest_path_with_lookahead(adj, neighbor_fc, destination_fc,
                                                              connection_usage, state, package)
            if remaining_path:
                path_score += len(remaining_path) * 0.1  # Small penalty for longer remaining paths

        if path_score < best_score:
            best_score = path_score
            best_next_hop = neighbor_fc

    return best_next_hop

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

    # Adjust adaptive parameters based on current network state
    _adaptive_parameter_adjustment(state)

    # First, try the optimal path with full lookahead using A* algorithm
    next_fc = _get_optimal_next_hop(state, package)

    # Validate the move
    if next_fc and _is_valid_move(state, package, next_fc):
        return next_fc

    # If optimal path is blocked or not found, evaluate alternative paths
    # This provides a more sophisticated fallback strategy
    alternative_next_hop = _evaluate_alternative_paths(state, package)

    if alternative_next_hop and _is_valid_move(state, package, alternative_next_hop):
        return alternative_next_hop

    # Final fallback: try any available connection with improved congestion handling
    adj = _build_adjacency_list(state)
    connection_usage = _get_connection_usage(state)

    # Sort connections by quality (weight + congestion penalty)
    connections_with_scores = []
    for neighbor_fc, weight, bandwidth in adj.get(package.current_fc, []):
        if not _is_valid_move(state, package, neighbor_fc):
            continue

        connection_key = (package.current_fc, neighbor_fc)
        usage = connection_usage.get(connection_key, 0)

        # Calculate connection score (lower is better)
        score = weight

        # Add congestion penalty
        if bandwidth != float('inf') and usage > 0:
            congestion_ratio = usage / bandwidth
            score += weight * congestion_ratio * _adaptive_params['congestion_penalty']

        # Bonus for direct connection to destination
        if neighbor_fc == package.destination_fc:
            score *= 0.1  # Strong preference for direct connections

        connections_with_scores.append((score, neighbor_fc, weight, bandwidth))

    # Sort by score and try the best connections first
    connections_with_scores.sort(key=lambda x: x[0])

    for score, neighbor_fc, weight, bandwidth in connections_with_scores:
        if _is_valid_move(state, package, neighbor_fc):
            return neighbor_fc

    # No valid moves available
    return None
