"""Shortest paths and path lengths using the A* ("A star") algorithm.
"""
from heapq import heappush, heappop
from itertools import count

import networkx as nx
from networkx.utils import not_implemented_for
from networkx.algorithms.shortest_paths.weighted import _weight_function

__all__ = ['greedy_path', 'greedy_path_length']


def greedy_path(G, source, target, heuristic=None, weight='weight'):
    """Returns a list of nodes in a shortest path between source and target
    using the greedy algorithm.

    There may be more than one shortest path.  This returns only one.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : node
       Ending node for path

    heuristic : function
       A function to evaluate the estimate of the distance
       from the a node to the target.  The function takes
       two nodes arguments and must return a number.

    weight: string, optional (default='weight')
       Edge data key corresponding to the edge weight.

    Raises
    ------
    NetworkXNoPath
        If no path exists between source and target.


    """
    if source not in G or target not in G:
        msg = f"Either source {source} or target {target} is not in G"
        raise nx.NodeNotFound(msg)

    if heuristic is None:
        msg = f"No heuristic is defined"
        raise nx.NetworkXException(msg)

    push = heappush
    pop = heappop
    weight = _weight_function(G, weight)

    # The queue stores priority, node and parent.
    # Uses Python heapq to keep in priority order.
    # Add a counter to the queue to prevent the underlying heap from
    # attempting to compare the nodes themselves. The hash breaks ties in the
    # priority and is guaranteed unique for all nodes in the graph.
    c = count()
    queue = [(0, next(c), source, None)]

    # Maps enqueued nodes to distance of discovered paths and the
    # computed heuristics to target. We avoid computing the heuristics
    # more than once and inserting the node into the queue too many times.
    enqueued = {}
    # Maps explored nodes to parent closest to the source.
    explored = {}

    while queue:
        # Pop the smallest item from queue.
        _, __, curnode, parent = pop(queue)

        if curnode == target:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            return path

        if curnode in explored:
            # Do not override the parent of starting node
            if explored[curnode] is None:
                continue

        explored[curnode] = parent

        for neighbor, _ in G[curnode].items():
            h = heuristic(neighbor, target)
            enqueued[neighbor] = h
            push(queue, (h, next(c), neighbor, curnode))

    raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")

def greedy_path_length(G, source, target, heuristic=None, weight='weight'):
    """Returns the length of the shortest path between source and target using
    the greedy algorithm.

    Parameters
    ----------
    G : NetworkX graph

    source : node
       Starting node for path

    target : node
       Ending node for path

    heuristic : function
       A function to evaluate the estimate of the distance
       from the a node to the target.  The function takes
       two nodes arguments and must return a number.

    Raises
    ------
    NetworkXNoPath
        If no path exists between source and target.

    See Also
    --------
    greedy_path

    """
    if source not in G or target not in G:
        msg = f"Either source {source} or target {target} is not in G"
        raise nx.NodeNotFound(msg)

    weight = _weight_function(G, weight)
    path = greedy_path(G, source, target, heuristic, weight)
    return sum(weight(u, v, G[u][v]) for u, v in zip(path[:-1], path[1:]))

