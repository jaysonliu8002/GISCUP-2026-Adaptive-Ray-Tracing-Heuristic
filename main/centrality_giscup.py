"""
vantage_graph.py — vertex-to-vertex VISIBILITY GRAPH and CENTRALITY-based
starting-antenna selection ("vantage seeds").

This is a SEPARATE, coarser notion of "visibility" than the coverage-scoring
viewCache built elsewhere (lib_giscup.candidateViewshed / sweepAllEdges),
which tracks candidate-to-BUILDING-EDGE visibility for tau/coverage scoring.
Everything in this module only asks one binary question per pair of
vertices: "is the straight line between these two corners obstructed by any
building?" It never touches which building, which edge, or how much
perimeter is involved -- that's intentional, since the only thing this
module is used for is picking a small, spatially-spread set of well-
connected corners (by visibility-graph DEGREE, via iterative max-degree-
then-remove-its-edges selection) to hand to greedyCornersCached as fixed
starting antennas. See run_giscup.py's _pickVantageSeeds for how it's wired in.

This file was trimmed down to ONE build strategy on purpose:
buildVisibilityGraphRadiusBVH -- radius-restricted (bounds cost by LOCAL
density, not total dataset size -- an unrestricted V^2 all-pairs graph is
a hard wall at real dataset scale, see that function's docstring), but
UNCAPPED within that radius (every candidate found gets tested, nothing
silently truncated the way a maxNeighbors/maxPolygons cap would). It uses
a shared global shapely STRtree (a bounding volume hierarchy) for coarse
candidate-building lookup, the slab method for a tighter AABB test, early
exit on the first confirmed real intersection, and multiprocessing with
honest pairs/s progress reporting (pool startup timed separately from
actual per-pair processing). Several earlier alternatives (plain bbox-only,
STRtree-indexed-but-still-radius-limited, a fully unrestricted V^2 version,
an angular-sweep version) were built, benchmarked, and removed once this
one was settled on as the one actually in use -- if you want the reasoning
and benchmark history behind that choice, it's in the conversation that
built this file, not preserved here as dead code.

Depends on lib_giscup for: EPS, pointInsidePolygon, buildScene.
buildVisibilityGraphRadiusBVH additionally needs shapely
(pip install shapely), imported lazily inside the functions that need it
so the rest of this module still works without it installed. Its
multiprocessing worker functions (_initRadiusBVHWorker, _radiusBVHEdgesTask)
must be importable as module-level names, same requirement as
lib_giscup's own precomputeViewshedsParallel -- run from a .py script (or
ensure notebooks import this module rather than defining things inline)
and guard the call site with `if __name__ == "__main__":`.
"""

import os

from collections import defaultdict
from lib_giscup import EPS, pointInsidePolygon, buildScene


# ---------------------------------------------------------------------------
# Vertex-to-vertex visibility graph — a SEPARATE, coarser notion of
# "visibility" than the viewCache above (which tracks candidate-to-BUILDING-
# EDGE visibility for coverage scoring). Here an edge means "these two
# corners can see each other in a straight line", used to pick a small,
# spatially-spread set of high-connectivity vertices as good STARTING
# antennas for the real search — not a replacement for it.
# ---------------------------------------------------------------------------

def _properIntersect(p1, p2, p3, p4):
    """
    Strict (open-segment) intersection test between segment p1-p2 and
    segment p3-p4, via orientation cross-products. Returns False for any
    touch confined to an endpoint or exact collinear overlap — this is
    what lets two candidate vertices that share an incident edge (adjacent
    corners of the same polygon) avoid being falsely marked "blocked" by
    that very edge.
    """
    ax, ay = p1; bx, by = p2; cx, cy = p3; dx, dy = p4
    d1 = (dx - cx) * (ay - cy) - (dy - cy) * (ax - cx)
    d2 = (dx - cx) * (by - cy) - (dy - cy) * (bx - cx)
    d3 = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    d4 = (bx - ax) * (dy - ay) - (by - ay) * (dx - ax)
    return (((d1 > EPS and d2 < -EPS) or (d1 < -EPS and d2 > EPS)) and
            ((d3 > EPS and d4 < -EPS) or (d3 < -EPS and d4 > EPS)))

def _segmentIntersectsAABB(ox, oy, dx, dy, minx, miny, maxx, maxy):
    """
    Slab method: does the segment from (ox,oy) to (ox+dx,oy+dy) (i.e.
    parametrized t in [0,1]) pass through the axis-aligned box
    [minx,maxx] x [miny,maxy]? O(1), pure arithmetic -- for each axis,
    compute the t-interval where the segment's LINE is within that axis's
    slab, intersect the two intervals (and [0,1]), and check the result
    is non-empty.

    Tighter than bbox-vs-bbox overlap: a diagonal segment's own bounding
    box can overlap a building's bbox in a region the actual line never
    passes through (segment along y=x has bbox [0,10]x[0,10], which
    overlaps a box at [8,10]x[0,2] even though the diagonal itself never
    enters that corner) -- this test correctly rejects that case.
    """
    tmin, tmax = 0.0, 1.0
    if abs(dx) < EPS:
        if ox < minx or ox > maxx:
            return False
    else:
        t1 = (minx - ox) / dx
        t2 = (maxx - ox) / dx
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin: tmin = t1
        if t2 < tmax: tmax = t2
        if tmin > tmax:
            return False
    if abs(dy) < EPS:
        if oy < miny or oy > maxy:
            return False
    else:
        t1 = (miny - oy) / dy
        t2 = (maxy - oy) / dy
        if t1 > t2:
            t1, t2 = t2, t1
        if t1 > tmin: tmin = t1
        if t2 < tmax: tmax = t2
        if tmin > tmax:
            return False
    return True

def _segmentBlockedIndexed(pA, pB, candidateEntries):
    """
    Like _segmentBlocked, but operates on a candidate list ALREADY
    narrowed by a BVH/STRtree bbox query (candidateEntries), rather than
    doing its own bbox-vs-bbox pass first -- the tree already did that
    part. Adds the slab test as a second-tier filter before the exact
    per-edge check, and returns True (early exit) the instant a real
    crossing is confirmed against actual building geometry, without
    checking any remaining candidates.
    """
    ax, ay = pA
    bx, by = pB
    dx, dy = bx - ax, by - ay

    for verts, edges, bbox in candidateEntries:
        minx, miny, maxx, maxy = bbox
        if not _segmentIntersectsAABB(ax, ay, dx, dy, minx, miny, maxx, maxy):
            continue   # slab rejection -- bbox overlapped, actual line doesn't
        for ea, eb in edges:
            if _properIntersect(pA, pB, ea, eb):
                return True   # early exit: confirmed real intersection

    for t in (0.15, 0.3, 0.45, 0.55, 0.7, 0.85):
        mx, my = ax + dx * t, ay + dy * t
        for verts, edges, bbox in candidateEntries:
            minx, miny, maxx, maxy = bbox
            if mx < minx or mx > maxx or my < miny or my > maxy:
                continue
            if pointInsidePolygon(mx, my, verts, edges):
                return True
    return False

# ---------------------------------------------------------------------------
# Radius-restricted (but NOT K-nearest capped) BVH+slab+multiprocessing
# visibility graph. Same machinery as buildVisibilityGraphUnrestrictedBVH
# (shared global STRtree, slab-method AABB test, early exit, multiprocess
# workers, pairs/s progress reporting with startup timed separately) -- the
# ONLY difference is that pair testing is restricted to candidates within
# `radius` of each other, the same locality assumption buildVisibilityGraph
# uses. This bounds cost by LOCAL density instead of total V (turning the
# O(V^2) wall of the unrestricted version into something governed by how
# many buildings/candidates actually sit near a given vertex), while still
# testing EVERY candidate within that radius -- unlike maxNeighbors/
# maxPolygons capping, nothing within radius is silently skipped.
# ---------------------------------------------------------------------------

_RBW = {}   # per-worker state for the radius-restricted BVH multiprocessing path

def _initRadiusBVHWorker(polygons, candidates, radius, gridCell, node_capacity):
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.strtree import STRtree
    _, _, allVertsEdges = buildScene(polygons)
    shapelyPolys = [ShapelyPolygon(verts) for verts, edges, bbox in allVertsEdges]
    _RBW["allVertsEdges"] = allVertsEdges
    _RBW["tree"] = STRtree(shapelyPolys, node_capacity=node_capacity)
    _RBW["candidates"] = candidates
    _RBW["radius"] = radius
    _RBW["gridCell"] = gridCell
    candBuckets = defaultdict(list)
    for idx, c in enumerate(candidates):
        candBuckets[(int(c[0] // gridCell), int(c[1] // gridCell))].append(idx)
    _RBW["candBuckets"] = candBuckets

def _radiusBVHEdgesTask(i):
    """
    One task = one vertex's outgoing edges to every OTHER vertex with a
    higher index that's also within `radius` (so each pair is computed by
    exactly one task). Gathering nearby candidates uses the same grid-
    bucket approach as buildVisibilityGraphParallel; blocking uses the
    shared global STRtree + slab test + early exit, same as
    buildVisibilityGraphUnrestrictedBVH. NO cap -- every candidate found
    within radius gets tested, none silently dropped.

    Returns (u, visibleTo, nPairsInThisTask) -- see
    _unrestrictedBVHEdgesTask for why pair count (not just vertex count)
    matters for progress reporting.
    """
    import numpy as np
    from shapely import box as shapely_box

    candidates = _RBW["candidates"]
    radius = _RBW["radius"]
    gridCell = _RBW["gridCell"]
    candBuckets = _RBW["candBuckets"]
    tree = _RBW["tree"]
    allVertsEdges = _RBW["allVertsEdges"]

    u = candidates[i]
    cx, cy = int(u[0] // gridCell), int(u[1] // gridCell)
    span = int(radius // gridCell) + 1
    r2 = radius * radius

    nearbyVIdx = []
    for gx in range(cx - span, cx + span + 1):
        for gy in range(cy - span, cy + span + 1):
            for j in candBuckets.get((gx, gy), ()):
                if j <= i:
                    continue
                v = candidates[j]
                if (v[0] - u[0])**2 + (v[1] - u[1])**2 > r2:
                    continue
                nearbyVIdx.append(j)

    if not nearbyVIdx:
        return u, [], 0

    others = [candidates[j] for j in nearbyVIdx]
    vxs = np.array([v[0] for v in others])
    vys = np.array([v[1] for v in others])
    segs = shapely_box(np.minimum(u[0], vxs), np.minimum(u[1], vys),
                       np.maximum(u[0], vxs), np.maximum(u[1], vys))
    pairIdx, polyIdx = tree.query(segs)

    candByPair = defaultdict(list)
    for p, b in zip(pairIdx.tolist(), polyIdx.tolist()):
        candByPair[p].append(allVertsEdges[b])

    visibleTo = [v for pos, v in enumerate(others)
                if not _segmentBlockedIndexed(u, v, candByPair.get(pos, []))]
    return u, visibleTo, len(others)

def buildVisibilityGraphRadiusBVH(polygons, candidates, radius, gridCell=None,
                                  node_capacity=10, processes=None, chunksize=20,
                                  progressEvery=None, verbose=True):
    """
    Radius-restricted, UNCAPPED visibility graph using the same BVH+slab+
    early-exit+multiprocessing machinery as buildVisibilityGraphUnrestrictedBVH,
    bounding cost by local density instead of total V^2. See the module
    comment block above this function for the full reasoning.

    radius        : max distance between two vertices to test them at all
                    -- this is what bounds cost, replacing V^2 with
                    V * local_density. Every candidate within radius gets
                    tested (no maxNeighbors/maxPolygons truncation) --
                    unlike buildVisibilityGraph/buildVisibilityGraphParallel's
                    capping option, nothing within radius is skipped here.
    gridCell      : bucket size for the candidate-point grid. Defaults to radius.
    node_capacity : STRtree leaf capacity, see buildVisibilityGraphLocalIndexed.
    processes     : worker count, defaults to os.cpu_count().
    chunksize     : vertices per worker dispatch. Larger than the
                    unrestricted version's default (20 vs 4) since
                    per-vertex cost is much more UNIFORM here -- radius
                    restriction means a vertex's cost depends on LOCAL
                    density near it, not on its position in the candidate
                    list, so there's no systematic early-vertex-is-
                    expensive skew to rebalance against.
    progressEvery : print a progress line every N completed vertices.

    Note on progress metrics: unlike the unrestricted version, vertices/s
    is NOT systematically biased here (no positional skew -- see chunksize
    above), so it's a reasonable ETA basis on its own; pairs/s is still
    reported for consistency and as a cross-check against local density
    variation across your dataset.

    IMPORTANT: requires an `if __name__ == "__main__":` guard at the call
    site when run as a plain script. Requires shapely.

    Returns {vertex: set(vertex, ...)} -- same shape as buildVisibilityGraph.
    """
    import multiprocessing as mp
    import time

    if gridCell is None:
        gridCell = radius
    if processes is None:
        processes = os.cpu_count() or 1
    V = len(candidates)
    if progressEvery is None:
        progressEvery = max(1, V // 20)

    if verbose:
        print(f"Starting radius-restricted BVH+slab visibility graph: {V} vertices, "
             f"radius={radius}, {processes} processes, node_capacity={node_capacity} "
             f"(uncapped within radius)")
        print(f"  spawning worker pool and building each worker's STRtree "
             f"(over all {len(polygons)} buildings) -- one-time startup, timed separately...")

    adjacency = defaultdict(set)
    poolStartTime = time.perf_counter()
    processingStartTime = None
    completed = 0
    pairsCompleted = 0

    with mp.Pool(processes=processes, initializer=_initRadiusBVHWorker,
                initargs=(polygons, candidates, radius, gridCell, node_capacity)) as pool:
        for u, visibleTo, nPairsInTask in pool.imap_unordered(_radiusBVHEdgesTask, range(V),
                                                              chunksize=chunksize):
            if processingStartTime is None:
                processingStartTime = time.perf_counter()
                if verbose:
                    startupTime = processingStartTime - poolStartTime
                    print(f"  worker pool ready, first result in ({startupTime:.1f}s startup) -- "
                         f"rates below are computed from here, excluding that startup time")
            for v in visibleTo:
                adjacency[u].add(v)
                adjacency[v].add(u)
            completed += 1
            pairsCompleted += nPairsInTask
            if verbose and (completed % progressEvery == 0 or completed == V):
                elapsed = time.perf_counter() - processingStartTime
                vertexRate = completed / elapsed if elapsed > 0 else 0.0
                pairRate = pairsCompleted / elapsed if elapsed > 0 else 0.0
                remaining = (V - completed) / vertexRate if vertexRate > 0 else float('inf')
                nEdgesSoFar = sum(len(nbrs) for nbrs in adjacency.values()) // 2
                print(f"  ... {completed}/{V} vertices ({completed/V*100:.1f}%), "
                     f"{pairsCompleted} pairs tested so far, "
                     f"{elapsed:.1f}s processing, ~{remaining:.1f}s remaining, "
                     f"{vertexRate:.2f} vertices/s ({pairRate:.0f} pairs/s), "
                     f"{nEdgesSoFar} edges found so far")

    if verbose:
        endTime = time.perf_counter()
        processingElapsed = endTime - processingStartTime if processingStartTime else 0.0
        startupElapsed = (processingStartTime or endTime) - poolStartTime
        totalElapsed = endTime - poolStartTime
        nEdges = sum(len(nbrs) for nbrs in adjacency.values()) // 2
        print(f"Done: {V} vertices, {pairsCompleted} pairs tested, {nEdges} edges, "
             f"took {totalElapsed:.1f}s total ({startupElapsed:.1f}s pool startup + "
             f"{processingElapsed:.1f}s processing), "
             f"{V/processingElapsed if processingElapsed>0 else 0:.2f} vertices/s during processing")
    return adjacency

def topKVantageVertices(adjacency, k=5, convexity=None):
    """
    Greedily pick the k highest-DEGREE vertices from a visibility graph.
    After each pick, that vertex's edges are removed from the graph (so its
    former neighbors' degrees drop) BEFORE the next pick is scored — this
    spreads the k picks out spatially instead of clustering them all in a
    single densest neighborhood, since a neighbor "used up" by an earlier
    pick no longer inflates anyone else's count.

    adjacency  : {vertex: set(vertex,...)} from buildVisibilityGraph. Not
                 mutated — this function works on an internal copy.
    convexity  : optional {vertex: score}; breaks ties among equal-degree
                 vertices in favor of sharper outward corners (same
                 tiebreak role it plays in greedyCornersCached).

    Returns [(vertex, degreeAtSelectionTime), ...], length min(k, |V|).
    """
    adj = {v: set(nbrs) for v, nbrs in adjacency.items()}
    if convexity is None:
        convexity = {}
    chosen = []
    for _ in range(k):
        if not adj:
            break
        bestDeg = max(len(nbrs) for nbrs in adj.values())
        tied = [v for v, nbrs in adj.items() if len(nbrs) == bestDeg]
        best = max(tied, key=lambda v: convexity.get(v, 0.0))
        chosen.append((best, bestDeg))
        for nb in adj[best]:
            adj[nb].discard(best)
        del adj[best]
    return chosen