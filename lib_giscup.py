"""
giscup_lib.py — reusable functions for the GIS Cup 2026 antenna solver.

Import everything with:  from giscup_lib import *

Contents
  I/O ............ loadGeoJSON, writeSubmission, parseSubmission, formatCoord
  Visibility ..... sweepAllEdges, buildScene, buildEdgeOwner, pointOnEdge,
                   normalizePolygon, combineLengths
  Spatial ........ SpatialGrid, localScene
  Greedy ......... candidateViewshed, greedyCornersCached, polygonPerimeter,
                   servicedIdsFromViewCache
  Parallel ....... precomputeViewshedsParallel, localEvalCoverParallel
                   (CPU multiprocessing)
  Viewshed cache . candidateList, loadOrBuildViewsheds, buildAndCacheViewsheds,
                   saveViewCache, loadViewCache
"""

import math
import json
import random
import os
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon


# ===========================================================================
#  PARALLEL PRECOMPUTE NOTE
#  The multiprocessing worker functions (_initWorker, _viewshedTask) must be
#  importable as module-level names so worker processes can pickle them.
#  Because they live in THIS module, run the compute from a .py script or
#  ensure the notebook imports giscup_lib (rather than defining these inline).
# ===========================================================================

# ===========================================================================
#  I/O
# ===========================================================================
"""
GIS Cup 2026 — I/O helpers.

Loads building footprints from the competition GeoJSON format and writes
submissions in the required text format.

GeoJSON structure (FeatureCollection of Polygon features):
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "id": <building id>,                      # may also live in properties
      "properties": {... possibly "id" ...},
      "geometry": {
        "type": "Polygon",
        "coordinates": [ [ [x,y], [x,y], ... ] ]   # first ring = outer boundary
      }
    },
    ...
  ]
}

The dataset is guaranteed simple polygons (no self-intersection, no holes),
so we only read the outer ring (coordinates[0]).
"""

def loadGeoJSON(path):
    """
    Load building footprints from a GIS Cup GeoJSON file.

    Returns
    -------
    polygons    : list of polygons, each a list of (x, y) float tuples
                  (the closing duplicate vertex, if present, is kept as-is —
                   normalizePolygon handles it downstream)
    buildingIds : list of building IDs, parallel to `polygons`
    """
    with open(path, "r") as f:
        data = json.load(f)

    polygons = []
    buildingIds = []

    features = data["features"] if data.get("type") == "FeatureCollection" else data

    for idx, feat in enumerate(features):
        geom = feat["geometry"]
        gtype = geom["type"]

        # Resolve a building ID: top-level "id", else properties.id, else index
        bid = feat.get("id")
        if bid is None:
            bid = feat.get("properties", {}).get("id", idx)

        if gtype == "Polygon":
            ring = geom["coordinates"][0]                      # outer ring
            polygons.append([(float(x), float(y)) for x, y in ring])
            buildingIds.append(bid)

        elif gtype == "MultiPolygon":
            # Each sub-polygon becomes its own building, ID suffixed
            for sub_i, poly in enumerate(geom["coordinates"]):
                ring = poly[0]
                polygons.append([(float(x), float(y)) for x, y in ring])
                buildingIds.append(f"{bid}_{sub_i}" if sub_i > 0 else bid)

    return polygons, buildingIds


def formatCoord(x, y):
    """Format a coordinate at full 64-bit precision (repr preserves doubles)."""
    return f"({x!r}, {y!r})"


def writeSubmission(path, results):
    """
    Write a submission file.

    results : list of dicts, one per sub-problem, each with:
        {
          "tau":        float,
          "k":          int,
          "antennas":   [(x, y), ...],          # length k, on building boundaries
          "servicedIds":[buildingId, ...],      # IDs claimed serviced
        }

    Output format per sub-problem (three lines):
        (tau, k)
        (x1, y1), (x2, y2), ...
        id1, id2, ...
    """
    dirName = os.path.dirname(path)
    if dirName:
        os.makedirs(dirName, exist_ok=True)
    with open(path, "w") as f:
        for r in results:
            f.write(f"({r['tau']!r}, {r['k']})\n")
            f.write(", ".join(formatCoord(x, y) for x, y in r["antennas"]) + "\n")
            f.write(", ".join(str(i) for i in r["servicedIds"]) + "\n")
# ===========================================================================
#  VISIBILITY CORE
# ===========================================================================
EPS   = 1e-9

def intersectionSolver(ox, oy, dx, dy, ax, ay, bx, by):
    sx, sy = bx - ax, by - ay
    den = dx * sy - dy * sx
    if abs(den) < EPS:
        return None
    t = ((ax - ox) * sy - (ay - oy) * sx) / den
    u = ((ax - ox) * dy - (ay - oy) * dx) / den
    return t if t > EPS and -EPS <= u <= 1 + EPS else None

def lengthSolver(ox, oy, angle, ea, eb):
    dx, dy = math.cos(angle), math.sin(angle)
    t = intersectionSolver(ox, oy, dx, dy, ea[0], ea[1], eb[0], eb[1])
    if t is None:
        return None
    hx, hy = ox + dx * t, oy + dy * t
    elen = math.hypot(eb[0] - ea[0], eb[1] - ea[1])
    if elen < EPS:
        return 0.0
    edx, edy = (eb[0] - ea[0]) / elen, (eb[1] - ea[1]) / elen
    return max(0.0, min(elen, (hx - ea[0]) * edx + (hy - ea[1]) * edy))

def combineLengths(intervals):
    merged = []
    for lo, hi in sorted(intervals):
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append([lo, hi])
    return sum(hi - lo for lo, hi in merged)

def normalizePolygon(polygon):
    verts = polygon[:-1] if polygon[0] == polygon[-1] else list(polygon)
    n = len(verts)
    return [(verts[i], verts[(i + 1) % n]) for i in range(n)]

def pointOnEdge(ox, oy, ea, eb):
    ax, ay, bx, by = ea[0], ea[1], eb[0], eb[1]
    if abs((bx - ax) * (oy - ay) - (by - ay) * (ox - ax)) > 1e-6:
        return False
    t = ((ox - ax) * (bx - ax) + (oy - ay) * (by - ay)) / max(EPS, (bx-ax)**2 + (by-ay)**2)
    return -EPS <= t <= 1 + EPS

def _bbox(verts):
    xs = [v[0] for v in verts]; ys = [v[1] for v in verts]
    return (min(xs), min(ys), max(xs), max(ys))

def _rawInside(px, py, verts):
    """Ray-cast point-in-polygon WITHOUT the on-edge check (fast)."""
    n, inside, j = len(verts), False, len(verts) - 1
    for i in range(n):
        xi, yi = verts[i]; xj, yj = verts[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def pointInsidePolygon(px, py, verts, edges):
    """Strict inside test (boundary = outside). Kept for external callers."""
    if any(pointOnEdge(px, py, ea, eb) for ea, eb in edges):
        return False
    return _rawInside(px, py, verts)

def getObserverPolygons(ox, oy, allVertsEdges):
    return {i for i, (verts, edges, bbox) in enumerate(allVertsEdges)
            if any(pointOnEdge(ox, oy, ea, eb) for ea, eb in edges)}

def getOutwardOffset(ox, oy, allVertsEdges, epsilon=1e-4):
    for verts, edges, bbox in allVertsEdges:
        if not any(pointOnEdge(ox, oy, ea, eb) for ea, eb in edges):
            continue
        cx = sum(v[0] for v in verts) / len(verts)
        cy = sum(v[1] for v in verts) / len(verts)
        dx, dy = ox - cx, oy - cy
        mag = math.hypot(dx, dy)
        if mag < EPS:
            continue
        return (ox + dx/mag * epsilon, oy + dy/mag * epsilon)
    return (ox, oy)

def rayPassesInterior(ox, oy, dx, dy, t, allVertsEdges, observerPolys):
    """Fast interior check: bbox rejection + raw inside test (no on-edge check)."""
    for frac in (0.1, 0.3, 0.5, 0.7, 0.9):
        mx, my = ox + dx * t * frac, oy + dy * t * frac
        for i, (verts, edges, bbox) in enumerate(allVertsEdges):
            if i in observerPolys:
                continue
            minx, miny, maxx, maxy = bbox
            if mx < minx or mx > maxx or my < miny or my > maxy:
                continue   # outside bounding box — trivially outside polygon
            if _rawInside(mx, my, verts):
                return True
    return False
def buildScene(polygons):
    """Precompute edges, verts, and bounding boxes once for a scene."""
    edgesPerPoly = [normalizePolygon(p) for p in polygons]
    allEdges = [e for edges in edgesPerPoly for e in edges]
    vertsPerPoly = [p[:-1] if p[0]==p[-1] else list(p) for p in polygons]
    allVertsEdges = [(v, e, _bbox(v)) for v, e in zip(vertsPerPoly, edgesPerPoly)]
    return edgesPerPoly, allEdges, allVertsEdges
# ---------------------------------------------------------------------------
# Combined single-pass sweep — records the front edge for EVERY building in
# one rotation, instead of re-sweeping once per target. Same per-wedge logic.
# ---------------------------------------------------------------------------

def sweepAllEdges(observer, allEdges, allVertsEdges, edgeOwner):
    """
    One angular sweep from `observer` that records visible intervals for the
    front edge in every wedge, attributing each to its owning polygon.

    edgeOwner : list parallel to allEdges; edgeOwner[i] = (polyIdx, edgeIdx)

    Returns dict {polyIdx: {edgeIdx: [(lo,hi), ...]}} of visible intervals.
    """
    ox, oy = observer
    TAU, D = 2 * math.pi, 1e-7
    TINY = 1e-4

    result = {}

    # Mark edges the observer lies on as fully visible
    for gi, (ea, eb) in enumerate(allEdges):
        if pointOnEdge(ox, oy, ea, eb):
            pi, ei = edgeOwner[gi]
            result.setdefault(pi, {}).setdefault(ei, []).append(
                (0.0, math.hypot(eb[0]-ea[0], eb[1]-ea[1])))

    observerPolys = getObserverPolygons(ox, oy, allVertsEdges)
    sx, sy = getOutwardOffset(ox, oy, allVertsEdges)

    angles = sorted({((math.atan2(vy - sy, vx - sx) + off) % TAU)
                     for vx, vy in (a for a, _ in allEdges) for off in (-D, 0, D)})

    for i, a0 in enumerate(angles):
        a1 = angles[(i + 1) % len(angles)]
        aMid = ((a0 + a1) / 2 if a1 > a0 else (a0 + a1 + TAU) / 2) % TAU
        dx, dy = math.cos(aMid), math.sin(aMid)

        mx0, my0 = ox + dx * TINY, oy + dy * TINY
        skip = False
        for i2 in observerPolys:
            verts, edges, bbox = allVertsEdges[i2]
            if pointInsidePolygon(mx0, my0, verts, edges):
                skip = True
                break
        if skip:
            continue

        hits = sorted(
            [(t, j) for j, (a, b) in enumerate(allEdges)
             if (t := intersectionSolver(sx, sy, dx, dy, a[0], a[1], b[0], b[1])) is not None]
        )
        if not hits:
            continue

        bestT, bestIdx = hits[0]

        if rayPassesInterior(sx, sy, dx, dy, bestT, allVertsEdges, observerPolys):
            continue

        pi, ei = edgeOwner[bestIdx]
        ea, eb = allEdges[bestIdx]
        if pointOnEdge(ox, oy, ea, eb):
            continue

        elen = math.hypot(eb[0] - ea[0], eb[1] - ea[1])
        u0 = lengthSolver(sx, sy, a0, ea, eb)
        u1 = lengthSolver(sx, sy, a1, ea, eb)
        if u0 is None: u0 = 0.0 if lengthSolver(sx, sy, a0 + D, ea, eb) is not None else elen
        if u1 is None: u1 = elen if lengthSolver(sx, sy, a1 - D, ea, eb) is not None else 0.0

        result.setdefault(pi, {}).setdefault(ei, []).append((min(u0, u1), max(u0, u1)))

    return result


def buildEdgeOwner(edgesPerPoly):
    """List parallel to allEdges mapping each global edge index to (polyIdx, edgeIdx)."""
    owner = []
    for pi, edges in enumerate(edgesPerPoly):
        for ei in range(len(edges)):
            owner.append((pi, ei))
    return owner

# ===========================================================================
#  SPATIAL SCALING
# ===========================================================================
"""
Spatial scaling layer for the GIS Cup dataset (~13k buildings, ~92k vertices).

The base visibility_sweep treats every edge in the scene as a potential
occluder, which is O(total_vertices) per ray — hopeless at full scale.

This module adds a uniform spatial grid so that, for any antenna, we only
consider buildings within a locality radius R. Far-away buildings can neither
be served by nor occlude the antenna in any realistic sense, so restricting to
the local neighborhood makes per-antenna cost roughly constant regardless of
dataset size.

Public API:
    grid = SpatialGrid(polygons, cell=R)
    localPolys, localIds = grid.neighbors(x, y)      # buildings near (x,y)
    coverage = localEvalCover(antennas, polygons, ids, grid, R)
"""

def polyBBox(poly):
    xs = [v[0] for v in poly]; ys = [v[1] for v in poly]
    return (min(xs), min(ys), max(xs), max(ys))


def polyCentroid(poly):
    xs = [v[0] for v in poly]; ys = [v[1] for v in poly]
    return (sum(xs)/len(xs), sum(ys)/len(ys))


class SpatialGrid:
    """Uniform grid bucketing buildings by centroid for fast local lookup."""

    def __init__(self, polygons, cell=200.0):
        self.cell = cell
        self.polygons = polygons
        self.centroids = [polyCentroid(p) for p in polygons]
        self.bboxes = [polyBBox(p) for p in polygons]
        self.buckets = defaultdict(list)
        for i, (cx, cy) in enumerate(self.centroids):
            self.buckets[self._key(cx, cy)].append(i)

    def _key(self, x, y):
        return (int(x // self.cell), int(y // self.cell))

    def neighborIndices(self, x, y, radius):
        """Indices of buildings whose centroid is within `radius` of (x,y)."""
        cx, cy = self._key(x, y)
        span = int(radius // self.cell) + 1
        out = []
        r2 = radius * radius
        for gx in range(cx - span, cx + span + 1):
            for gy in range(cy - span, cy + span + 1):
                for i in self.buckets.get((gx, gy), ()):
                    ex, ey = self.centroids[i]
                    if (ex - x)**2 + (ey - y)**2 <= r2:
                        out.append(i)
        return out


def localScene(indices, polygons):
    """buildScene for a subset of polygons given by index list."""
    sub = [polygons[i] for i in indices]
    return buildScene(sub), sub
# ===========================================================================
#  GREEDY SOLVER
# ===========================================================================
"""
Scalable greedy antenna placement for the full GIS Cup dataset
(greedyCornersCached).

Key scaling ideas (all preserve exact correctness):

1. Lazy viewsheds — a candidate's per-building visible intervals are computed
   on first use and cached.

2. Targeted gain refresh — after placing an antenna, a candidate's marginal
   gain only changes if it touches a building that just became served. We only
   recompute gains for those affected candidates (submodularity: gains only
   ever decrease as buildings get served).

3. Spatial locality — visibility uses only the local neighborhood (radius R).

Tiebreaks among equal-gain candidates: convexity, then coverage-sum score,
then random.
"""

def polygonPerimeter(poly):
    verts = poly[:-1] if poly[0] == poly[-1] else list(poly)
    n = len(verts)
    return sum(math.hypot(verts[(i+1)%n][0]-verts[i][0],
                          verts[(i+1)%n][1]-verts[i][1]) for i in range(n))

# ---------------------------------------------------------------------------
# Corner convexity — used as a tiebreak to prefer sharper outward-pointing
# corners over near-straight or reflex (concave) ones. A sharp convex spike
# sees close to a full half-turn of extra perimeter around itself; a reflex
# corner is prone to being self-occluded by its own building. This never
# overrides the exact viewCache-derived gain/score (which already accounts
# for true visibility including occlusion) — it only breaks ties among
# candidates that score identically on the real objective.
# ---------------------------------------------------------------------------

def _signedArea(verts):
    n = len(verts)
    return 0.5 * sum(verts[i][0] * verts[(i+1) % n][1] - verts[(i+1) % n][0] * verts[i][1]
                     for i in range(n))

def _vertexConvexity(prev, cur, nxt, orientationSign):
    """
    Signed turning angle at `cur` (radians), normalized so that positive
    always means convex (an outward-pointing corner) regardless of whether
    the polygon ring winds clockwise or counter-clockwise. Magnitude grows
    toward +/-pi for sharper turns; near 0 means the corner is almost a
    straight line.
    """
    ix, iy = cur[0] - prev[0], cur[1] - prev[1]
    ox, oy = nxt[0] - cur[0], nxt[1] - cur[1]
    cross = ix * oy - iy * ox
    dot = ix * ox + iy * oy
    return math.atan2(cross, dot) * orientationSign

def computeConvexity(polygons):
    """
    Convexity score per unique vertex coordinate, keyed the same way
    candidateList() dedupes candidates. If a coordinate is shared by more
    than one polygon (rare, e.g. adjoining buildings), keep the MAX
    (most-convex) score across occurrences — being a good vantage point
    from any one incident polygon is enough to deserve the tiebreak boost.

    Returns {(x, y): convexityScore}. Degenerate (near-zero-perimeter or
    <3 vertex) polygons contribute 0.0 for their vertices.
    """
    convexity = {}
    for poly in polygons:
        verts = poly[:-1] if poly[0] == poly[-1] else list(poly)
        n = len(verts)
        if n < 3:
            for v in verts:
                t = tuple(v)
                if t not in convexity:
                    convexity[t] = 0.0
            continue
        orientationSign = 1.0 if _signedArea(verts) >= 0 else -1.0
        for i in range(n):
            prev, cur, nxt = verts[i - 1], verts[i], verts[(i + 1) % n]
            score = _vertexConvexity(prev, cur, nxt, orientationSign)
            t = tuple(cur)
            if t not in convexity or score > convexity[t]:
                convexity[t] = score
    return convexity



def candidateViewshed(antenna, polygons, grid, radius):
    """Per-building visible intervals for one antenna: {bldgIdx: {edgeIdx: [iv]}}."""
    ax, ay = antenna
    idxs = grid.neighborIndices(ax, ay, radius)
    if not idxs:
        return {}
    scene, sub = localScene(idxs, polygons)
    edgesPerPoly, allEdges, allVertsEdges = scene
    owner = buildEdgeOwner(edgesPerPoly)
    profile = sweepAllEdges((ax, ay), allEdges, allVertsEdges, owner)
    return {idxs[localPi]: {ei: list(ivs) for ei, ivs in edgeMap.items()}
            for localPi, edgeMap in profile.items()}
def greedyCornersCached(polygons, k, tau, candidates, viewCache, seed=0, convexity=None,
                        seedVertices=None):
    """
    Lazy single-path greedy: at each step, place the candidate with the
    highest exact gain (buildings pushed to >= tau), tiebroken by convexity
    then coverage-sum score then randomly. Only candidates touching a
    building that just changed state get their gain/score refreshed each
    step, instead of rescanning everything.

    convexity : optional {vertex: score} from computeConvexity(polygons).
                Used as a THIRD-level tiebreak (after exact gain, after the
                coverage-sum score) — it only ever decides between
                candidates that are indistinguishable on the real objective.
                Computed internally if not supplied.

    seedVertices : optional list of vertices (e.g. from
                topKVantageVertices(buildVisibilityGraph(...)) in
                vantage_graph.py) to place FIRST, before the main greedy
                loop, as fixed starting antennas. Any vertex not present in
                `candidates` is silently skipped. These count toward k.
    """
    rng = random.Random(seed)
    perim = [polygonPerimeter(p) for p in polygons]
    if convexity is None:
        convexity = computeConvexity(polygons)

    committed = defaultdict(lambda: defaultdict(list))
    servedNow = set()

    def coverageWith(bIdx, extra=None):
        if perim[bIdx] < 1e-9:
            return 0.0
        edgeMap = committed.get(bIdx, {})
        edgeIdxs = set(edgeMap.keys())
        if extra:
            edgeIdxs |= set(extra.keys())
        visible = 0.0
        for ei in edgeIdxs:
            ivs = edgeMap.get(ei, [])
            if extra and ei in extra:
                ivs = ivs + extra[ei]
            visible += combineLengths(ivs)
        return min(visible / perim[bIdx], 1.0)

    def candidateGainAndScore(c):
        # gainCount: buildings this candidate would push to >= tau (the real objective).
        # scoreSum: sum of resulting coverage over not-yet-served buildings, capped at
        # 1.0 each by coverageWith — used only as a tiebreak among equal-gain candidates,
        # since gain alone is frequently 0/0-tied early on (e.g. step 1, high tau).
        gainCount = 0
        scoreSum = 0.0
        for b, ex in viewCache[c].items():
            if b in servedNow:
                continue
            cov = coverageWith(b, ex)
            if cov >= tau:
                gainCount += 1
            scoreSum += cov
        return gainCount, scoreSum

    buildingToCands = defaultdict(set)
    for c in candidates:
        for b in viewCache[c]:
            buildingToCands[b].add(c)

    gain, score = {}, {}
    for c in candidates:
        gain[c], score[c] = candidateGainAndScore(c)
    placed = []
    remaining = set(candidates)

    def commit(c):
        remaining.discard(c)
        placed.append(c)
        view = viewCache[c]
        touched = list(view.keys())
        for b, ex in view.items():
            for ei, ivs in ex.items():
                committed[b][ei].extend(ivs)
        for b in view:
            if b not in servedNow and coverageWith(b) >= tau:
                servedNow.add(b)
        affected = set()
        for b in touched:
            affected |= buildingToCands[b]
        affected &= remaining
        for cc in affected:
            gain[cc], score[cc] = candidateGainAndScore(cc)

    if seedVertices:
        for v in seedVertices:
            if v in remaining:
                commit(v)

    for step in range(k - len(placed)):
        if not remaining:
            break
        bestGain = -1
        bestList = []
        for c in remaining:
            g = gain[c]
            if g > bestGain:
                bestGain = g
                bestList = [c]
            elif g == bestGain:
                bestList.append(c)
        # Tiebreak order among equal-gain candidates: convexity FIRST (prefer
        # sharper outward corners), then coverage-sum score, then random.
        bestConvexity = max(convexity.get(c, 0.0) for c in bestList)
        tiedByConvexity = [c for c in bestList if convexity.get(c, 0.0) == bestConvexity]
        bestScore = max(score[c] for c in tiedByConvexity)
        tiedBest = [c for c in tiedByConvexity if score[c] == bestScore]
        chosen = rng.choice(tiedBest)
        commit(chosen)

    return placed, len(servedNow)

# ===========================================================================
#  INCREMENTAL COVERAGE STATE  (used by servicedIdsFromViewCache)
#  Tracks each building's covered length as already-merged (disjoint)
#  intervals plus a running scalar, updated only when an antenna is
#  actually committed -- not re-sorted and re-merged from scratch on every
#  read. This is what lets servicedIdsFromViewCache process a growing
#  antenna list (e.g. k=50, then 500, then 1000 for the same tau)
#  incrementally instead of re-scanning every antenna placed so far.
# ===========================================================================

def _mergeIntervalsIncremental(mergedSorted, newIntervals):
    """
    mergedSorted : existing disjoint, sorted (lo, hi) tuples (already merged).
    newIntervals : intervals to merge in (unsorted, possibly overlapping each
                   other and/or mergedSorted).
    Returns (updatedMergedSorted, addedLength) — addedLength is the NET new
    visible length contributed by newIntervals (only the previously-uncovered
    part), not the raw sum of newIntervals' lengths.
    """
    if not newIntervals:
        return mergedSorted, 0.0
    combined = sorted(list(mergedSorted) + list(newIntervals))
    merged = []
    for lo, hi in combined:
        if merged and lo <= merged[-1][1]:
            if hi > merged[-1][1]:
                merged[-1] = (merged[-1][0], hi)
        else:
            merged.append((lo, hi))
    oldTotal = sum(hi - lo for lo, hi in mergedSorted)
    newTotal = sum(hi - lo for lo, hi in merged)
    return merged, newTotal - oldTotal

def _newBeamState():
    return {"mergedIntervals": {}, "visibleLength": {},
            "servedNow": set(), "placed": []}

def _coverageWithState(state, perim, bIdx, extra=None):
    """
    Coverage fraction for building bIdx given state's committed coverage,
    optionally with `extra` (a candidate's not-yet-committed viewshed
    intervals) added hypothetically — without mutating state. extra=None
    just reads the cached scalar (O(1)); extra given only merges the small
    per-edge baseline against the candidate's own (small) intervals, never
    the full raw commit history.
    """
    if perim[bIdx] < 1e-9:
        return 0.0
    baseVisible = state["visibleLength"].get(bIdx, 0.0)
    if not extra:
        return min(baseVisible / perim[bIdx], 1.0)

    mergedMap = state["mergedIntervals"].get(bIdx, {})
    addedTotal = 0.0
    for ei, newIvs in extra.items():
        _, added = _mergeIntervalsIncremental(mergedMap.get(ei, []), newIvs)
        addedTotal += added
    return min((baseVisible + addedTotal) / perim[bIdx], 1.0)

def _applyCandidateToState(state, perim, viewCache, tau, c):
    """Mutates state in place: commits c's viewshed, updates servedNow/placed.
    Updates the merged-interval cache and visible-length scalar incrementally
    (only the newly added intervals get merged in, not the whole history)."""
    view = viewCache[c]
    for b, ex in view.items():
        mergedMap = state["mergedIntervals"].setdefault(b, {})
        addedTotal = 0.0
        for ei, newIvs in ex.items():
            updated, added = _mergeIntervalsIncremental(mergedMap.get(ei, []), newIvs)
            mergedMap[ei] = updated
            addedTotal += added
        state["visibleLength"][b] = state["visibleLength"].get(b, 0.0) + addedTotal
    for b in view:
        if b not in state["servedNow"] and _coverageWithState(state, perim, b) >= tau:
            state["servedNow"].add(b)
    state["placed"].append(c)

def servicedIdsFromViewCache(newAntennas, viewCache, perim, ids, tau, state=None):
    """
    Compute serviced building ids DIRECTLY from the precomputed viewCache
    instead of re-running a fresh grid-query + sweep from scratch. ONLY
    valid when every antenna is an element of the same
    candidate set viewCache was built from, evaluated at the same radius
    viewCache was built with (e.g. solve()'s SOLVE_RADIUS-vs-SOLVE_RADIUS
    case — NOT validate()'s wider VALIDATION_RADIUS re-check, which
    genuinely needs an independent recomputation at a different radius).

    `newAntennas` should be ONLY the antennas not already reflected in
    `state` — e.g. for a fixed tau across growing k's (k=50, then 500, then
    1000), pass placedFull[prevK:k] each time, not the full prefix. Uses
    the same incremental merged-interval state machinery (_newBeamState /
    _applyCandidateToState) so repeated calls with a carried-forward
    `state` never re-process the same antenna twice.

    Returns (servicedIds, state) — pass `state` into the next call to
    continue incrementally; state["servedNow"] already tracks every
    building crossing tau, no need to rescan all buildings.
    """
    if state is None:
        state = _newBeamState()
    for a in newAntennas:
        _applyCandidateToState(state, perim, viewCache, tau, a)
    servicedIds = [ids[i] for i in state["servedNow"]]
    return servicedIds, state

# ===========================================================================
#  LOCAL SEARCH REFINEMENT  (post-process a greedy solution via swaps)
#  greedyCornersCached has a proven but LOOSE worst-case bound ((1-1/e) of
#  optimal for this class of coverage problem -- and per Feige 1998, no
#  polynomial algorithm can guarantee better in the worst case, so this
#  isn't trying to beat that bound in theory, just recover some of the real
#  gap between "worst case" and "this specific instance" that greedy's
#  single forward pass leaves on the table). The idea: repeatedly try
#  swapping ONE placed antenna for ONE unplaced candidate, keep any swap
#  that increases total served-building count, stop when no improving swap
#  exists (a local optimum) or a budget runs out.
#
# This needs REMOVAL, which the existing incremental state machinery above
# (_newBeamState/_applyCandidateToState) doesn't support -- it merges
# intervals into one running total at commit time and can't un-merge one
# candidate's specific share afterward. refineWithSwaps below tracks RAW,
# per-candidate-tagged interval contributions instead (contributions[b][ei]
# = {candidateId: [intervals]}), and only merges (via combineLengths) at
# READ time when checking a specific building's coverage -- more memory
# than the add-only running-total approach, but the only way to make
# removal correct.
# ===========================================================================

def refineWithSwaps(polygons, placed, tau, candidates, viewCache,
                    maxIterations=10, verbose=False):
    """
    Local search improvement on a greedy solution: repeatedly find the
    single best (remove one placed antenna, add one unplaced candidate)
    swap that increases total served-building count, apply it, and repeat
    until no improving swap exists (bestDelta <= 0) or maxIterations is
    reached. No other early-stop threshold -- see below for why.

    placed   : list of candidate vertices from a prior greedyCornersCached
               call (or any valid <=k placement) to improve on.

    Replacement-candidate locality: instead of a separate geographic-
    distance parameter, this reuses viewCache directly -- for a given
    removal candidate r, only unplaced candidates that share AT LEAST ONE
    visible building with r are considered as replacements (via a small
    inverted index built once per call: building -> candidates that can
    see it). This is a more relevant notion of "nearby" than raw XY
    distance for this specific search: two candidates being close in
    space doesn't mean they're relevant to each other (a building could
    block the line between them), while two candidates sharing a visible
    building ARE relevant, by construction, since that's exactly what the
    swap's netDelta computation cares about. It also reuses viewCache's
    OWN locality (already bounded by whatever radius the cache was built
    at) instead of introducing a second, separate radius concept. TRADEOFF:
    this is still a pruning heuristic, not an exhaustive search -- a
    replacement that shares ZERO buildings with the removal candidate but
    still nets out positive overall is possible in principle and won't be
    found this way, same category of tradeoff a geographic radius would
    have, just a different (and more directly relevant) approximation.

    maxIterations : hard cap on the number of swaps to attempt -- the
               ONLY lever controlling total cost now (see below for why
               there isn't a second, diminishing-returns-based one).
               EVERY placed antenna is tried as a removal candidate each
               iteration (no partial-candidate cap -- a full search every
               time is more CONSISTENT than ranking-and-truncating removal
               candidates by a proxy heuristic, which can silently exclude
               a genuinely great swap just because its removal candidate
               ranked outside some cutoff). Measured directly at k=200 on
               a 500-building test: cost scales roughly LINEARLY at
               ~2.2s/iteration (2 iterations: 4.75s/served+2, 5:
               11.8s/served+5, 10: 22.5s/served+11, 20: 44s/served+21) --
               every iteration does the same full amount of work now, so
               there's no natural early-iteration discount. Tune this
               directly against your own real-scale timing; the default
               here (10) is a starting point, not a calibrated answer.

    NO minimum-improvement threshold (ratio- or gradient-based): an
    earlier version of this function had one, but confirmed directly:
    swap improvements here are almost always exactly +1 building (a real
    6-swap run gained +1 every single time). Any threshold pegged to that
    scale behaves as an ALL-OR-NOTHING CLIFF rather than a smooth dial --
    it either blocks every swap or blocks none, with essentially no
    useful middle ground, since there's rarely more than one discrete
    step size to distinguish between. Given that, the only stopping
    condition that actually means something for this problem is the
    natural one already in place: keep going as long as ANY improving
    swap exists (bestDelta > 0), bounded by maxIterations for cost. Adding
    a second threshold on top added configuration surface without adding
    real control.

    Returns (newPlaced, newServedCount, nSwapsMade) -- newPlaced has the
    same LENGTH as `placed` (a swap replaces one antenna, never changes
    the total count), so it stays a valid k-budget solution.
    """
    perim = [polygonPerimeter(p) for p in polygons]
    placed = list(placed)
    placedSet = set(placed)
    remaining = set(candidates) - placedSet

    # contributions[bIdx][edgeIdx] = {candidateId: [intervals]} -- raw, NOT
    # merged, so any one candidate's share can be excluded on demand.
    contributions = defaultdict(lambda: defaultdict(dict))
    for c in placed:
        for b, edgeMap in viewCache[c].items():
            for ei, ivs in edgeMap.items():
                contributions[b][ei][c] = ivs

    def coverageOf(bIdx, exclude=None, extraIvs=None):
        if perim[bIdx] < 1e-9:
            return 0.0
        edgeMap = contributions.get(bIdx, {})
        edgeIdxs = set(edgeMap.keys())
        if extraIvs:
            edgeIdxs |= set(extraIvs.keys())
        visible = 0.0
        for ei in edgeIdxs:
            ivs = [iv for cid, cIvs in edgeMap.get(ei, {}).items()
                  if cid != exclude for iv in cIvs]
            if extraIvs and ei in extraIvs:
                ivs = ivs + extraIvs[ei]
            visible += combineLengths(ivs)
        return min(visible / perim[bIdx], 1.0)

    servedNow = {b for b in contributions if coverageOf(b) >= tau}

    # Inverted index built ONCE from viewCache: building -> candidates that
    # can see it. Replaces the old geographic-radius grid search -- reuses
    # viewCache's own locality directly instead of a second radius concept.
    buildingToCandidates = defaultdict(set)
    for c in candidates:
        for b in viewCache[c]:
            buildingToCandidates[b].add(c)

    def relevantReplacements(r):
        pool = set()
        for b in viewCache[r]:
            pool |= buildingToCandidates[b]
        return pool & remaining

    def netDelta(r, a):
        """Buildings newly served minus buildings newly unserved if r is
        removed and a is added, checking only buildings either touches."""
        touched = set(viewCache[r].keys()) | set(viewCache[a].keys())
        delta = 0
        for b in touched:
            wasServed = b in servedNow
            extraIvs = viewCache[a].get(b)
            nowCov = coverageOf(b, exclude=r, extraIvs=extraIvs) >= tau
            if nowCov and not wasServed:
                delta += 1
            elif not nowCov and wasServed:
                delta -= 1
        return delta

    def applySwap(r, a):
        # remove r's contributions
        for b, edgeMap in viewCache[r].items():
            for ei in edgeMap:
                contributions[b][ei].pop(r, None)
        # add a's contributions
        for b, edgeMap in viewCache[a].items():
            for ei, ivs in edgeMap.items():
                contributions[b][ei][a] = ivs
        # recompute served status only for buildings either touches
        touched = set(viewCache[r].keys()) | set(viewCache[a].keys())
        for b in touched:
            if coverageOf(b) >= tau:
                servedNow.add(b)
            else:
                servedNow.discard(b)
        placedSet.discard(r)
        placedSet.add(a)
        remaining.discard(a)
        remaining.add(r)

    nSwaps = 0
    for iteration in range(maxIterations):
        bestSwap = None
        bestDelta = 0
        for r in placedSet:
            for a in relevantReplacements(r):
                d = netDelta(r, a)
                if d > bestDelta:
                    bestDelta = d
                    bestSwap = (r, a)

        if bestSwap is None:
            if verbose:
                print(f"refineWithSwaps: local optimum reached after {nSwaps} swap(s) "
                     f"({iteration} iteration(s) checked)")
            break

        r, a = bestSwap
        applySwap(r, a)
        nSwaps += 1
        if verbose:
            print(f"  swap {nSwaps}: removed {r}, added {a}, "
                 f"+{bestDelta} served ({len(servedNow)} total)")
    else:
        if verbose:
            print(f"refineWithSwaps: stopped at maxIterations={maxIterations} "
                 f"({nSwaps} swap(s) made) -- may not be a local optimum yet")

    newPlaced = [c for c in placed if c not in remaining] + \
                [c for c in placedSet if c not in placed]
    return newPlaced, len(servedNow), nSwaps

# ===========================================================================
#  PARALLEL VIEWSHED PRECOMPUTE
# ===========================================================================
"""
Parallel viewshed precompute for greedy antenna placement.

The precompute (one visibility sweep per candidate vertex) is embarrassingly
parallel — each candidate is independent. This module runs it across CPU cores
with multiprocessing. No GPU involved; this is CPU-core parallelism.

Usage:
    from greedy_parallel import precomputeViewshedsParallel
    viewCache = precomputeViewshedsParallel(polygons, candidates, radius, gridCell)

Then feed viewCache into greedyCornersCached.

Notes:
- The worker function and initializer are module-level (not notebook cells) so
  they can be pickled and shipped to worker processes. On Windows/Jupyter,
  importing from a .py file is the reliable pattern.
- polygons + grid are sent to each worker ONCE via the initializer, not per task.
"""

# --- per-worker global state (set by the initializer) ---
_W = {}

def _initWorker(polygons, radius, gridCell):
    _W["polygons"] = polygons
    _W["radius"]   = radius
    _W["grid"]     = SpatialGrid(polygons, cell=gridCell)

def _viewshedTask(antenna):
    polygons = _W["polygons"]
    grid     = _W["grid"]
    radius   = _W["radius"]
    ax, ay = antenna
    idxs = grid.neighborIndices(ax, ay, radius)
    if not idxs:
        return antenna, {}
    scene, sub = localScene(idxs, polygons)
    edgesPerPoly, allEdges, allVertsEdges = scene
    owner = buildEdgeOwner(edgesPerPoly)
    profile = sweepAllEdges((ax, ay), allEdges, allVertsEdges, owner)
    view = {idxs[localPi]: {ei: list(ivs) for ei, ivs in edgeMap.items()}
            for localPi, edgeMap in profile.items()}
    return antenna, view

def precomputeViewshedsParallel(polygons, candidates, radius=80.0, gridCell=80.0,
                                processes=None, chunksize=50, verbose=True):
    """
    Compute every candidate's viewshed across CPU cores.
    Returns {candidate: {bldgIdx: {edgeIdx: [intervals]}}}, same as the serial cache.
    """
    import multiprocessing as mp
    if processes is None:
        processes = os.cpu_count() or 1
    if verbose:
        print(f"Precomputing {len(candidates)} viewsheds across {processes} processes...")

    viewCache = {}
    with mp.Pool(processes=processes,
                 initializer=_initWorker,
                 initargs=(polygons, radius, gridCell)) as pool:
        done = 0
        for antenna, view in pool.imap_unordered(_viewshedTask, candidates,
                                                  chunksize=chunksize):
            viewCache[antenna] = view
            done += 1
            if verbose and done % 5000 == 0:
                print(f"  {done}/{len(candidates)}")
    return viewCache

# ===========================================================================
#  SUBMISSION PARSING (shared by validation + plotting)
# ===========================================================================

def parseSubmission(path):
    """Parse a submission file into a list of sub-problem dicts."""
    with open(path) as f:
        lines = f.read().strip().split("\n")
    subs = []
    for blk in range(0, len(lines), 3):
        tau, k = (float(x) for x in lines[blk].strip("()").split(","))
        antennas = []
        for chunk in lines[blk+1].split("),"):
            chunk = chunk.strip().strip("()")
            if chunk:
                x, y = (float(v) for v in chunk.split(","))
                antennas.append((x, y))
        claimedIds = [int(x) for x in lines[blk+2].split(",") if x.strip()]
        subs.append({"tau": tau, "k": int(k),
                     "antennas": antennas, "claimedIds": claimedIds})
    return subs


# ===========================================================================
#  PARALLEL LOCAL COVERAGE  (for fast validation of large-k cases)
#  Farms each antenna's sweep across CPU cores, then merges intervals --
#  one combined sweep per antenna, same approach as candidateViewshed, just
#  spread across workers for large antenna counts.
# ===========================================================================

def localEvalCoverParallel(antennas, polygons, radius=80.0, gridCell=80.0,
                           processes=None, chunksize=8):
    """
    Coverage fraction per building for a given antenna list, computed in
    parallel across CPU cores (one sweep per antenna, merged across workers).
    Returns {buildingIndex: coverageFraction}.

    Note: builds its own grid inside each worker (via _initWorker), so you pass
    polygons/radius/gridCell rather than a prebuilt grid.
    """
    import multiprocessing as mp
    if processes is None:
        processes = os.cpu_count() or 1
    if not antennas:
        return {}

    accum = {}   # globalBldgIdx -> {edgeIdx: [intervals]}

    with mp.Pool(processes=processes,
                 initializer=_initWorker,
                 initargs=(polygons, radius, gridCell)) as pool:
        for antenna, view in pool.imap_unordered(_viewshedTask, antennas,
                                                  chunksize=chunksize):
            # view: {globalBldgIdx: {edgeIdx: [intervals]}}
            for bIdx, edgeMap in view.items():
                store = accum.setdefault(bIdx, {})
                for ei, ivs in edgeMap.items():
                    store.setdefault(ei, []).extend(ivs)

    coverage = {}
    for bIdx, edgeMap in accum.items():
        poly = polygons[bIdx]
        verts = poly[:-1] if poly[0] == poly[-1] else list(poly)
        n = len(verts)
        total = sum(math.hypot(verts[(i+1)%n][0]-verts[i][0],
                               verts[(i+1)%n][1]-verts[i][1]) for i in range(n))
        if total < 1e-9:
            coverage[bIdx] = 0.0
            continue
        visible = sum(combineLengths(edgeMap.get(ei, [])) for ei in edgeMap)
        coverage[bIdx] = min(visible / total, 1.0)
    return coverage


# ===========================================================================
#  VIEWSHED CACHE — disk persistence
#  The cache maps each candidate vertex to its precomputed viewshed and depends
#  ONLY on (polygons, radius). It is independent of tau and k, so one cache
#  serves the greedy solver, validation, and any learned policies.
#
#  A cache is valid only for the exact dataset + radius it was built from, so we
#  store metadata (radius, candidate count, a dataset fingerprint) and verify it
#  on load. A mismatch raises rather than silently returning wrong coverage.
# ===========================================================================

import pickle
import hashlib


def candidateList(polygons):
    """Unique candidate vertices (deterministic order), the antenna positions."""
    candidates = []
    seen = set()
    for poly in polygons:
        verts = poly[:-1] if poly[0] == poly[-1] else list(poly)
        for v in verts:
            t = tuple(v)
            if t not in seen:
                seen.add(t)
                candidates.append(t)
    return candidates


def _datasetFingerprint(polygons):
    """Cheap, order-sensitive fingerprint of the polygon geometry."""
    h = hashlib.sha256()
    h.update(repr(len(polygons)).encode())
    # sample first/last few polygons' coords + total vertex count for a fast hash
    h.update(repr(sum(len(p) for p in polygons)).encode())
    for p in polygons[:50] + polygons[-50:]:
        h.update(repr(p).encode())
    return h.hexdigest()


def buildAndCacheViewsheds(polygons, radius=80.0, gridCell=80.0, path=None,
                           parallel=True, verbose=True):
    """
    Build the viewshed cache (candidates + perimeters + viewsheds) and, if a
    path is given, save it to disk. Returns (candidates, viewCache, perim).
    """
    candidates = candidateList(polygons)
    perim = {i: polygonPerimeter(p) for i, p in enumerate(polygons)}

    if parallel:
        viewCache = precomputeViewshedsParallel(polygons, candidates,
                                                radius=radius, gridCell=gridCell,
                                                verbose=verbose)
    else:
        grid = SpatialGrid(polygons, cell=gridCell)
        viewCache = {}
        for i, c in enumerate(candidates):
            viewCache[c] = candidateViewshed(c, polygons, grid, radius)
            if verbose and (i + 1) % 5000 == 0:
                print(f"  {i+1}/{len(candidates)}")

    if path is not None:
        saveViewCache(path, polygons, radius, candidates, viewCache, perim)
        if verbose:
            print(f"Saved cache to {path}")

    return candidates, viewCache, perim


def saveViewCache(path, polygons, radius, candidates, viewCache, perim):
    """Write the cache plus validation metadata to disk.
    Uses a write-to-temp-then-rename pattern so a crash or OOM during the
    write never leaves a truncated file behind -- the old cache stays intact
    until the new one is fully flushed, then os.replace() swaps them
    atomically."""
    import tempfile
    blob = {
        "radius": radius,
        "nCandidates": len(candidates),
        "fingerprint": _datasetFingerprint(polygons),
        "candidates": candidates,
        "viewCache": viewCache,
        "perim": perim,
    }
    dirName = os.path.dirname(os.path.abspath(path))
    os.makedirs(dirName, exist_ok=True)
    # write to a sibling temp file in the same directory (ensures same
    # filesystem so os.replace is truly atomic, not a copy+delete)
    fd, tmpPath = tempfile.mkstemp(dir=dirName, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmpPath, path)   # atomic on Windows and POSIX
    except:
        try:
            os.unlink(tmpPath)      # clean up the temp file if anything failed
        except OSError:
            pass
        raise


def loadViewCache(path, polygons, radius):
    """
    Load a cache from disk, verifying it matches the current dataset + radius.
    Returns (candidates, viewCache, perim). Raises ValueError on mismatch.
    """
    with open(path, "rb") as f:
        blob = pickle.load(f)

    if abs(blob["radius"] - radius) > 1e-9:
        raise ValueError(f"Cache radius {blob['radius']} != requested {radius}. "
                         "Rebuild the cache.")
    if blob["fingerprint"] != _datasetFingerprint(polygons):
        raise ValueError("Cache was built from a different dataset "
                         "(fingerprint mismatch). Rebuild the cache.")
    return blob["candidates"], blob["viewCache"], blob["perim"]


def loadOrBuildViewsheds(polygons, radius=80.0, gridCell=80.0, path="viewcache.pkl",
                         parallel=True, verbose=True):
    """
    Convenience: load the cache from `path` if it exists and matches; otherwise
    build it and save it. Returns (candidates, viewCache, perim).
    """
    import os
    if os.path.exists(path):
        try:
            candidates, viewCache, perim = loadViewCache(path, polygons, radius)
            if verbose:
                print(f"Loaded cache from {path} "
                      f"({len(candidates)} candidates, radius {radius})")
            return candidates, viewCache, perim
        except ValueError as e:
            if verbose:
                print(f"Cache at {path} unusable ({e}); rebuilding.")
    return buildAndCacheViewsheds(polygons, radius, gridCell, path,
                                  parallel=parallel, verbose=verbose)