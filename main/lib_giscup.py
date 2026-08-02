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
                   Every parallel stage is unconditionally parallel -- there
                   is no serial mode to fall back to -- and gets its worker
                   count from compute.py (settings.CPU_PROCESSES), never
                   from os.cpu_count() here.
  Viewshed cache . candidateList, loadOrBuildViewsheds, buildAndCacheViewsheds,
                   saveViewCache, loadViewCache
"""

import math
import json
import random
import os
import time
import numpy as np
from collections import defaultdict

import compute

# matplotlib is NOT imported at module scope, and must not be: on Windows
# every multiprocessing worker re-imports this module at spawn, and a solve
# creates ~19 pools (precompute + one validation and one refine pool per
# subcase) x os.cpu_count() workers. Importing matplotlib.pyplot costs ~1.0s
# per worker on top of numpy's 0.49s -- pure startup tax paid by processes
# that only ever run geometry. Nothing in this module actually draws, so the
# names below are re-exported lazily (PEP 562) purely so existing callers'
# `from lib_giscup import plt` keeps working; the import happens on first
# attribute access, i.e. only in a process that really plots.

def __getattr__(name):
    if name == "plt":
        import matplotlib.pyplot as plt
        return plt
    if name == "MplPolygon":
        from matplotlib.patches import Polygon as MplPolygon
        return MplPolygon
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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

    Crash-safe: writes to a sibling temp file and os.replace()s it into
    place, so a crash mid-write can never leave a truncated submission --
    the previous version (if any) survives intact. This matters now that
    solve() rewrites the submission after EVERY subcase.
    """
    import tempfile
    dirName = os.path.dirname(path)
    if dirName:
        os.makedirs(dirName, exist_ok=True)
    fd, tmpPath = tempfile.mkstemp(dir=dirName or ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            for r in results:
                f.write(f"({r['tau']!r}, {r['k']})\n")
                f.write(", ".join(formatCoord(x, y) for x, y in r["antennas"]) + "\n")
                f.write(", ".join(str(i) for i in r["servicedIds"]) + "\n")
        os.replace(tmpPath, path)
    except:
        try:
            os.unlink(tmpPath)
        except OSError:
            pass
        raise
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

def sweepAllEdgesReference(observer, allEdges, allVertsEdges, edgeOwner):
    """
    REFERENCE implementation (pure Python, slow). Kept only for
    cross-checking sweepAllEdges (the numpy version below), which was
    verified to produce identical viewsheds on random candidates at both
    solve and commit radii. Not used by the pipeline.

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


def sweepAllEdges(observer, allEdges, allVertsEdges, edgeOwner,
                  radius=None, angleWindows=None):
    """
    Numpy-vectorized angular sweep -- identical wedge logic and output to
    sweepAllEdgesReference (verified bit-compatible on random candidates
    at radii 80 and 160), roughly 2.5x faster at solve radius and 4.5x
    faster at commit radius. The three hot loops are vectorized:

      1. ray-vs-every-edge intersection (the old per-ray Python loop over
         allEdges was ~85% of total sweep cost) -- computed as (angles x
         edges) matrices, chunked to bound temp memory;
      2. the interior-blocking test (5 sample points per winning ray,
         bbox-filtered against every polygon in one boolean matrix, exact
         _rawInside only on surviving pairs);
      3. the "wedge starts inside the observer's own polygon" skip
         (batch ray-cast over all wedge start points at once).

    Corridor support (both optional, back-compatible when omitted):
      radius       : when given, wedges whose nearest hit is beyond this
                     distance (or that hit nothing in the scene) are OPEN
                     -- escape directions down streets/plazas. They are
                     NOT credited (their true content needs a bigger
                     scene) and the function returns (result, openWindows)
                     where openWindows is a merged list of (angStart,
                     angEnd) escape sectors (angEnd may exceed 2*pi to
                     express wraparound).
      angleWindows : restrict the sweep to wedges intersecting these
                     angular windows (same format). Used by the adaptive
                     expansion to probe ONLY previously-open corridors
                     against a larger scene -- the (angles x edges)
                     matrix shrinks to the corridor's angular share.
    """
    ox, oy = observer
    TAU, D = 2 * math.pi, 1e-7
    TINY = 1e-4

    result = {}

    # --- edge arrays (E,) ---
    E = len(allEdges)
    eax = np.fromiter((e[0][0] for e in allEdges), dtype=np.float64, count=E)
    eay = np.fromiter((e[0][1] for e in allEdges), dtype=np.float64, count=E)
    ebx = np.fromiter((e[1][0] for e in allEdges), dtype=np.float64, count=E)
    eby = np.fromiter((e[1][1] for e in allEdges), dtype=np.float64, count=E)
    esx = ebx - eax          # edge direction vectors
    esy = eby - eay

    # vectorized pointOnEdge(ox, oy, edge) for ALL edges at once
    crossO = np.abs(esx * (oy - eay) - esy * (ox - eax))
    lenSq = np.maximum(esx * esx + esy * esy, EPS)
    tO = ((ox - eax) * esx + (oy - eay) * esy) / lenSq
    onEdgeMask = (crossO <= 1e-6) & (tO >= -EPS) & (tO <= 1 + EPS)

    for gi in np.nonzero(onEdgeMask)[0]:
        pi, ei = edgeOwner[gi]
        ea, eb = allEdges[gi]
        result.setdefault(pi, {}).setdefault(ei, []).append(
            (0.0, math.hypot(eb[0]-ea[0], eb[1]-ea[1])))

    observerPolys = {edgeOwner[gi][0] for gi in np.nonzero(onEdgeMask)[0]}
    sx, sy = getOutwardOffset(ox, oy, allVertsEdges)

    rax = eax - sx           # observer -> edge start
    ray_ = eay - sy

    # --- wedge midpoint angles (vectorized atan2 + unique) ---
    base = np.arctan2(eay - sy, eax - sx)
    ang = np.unique(np.concatenate([(base - D) % TAU, base % TAU, (base + D) % TAU]))
    A = len(ang)
    nxt = np.roll(ang, -1)
    if angleWindows:
        # align wedge boundaries with the windows, then keep only wedges
        # overlapping a window
        extra = []
        for w0, w1 in angleWindows:
            extra.append(w0 % TAU); extra.append(w1 % TAU)
        ang = np.unique(np.concatenate([ang, np.asarray(extra, dtype=np.float64)]))
        A = len(ang)
        nxt = np.roll(ang, -1)
    aMid = np.where(nxt > ang, (ang + nxt) / 2.0, (ang + nxt + TAU) / 2.0) % TAU
    dxs = np.cos(aMid)
    dys = np.sin(aMid)

    if angleWindows:
        ws = ang
        we = np.where(nxt > ang, nxt, nxt + TAU)          # wedge [ws, we)
        keep = np.zeros(A, dtype=bool)
        for w0, w1 in angleWindows:
            a = w0 % TAU
            b = a + ((w1 - w0) % TAU or (TAU if (w1 - w0) else 0.0))
            for shift in (-TAU, 0.0, TAU):
                keep |= (ws < b + shift - D) & (we > a + shift + D)
        keepIdx = np.nonzero(keep)[0]
    else:
        keep = np.ones(A, dtype=bool)
        keepIdx = np.arange(A)

    # --- nearest edge per midpoint ray, vectorized, chunked over angles ---
    bestT = np.full(A, np.inf)
    bestIdx = np.full(A, -1, dtype=np.int64)
    K = len(keepIdx)
    dxsK = dxs[keepIdx]; dysK = dys[keepIdx]
    CH = max(1, min(K if K else 1, 8_000_000 // max(E, 1)))   # keep temp arrays modest
    for lo in range(0, K, CH):
        hi = lo + CH
        sel = keepIdx[lo:hi]
        dx = dxsK[lo:hi][:, None]     # (a,1)
        dy = dysK[lo:hi][:, None]
        den = dx * esy[None, :] - dy * esx[None, :]          # (a,E)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (rax[None, :] * esy[None, :] - ray_[None, :] * esx[None, :]) / den
            u = (rax[None, :] * dy - ray_[None, :] * dx) / den
        valid = (np.abs(den) >= EPS) & (t > EPS) & (u >= -EPS) & (u <= 1 + EPS)
        t = np.where(valid, t, np.inf)
        idx = np.argmin(t, axis=1)
        tmin = t[np.arange(t.shape[0]), idx]
        bestT[sel] = tmin
        bestIdx[sel] = np.where(np.isfinite(tmin), idx, -1)

    # --- vectorized interior-blocking test for the winning rays ---
    # restricted to KEPT wedges that actually hit something: at corridor
    # expansions the full angle set A is the whole big scene's event
    # angles while only the corridor's wedges matter, so building (A*5, P)
    # matrices was the dominant cost of the whole adaptive sweep
    hitMask = bestIdx >= 0
    winIdx = np.nonzero(hitMask & keep)[0]                 # (W,)
    W = len(winIdx)
    fracs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    tW = bestT[winIdx]
    mxs = sx + dxs[winIdx, None] * (tW[:, None] * fracs[None, :])   # (W,5)
    mys = sy + dys[winIdx, None] * (tW[:, None] * fracs[None, :])

    P = len(allVertsEdges)
    bminx = np.fromiter((b[0] for _, _, b in allVertsEdges), dtype=np.float64, count=P)
    bminy = np.fromiter((b[1] for _, _, b in allVertsEdges), dtype=np.float64, count=P)
    bmaxx = np.fromiter((b[2] for _, _, b in allVertsEdges), dtype=np.float64, count=P)
    bmaxy = np.fromiter((b[3] for _, _, b in allVertsEdges), dtype=np.float64, count=P)
    obsMask = np.zeros(P, dtype=bool)
    for i2 in observerPolys:
        obsMask[i2] = True

    blockedArr = np.zeros(A, dtype=bool)
    flatX = mxs.ravel(); flatY = mys.ravel()               # (W*5,)
    S = flatX.shape[0]
    CHS = max(1, min(S if S else 1, 4_000_000 // max(P, 1)))
    for lo in range(0, S, CHS):
        hi = lo + CHS
        inBox = ((flatX[lo:hi, None] >= bminx[None, :]) & (flatX[lo:hi, None] <= bmaxx[None, :]) &
                 (flatY[lo:hi, None] >= bminy[None, :]) & (flatY[lo:hi, None] <= bmaxy[None, :]) &
                 (~obsMask[None, :]))
        sIdx, pIdx = np.nonzero(inBox)
        for s_, p_ in zip((sIdx + lo).tolist(), pIdx.tolist()):
            wedge = winIdx[s_ // 5]
            if blockedArr[wedge]:
                continue
            verts = allVertsEdges[p_][0]
            if _rawInside(flatX[s_], flatY[s_], verts):
                blockedArr[wedge] = True

    # --- vectorized "wedge starts inside an observer polygon" skip ---
    startInside = np.zeros(A, dtype=bool)
    if observerPolys:
        px0 = ox + dxsK * TINY                             # kept wedges only
        py0 = oy + dysK * TINY
        for i2 in observerPolys:
            verts, edges, bbox = allVertsEdges[i2]
            vx = np.fromiter((v[0] for v in verts), dtype=np.float64, count=len(verts))
            vy = np.fromiter((v[1] for v in verts), dtype=np.float64, count=len(verts))
            vxj = np.roll(vx, 1); vyj = np.roll(vy, 1)
            inside = np.zeros(K, dtype=bool)
            for xi, yi, xj, yj in zip(vx, vy, vxj, vyj):
                cond = ((yi > py0) != (yj > py0))
                if not cond.any():
                    continue
                with np.errstate(divide='ignore', invalid='ignore'):
                    xint = (xj - xi) * (py0 - yi) / (yj - yi) + xi
                inside ^= cond & (px0 < xint)
            # preserve exact pointInsidePolygon semantics: boundary = outside.
            # vectorized pointOnEdge over (flagged points x polygon edges):
            # |cross| <= 1e-6 and projection parameter within [-EPS, 1+EPS]
            flag = np.nonzero(inside)[0]
            if len(flag):
                fx = px0[flag][:, None]; fy = py0[flag][:, None]
                eax = np.fromiter((ea[0] for ea, eb in edges), dtype=np.float64, count=len(edges))[None, :]
                eay = np.fromiter((ea[1] for ea, eb in edges), dtype=np.float64, count=len(edges))[None, :]
                ebx = np.fromiter((eb[0] for ea, eb in edges), dtype=np.float64, count=len(edges))[None, :]
                eby = np.fromiter((eb[1] for ea, eb in edges), dtype=np.float64, count=len(edges))[None, :]
                dxe = ebx - eax; dye = eby - eay
                cross = np.abs(dxe * (fy - eay) - dye * (fx - eax))
                tproj = ((fx - eax) * dxe + (fy - eay) * dye) / \
                        np.maximum(EPS, dxe * dxe + dye * dye)
                onEdge = (cross <= 1e-6) & (tproj >= -EPS) & (tproj <= 1 + EPS)
                inside[flag[onEdge.any(axis=1)]] = False
            startInside[keepIdx[inside]] = True

    # --- open (escape) wedges: nothing blocks within `radius` -- these
    # are corridor directions; defer them to a bigger-scene sweep ---
    openMask = np.zeros(A, dtype=bool)
    if radius is not None:
        openMask = keep & ~blockedArr & ~startInside & \
                   ((bestIdx < 0) | (bestT > radius))

    # --- per accepted wedge: interval endpoints (few, cheap) ---
    accepted = np.nonzero(keep & (bestIdx >= 0) & ~blockedArr &
                          ~startInside & ~openMask)[0]
    for i in accepted:
        bIdx = bestIdx[i]

        pi, ei = edgeOwner[bIdx]
        ea, eb = allEdges[bIdx]
        if pointOnEdge(ox, oy, ea, eb):
            continue

        a0 = ang[i]
        a1 = ang[(i + 1) % A]
        elen = math.hypot(eb[0] - ea[0], eb[1] - ea[1])
        u0 = lengthSolver(sx, sy, a0, ea, eb)
        u1 = lengthSolver(sx, sy, a1, ea, eb)
        if u0 is None: u0 = 0.0 if lengthSolver(sx, sy, a0 + D, ea, eb) is not None else elen
        if u1 is None: u1 = elen if lengthSolver(sx, sy, a1 - D, ea, eb) is not None else 0.0

        result.setdefault(pi, {}).setdefault(ei, []).append((min(u0, u1), max(u0, u1)))

    if radius is not None:
        openWindows = []
        for i in np.nonzero(openMask)[0]:
            a0 = ang[i]
            a1e = nxt[i] if nxt[i] > ang[i] else nxt[i] + TAU
            if openWindows and abs((openWindows[-1][1] % TAU) - a0) < 1e-12:
                openWindows[-1] = (openWindows[-1][0],
                                   openWindows[-1][1] + (a1e - a0))
            else:
                openWindows.append((a0, a1e))
        return result, openWindows

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
    """Uniform grid bucketing buildings by BOUNDING BOX for fast local
    lookup. (Earlier versions bucketed and filtered by CENTROID, which
    silently dropped any large building whose centroid lay beyond the
    query radius but whose body intruded into the local scene -- losing
    it both as an occluder and as a coverage target, i.e. potential
    false claims. bbox bucketing is a strict superset: viewsheds can
    only get more accurate. Delete stale viewshed caches after this
    change -- the cache validity check keys on dataset+radius only.)"""

    def __init__(self, polygons, cell=200.0):
        self.cell = cell
        self.polygons = polygons
        self.centroids = [polyCentroid(p) for p in polygons]
        self.bboxes = [polyBBox(p) for p in polygons]
        self.buckets = defaultdict(list)
        for i, (minx, miny, maxx, maxy) in enumerate(self.bboxes):
            gx0, gy0 = self._key(minx, miny)
            gx1, gy1 = self._key(maxx, maxy)
            for gx in range(gx0, gx1 + 1):
                for gy in range(gy0, gy1 + 1):
                    self.buckets[(gx, gy)].append(i)

    def _key(self, x, y):
        return (int(x // self.cell), int(y // self.cell))

    def neighborIndices(self, x, y, radius):
        """Indices of buildings whose bounding box comes within `radius`
        of (x, y) -- deduplicated, in ascending index order."""
        cx, cy = self._key(x, y)
        span = int(radius // self.cell) + 1
        seen = set()
        r2 = radius * radius
        for gx in range(cx - span, cx + span + 1):
            for gy in range(cy - span, cy + span + 1):
                for i in self.buckets.get((gx, gy), ()):
                    if i in seen:
                        continue
                    minx, miny, maxx, maxy = self.bboxes[i]
                    # distance from (x,y) to the bbox (clamp then measure)
                    ddx = (minx - x) if x < minx else ((x - maxx) if x > maxx else 0.0)
                    ddy = (miny - y) if y < miny else ((y - maxy) if y > maxy else 0.0)
                    if ddx * ddx + ddy * ddy <= r2:
                        seen.add(i)
        return sorted(seen)


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
    # radius cap: the scene only contains buildings whose bbox comes within
    # `radius`, so a wedge hit BEYOND it could be missing its true occluder
    # entirely -- crediting it risks a false claim. Beyond-radius wedges are
    # dropped (the adaptive path re-solves them against a bigger scene).
    profile, _beyondRadius = sweepAllEdges((ax, ay), allEdges, allVertsEdges,
                                           owner, radius=radius)
    return normalizeView(
        {idxs[localPi]: {ei: list(ivs) for ei, ivs in edgeMap.items()}
         for localPi, edgeMap in profile.items()})
def buildExtraLen(candidates, viewCache):
    """Static per-(candidate, building) UNMERGED interval-length totals.
    baseVis + extraLen is a sound upper bound on merged coverage, letting
    threshold checks skip the exact merge for sliver touches. Depends only
    on the viewCache, so build it ONCE per cache and pass it to
    greedyCornersCached / greedyLookahead / refineWithSwapsV2 instead of
    letting each of them rebuild it per subcase."""
    out = {}
    for c in candidates:
        d = {}
        for b, em in viewCache[c].items():
            d[b] = sum(hi - lo for ivs in em.values() for lo, hi in ivs)
        out[c] = d
    return out


class _CoverageState:
    """Incremental committed-coverage state shared by greedyCornersCached
    and greedyLookahead.

    Per building: merged committed intervals + visible length (merged once
    per commit, never re-merged from scratch). Per candidate: the exact
    tip set (unserved buildings it would push >= tau; gain = its size) and
    the coverage-sum tiebreak score, both updated per (candidate,
    building) when a commit changes that building -- so commit cost is
    bounded by (touched buildings) x (candidates per building), instead of
    the old full-candidate rescans of every affected candidate."""

    def __init__(self, polygons, tau, candidates, viewCache, extraLen=None,
                 baseCov=None, buildingToCands=None):
        self.tau = tau
        self.viewCache = viewCache
        self.perim = [polygonPerimeter(p) for p in polygons]
        self.extraLen = extraLen if extraLen is not None \
                        else buildExtraLen(candidates, viewCache)
        self.merged = {}          # b -> {ei: mergedSorted intervals}
        self.vis = {}             # b -> committed visible length
        self.servedNow = set()
        self.remaining = set(candidates)
        if buildingToCands is not None:
            # shared read-only index built once per solve (never mutated here)
            self.buildingToCands = buildingToCands
        else:
            self.buildingToCands = defaultdict(set)
            for c in candidates:
                for b in viewCache[c]:
                    self.buildingToCands[b].add(c)
        self.tips, self.gain, self.score, self.lastCov = {}, {}, {}, {}
        if baseCov is not None:
            # tau-independent standalone coverages precomputed once per
            # cache (baseCovFromPacked) -- no per-subcase interval merging
            for c in candidates:
                t = set(); ssum = 0.0
                bc = baseCov[c]
                for b, cov in bc.items():
                    ssum += cov
                    if cov >= tau:
                        t.add(b)
                self.tips[c] = t
                self.gain[c] = len(t)
                self.score[c] = ssum
                self.lastCov[c] = dict(bc)    # copy: commit() mutates it
            return
        for c in candidates:
            t = set(); ssum = 0.0; lc = {}
            for b, ex in viewCache[c].items():
                pb = self.perim[b]
                if pb < 1e-9:
                    lc[b] = 0.0
                    continue
                cov = _covWithExtra({}, 0.0, ex, pb)
                lc[b] = cov
                ssum += cov
                if cov >= tau:
                    t.add(b)
            self.tips[c] = t
            self.gain[c] = len(t)
            self.score[c] = ssum
            self.lastCov[c] = lc

    def commit(self, c):
        """Merge candidate c's viewshed into the committed state and update
        every touched (candidate, building) pair. Returns newly served."""
        self.remaining.discard(c)
        view = self.viewCache[c]
        newlyServed = set()
        for b, ex in view.items():
            m = self.merged.setdefault(b, {})
            v = self.vis.get(b, 0.0)
            for ei, ivs in ex.items():
                lst, added = _mergeIntervalsIncremental(m.get(ei, []), ivs)
                m[ei] = lst
                v += added
            self.vis[b] = v
            pb = self.perim[b]
            if b not in self.servedNow and pb > 1e-9 and v / pb >= self.tau:
                self.servedNow.add(b)
                newlyServed.add(b)
        for b in view:
            pb = self.perim[b]
            if pb < 1e-9:
                continue
            if b in self.servedNow:
                if b in newlyServed:      # leaves every score/tip set once
                    for a in self.buildingToCands.get(b, ()):
                        if a not in self.remaining:
                            continue
                        old = self.lastCov[a].pop(b, None)
                        if old is not None:
                            self.score[a] -= old
                        t = self.tips[a]
                        if b in t:
                            t.discard(b)
                            self.gain[a] -= 1
                continue
            mB, vB = self.merged[b], self.vis[b]
            for a in self.buildingToCands.get(b, ()):
                if a not in self.remaining:
                    continue
                cov = _covWithExtra(mB, vB, self.viewCache[a].get(b), pb)
                lc = self.lastCov[a]
                self.score[a] += cov - lc.get(b, 0.0)
                lc[b] = cov
                t = self.tips[a]
                if cov >= self.tau:
                    if b not in t:
                        t.add(b)
                        self.gain[a] += 1
                elif b in t:
                    t.discard(b)
                    self.gain[a] -= 1
        return newlyServed


def greedyCornersCached(polygons, k, tau, candidates, viewCache, seed=0, convexity=None,
                        seedVertices=None, extraLen=None, baseCov=None,
                        buildingToCands=None):
    """
    Lazy single-path greedy: at each step, place the candidate with the
    highest exact gain (buildings pushed to >= tau), tiebroken by convexity
    then coverage-sum score then randomly. Runs on _CoverageState, so each
    placement only updates the (candidate, building) pairs the commit
    actually touched -- the old version fully rescanned every affected
    candidate per commit, which dominated greedy time at scale.

    convexity : optional {vertex: score} from computeConvexity(polygons).
                Third-level tiebreak only.
    seedVertices : optional vertices to place FIRST (count toward k);
                anything not in `candidates` is silently skipped.
    extraLen  : optional precomputed buildExtraLen(candidates, viewCache);
                pass it when calling repeatedly on the same cache.
    baseCov, buildingToCands : optional shared precomputes (see
                baseCovFromPacked / buildBuildingToCandidates) that make
                _CoverageState construction cheap on repeated calls.
    """
    rng = random.Random(seed)
    if convexity is None:
        convexity = computeConvexity(polygons)
    st = _CoverageState(polygons, tau, candidates, viewCache, extraLen,
                        baseCov=baseCov, buildingToCands=buildingToCands)
    placed = []

    def commit(c):
        st.commit(c)
        placed.append(c)

    if seedVertices:
        for v in seedVertices:
            if v in st.remaining:
                commit(v)

    gain, score, remaining = st.gain, st.score, st.remaining
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

    return placed, len(st.servedNow)

def greedyLookahead(polygons, k, tau, candidates, viewCache, beamWidth=6,
                    depth=2, seed=0, convexity=None, seedVertices=None,
                    extraLen=None, verbose=False, innerBeam=None,
                    baseCov=None, buildingToCands=None):
    """
    Beam-lookahead greedy with configurable ply depth: each step scores
    the `beamWidth` most promising candidates as
        totalScore(c) = gain(c) + best gain chain of (depth - 1) further
                        placements, each evaluated EXACTLY against the
                        hypothetical state
    and places the argmax. depth=1 degenerates to plain greedy (and just
    delegates to greedyCornersCached, zero overhead); depth=2 sees pair
    synergies; depth=3 sees triples. See README for the cost model and
    when each depth pays off.

    Fast because nothing is ever mutated or undone: a hypothetical
    placement lives in an OVERLAY {building: (merged, vis)} covering only
    its touched buildings, chained overlays stack for deeper plies, and a
    candidate's gain under an overlay is derived from its global tip set:
        g(x | overlay) = gain(x) - |tips(x) & servedByOverlay| + newly
                         enabled tips on overlay buildings
    where only overlay buildings need per-building checks (with the
    extraLen upper bound skipping slivers) and untouched candidates keep
    their global gain exactly, so their maximum comes from one sorted top
    slice per step. Inner plies use a beam of max(2, beamWidth // 2).
    """
    if depth <= 1:
        return greedyCornersCached(polygons, k, tau, candidates, viewCache,
                                   seed=seed, convexity=convexity,
                                   seedVertices=seedVertices, extraLen=extraLen,
                                   baseCov=baseCov, buildingToCands=buildingToCands)
    import heapq
    rng = random.Random(seed)
    if convexity is None:
        convexity = computeConvexity(polygons)
    st = _CoverageState(polygons, tau, candidates, viewCache, extraLen,
                        baseCov=baseCov, buildingToCands=buildingToCands)
    perim, viewC = st.perim, st.viewCache
    extraLen = st.extraLen
    gain, score, tips, remaining = st.gain, st.score, st.tips, st.remaining
    if innerBeam is None:
        # depth >= 3 needs a full-width inner beam: a truncated one makes
        # the deep estimate noisy, and noisy-deep loses to accurate-shallow
        # (measured: depth 3 with half-width inner beam scored BELOW depth
        # 2; with full width it scored above)
        innerBeam = beamWidth if depth >= 3 else max(2, beamWidth // 2)
    placed = []

    if seedVertices:
        for v in seedVertices:
            if v in st.remaining:
                st.commit(v)
                placed.append(v)

    def extendOverlay(overlayMap, x):
        """Overlay with x's viewshed merged on top (stacking on any
        buildings already overlaid); also returns x's exact tip set under
        the PREVIOUS overlay -- the buildings x newly serves there."""
        out = dict(overlayMap)
        for b, ex in viewC[x].items():
            pb = perim[b]
            if pb < 1e-9 or b in st.servedNow:
                continue
            m, v = out.get(b) or (st.merged.get(b, {}), st.vis.get(b, 0.0))
            m = dict(m)
            for ei, ivs in ex.items():
                lst, added = _mergeIntervalsIncremental(m.get(ei, []), ivs)
                m[ei] = lst
                v += added
            out[b] = (m, v)
        return out

    def tipSetUnder(x, overlayMap, servedOv):
        """Exact tip set of x under the overlay state."""
        out = {b for b in tips[x] if b not in servedOv}
        for b, (mB, vB) in overlayMap.items():
            if b in servedOv or b in out or b in tips[x]:
                continue
            exB = viewC[x].get(b)
            if exB is None:
                continue
            pb = perim[b]
            if (vB + extraLen[x].get(b, 0.0)) / pb < tau:
                continue
            if _covWithExtra(mB, vB, exB, pb) >= tau:
                out.add(b)
        return out

    def bestChain(overlayMap, servedOv, exclude, depthLeft, topGlobal):
        """Max total gain of depthLeft further placements, each measured
        exactly against the (stacked) overlay state."""
        # gains under overlay, via per-building inversion
        dec = defaultdict(int)
        inc = defaultdict(int)
        for b in servedOv:
            for a in st.buildingToCands.get(b, ()):
                if a in remaining and a not in exclude and b in tips[a]:
                    dec[a] += 1
        for b, (mB, vB) in overlayMap.items():
            if b in servedOv:
                continue
            pb = perim[b]
            for a in st.buildingToCands.get(b, ()):
                if a not in remaining or a in exclude or b in tips[a]:
                    continue
                if (vB + extraLen[a].get(b, 0.0)) / pb < tau:
                    continue
                if _covWithExtra(mB, vB, viewC[a].get(b), pb) >= tau:
                    inc[a] += 1
        changed = set(dec) | set(inc)
        gUnder = {a: gain[a] - dec.get(a, 0) + inc.get(a, 0) for a in changed}
        # untouched candidates keep their global gain exactly
        for a in topGlobal:
            if a not in changed and a not in exclude:
                gUnder[a] = gain[a]
                break
        if not gUnder:
            return 0
        if depthLeft <= 1:
            return max(gUnder.values())
        best = 0
        for x in heapq.nlargest(innerBeam, gUnder, key=gUnder.get):
            gx = gUnder[x]
            ov2 = extendOverlay(overlayMap, x)
            sv2 = servedOv | tipSetUnder(x, overlayMap, servedOv)
            total = gx + bestChain(ov2, sv2, exclude | {x},
                                   depthLeft - 1, topGlobal)
            if total > best:
                best = total
        return best

    for step in range(k - len(placed)):
        if not remaining:
            break
        beam = heapq.nlargest(
            beamWidth, remaining,
            key=lambda c: (gain[c], convexity.get(c, 0.0), score[c]))
        topGlobal = heapq.nlargest(beamWidth + 64, remaining,
                                   key=lambda c: gain[c])
        bestKey = None
        bestC = None
        for c in beam:
            ov = extendOverlay({}, c)
            chain = bestChain(ov, set(tips[c]), {c}, depth - 1, topGlobal)
            key = (gain[c] + chain, gain[c], convexity.get(c, 0.0),
                   score[c], rng.random())
            if bestKey is None or key > bestKey:
                bestKey = key
                bestC = c
        if verbose:
            print(f"    lookahead step {len(placed)+1} (depth {depth}): "
                  f"total {bestKey[0]} (immediate {bestKey[1]})")
        st.commit(bestC)
        placed.append(bestC)

    return placed, len(st.servedNow)


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

    Use this when you need the merged LIST back. When only addedLength is
    wanted (the overwhelmingly common case -- see _covWithExtra), call
    _addedLengthNorm instead: it skips building the list and re-summing the
    baseline, which measured 3.7x faster on real cache intervals.
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


def _addedLengthNorm(mergedSorted, ivs):
    """
    Length of `ivs` NOT already covered by `mergedSorted` -- i.e. exactly
    the addedLength _mergeIntervalsIncremental returns, without building
    the merged list or re-summing the baseline.

    REQUIRES both arguments sorted ascending and internally disjoint.
    `mergedSorted` always is (it is produced by merging). `ivs` always is
    too, because every viewshed is passed through normalizeView() when it
    is built and when an older cache is loaded -- see that function for
    why that invariant is enforced at the source rather than checked here
    (a per-call sortedness check would cost the very O(n) pass this
    avoids).

    Measured 3.7x faster than _mergeIntervalsIncremental on real cache
    intervals, verified identical on 300K randomized cases plus every
    (candidate, building, edge) list in a full crop cache.
    """
    n = len(ivs)
    if n == 0:
        return 0.0
    if not mergedSorted:
        total = 0.0
        for lo, hi in ivs:
            total += hi - lo
        return total
    total = 0.0
    nb = len(mergedSorted)
    j = 0
    for curLo, curHi in ivs:
        lo = curLo
        # baseline runs entirely left of this interval never matter again
        while j < nb and mergedSorted[j][1] <= lo:
            j += 1
        jj = j
        while jj < nb:
            blo, bhi = mergedSorted[jj]
            if blo >= curHi:
                break
            if blo > lo:
                total += blo - lo
            if bhi > lo:
                lo = bhi
            if lo >= curHi:
                break
            jj += 1
        if lo < curHi:
            total += curHi - lo
    return total


def normalizeView(view):
    """
    Sort and union every edge's interval list in one candidate's viewshed,
    in place. The intervals a sweep emits are per-wedge fragments -- often
    adjacent or overlapping (the adaptive corridor passes deliberately
    produce overlaps at window boundaries), averaging ~11.6 fragments per
    (building, edge) and already 97.6% sorted.

    Every consumer of a viewshed treats its intervals as a UNION, so
    collapsing them here changes no result -- it only makes the union
    explicit, once, instead of re-deriving it on each of the ~23 million
    coverage checks a solve performs. That is what lets the hot path use
    _addedLengthNorm. It also shrinks the cache, since merged fragments
    collapse into fewer tuples.
    """
    for edgeMap in view.values():
        for ei, ivs in edgeMap.items():
            if len(ivs) == 1:
                continue
            merged = []
            for lo, hi in sorted(ivs):
                if merged and lo <= merged[-1][1]:
                    if hi > merged[-1][1]:
                        merged[-1] = (merged[-1][0], hi)
                else:
                    merged.append((lo, hi))
            edgeMap[ei] = merged
    return view


def normalizeViewCache(viewCache, verbose=True):
    """normalizeView over a whole cache (used to upgrade a cache built
    before normalization was introduced). Returns the interval count
    before and after."""
    t0 = time.perf_counter()
    before = after = 0
    for view in viewCache.values():
        for edgeMap in view.values():
            for ivs in edgeMap.values():
                before += len(ivs)
        normalizeView(view)
        for edgeMap in view.values():
            for ivs in edgeMap.values():
                after += len(ivs)
    if verbose:
        print(f"  normalized viewsheds: {before} -> {after} intervals "
              f"({time.perf_counter() - t0:.1f}s)")
    return before, after

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
    candidate set viewCache was built from, evaluated at the same reach
    viewCache was built with (e.g. solve()'s claim computation — NOT the
    validation re-sweep, which genuinely needs an independent
    recomputation at its wider reach).

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
    DEPRECATED -- use refineWithSwapsV2 (same results in narrow mode,
    measured ~16x faster via per-removal baselines + bound pruning).
    Kept as a reference implementation.

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
parallel — each candidate is independent, and it always runs in parallel
(there is no serial path). Worker count comes from compute.unitCount().

This is the single most expensive stage in the pipeline (~36 min
single-core at r=160 on the full dataset, ~77 min at two doublings), which
is why it is also the one worth understanding before optimizing anything
else.

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

def _initWorker(polygons, radius, gridCell, maxRadius=None):
    _W["polygons"]  = polygons
    _W["radius"]    = radius
    _W["maxRadius"] = maxRadius
    _W["grid"]      = SpatialGrid(polygons, cell=gridCell)

def _viewshedTask(antenna):
    polygons  = _W["polygons"]
    grid      = _W["grid"]
    radius    = _W["radius"]
    maxRadius = _W.get("maxRadius")
    if maxRadius and maxRadius > radius:
        return antenna, candidateViewshedAdaptive(
            antenna, polygons, grid, radius, maxRadius)
    ax, ay = antenna
    idxs = grid.neighborIndices(ax, ay, radius)
    if not idxs:
        return antenna, {}
    scene, sub = localScene(idxs, polygons)
    edgesPerPoly, allEdges, allVertsEdges = scene
    owner = buildEdgeOwner(edgesPerPoly)
    # radius cap: the scene only contains buildings whose bbox comes within
    # `radius`, so a wedge hit BEYOND it could be missing its true occluder
    # entirely -- crediting it risks a false claim. Beyond-radius wedges are
    # dropped (the adaptive path re-solves them against a bigger scene).
    profile, _beyondRadius = sweepAllEdges((ax, ay), allEdges, allVertsEdges,
                                           owner, radius=radius)
    view = normalizeView(
        {idxs[localPi]: {ei: list(ivs) for ei, ivs in edgeMap.items()}
         for localPi, edgeMap in profile.items()})
    return antenna, view

# ---------------------------------------------------------------------------
#  CRASH CHECKPOINTS  (shared by the viewshed precompute and the
#  centrality visibility-graph build -- both are long parallel loops over
#  independent items, and both must survive an interruption without
#  redoing hours of work)
# ---------------------------------------------------------------------------

class _CheckpointWriter:
    """
    Append-only crash checkpoint for a long parallel loop.

    Format: a metadata header frame, then one pickle frame per flush
    holding the (key, value) pairs finished since the previous flush.
    Appending whole frames (rather than rewriting a single blob) is what
    keeps the cost proportional to NEW work instead of total work, and it
    means a crash can damage at most the frame being written -- readers
    stop cleanly at a torn tail (see _loadCheckpointPairs).

    The flush interval is deliberately short relative to the run: each
    item is written exactly once no matter how often we checkpoint, so a
    shorter interval buys finer crash granularity for only extra
    flush/fsync calls -- while a long one risks never firing at all and
    holds more finished work in RAM waiting to be written.
    """

    def __init__(self, path, meta, every=120.0):
        self.path = path
        self.every = every
        self.pending = []
        self._last = time.monotonic()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        isNew = not os.path.exists(path)
        self.fh = open(path, "ab")
        if isNew:
            pickle.dump({"__checkpoint_meta__": meta or {}}, self.fh,
                        protocol=pickle.HIGHEST_PROTOCOL)
            self.fh.flush()

    def add(self, key, value):
        self.pending.append((key, value))

    def _write(self):
        if not self.pending:
            return False
        pickle.dump(self.pending, self.fh, protocol=pickle.HIGHEST_PROTOCOL)
        self.fh.flush()
        os.fsync(self.fh.fileno())
        self.pending = []
        return True

    def maybeFlush(self):
        """Write the pending batch if the interval has elapsed. Returns
        True if anything was actually written."""
        if time.monotonic() - self._last < self.every:
            return False
        wrote = self._write()
        self._last = time.monotonic()
        return wrote

    def close(self):
        """Flush whatever is left. Called from a finally: an interrupted
        run keeping its progress is the entire point."""
        try:
            self._write()
        finally:
            self.fh.close()


def _loadCheckpointPairs(path, meta, verbose=True):
    """
    Load the (key, value) pairs from a _CheckpointWriter file as a dict
    (empty when there is nothing usable).

    Reading stops cleanly at end-of-file OR at a frame that fails to
    unpickle -- the latter is the signature of a crash midway through an
    append, and everything written before it is still valid. A metadata
    mismatch (different dataset, radius, or settings) discards the whole
    file rather than resuming from work that no longer applies.
    """
    if not os.path.exists(path):
        return {}
    out = {}
    try:
        with open(path, "rb") as f:
            header = pickle.load(f)
            if not (isinstance(header, dict)
                    and header.get("__checkpoint_meta__") == meta):
                if verbose:
                    print(f"  checkpoint {path} is from a different "
                          f"dataset/config; ignoring it")
                return {}
            while True:
                for key, value in pickle.load(f):
                    out[key] = value
    except EOFError:
        pass                      # clean end of file
    except Exception:
        pass                      # torn final frame from a crash mid-append
    return out


def precomputeViewshedsParallel(polygons, candidates, radius=80.0, gridCell=80.0,
                                chunksize=50, verbose=True,
                                maxRadius=None, checkpointPath=None,
                                checkpointMeta=None, checkpointEvery=120.0,
                                precomputed=None):
    """
    Compute every candidate's viewshed across the configured processing
    units (see compute.py; this is a geometry stage, so CPU worker
    processes under either backend).
    Returns {candidate: {bldgIdx: {edgeIdx: [intervals]}}}.

    Crash safety (all optional, off when checkpointPath is None):
      precomputed     : {candidate: view} already computed in an earlier,
                        interrupted run (from _loadCheckpointPairs) --
                        these candidates are skipped, not re-swept.
      checkpointPath  : append-only checkpoint file, written every
                        checkpointEvery seconds -- see _CheckpointWriter
                        for the format and why the interval is short. The
                        caller deletes this file once the real cache is
                        saved.
      checkpointMeta  : dict written as the file's header frame (dataset
                        fingerprint + radii) so a stale checkpoint from a
                        different dataset/config is refused on load.
    """
    viewCache = dict(precomputed) if precomputed else {}
    todo = [c for c in candidates if c not in viewCache] if viewCache else list(candidates)
    if verbose:
        skipNote = f" ({len(viewCache)} restored from checkpoint)" if viewCache else ""
        print(f"Precomputing {len(todo)} viewsheds across "
              f"{compute.unitCount()} processes...{skipNote}")
    if not todo:
        return viewCache

    ck = (_CheckpointWriter(checkpointPath, checkpointMeta, checkpointEvery)
          if checkpointPath is not None else None)
    try:
        with compute.workerPool(initializer=_initWorker,
                                initargs=(polygons, radius, gridCell,
                                          maxRadius)) as pool:
            done = 0
            for antenna, view in pool.imap_unordered(_viewshedTask, todo,
                                                      chunksize=chunksize):
                viewCache[antenna] = view
                done += 1
                if ck is not None:
                    ck.add(antenna, view)
                    if ck.maybeFlush() and verbose:
                        print(f"  checkpoint: {len(viewCache)}/{len(candidates)} "
                              f"viewsheds safe on disk")
                if verbose and done % 5000 == 0:
                    print(f"  {done}/{len(todo)}")
    finally:
        if ck is not None:
            ck.close()
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
                           chunksize=8, maxRadius=None):
    """
    Coverage fraction per building for a given antenna list, computed in
    parallel (one sweep per antenna, merged across workers), with the
    worker count from compute.unitCount().
    Returns {buildingIndex: coverageFraction}.

    Note: builds its own grid inside each worker (via _initWorker), so you pass
    polygons/radius/gridCell rather than a prebuilt grid.
    """
    if not antennas:
        return {}

    accum = {}   # globalBldgIdx -> {edgeIdx: [intervals]}

    with compute.workerPool(initializer=_initWorker,
                            initargs=(polygons, radius, gridCell,
                                      maxRadius)) as pool:
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
                           verbose=True, maxRadius=None):
    """
    Build the viewshed cache (candidates + perimeters + viewsheds) and, if a
    path is given, save it to disk. Returns (candidates, viewCache, perim).

    Always parallel -- the old `parallel=False` single-core loop is gone.
    It computed the identical cache tens of times slower and existed only
    as a debugging fallback; compute.py now decides the unit count.
    """
    candidates = candidateList(polygons)
    perim = {i: polygonPerimeter(p) for i, p in enumerate(polygons)}

    # Mid-precompute crash protection: results are appended to a sibling
    # .partial file every few minutes (see precomputeViewshedsParallel), and
    # a restart after a crash resumes from it instead of re-sweeping from
    # scratch. Deleted once the real cache is fully saved.
    partialPath = (path + ".partial") if path is not None else None
    checkpointMeta = {"fingerprint": _datasetFingerprint(polygons),
                      "radius": radius, "maxRadius": maxRadius,
                      "normalized": True}
    precomputed = (_loadCheckpointPairs(partialPath, checkpointMeta,
                                        verbose=verbose)
                   if partialPath else {})

    viewCache = precomputeViewshedsParallel(polygons, candidates,
                                            radius=radius, gridCell=gridCell,
                                            verbose=verbose, maxRadius=maxRadius,
                                            checkpointPath=partialPath,
                                            checkpointMeta=checkpointMeta,
                                            precomputed=precomputed)

    if path is not None:
        saveViewCache(path, polygons, radius, candidates, viewCache, perim,
                      maxRadius=maxRadius)
        if verbose:
            print(f"Saved cache to {path}")
        if partialPath and os.path.exists(partialPath):
            try:
                os.unlink(partialPath)   # full cache saved; checkpoint obsolete
            except OSError:
                pass

    return candidates, viewCache, perim


def saveViewCache(path, polygons, radius, candidates, viewCache, perim,
                  maxRadius=None):
    """Write the cache plus validation metadata to disk.
    Uses a write-to-temp-then-rename pattern so a crash or OOM during the
    write never leaves a truncated file behind -- the old cache stays intact
    until the new one is fully flushed, then os.replace() swaps them
    atomically."""
    import tempfile
    blob = {
        "radius": radius,
        "maxRadius": maxRadius,       # adaptive corridor reach (None = plain)
        "nCandidates": len(candidates),
        "fingerprint": _datasetFingerprint(polygons),
        # every interval list sorted + unioned (see normalizeView); a cache
        # without this flag is upgraded in place on load
        "normalized": True,
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


def loadViewCache(path, polygons, radius, maxRadius=None):
    """
    Load a cache from disk, verifying it matches the current dataset, radius,
    AND adaptive corridor reach (maxRadius) -- a cache built with different
    ADAPTIVE_DOUBLINGS settings holds genuinely different viewsheds, so it is
    refused rather than silently loaded (previously only the filename encoded
    this). Raises ValueError on mismatch.

    Returns (candidates, viewCache, perim, normalized), where `normalized`
    reports whether the intervals are already sorted+unioned. A False here
    is NOT a mismatch (the geometry is still correct and reusable) -- the
    caller upgrades the cache in place rather than rebuilding it.
    """
    # Unpickling a multi-GB cache creates hundreds of millions of small
    # tuples/lists/dicts, and the garbage collector tracks every one as it
    # is born -- disabling GC for the duration is a pure win (measured
    # 14.9x on a 220 MB cache: 20.0s -> 1.3s; the factor grows with size).
    import gc
    gcWasEnabled = gc.isenabled()
    gc.disable()
    try:
        with open(path, "rb") as f:
            blob = pickle.load(f)
    finally:
        if gcWasEnabled:
            gc.enable()

    if abs(blob["radius"] - radius) > 1e-9:
        raise ValueError(f"Cache radius {blob['radius']} != requested {radius}. "
                         "Rebuild the cache.")
    if "maxRadius" not in blob:
        raise ValueError("Cache predates maxRadius metadata (and the "
                         "beyond-radius credit fix in the sweep). Rebuild the cache.")
    cachedMax = blob["maxRadius"]
    if (cachedMax is None) != (maxRadius is None) or \
       (cachedMax is not None and abs(cachedMax - maxRadius) > 1e-9):
        raise ValueError(f"Cache maxRadius {cachedMax} != requested {maxRadius}. "
                         "Rebuild the cache.")
    if blob["fingerprint"] != _datasetFingerprint(polygons):
        raise ValueError("Cache was built from a different dataset "
                         "(fingerprint mismatch). Rebuild the cache.")
    return (blob["candidates"], blob["viewCache"], blob["perim"],
            bool(blob.get("normalized")))


def loadOrBuildViewsheds(polygons, radius=80.0, gridCell=80.0, path="viewcache.pkl",
                         verbose=True, maxRadius=None):
    """
    Convenience: load the cache from `path` if it exists and matches; otherwise
    build it and save it. Returns (candidates, viewCache, perim).
    """
    import os
    if os.path.exists(path):
        try:
            candidates, viewCache, perim, normalized = loadViewCache(
                path, polygons, radius, maxRadius=maxRadius)
            if verbose:
                print(f"Loaded cache from {path} "
                      f"({len(candidates)} candidates, radius {radius})")
            if not normalized:
                # One-time upgrade of a cache built before interval
                # normalization. The hot path assumes sorted, disjoint
                # interval lists (see _addedLengthNorm), so this is not
                # optional -- but it is a pure union-preserving rewrite,
                # so the cache stays valid for the same dataset/radius.
                # Re-saved (atomically) so future runs skip the pass.
                if verbose:
                    print("  cache predates interval normalization; "
                          "upgrading it in place")
                normalizeViewCache(viewCache, verbose=verbose)
                try:
                    saveViewCache(path, polygons, radius, candidates,
                                  viewCache, perim, maxRadius=maxRadius)
                    if verbose:
                        print(f"  re-saved normalized cache to {path}")
                except OSError as e:
                    if verbose:
                        print(f"  (could not re-save cache: {e}; "
                              f"continuing with the in-memory upgrade)")
            return candidates, viewCache, perim
        except ValueError as e:
            if verbose:
                print(f"Cache at {path} unusable ({e}); rebuilding.")
    return buildAndCacheViewsheds(polygons, radius, gridCell, path,
                                  verbose=verbose, maxRadius=maxRadius)

# ===========================================================================
#  PACKED VIEWSHED  (compact numpy mirror of the viewCache)
#
#  The pickled viewCache is nested dicts of tuples -- ~4x its 1 GB disk
#  size once unpickled, which is why refine workers could never each hold
#  a copy on a 16 GB machine. This packs the exact same intervals into
#  flat numpy arrays (CSR-style: candidate -> building -> edge -> interval
#  ranges) saved as .npy files. Worker processes np.load them with
#  mmap_mode="r", so all workers plus the parent share ONE physical copy
#  through the OS page cache -- that is what makes the parallel swap
#  refinement fit in memory.
#
#  Two derived per-(candidate, building) tables come free and vectorized:
#    cbExtra : unmerged interval-length sum  (== buildExtraLen's values)
#    cbCov   : merged standalone coverage fraction (tau-independent), so
#              _CoverageState.__init__ no longer re-merges every
#              (candidate, building) pair per subcase -- it was doing that
#              once per greedy call, nine times per solve.
# ===========================================================================

_PACKED_NAMES = ("candPtr", "cbBuilding", "cbEdgePtr", "cbeEdge",
                 "cbeIvPtr", "ivLo", "ivHi", "cbExtra", "cbCov", "perim")


def _segmentedUnionLengths(ivLo, ivHi, segPtr):
    """Union (merged) length of the intervals in each segment.
    segPtr : (S+1,) interval index boundaries; every segment non-empty.
    Exact segmented cummax via log2(maxSegLen) doubling passes -- no
    group-offset trick, so no precision loss from large offsets."""
    S = len(segPtr) - 1
    n = len(ivLo)
    if n == 0:
        return np.zeros(S, dtype=np.float64)
    gid = np.repeat(np.arange(S, dtype=np.int64), np.diff(segPtr))
    order = np.lexsort((ivLo, gid))
    glo = ivLo[order]
    ghi = ivHi[order]
    g = gid[order]
    maxSeg = int(np.diff(segPtr).max())
    runmax = ghi.copy()                    # cummax of hi within each segment
    shift = 1
    while shift < maxSeg:
        prev = np.full(n, -np.inf)
        prev[shift:] = runmax[:-shift]
        same = np.zeros(n, dtype=bool)
        same[shift:] = g[shift:] == g[:-shift]
        np.maximum(runmax, np.where(same, prev, -np.inf), out=runmax)
        shift *= 2
    prevHi = np.full(n, -np.inf)           # cummax over STRICTLY earlier elems
    prevHi[1:] = runmax[:-1]
    firstInSeg = np.ones(n, dtype=bool)
    firstInSeg[1:] = g[1:] != g[:-1]
    prevHi[firstInSeg] = -np.inf
    contrib = np.clip(ghi - np.maximum(glo, prevHi), 0.0, None)
    return np.add.reduceat(contrib, segPtr[:-1])


def buildPackedViewshed(candidates, viewCache, perim, nBuildings, verbose=True):
    """Flatten viewCache into the packed-array dict described above.
    Buildings are sorted ascending within each candidate (binary-search
    lookups); (candidate, building) pairs with no intervals are dropped
    (they contribute nothing anywhere)."""
    t0 = time.perf_counter()
    C = len(candidates)
    nCB = nCBE = nIV = 0
    for c in candidates:
        vc = viewCache[c]
        nCB += len(vc)
        for em in vc.values():
            nCBE += len(em)
            for ivs in em.values():
                nIV += len(ivs)

    candPtr = np.zeros(C + 1, dtype=np.int64)
    cbBuilding = np.zeros(nCB, dtype=np.int32)
    cbEdgePtr = np.zeros(nCB + 1, dtype=np.int64)
    cbeEdge = np.zeros(nCBE, dtype=np.int32)
    cbeIvPtr = np.zeros(nCBE + 1, dtype=np.int64)
    ivLo = np.zeros(nIV, dtype=np.float64)
    ivHi = np.zeros(nIV, dtype=np.float64)

    iCB = iCBE = iIV = 0
    for ci, c in enumerate(candidates):
        vc = viewCache[c]
        for b in sorted(vc):
            edgeStart = iCBE
            for ei, ivs in vc[b].items():
                nI = len(ivs)
                if nI == 0:
                    continue
                cbeEdge[iCBE] = ei
                if nI == 1:
                    ivLo[iIV] = ivs[0][0]
                    ivHi[iIV] = ivs[0][1]
                else:
                    los, his = zip(*ivs)
                    ivLo[iIV:iIV + nI] = los
                    ivHi[iIV:iIV + nI] = his
                iIV += nI
                iCBE += 1
                cbeIvPtr[iCBE] = iIV
            if iCBE == edgeStart:
                continue                       # empty pair -- drop it
            cbBuilding[iCB] = b
            iCB += 1
            cbEdgePtr[iCB] = iCBE
        candPtr[ci + 1] = iCB

    cbBuilding = cbBuilding[:iCB]
    cbEdgePtr = cbEdgePtr[:iCB + 1]
    cbeEdge = cbeEdge[:iCBE]
    cbeIvPtr = cbeIvPtr[:iCBE + 1]
    ivLo = ivLo[:iIV]
    ivHi = ivHi[:iIV]

    perimArr = np.array([perim[i] for i in range(nBuildings)], dtype=np.float64)

    # ---- derived per-(candidate, building) tables ----
    # Vectorized rather than parallelized on purpose: this is one pass of
    # flat-array numpy over the whole cache, and it runs in seconds against
    # the tens of minutes the sweeps cost. Splitting it across processes
    # would spend more on pickling the arrays than on the arithmetic.

    # unmerged per-(c,b) sums: segment-local reduceat, no long cumsums
    cbIvStart = cbeIvPtr[cbEdgePtr[:-1]] if iCB else np.zeros(0, dtype=np.int64)
    cbExtra = (np.add.reduceat(ivHi - ivLo, cbIvStart)
               if iCB else np.zeros(0, dtype=np.float64))

    # merged per-(c,b) coverage: union per edge group, summed per pair
    edgeUnion = _segmentedUnionLengths(ivLo, ivHi, cbeIvPtr)
    cbUnion = (np.add.reduceat(edgeUnion, cbEdgePtr[:-1])
               if iCB else np.zeros(0, dtype=np.float64))
    pb = perimArr[cbBuilding]
    with np.errstate(divide="ignore", invalid="ignore"):
        cbCov = np.where(pb >= 1e-9,
                         np.minimum(cbUnion / np.maximum(pb, 1e-30), 1.0),
                         0.0)

    if verbose:
        print(f"  packed viewshed: {iCB} (cand,bldg) pairs, {iIV} intervals "
              f"({time.perf_counter() - t0:.1f}s)")
    return {"candPtr": candPtr, "cbBuilding": cbBuilding,
            "cbEdgePtr": cbEdgePtr, "cbeEdge": cbeEdge,
            "cbeIvPtr": cbeIvPtr, "ivLo": ivLo, "ivHi": ivHi,
            "cbExtra": cbExtra, "cbCov": cbCov, "perim": perimArr}


def savePackedViewshed(prefix, packed, meta):
    """One .npy per array (so they can be memory-mapped individually) plus
    a .meta.json used to validate on load. Arrays are written to temp
    names and renamed into place; the meta file is written LAST, so a
    crash mid-save just looks like 'no valid packed cache' next run."""
    dirName = os.path.dirname(os.path.abspath(prefix))
    os.makedirs(dirName, exist_ok=True)
    metaPath = prefix + ".meta.json"
    if os.path.exists(metaPath):
        os.unlink(metaPath)                   # invalidate before touching arrays
    for name in _PACKED_NAMES:
        tmp = f"{prefix}.{name}.tmp.npy"   # ends in .npy so np.save keeps it
        np.save(tmp, packed[name])
        os.replace(tmp, f"{prefix}.{name}.npy")
    with open(metaPath + ".tmp", "w") as f:
        json.dump(meta, f)
    os.replace(metaPath + ".tmp", metaPath)


def loadPackedViewshed(prefix, mmapMode="r"):
    """Load the packed arrays, memory-mapped by default (near-zero RAM and
    load time; pages are shared between every process that maps them)."""
    return {name: np.load(f"{prefix}.{name}.npy", mmap_mode=mmapMode)
            for name in _PACKED_NAMES}


def loadOrBuildPackedViewshed(prefix, candidates, viewCache, perim, polygons,
                              verbose=True):
    """Load the packed mirror if present and matching this dataset/cache;
    otherwise build it from viewCache, save it, and reload memory-mapped
    (dropping the in-heap build copies)."""
    # "normalized" pins the packed arrays to normalized intervals: cbExtra
    # is the UNION length there (a tighter, still-sound upper bound) rather
    # than the raw fragment sum, so a packed file built before normalization
    # must be rebuilt rather than reused.
    meta = {"nCandidates": len(candidates),
            "nBuildings": len(polygons),
            "normalized": True,
            "fingerprint": _datasetFingerprint(polygons)}
    metaPath = prefix + ".meta.json"
    if os.path.exists(metaPath):
        try:
            with open(metaPath) as f:
                onDisk = json.load(f)
            if onDisk == meta:
                packed = loadPackedViewshed(prefix, mmapMode="r")
                if verbose:
                    print(f"  packed viewshed loaded (mmap) from {prefix}.*.npy")
                return packed
            if verbose:
                print("  packed viewshed on disk is stale; rebuilding")
        except Exception as e:
            if verbose:
                print(f"  packed viewshed unusable ({e}); rebuilding")
    packed = buildPackedViewshed(candidates, viewCache, perim, len(polygons),
                                 verbose=verbose)
    savePackedViewshed(prefix, packed, meta)
    del packed
    return loadPackedViewshed(prefix, mmapMode="r")


def extraLenFromPacked(candidates, packed):
    """{c: {b: unmergedLengthSum}} -- same contents buildExtraLen computes,
    but read straight out of the packed arrays (fast)."""
    cp = np.asarray(packed["candPtr"]).tolist()
    cb = np.asarray(packed["cbBuilding"]).tolist()
    ce = np.asarray(packed["cbExtra"]).tolist()
    return {c: dict(zip(cb[cp[i]:cp[i + 1]], ce[cp[i]:cp[i + 1]]))
            for i, c in enumerate(candidates)}


def baseCovFromPacked(candidates, packed):
    """{c: {b: mergedStandaloneCoverage}} -- the tau-independent per-pair
    coverage _CoverageState.__init__ needs, computed once per cache
    instead of re-merged inside every greedy call."""
    cp = np.asarray(packed["candPtr"]).tolist()
    cb = np.asarray(packed["cbBuilding"]).tolist()
    cc = np.asarray(packed["cbCov"]).tolist()
    return {c: dict(zip(cb[cp[i]:cp[i + 1]], cc[cp[i]:cp[i + 1]]))
            for i, c in enumerate(candidates)}


def buildBuildingToCandidates(candidates, viewCache):
    """building -> set(candidates that see it). Built once per solve and
    shared read-only by every greedy/lookahead/refine call (each used to
    rebuild its own copy -- ~4M set inserts per subcase)."""
    d = defaultdict(set)
    for c in candidates:
        for b in viewCache[c]:
            d[b].add(c)
    return d


def buildRefineParallelContext(candidates, packedPrefix):
    """Everything refineWithSwapsV2 needs to run its pass scan in parallel.
    Built once per solve, reused by every refine call. REQUIRED -- the scan
    has no serial mode any more, so refineWithSwapsV2 raises without this.

      candidates   : the cache's candidate list (defines packed indices)
      packedPrefix : prefix the packed .npy files were saved under

    The worker count comes from compute.py, not from a parameter -- there
    is one knob for the whole program now.

    Contents: the mmap'd packed arrays, candidate->index map, and a
    building->candidate-index CSR (for fast per-pass loss-hit counting).
    """
    processes = compute.unitCount()
    packed = loadPackedViewshed(packedPrefix, mmapMode="r")
    candIndex = {c: i for i, c in enumerate(candidates)}
    cb = np.asarray(packed["cbBuilding"])
    candPtr = np.asarray(packed["candPtr"])
    candOfCb = np.repeat(np.arange(len(candidates), dtype=np.int64),
                         np.diff(candPtr))
    order = np.argsort(cb, kind="stable")
    bSorted = cb[order]
    bToCandFlat = candOfCb[order]
    nB = len(packed["perim"])
    bToCandPtr = np.searchsorted(bSorted, np.arange(nB + 1))
    return {"packed": packed, "prefix": packedPrefix, "processes": processes,
            "candidates": list(candidates), "candIndex": candIndex,
            "bToCandFlat": bToCandFlat, "bToCandPtr": bToCandPtr}


# --- worker side of the parallel refine pass scan ---
_RPW = {}

def _initRefinePackedWorker(prefix):
    _RPW["p"] = loadPackedViewshed(prefix, mmapMode="r")

def _packedFind(p, aIdx, b):
    """Index into the cb arrays for (candidate aIdx, building b), or -1."""
    lo = int(p["candPtr"][aIdx]); hi = int(p["candPtr"][aIdx + 1])
    cb = p["cbBuilding"]
    j = lo + int(np.searchsorted(cb[lo:hi], b))
    if j < hi and int(cb[j]) == b:
        return j
    return -1

def _packedCovAtLeast(p, aIdx, b, baseM, baseVis, pb, tau):
    """Does adding candidate aIdx's intervals for building b on top of the
    already-merged baseline (baseM per edge, baseVis total) reach tau?
    Mirrors _covWithExtra exactly: same upper-bound short-circuit, same
    _mergeIntervalsIncremental arithmetic on the same interval values."""
    if pb < 1e-9:
        return False
    j = _packedFind(p, aIdx, b)
    if j < 0:
        return baseVis / pb >= tau
    if (baseVis + float(p["cbExtra"][j])) / pb < tau:
        return False
    added = 0.0
    e0 = int(p["cbEdgePtr"][j]); e1 = int(p["cbEdgePtr"][j + 1])
    cbeEdge = p["cbeEdge"]; ivPtr = p["cbeIvPtr"]
    ivLo = p["ivLo"]; ivHi = p["ivHi"]
    for g in range(e0, e1):
        i0 = int(ivPtr[g]); i1 = int(ivPtr[g + 1])
        ivs = list(zip(ivLo[i0:i1].tolist(), ivHi[i0:i1].tolist()))
        added += _addedLengthNorm(baseM.get(int(cbeEdge[g]), []), ivs)
    return min((baseVis + added) / pb, 1.0) >= tau

def _refineScanTask(task):
    """Evaluate one removal candidate's surviving (r, a) pairs exactly.
    task = (tau, taskId, nLoss, floorDelta, baselines, pairs) where
      baselines : {b: (mergedByEdge, vis)}  -- the without-r baseline for
                  every building any pair below needs
      pairs     : [(aIdx, gainA, tipCheckBs, lossHitBs), ...]
    Returns (taskId, bestDelta, bestAIdx) with bestDelta > floorDelta, or
    (taskId, floorDelta, -1) if nothing beat the floor."""
    tau, taskId, nLoss, floorDelta, baselines, pairs = task
    p = _RPW["p"]
    perim = p["perim"]
    bestDelta = floorDelta
    bestA = -1
    for aIdx, gainA, tipBs, lossBs in pairs:
        if gainA + len(lossBs) - nLoss <= bestDelta:
            continue
        newly = gainA
        for b in tipBs:
            m, v = baselines[b]
            if not _packedCovAtLeast(p, aIdx, b, m, v, float(perim[b]), tau):
                newly -= 1
        delta = newly - nLoss
        for b in lossBs:
            pb = float(perim[b])
            if pb < 1e-9:
                continue
            m, v = baselines[b]
            if _packedCovAtLeast(p, aIdx, b, m, v, pb, tau):
                delta += 1
        if delta > bestDelta:
            bestDelta = delta
            bestA = aIdx
    return taskId, bestDelta, bestA


# ===========================================================================
#  SWAP REFINEMENT V2  (verified swaps + incremental pass engine)
#
#  Replaces refineWithSwaps (and the deleted dual-radius variants -- the
#  adaptive-corridor viewshed cache made search-narrow/commit-wide moot,
#  since the search now sees the full commit reach directly). Two pillars:
#
#  1. VERIFY BEFORE ACCEPT. The chosen swap's exact realized delta is
#     recomputed over every touched building before it is applied; the swap
#     only commits if realized delta > 0. Rejected (r, a) pairs are
#     blacklisted for the rest of the call. Predictions are exact by
#     construction here (same viewCache on both sides), so this is a cheap
#     safety net rather than a correction step.
#
#  2. SPEED (measured ~16-22x on a 605-building crop): per-removal-candidate
#     coverage baselines computed ONCE per iteration instead of re-merged
#     inside every (r, a) pair; an incremental global merged state for O(1)
#     current-coverage reads; and a set-arithmetic upper bound
#     (gains <= |unserved buildings a touches|, certain losses =
#     |lost buildings a cannot touch|) that prunes ~80% of pairs before
#     any interval math runs.
# ===========================================================================

def _mergedFromContribs(edgeMap, exclude=None):
    """{ei: mergedSortedIntervals}, total visible length -- merging every
    contribution on each edge except (optionally) one candidate's."""
    out = {}
    total = 0.0
    for ei, byCand in edgeMap.items():
        ivs = [iv for cid, cIvs in byCand.items() if cid != exclude for iv in cIvs]
        if not ivs:
            continue
        merged, _ = _mergeIntervalsIncremental([], ivs)
        out[ei] = merged
        total += sum(hi - lo for lo, hi in merged)
    return out, total


def _covWithExtra(baseMerged, baseVis, extra, perimB, tauHint=None):
    """Coverage fraction of one building with `extra` {ei: intervals}
    hypothetically merged on top of an already-merged baseline.

    tauHint: when given, short-circuits with a sound UPPER BOUND (base +
    unmerged sum of extra lengths, which overcounts overlap) whenever
    even that bound stays below tauHint*perimB -- the caller only needs
    ">= tau or not", and most candidate/building touches are slivers
    that cannot possibly tip the threshold, so this skips the merge for
    them. The returned bound is < tau exactly when the true value is,
    which is all the threshold comparisons observe."""
    if perimB < 1e-9:
        return 0.0
    if tauHint is not None:
        ub = baseVis + sum(b - a for ivs in extra.values() for a, b in ivs)
        if ub / perimB < tauHint:
            return ub / perimB
    added = 0.0
    for ei, newIvs in extra.items():
        added += _addedLengthNorm(baseMerged.get(ei, []), newIvs)
    return min((baseVis + added) / perimB, 1.0)


def refineWithSwapsV2(polygons, placed, tau, candidates, viewCache,
                      maxIterations=10, verbose=True,
                      logPrefix="    ", selfCheck=False, timeBudget=None,
                      extraLen=None, buildingToCands=None, parallelCtx=None):
    """
    Local-search swap refinement over the COMPLETE (remove r, add a) space,
    with verified commits and an INCREMENTAL pass engine.

    parallelCtx : REQUIRED buildRefineParallelContext(...) result. Each
    pass's expensive (r, a) scan runs in parallel, always: a vectorized
    numpy upper bound first prunes the pair space in the parent, then the
    surviving pairs are evaluated exactly by worker processes that read
    candidate intervals from the memory-mapped packed viewshed (one
    physical copy shared by all workers -- workers never hold the multi-GB
    viewCache). Verification and commits stay in the parent, so every
    committed swap is still exact-checked the same way.

    The old serial scan is gone. It computed the same maximum delta every
    pass (that equivalence was tested and is documented in README) at a
    fraction of the throughput, and keeping two implementations of the
    correctness-critical scan in sync was the real cost.

    buildingToCands : optional shared buildBuildingToCandidates() result
    (read-only), so repeated calls skip rebuilding it.

    Completeness (unchanged from the previous version): every (r, a) pair
    is considered every pass -- INTERACTING pairs (a's viewshed shares a
    building with r's) are evaluated exactly against the without-r
    baseline; NON-INTERACTING pairs factor exactly into
    standaloneGain(a) - loss(r), covered by one gain value per candidate.

    Incremental engine (new): the expensive pass state -- per-antenna
    removal baselines (baseWithout/lossSet) and per-candidate standalone
    gains -- is computed once up front and then only REFRESHED for the
    antennas and candidates whose viewsheds intersect the buildings a
    committed swap actually touched, instead of recomputed from scratch
    every pass. A swap touching ~100 buildings refreshes ~10-100 baseline
    entries and a few thousand gains rather than k full baselines and
    every candidate's gain, which is what made large-k swaps cost minutes
    each. First pass still pays full price; subsequent swaps are cheap.

    selfCheck : when True, every pass recomputes all incremental state
    from scratch and asserts it matches -- slow, for testing only.

    Returns (newPlaced, servedCount, nSwaps, committedViewCache) --
    committedViewCache maps every candidate in newPlaced to its viewCache
    entry.
    """
    perim = [polygonPerimeter(p) for p in polygons]
    placed = list(placed)
    placedSet = set(placed)
    remaining = set(candidates) - placedSet

    committedView = {c: viewCache[c] for c in placed}

    if not parallelCtx:
        raise ValueError(
            "refineWithSwapsV2 requires parallelCtx (the swap scan is always "
            "parallel now -- build one with buildRefineParallelContext).")

    _pk = parallelCtx["packed"]
    _candIndex = parallelCtx["candIndex"]
    _candList = parallelCtx["candidates"]
    _pkCandPtr = np.asarray(_pk["candPtr"])
    _pkCbB = np.asarray(_pk["cbBuilding"])
    _bToCandFlat = parallelCtx["bToCandFlat"]
    _bToCandPtr = parallelCtx["bToCandPtr"]
    nProcs = parallelCtx["processes"]

    def _candSlice(i):
        return int(_pkCandPtr[i]), int(_pkCandPtr[i + 1])

    def _seesB(i, b):
        lo, hi = _candSlice(i)
        j = lo + int(np.searchsorted(_pkCbB[lo:hi], b))
        return j < hi and int(_pkCbB[j]) == b

    def _seesAnyOf(a, bset):
        if not bset:
            return False
        lo, hi = _candSlice(_candIndex[a])
        if hi == lo:
            return False
        sub = _pkCbB[lo:hi]
        arr = np.fromiter(bset, dtype=np.int64, count=len(bset))
        pos = np.searchsorted(sub, arr)
        valid = pos < (hi - lo)
        return bool(np.any(sub[pos[valid]] == arr[valid]))

    import multiprocessing as mp
    pool = mp.Pool(processes=nProcs,
                   initializer=_initRefinePackedWorker,
                   initargs=(parallelCtx["prefix"],))

    # static upper-bound helper: total (unmerged) interval length candidate c
    # contributes to building b -- baseVis + extraLen is a sound upper bound
    # on merged coverage, letting threshold checks skip the exact merge for
    # the many sliver touches that cannot possibly tip a building over tau
    if extraLen is None:
        extraLen = buildExtraLen(candidates, viewCache)

    contributions = defaultdict(lambda: defaultdict(dict))
    for c in placed:
        for b, edgeMap in committedView[c].items():
            for ei, ivs in edgeMap.items():
                contributions[b][ei][c] = ivs

    globalMerged, globalVis = {}, {}

    def rebuildBuilding(b):
        m, v = _mergedFromContribs(contributions.get(b, {}))
        globalMerged[b] = m
        globalVis[b] = v

    for b in list(contributions):
        rebuildBuilding(b)

    servedNow = {b for b, v in globalVis.items()
                 if perim[b] > 1e-9 and v / perim[b] >= tau}

    if buildingToCands is not None:
        buildingToCandidates = buildingToCands   # shared, read-only here
    else:
        buildingToCandidates = defaultdict(set)
        for c in candidates:
            for b in viewCache[c]:
                buildingToCandidates[b].add(c)

    committedViewSets = {c: frozenset(committedView[c].keys()) for c in placed}

    # ---------------- incremental pass state ----------------
    unservedSet = {b for b in buildingToCandidates if b not in servedNow}

    bCoveredBy = defaultdict(set)          # building -> placed antennas covering it
    for c in placed:
        for b in committedViewSets[c]:
            bCoveredBy[b].add(c)

    baseWithout = {}                       # r -> {b: (merged, vis) without r}
    lossSet = {}                           # r -> buildings dropped if r removed

    def refreshBaseline(r, bs):
        bw = baseWithout.setdefault(r, {})
        loss = lossSet.setdefault(r, set())
        for b in bs:
            m, v = _mergedFromContribs(contributions.get(b, {}), exclude=r)
            bw[b] = (m, v)
            if b in servedNow and (perim[b] < 1e-9 or v / perim[b] < tau):
                loss.add(b)
            else:
                loss.discard(b)

    for r in placedSet:
        refreshBaseline(r, committedViewSets[r])

    poolCache = {}   # r -> sorted ndarray of the packed INDICES of the
                     # candidates sharing >= 1 building with r

    def buildPool(r):
        bs = committedViewSets[r]
        if bs:
            parts = [_bToCandFlat[_bToCandPtr[b]:_bToCandPtr[b + 1]]
                     for b in bs]
            poolCache[r] = np.unique(np.concatenate(parts))
        else:
            poolCache[r] = np.zeros(0, dtype=np.int64)

    for r in placedSet:
        buildPool(r)

    gains = {}                             # candidate -> exact standalone gain

    aTips = {}          # a -> set of unserved buildings a would tip over tau

    def _tips(a, b):
        """Would adding candidate a tip (currently unserved) building b?"""
        pb = perim[b]
        if pb < 1e-9:
            return False
        v = globalVis.get(b, 0.0)
        if (v + extraLen[a].get(b, 0.0)) / pb < tau:   # sound upper bound
            return False
        return _covWithExtra(globalMerged.get(b, {}), v,
                             viewCache[a].get(b), pb) >= tau

    def computeGain(a):
        lo, hi = _candSlice(_candIndex[a])
        touched = [b for b in _pkCbB[lo:hi].tolist() if b in unservedSet]
        t = {b for b in touched if _tips(a, b)}
        aTips[a] = t
        gains[a] = len(t)

    # Vectorized gain init: ONE numpy upper-bound filter over every
    # (candidate, building) pair in the packed arrays, exact _tips
    # checks only on the survivors -- replaces a python loop over
    # all candidates (same bound _tips applies, so survivors match).
    # The largest single array operation in the solver, and the reason
    # this stage is vectorized rather than farmed out: the mask is a
    # handful of full-array passes, while the surviving _tips checks are
    # the part that actually costs anything.
    nB = len(_pk["perim"])
    visArr = np.zeros(nB)
    for b, v in globalVis.items():
        visArr[b] = v
    perimArr = np.asarray(_pk["perim"])
    unservedArr = np.zeros(nB, dtype=bool)
    if unservedSet:
        unservedArr[np.fromiter(unservedSet, dtype=np.int64,
                                count=len(unservedSet))] = True
    pbAll = perimArr[_pkCbB]
    mask = unservedArr[_pkCbB] & (pbAll >= 1e-9)
    mask &= (visArr[_pkCbB] + np.asarray(_pk["cbExtra"])) >= tau * pbAll
    survivors = np.flatnonzero(mask)
    for a in remaining:
        aTips[a] = set()
        gains[a] = 0
    if len(survivors):
        candOf = (np.searchsorted(_pkCandPtr, survivors, side="right") - 1)
        bList = np.asarray(_pkCbB)[survivors].tolist()
        cList = candOf.tolist()
        prevC = -1
        curA = None
        inRem = False
        for pos in range(len(cList)):
            ci = cList[pos]
            if ci != prevC:
                prevC = ci
                curA = _candList[ci]
                inRem = curA in remaining
            if not inRem:
                continue
            b = bList[pos]
            if _tips(curA, b):
                aTips[curA].add(b)
        for a, t in aTips.items():
            gains[a] = len(t)

    rejected = set()
    nSwaps = 0
    served0 = len(servedNow)
    tStart = time.perf_counter()
    hitCap = False

    hitBudget = False
    while True:
        if nSwaps >= maxIterations:
            hitCap = True
            break
        if timeBudget is not None and time.perf_counter() - tStart > timeBudget:
            hitBudget = True
            break
        itT0 = time.perf_counter()

        if selfCheck:
            for r in placedSet:
                freshLoss = set()
                for b in committedViewSets[r]:
                    m, v = _mergedFromContribs(contributions.get(b, {}), exclude=r)
                    bwv = baseWithout[r][b][1]
                    assert abs(bwv - v) < 1e-9, (r, b, bwv, v)
                    if b in servedNow and (perim[b] < 1e-9 or v / perim[b] < tau):
                        freshLoss.add(b)
                assert freshLoss == lossSet[r], (r, freshLoss, lossSet[r])
            for a in remaining:
                gOld = gains[a]
                computeGain(a)
                assert gains[a] == gOld, (a, gOld, gains[a])

        gainOrder = sorted((a for a in remaining if gains.get(a, 0) > 0),
                           key=gains.get, reverse=True)

        bestDelta = 0
        bestSwap = None
        bestCross = False
        pairsEvaluated = pairsPruned = 0

        # ==== PASS SCAN ====
        # Three ingredients: the exact non-interacting formula, the
        # gains+lossHit upper bound, and exact evaluation of whatever
        # survives. The bound runs vectorized in the parent; the exact
        # evaluations run across worker processes reading the mmap'd
        # packed viewshed.
        #
        # Removal candidates are visited in ascending removal-loss order:
        # cheap removals produce the biggest deltas, raising the floor
        # early so the bound prunes harder on everything scanned after.
        # ---- phase 0: best NON-interacting swap (cheap, exact) ----
        rOrder = sorted(placedSet, key=lambda x: len(lossSet[x]))
        for r in rOrder:
            nLoss = len(lossSet[r])
            for a in gainOrder:
                d = gains[a] - nLoss
                if d <= bestDelta:
                    break
                if (r, a) in rejected or _seesAnyOf(a, committedViewSets[r]):
                    continue
                bestDelta = d
                bestSwap = (r, a)
                bestCross = True
                break
        floorDelta = bestDelta
        # ---- phase 1: vectorized upper bound over each r's pool ----
        # Stays in the parent rather than being farmed out: these are
        # per-removal-candidate slices of a few thousand entries, and
        # shipping each one to a worker would cost more in pickling than
        # the arithmetic saves. Only the survivors -- which need the
        # expensive exact interval math -- are worth dispatching.
        gainsArr = np.fromiter((gains.get(c, 0) for c in _candList),
                               dtype=np.int64, count=len(_candList))
        tasks, taskRs = [], []
        for r in rOrder:
            loss = lossSet[r]
            nLoss = len(loss)
            poolIdx = poolCache[r]
            if len(poolIdx) == 0:
                continue
            if loss:
                hitIdx = np.concatenate(
                    [_bToCandFlat[_bToCandPtr[b]:_bToCandPtr[b + 1]]
                     for b in loss])
                pos = np.searchsorted(poolIdx, hitIdx)
                valid = pos < len(poolIdx)
                match = np.zeros(len(hitIdx), dtype=bool)
                match[valid] = poolIdx[pos[valid]] == hitIdx[valid]
                counts = np.bincount(pos[match], minlength=len(poolIdx))
                ub = gainsArr[poolIdx] + counts - nLoss
            else:
                ub = gainsArr[poolIdx] - nLoss
            surv = poolIdx[ub > floorDelta]
            pairsPruned += len(poolIdx) - len(surv)
            if len(surv) == 0:
                continue
            rView = committedViewSets[r]
            bw = baseWithout[r]
            pairs = []
            neededBs = set()
            for i in surv.tolist():
                a = _candList[i]
                if a not in remaining or (r, a) in rejected:
                    continue
                tipBs = list(aTips[a] & rView)
                lossBs = [b for b in loss if _seesB(i, b)]
                if gains[a] + len(lossBs) - nLoss <= floorDelta:
                    continue
                pairs.append((i, gains[a], tipBs, lossBs))
                neededBs.update(tipBs)
                neededBs.update(lossBs)
            if not pairs:
                continue
            tasks.append((tau, len(tasks), nLoss, floorDelta,
                          {b: bw[b] for b in neededBs}, pairs))
            taskRs.append(r)
            pairsEvaluated += len(pairs)
        # ---- phase 2: exact evaluation across worker processes ----
        if tasks:
            if pairsEvaluated >= 64:
                outcomes = list(pool.imap_unordered(
                    _refineScanTask, tasks,
                    chunksize=max(1, len(tasks) // (nProcs * 4))))
            else:
                # Batch too small to be worth the dispatch: identical code,
                # run in-process. Not a serial MODE -- there is no setting
                # that reaches it, and it cannot change the result.
                _RPW["p"] = _pk
                outcomes = [_refineScanTask(t) for t in tasks]
            # deterministic pick: highest delta, then lowest-loss r
            # (task order), then lowest candidate index
            outcomes.sort(key=lambda t: (-t[1], t[0], t[2]))
            tId, d, aIdx = outcomes[0]
            if aIdx >= 0 and d > bestDelta:
                bestDelta = d
                bestSwap = (taskRs[tId], _candList[aIdx])
                bestCross = False

        if bestSwap is None:
            break

        r, a = bestSwap
        newView = viewCache[a]

        # exact realized delta over every building r or a touches
        touched = set(committedView[r].keys()) | set(newView.keys())
        realized = 0
        newlyServed, newlyLost = [], []
        for b in touched:
            if perim[b] < 1e-9:
                continue
            ivsByEdge = defaultdict(list)
            for ei, byCand in contributions.get(b, {}).items():
                for cid, cIvs in byCand.items():
                    if cid != r:
                        ivsByEdge[ei].extend(cIvs)
            for ei, ivs in newView.get(b, {}).items():
                ivsByEdge[ei].extend(ivs)
            vis = sum(combineLengths(v) for v in ivsByEdge.values())
            nowServed = min(vis / perim[b], 1.0) >= tau
            wasServed = b in servedNow
            if nowServed and not wasServed:
                realized += 1
                newlyServed.append(b)
            elif wasServed and not nowServed:
                realized -= 1
                newlyLost.append(b)

        if realized <= 0:
            rejected.add((r, a))
            if verbose:
                print(f"{logPrefix}  swap candidate rejected on realized-delta "
                      f"verification (predicted {bestDelta:+d}, realized "
                      f"{realized:+d}) -- pair blacklisted, rescanning")
            continue

        # ---- commit: core state ----
        oldRSet = committedViewSets[r]
        for b, edgeMap in committedView[r].items():
            for ei in edgeMap:
                contributions[b][ei].pop(r, None)
        for b, edgeMap in newView.items():
            for ei, ivs in edgeMap.items():
                contributions[b][ei][a] = ivs
        for b in touched:
            rebuildBuilding(b)
        newlyServedSet = set(newlyServed)
        for b in newlyServed:
            servedNow.add(b)
            unservedSet.discard(b)
        for b in newlyLost:
            servedNow.discard(b)
            unservedSet.add(b)

        placedSet.discard(r); placedSet.add(a)
        # a is now placed and leaves the candidate pool; the removed r
        # becomes available again (this was previously reversed -- masked
        # by the old full-candidate gain recompute, caught by selfCheck
        # once gains went incremental)
        remaining.discard(a); remaining.add(r)
        del committedView[r]; del committedViewSets[r]
        committedView[a] = newView
        newASet = frozenset(newView.keys())
        committedViewSets[a] = newASet
        placed[placed.index(r)] = a
        nSwaps += 1

        # ---- commit: incremental pass-state refresh ----
        for b in oldRSet:
            bCoveredBy[b].discard(r)
        for b in newASet:
            bCoveredBy[b].add(a)

        del baseWithout[r], lossSet[r], poolCache[r]
        refreshBaseline(a, newASet)
        buildPool(a)
        need = defaultdict(list)
        for b in touched:
            for r2 in bCoveredBy[b]:
                if r2 != a:
                    need[r2].append(b)
        for r2, bs in need.items():
            refreshBaseline(r2, bs)

        gains.pop(a, None)
        aTips.pop(a, None)
        # Per-building gain updates: only buildings that are unserved after
        # the swap (tip status may have flipped either way) or that JUST
        # became served (must leave every tip set) can change any
        # candidate's gain. Touched buildings served both before and after
        # were never in a tip set and never enter one -- no work. This is
        # what keeps swap cost bounded by the changed neighborhood even at
        # low tau, where most touched buildings are already served.
        for b in touched:
            if b in servedNow:
                if b in newlyServedSet:            # leave all tip sets
                    for c in buildingToCandidates.get(b, ()):   
                        t = aTips.get(c)
                        if t is not None and b in t:
                            t.discard(b)
                            gains[c] -= 1
                continue
            for c in buildingToCandidates.get(b, ()):
                t = aTips.get(c)
                if t is None:
                    continue
                if _tips(c, b):
                    if b not in t:
                        t.add(b); gains[c] += 1
                elif b in t:
                    t.discard(b); gains[c] -= 1
        computeGain(r)     # the removed antenna is a candidate again

        if verbose:
            kind = "cross-region" if bestCross else "local"
            bonus = f", {realized - bestDelta:+d} vs predicted" \
                    if realized != bestDelta else ""
            print(f"{logPrefix}  swap {nSwaps} ({kind}): {realized:+d} served"
                  f"{bonus} -> {len(servedNow)} "
                  f"[{time.perf_counter()-itT0:.2f}s]")

    pool.close()
    pool.join()

    if verbose:
        if hitCap:
            stopReason = (f"swap cap hit ({maxIterations}) while still "
                          f"improving -- raise REFINE_MAX_ITERATIONS for more")
        elif hitBudget:
            stopReason = (f"time budget hit ({timeBudget:.0f}s) while still "
                          f"improving -- raise REFINE_TIME_BUDGET for more")
        else:
            stopReason = "local optimum, no improving swap exists"
        print(f"{logPrefix}  refined {served0} -> {len(servedNow)} "
              f"({len(servedNow)-served0:+d}) in {nSwaps} swap(s), "
              f"{time.perf_counter()-tStart:.2f}s -- {stopReason}")

    return placed, len(servedNow), nSwaps, dict(committedView)


# ===========================================================================
#  ADAPTIVE CORRIDOR VIEWSHED  (long streets / plazas past the base radius)
# ===========================================================================

def candidateViewshedAdaptive(pt, polygons, grid, radius, maxRadius,
                              factor=2.0):
    """
    Viewshed with automatic long-street handling. Sweeps the base scene at
    `radius`; any OPEN wedge (an angular sector where nothing blocks within
    the radius -- i.e. the candidate is looking down a street, plaza, or
    other corridor) is re-swept against a scene of doubled radius,
    RESTRICTED to those corridor directions, repeating until every
    direction is blocked or `maxRadius` is reached. The final pass credits
    whatever the corridors contain out to maxRadius.

    Cost model: candidates enclosed by buildings pay exactly one base
    sweep (their wedges all close early). Street candidates pay extra
    sweeps whose (angles x edges) work scales with the corridor's ANGULAR
    share of the circle, not with the full bigger scene -- which is what
    makes maxRadius of 4-8x the base affordable.

    Correctness: a wedge blocked within r is final (any occluder crossing
    nearer than the hit lies within r of the observer, so it is already
    in the scene); only open wedges are deferred, and they are re-solved
    exactly against the larger scene. Interval overlaps at window
    boundaries are harmless -- every consumer merges intervals via
    combineLengths (union semantics).

    Returns the same {buildingIdx: {edgeIdx: [(u0,u1), ...]}} mapping as
    candidateViewshed.
    """
    out = {}
    r = float(radius)
    windows = None
    while True:
        final = r >= maxRadius - 1e-9
        idxs = grid.neighborIndices(pt[0], pt[1], r)
        if idxs:
            scene, sub = localScene(idxs, polygons)
            epp, ae, ave = scene
            eo = buildEdgeOwner(epp)
            if final:
                # cap the FINAL pass at r too: this scene only contains
                # buildings whose bbox comes within r, so a wedge hit beyond
                # r could be missing its true occluder (which may lie wholly
                # outside the scene) -- crediting it risks a false claim.
                # Beyond-r wedges are dropped, not deferred (no bigger pass).
                res, _beyondMax = sweepAllEdges(pt, ae, ave, eo,
                                                radius=r, angleWindows=windows)
                openW = []
            else:
                res, openW = sweepAllEdges(pt, ae, ave, eo,
                                           radius=r, angleWindows=windows)
            for p, em in res.items():
                g = idxs[p]
                dst = out.setdefault(g, {})
                for ei, ivs in em.items():
                    dst.setdefault(ei, []).extend(ivs)
        else:
            openW = [(0.0, 2 * math.pi)] if not final else []
        if final or not openW:
            # corridor passes deliberately overlap at window boundaries;
            # collapse every edge's fragments into a sorted disjoint union
            return normalizeView(out)
        windows = openW
        r = min(r * factor, maxRadius)


# ===========================================================================
#  FAST BOUNDARY CHECK  (grid-bucketed edges; replaces the O(allEdges)
#  scan per antenna, which at full scale is ~92K pointOnEdge calls PER
#  ANTENNA -- ~92M calls per k=1000 subcase)
# ===========================================================================

def buildBoundaryChecker(polygons, cell=50.0):
    """
    Returns onAnyBoundary(pt) -> bool. Fast path: exact membership in the
    candidate-vertex set (antennas produced by this pipeline are always
    polygon vertices, and repr round-trips doubles exactly, so parsed
    submissions still hit it). Fallback: exact pointOnEdge against only
    the edges whose bounding boxes intersect the point's grid cell.
    """
    vertexSet = set(candidateList(polygons))
    edgeBuckets = defaultdict(list)
    inv = 1.0 / cell
    for poly in polygons:
        for ea, eb in normalizePolygon(poly):
            gx0 = int(min(ea[0], eb[0]) * inv); gx1 = int(max(ea[0], eb[0]) * inv)
            gy0 = int(min(ea[1], eb[1]) * inv); gy1 = int(max(ea[1], eb[1]) * inv)
            for gx in range(gx0, gx1 + 1):
                for gy in range(gy0, gy1 + 1):
                    edgeBuckets[(gx, gy)].append((ea, eb))

    def onAnyBoundary(pt):
        t = (pt[0], pt[1])
        if t in vertexSet:
            return True
        gx, gy = int(pt[0] * inv), int(pt[1] * inv)
        for dgx in (-1, 0, 1):
            for dgy in (-1, 0, 1):
                for ea, eb in edgeBuckets.get((gx + dgx, gy + dgy), ()):
                    if pointOnEdge(pt[0], pt[1], ea, eb):
                        return True
        return False

    return onAnyBoundary