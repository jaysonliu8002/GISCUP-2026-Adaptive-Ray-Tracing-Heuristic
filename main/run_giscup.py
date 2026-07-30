"""
run_giscup.py — compute, validate, and plot for the GIS Cup 2026 solver.
 
All reusable functions live in giscup_lib.py; this file is just the driver.
Run the sections in order (or as notebook cells). Each section is self-contained
and reads from files, so you can re-run validation/plotting without re-solving.
 
IMPORTANT (multiprocessing): the parallel precompute spawns worker processes
that import giscup_lib. When running as a plain script, guard the entry point
with `if __name__ == "__main__":` (done below). In Jupyter, parallel=True works
because the workers import the installed giscup_lib module.
"""
 
from lib_giscup import (
    greedyCornersCached,
    loadGeoJSON, loadOrBuildViewsheds, servicedIdsFromViewCache, writeSubmission, parseSubmission,
    greedyLookahead, buildExtraLen,
    localEvalCoverParallel, refineWithSwapsV2, buildBoundaryChecker,
    normalizePolygon, pointOnEdge, computeConvexity, SpatialGrid,
    loadOrBuildPackedViewshed, extraLenFromPacked, baseCovFromPacked,
    buildBuildingToCandidates, buildRefineParallelContext,
)
# matplotlib is imported INSIDE the plotting functions, never at module
# scope: multiprocessing spawn re-imports this file in every worker, so a
# module-level matplotlib import would charge ~1s of startup to each of the
# hundreds of workers a solve creates -- none of which plot anything.
from centrality_giscup import loadOrPickVantageSeeds
from datetime import datetime
import os
import time

# Data files (dataset, cache/, submission/) live at the repo root, one level
# above this script's folder, so anchor paths there instead of relying on the
# current working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Config -- every switch is documented in README.md, including the
# measurements behind each default. Keep this block terse; put prose there.
# ---------------------------------------------------------------------------

# -- problem ----------------------------------------------------------------
TAUS = [0.25, 0.5, 0.75]
KS   = [50, 500, 1000]
GEOJSON_PATH = os.path.join(REPO_ROOT, "GIS-cup-sample-dataset.geojson")

# -- radii / viewshed reach --------------------------------------------------
SOLVE_RADIUS       = 80.0
ADAPTIVE_DOUBLINGS = 5                     # corridor reach = SOLVE_RADIUS * 2**n; 0/None = off
SOLVE_GRIDCELL     = SOLVE_RADIUS
VALIDATION_STEP    = 1.25                  # validation re-sweep reach = claim reach * this;
                                           # > 1 so the re-sweep can both re-check claims
                                           # independently AND harvest extra true ones

# -- main strategy ------------------------------------------------------------
LOOKAHEAD_GREEDY       = True             # chess-style placement search (see README)
LOOKAHEAD_BEAM         = 5                 # candidates given the deep evaluation per step
LOOKAHEAD_DEPTH_BY_TAU = {0.25: 1, 0.5: 2, 0.75: 3}   # tau -> ply gradient; None = fixed
LOOKAHEAD_DEPTH        = 2                 # used when the gradient is None

# -- refinement ---------------------------------------------------------------
REFINE_WITH_SWAPS     = True
REFINE_MAX_ITERATIONS = 10000                # cap on COMMITTED swaps
REFINE_TIME_BUDGET    = 3600                # seconds per subcase; None = unlimited
REFINE_VERBOSE        = True
REFINE_PROCESSES      = None               # workers for the parallel swap-scan
                                           # (mmap'd packed viewshed, see README);
                                           # None -> os.cpu_count(), 0 = serial

# -- crash safety --------------------------------------------------------------
RESUME_PARTIAL_SUBMISSION = True           # a partially-written submission_N.txt
                                           # (crash mid-solve) is continued, not
                                           # restarted; complete files stay
                                           # protected as before

# -- validation ---------------------------------------------------------------
SKIP_REDUNDANT_VALIDATION = True
CLAIM_MARGIN = 1e-6      # search, claim, and validate at tau + this, keeping
                         # every claim clear of the knife edge (see README)

# -- vantage seeding (centrality_alg.py) ---------------------------------------
VANTAGE_SEED_PERCENT  = 0.01               # 0 = off (enables the faster slicing path)
VANTAGE_RADIUS        = 1280.0
VANTAGE_NODE_CAPACITY = 25
VANTAGE_PROCESSES     = None               # None -> os.cpu_count()

SUBMISSION_DIR = os.path.join(REPO_ROOT, "submission")

def _submissionPath(n):
    return os.path.join(SUBMISSION_DIR, f"submission_{n}.txt")

def nextSubmissionPath():
    """First submission_N.txt (N = 1, 2, ...) that doesn't exist yet, so a
    new solve never overwrites an earlier submission."""
    n = 1
    while os.path.exists(_submissionPath(n)):
        n += 1
    return _submissionPath(n)

def latestSubmissionPath():
    """Highest-numbered existing submission_N.txt, or None if there are none."""
    n = 1
    while os.path.exists(_submissionPath(n)):
        n += 1
    return _submissionPath(n - 1) if n > 1 else None

def _resumableSubmission():
    """
    Where this solve should write, and which subcases are already done.

    If the LATEST submission_N.txt holds a strict subset of this run's
    (tau, k) grid -- i.e. a previous solve crashed or was interrupted
    after writing some but not all subcases -- resume it: reuse its
    finished subcases verbatim and keep writing to the same file.
    Anything else (no file, a complete file, a file whose subcases don't
    match the current TAUS/KS config, an unparseable file) starts a fresh
    numbered file, so complete submissions stay protected exactly as
    before.

    Returns (outPath, doneResults) with doneResults in writeSubmission's
    format.
    """
    expected = {(t, k) for t in TAUS for k in KS}
    path = latestSubmissionPath()
    if not RESUME_PARTIAL_SUBMISSION or path is None:
        return nextSubmissionPath(), []
    try:
        subs = parseSubmission(path)
    except Exception:
        return nextSubmissionPath(), []
    keys = [(s["tau"], s["k"]) for s in subs]
    if len(keys) == len(set(keys)) and 0 < len(keys) and set(keys) < expected:
        done = [{"tau": s["tau"], "k": s["k"], "antennas": s["antennas"],
                 "servicedIds": s["claimedIds"]} for s in subs]
        return path, done
    return nextSubmissionPath(), []

# ===========================================================================
#  SHARED VALIDATION HELPER  (used by validate() and solve() so both
#  independently re-check claims the exact same way — one definition of
#  "verified", not two copies that could drift apart)
# ===========================================================================

def _buildBoundaryChecker(polygons):
    """Returns onAnyBoundary(pt) -> bool. Delegates to the lib's
    grid-bucketed checker: O(1) vertex-set hit for pipeline-produced
    antennas plus a local-cell edge fallback, instead of the old
    every-edge scan (~92K pointOnEdge calls PER ANTENNA at full scale)."""
    return buildBoundaryChecker(polygons)


def _validateSubproblem(tau, k, antennas, claimedIds, polygons, ids, idToIndex,
                        onAnyBoundary, radius, maxRadius=None, gridCell=None,
                        skipCoverageRecheck=False, verbose=True):
    """
    Independently re-check ONE (tau, k) sub-problem's claims with the same
    adaptive corridor sweep the solve cache was built with (base `radius`,
    corridors followed out to `maxRadius` -- normally _validationMax(),
    one VALIDATION_STEP past the claim reach). Checks: antenna count
    matches k, every antenna sits on a building boundary, and every
    claimed building's re-swept coverage actually reaches tau. Uses
    localEvalCoverParallel (independent sweep, CPU-core parallel) rather
    than anything from the solve-side viewCache, so it also catches an
    implementation bug in either code path. Because it reaches PAST the
    claim reach it additionally HARVESTS: buildings the extra reach tips
    over tau join verifiedIds and get submitted (the contest scores
    visibility with no distance limit, so these are free true claims).

    skipCoverageRecheck : skip the EXPENSIVE localEvalCoverParallel
        re-sweep (the antenna-count and on-boundary checks still run
        either way -- those are cheap and catch a different class of
        problem). solve() sets this from _shouldSkipCoverageRecheck(),
        i.e. only when the re-sweep would reach no further than the
        claims themselves (VALIDATION_STEP <= 1).

    All coverage comparisons here use tau + CLAIM_MARGIN, matching how the
    claims were generated -- the contest evaluator recomputes visibility
    with its own floating-point path, so a claim within float noise of the
    bare threshold could flip there; the margin keeps every submitted claim
    clear of that edge.

    Returns (verifiedResult, ok):
      verifiedResult : {"tau", "k", "antennas", "servicedIds"} using the
                        VERIFIED ids — safe to feed straight into
                        writeSubmission.
      ok              : False if any problem was found (antenna count,
                        off-boundary, or false claims).
    """
    if gridCell is None:
        gridCell = radius
    effTau = tau + CLAIM_MARGIN
    problems = []

    if len(antennas) != k:
        problems.append(f"antenna count {len(antennas)} != k={k}")

    offBoundary = sum(1 for a in antennas if not onAnyBoundary(a))
    if offBoundary:
        problems.append(f"{offBoundary} antennas off-boundary")

    if skipCoverageRecheck:
        verifiedIds = list(claimedIds)
        status = "OK" if not problems else "FAIL"
        if verbose:
            if status != "OK":
                print(f"    verify tau={tau}, k={k}: {len(antennas)} antennas, "
                      f"{len(claimedIds)} claimed -> {status}")
            for p in problems:
                print(f"        - {p}")
        verifiedResult = {"tau": tau, "k": k, "antennas": antennas, "servicedIds": verifiedIds}
        return verifiedResult, (not problems)

    # parallel coverage re-check (one sweep per antenna, across CPU cores)
    covDict = localEvalCoverParallel(antennas, polygons, radius=radius, gridCell=gridCell,
                                     maxRadius=maxRadius)

    falseClaims = []
    for bid in claimedIds:
        idx = idToIndex.get(bid)
        if idx is None:
            falseClaims.append((bid, "unknown id"))
        elif covDict.get(idx, 0.0) < effTau:
            falseClaims.append((bid, f"{covDict.get(idx,0.0):.6f} < {tau} + margin"))
    if falseClaims:
        problems.append(f"{len(falseClaims)} false claims")

    verifiedIds = [ids[i] for i in range(len(polygons)) if covDict.get(i, 0.0) >= effTau]

    status = "OK" if not problems else "FAIL"
    if verbose and (status != "OK" or len(verifiedIds) < len(claimedIds)):
        print(f"    verify (tau={tau}, k={k}) @ reach {int(maxRadius or radius)}: {len(antennas)} antennas, "
              f"{len(claimedIds)} claimed / {len(verifiedIds)} verified -> {status}, "
              f"diff = {len(verifiedIds) - len(claimedIds)}")
        for p in problems:
            print(f"        - {p}")
        for bid, reason in falseClaims[:5]:
            print(f"          false: building {bid} ({reason})")

    verifiedResult = {"tau": tau, "k": k, "antennas": antennas, "servicedIds": verifiedIds}
    return verifiedResult, (not problems)

def _lookaheadDepth(tau):
    """Ply depth for greedyLookahead at this tau: nearest key in the
    LOOKAHEAD_DEPTH_BY_TAU gradient, or the fixed LOOKAHEAD_DEPTH."""
    if LOOKAHEAD_DEPTH_BY_TAU:
        key = min(LOOKAHEAD_DEPTH_BY_TAU, key=lambda t: abs(t - tau))
        return LOOKAHEAD_DEPTH_BY_TAU[key]
    return LOOKAHEAD_DEPTH

def _adaptiveMax():
    """The corridor-following reach implied by ADAPTIVE_DOUBLINGS, as an
    absolute radius (SOLVE_RADIUS * 2**doublings), or None when disabled."""
    if ADAPTIVE_DOUBLINGS and ADAPTIVE_DOUBLINGS > 0:
        return SOLVE_RADIUS * (2 ** ADAPTIVE_DOUBLINGS)
    return None

def _validationMax():
    """Reach of the validation re-sweep: one extra corridor step,
    VALIDATION_STEP x past the claim reach (the corridor doublings' reach,
    or SOLVE_RADIUS when adaptive is off). The re-sweep is the SAME
    adaptive method as the solve cache -- same base radius, same corridor
    doublings -- just allowed to follow still-open corridors this one
    step further, so it re-derives every claim independently AND can tip
    extra buildings with the perimeter it newly sees past the claim
    reach. A wider sweep can only ADD coverage (any occluder of a point
    at distance d lies within d, so claims within reach are exact by
    construction), so the harvest is as trustworthy as the re-checked
    claims themselves."""
    return (_adaptiveMax() or SOLVE_RADIUS) * VALIDATION_STEP

def _shouldSkipCoverageRecheck():
    """
    Whether the independent coverage re-sweep would be redundant: True when
    its reach does not exceed the reach the claims were computed at (same
    method sweeping the same geometry could only re-confirm, never falsify
    or extend). With VALIDATION_STEP > 1 the re-sweep always reaches
    further, so this stays False and validation runs every subcase; set
    VALIDATION_STEP <= 1 to skip it (keeping only the cheap antenna-count
    and on-boundary checks).
    """
    claimReach = max(SOLVE_RADIUS, _adaptiveMax() or 0)
    return SKIP_REDUNDANT_VALIDATION and claimReach >= _validationMax()

# ===========================================================================
#  VANTAGE SEEDS  (visibility-graph starting antennas, tau-independent)
#  Build the undirected vertex-to-vertex visibility graph once, then pick
#  n high-degree, spatially-spread corners via iterative
#  max-degree-then-remove-its-edges selection. Passed into a
#  greedyCornersCached call as fixed starting antennas -- the real search
#  fills in the rest around them. See VANTAGE_SEED_PERCENT's config
#  comment for how many seeds each subcase actually asks for.
# ===========================================================================

def _pickVantageSeeds(polygons, convexity, n=0, radius=VANTAGE_RADIUS,
                      nodeCapacity=VANTAGE_NODE_CAPACITY, processes=VANTAGE_PROCESSES,
                      verbose=True):
    if n <= 0:
        # Skip building the visibility graph entirely -- no point paying
        # for it just to discard the result.
        if verbose:
            print("Vantage seeds: n=0 requested, skipping (plain greedy).")
        return []
    from lib_giscup import candidateList   # was `from lib_giscup import` -- broken name
    candidates = candidateList(polygons)
    # Cached and crash-safe: the graph build behind these seeds is the
    # longest single stage in the pipeline (~22 min at radius 640), so it
    # checkpoints as it goes and both it and the picked seeds persist to
    # cache/. A re-run with the same settings loads the seeds outright; a
    # re-run with a different seed count reuses the graph; a run resumed
    # after a crash mid-build picks up from the checkpoint.
    # Cache filenames encode exactly the settings the CONTENTS depend on --
    # here just the radius (plus the seed count, on the seeds file only).
    # VANTAGE_NODE_CAPACITY, gridCell and the process count are performance
    # knobs that cannot change an edge, so they are deliberately absent from
    # both the name and the metadata check (see vantageMeta): tuning them
    # must not cost a 22-minute rebuild.
    tag = f"r{int(radius)}"
    top = loadOrPickVantageSeeds(
        polygons, candidates, n=n, radius=radius, nodeCapacity=nodeCapacity,
        convexity=convexity, processes=processes, verbose=verbose,
        seedsPath=os.path.join(REPO_ROOT, "cache", f"vantage_seeds_{tag}_n{n}.pkl"),
        graphPath=os.path.join(REPO_ROOT, "cache", f"vantage_graph_{tag}.pkl"))
    if verbose:
        print(f"Vantage seeds ({len(top)}/{n} requested):")
        for v, deg in top:
            print(f"    {v}  (visibility-graph degree at selection: {deg})")
    return [v for v, _deg in top]

# Static, cache-derived tables shared by every tau/k subcase -- solve()
# builds them once (from the packed viewshed arrays, see README) and
# greedy/lookahead/refine all read them from here:
#   "v"        (in _EXTRA_LEN) : per-(candidate, building) unmerged-length
#                                upper bound (buildExtraLen equivalent)
#   "baseCov"  : per-(candidate, building) merged standalone coverage
#   "btc"      : building -> candidates-that-see-it index
#   "refineCtx": mmap'd packed arrays + index maps for the PARALLEL
#                refine swap-scan (None/absent -> refine runs serial)
_EXTRA_LEN = {}
_SHARED = {}

def _runGreedy(polygons, kTarget, tau, candidates, viewCache, seed, convexity, seedVertices):
    """
    Runs the configured placement engine (plain greedy, or the lookahead
    when LOOKAHEAD_GREEDY is on) behind one consistent interface.
    Returns (placed, served). Final coverage is always read from the
    shared `viewCache` -- the adaptive corridor cache IS the commit-
    accurate viewshed, there is no separate wide commit step anymore.
    """
    if LOOKAHEAD_GREEDY:
        return greedyLookahead(
            polygons, kTarget, tau, candidates, viewCache,
            beamWidth=LOOKAHEAD_BEAM, depth=_lookaheadDepth(tau),
            seed=seed, convexity=convexity,
            seedVertices=seedVertices, extraLen=_EXTRA_LEN.get("v"),
            baseCov=_SHARED.get("baseCov"), buildingToCands=_SHARED.get("btc"))
    return greedyCornersCached(
        polygons, kTarget, tau, candidates, viewCache,
        seed=seed, convexity=convexity, seedVertices=seedVertices,
        extraLen=_EXTRA_LEN.get("v"),
        baseCov=_SHARED.get("baseCov"), buildingToCands=_SHARED.get("btc"))

def _runRefine(polygons, placed, tau, candidates, viewCache):
    """
    Swap refinement behind one consistent interface. No-op if
    REFINE_WITH_SWAPS is off.

    Returns (newPlaced, nSwaps, placedViewCache) -- placedViewCache maps
    every candidate in newPlaced to its viewCache entry, ready for
    servicedIdsFromViewCache.
    """
    if not REFINE_WITH_SWAPS:
        return placed, 0, {c: viewCache[c] for c in placed}
    newPlaced, _served, nSwaps, finalVC = refineWithSwapsV2(
        polygons, placed, tau, candidates, viewCache,
        maxIterations=REFINE_MAX_ITERATIONS, verbose=REFINE_VERBOSE,
        timeBudget=REFINE_TIME_BUDGET, extraLen=_EXTRA_LEN.get("v"),
        buildingToCands=_SHARED.get("btc"),
        parallelCtx=_SHARED.get("refineCtx"))
    return newPlaced, nSwaps, finalVC

# ===========================================================================
#  SECTION 1 — SOLVE  (slow: the full precompute + nine sub-problems)
# ===========================================================================

_T0 = time.perf_counter()

def _log(msg, indent=0):
    """Uniform log line: [+elapsed-since-start] message."""
    print(f"[{time.perf_counter() - _T0:7.1f}s] {'    ' * indent}{msg}")

def _timed(label, fn, *args, **kwargs):
    """Run fn(*args, **kwargs), log how long it took, return its result."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    _log(f"{label}: {time.perf_counter() - t0:.2f}s")
    return result

def _printSummaryTable(rows):
    """rows: list of dicts with tau, k, claimed, verified, swaps, seconds."""
    print()
    print("    tau     k    served   swaps   time")
    print("    " + "-" * 40)
    for r in rows:
        print(f"    {r['tau']:<5} {r['k']:>5} {r['verified']:>9} "
              f"{r['swaps']:>7} {r['seconds']:>6.1f}s")
    print("    " + "-" * 40)
    print(f"    total served: {sum(r['verified'] for r in rows)}")
    print()

def solve(seed=0):
    solveStart = time.perf_counter()

    polygons, ids = _timed("loadGeoJSON", loadGeoJSON, GEOJSON_PATH)
    idToIndex = {bid: i for i, bid in enumerate(ids)}
    onAnyBoundary = _timed("_buildBoundaryChecker", _buildBoundaryChecker, polygons)

    # build once or load from disk (the precompute checkpoints itself to a
    # .partial file every few minutes and resumes from it after a crash;
    # the finished cache is saved to disk BEFORE any solving starts)
    cacheBase = os.path.join(REPO_ROOT, "cache",
                             f"viewcache_r{int(SOLVE_RADIUS)}"
                             + (f"_d{ADAPTIVE_DOUBLINGS}" if _adaptiveMax() else ""))
    candidates, viewCache, perim = _timed(
        "loadOrBuildViewsheds", loadOrBuildViewsheds,
        polygons, radius=SOLVE_RADIUS, gridCell=SOLVE_GRIDCELL,
        maxRadius=_adaptiveMax(), path=cacheBase + ".pkl")

    # Packed numpy mirror of the cache (built once, then loaded mmap'd):
    # feeds the parallel refine workers AND yields the shared per-
    # (candidate, building) tables below as vectorized array reads.
    packed = _timed("loadOrBuildPackedViewshed", loadOrBuildPackedViewshed,
                    cacheBase + "_packed", candidates, viewCache, perim, polygons)
    _EXTRA_LEN["v"] = _timed("extraLenFromPacked (shared by greedy/lookahead/refine)",
                             extraLenFromPacked, candidates, packed)
    _SHARED["baseCov"] = _timed("baseCovFromPacked (shared standalone coverages)",
                                baseCovFromPacked, candidates, packed)
    _SHARED["btc"] = _timed("buildBuildingToCandidates (shared index)",
                            buildBuildingToCandidates, candidates, viewCache)
    if REFINE_WITH_SWAPS and (REFINE_PROCESSES is None or REFINE_PROCESSES > 0):
        _SHARED["refineCtx"] = _timed(
            "buildRefineParallelContext (parallel swap-scan)",
            buildRefineParallelContext, candidates, cacheBase + "_packed",
            REFINE_PROCESSES)
    else:
        _SHARED.pop("refineCtx", None)

    results = []
    allOk = True
    kMax = max(KS)
    # config-static: whether the independent coverage re-sweep will run at
    # all this run -- used both to skip it and to log HONESTLY about it
    skipRecheck = _shouldSkipCoverageRecheck()

    # Crash safety: pick (or resume) the output file NOW and rewrite it
    # after every verified subcase, so an interruption at any point keeps
    # everything finished so far and the next run continues from there.
    outPath, doneResults = _resumableSubmission()
    doneKeys = {(r["tau"], r["k"]) for r in doneResults}
    if doneResults:
        _log(f"Resuming partial {os.path.basename(outPath)}: "
             f"{sorted(doneKeys)} already done, skipping those subcases")

    def _flushSubmission():
        writeSubmission(outPath, sorted(doneResults + results,
                                        key=lambda r: (r["tau"], r["k"])))

    # Corner convexity: a cheap, purely geometric tiebreak (prefer sharper
    # outward corners over near-straight or reflex ones) applied ONLY when
    # candidates are otherwise tied on the real (gain, score) objective.
    # Computed once here and shared across every tau below.
    convexity = _timed("computeConvexity", computeConvexity, polygons)

    # Vantage seed POOL: computed ONCE (tau-independent) -- each subcase
    # below draws its own PERCENTAGE-sized prefix of this same pool, via
    # max(1, int(VANTAGE_SEED_PERCENT * k)) (see the floor's own comment
    # at the draw site). The pool only ever needs to be as large as the
    # BIGGEST single draw across every subcase, which happens at k=kMax --
    # requesting n=kMax unconditionally (an earlier version of this) asks
    # topKVantageVertices for far more than will ever actually be drawn
    # whenever VANTAGE_SEED_PERCENT is small (e.g. 1% of kMax=1000 only
    # ever draws 10 seeds, not 1000 -- requesting 1000 wastes ~100x the
    # necessary work in that selection loop). Skipped entirely (empty
    # pool, no visibility graph built) when VANTAGE_SEED_PERCENT is 0.
    if VANTAGE_SEED_PERCENT > 0:
        poolSize = max(1, int(VANTAGE_SEED_PERCENT * kMax))
        vantageSeedsPool = _timed(
            "_pickVantageSeeds", _pickVantageSeeds,
            polygons, convexity, n=poolSize, radius=VANTAGE_RADIUS,
            nodeCapacity=VANTAGE_NODE_CAPACITY, processes=VANTAGE_PROCESSES)
    else:
        vantageSeedsPool = []

    if VANTAGE_SEED_PERCENT <= 0:
        # FAST PATH -- identical to having no vantage-seed code at all.
        # Reuse across sub-problems: greedy's antenna ORDER for a fixed tau
        # doesn't depend on the target k (every step's choice only depends
        # on what's already placed, not on how many more will follow), so
        # for each tau, run the search ONCE at the largest k, then slice
        # that ordered list for every smaller k -- cuts the number of
        # greedy calls from len(TAUS)*len(KS) down to len(TAUS).
        #
        # servicedIds are computed incrementally too, reading from the
        # shared viewCache and only processing the NEW antennas since the
        # previous k (carrying the coverage state forward) -- UNLESS
        # refinement is active, which changes WHICH antennas are placed
        # (not just appends), breaking that trick.
        summaryRows = []
        for tau in TAUS:
            tauStart = time.perf_counter()
            if all((tau, k) in doneKeys for k in KS):
                _log(f"tau={tau}: every subcase resumed from file, "
                     f"skipping the greedy entirely")
                continue
            # Search at the SAME threshold claims are filtered at: the sample
            # data is full of rectangles where an antenna sees exactly two of
            # four sides -- coverage lands at tau + ~1e-7 at tau=0.5. At bare
            # tau the search would count those "served" and then lose them to
            # the claim margin; at tau + margin it spends its budget tipping
            # them robustly past the knife edge instead.
            tauEff = tau + CLAIM_MARGIN
            placedFull, servedFull = _runGreedy(
                polygons, kMax, tauEff, candidates, viewCache,
                seed, convexity, None)
            _log(f"tau={tau}: greedy at kMax={kMax} served {servedFull} "
                 f"({time.perf_counter() - tauStart:.1f}s)")

            coverageState = None
            prevK = 0
            for k in sorted(KS):
                if (tau, k) in doneKeys:
                    # resumed from the partial file; prevK/coverageState are
                    # deliberately NOT advanced, so the incremental branch
                    # below still processes placedFull[prevK:k] correctly
                    # for the next non-skipped k
                    _log(f"tau={tau}, k={k}: resumed from file, skipped")
                    continue
                start_time = time.perf_counter()

                if REFINE_WITH_SWAPS:
                    # Refinement changes WHICH antennas are placed (not just
                    # appends), so the incremental "newAntennas since prevK"
                    # trick no longer applies for this k -- compute claimedIds
                    # fresh from the refined set instead. coverageState is
                    # deliberately NOT carried forward across k here for the
                    # same reason (each k's refined set isn't a superset of
                    # the previous k's).
                    antennasK, nSwaps, sliceVC = _runRefine(
                        polygons, placedFull[:k], tauEff, candidates, viewCache)
                    claimedIds, _state = servicedIdsFromViewCache(
                        antennasK, sliceVC, perim, ids, tauEff, state=None)
                    label = "refined"
                else:
                    newAntennas = placedFull[prevK:k]
                    claimedIds, coverageState = servicedIdsFromViewCache(
                        newAntennas, viewCache, perim, ids, tauEff,
                        state=coverageState)
                    antennasK = placedFull[:k]
                    nSwaps = 0
                    label = "incremental"

                verifiedResult, ok = _validateSubproblem(
                    tau, k, antennasK, claimedIds, polygons, ids, idToIndex,
                    onAnyBoundary, radius=SOLVE_RADIUS, maxRadius=_validationMax(),
                    skipCoverageRecheck=skipRecheck)
                allOk = allOk and ok
                results.append(verifiedResult)
                _flushSubmission()   # crash-safe: this subcase is on disk now

                dt = time.perf_counter() - start_time
                nVerified = len(verifiedResult['servicedIds'])
                d = nVerified - len(claimedIds)
                note = "" if d == 0 else (
                    f", +{d} extra harvested by the validation re-sweep" if d > 0
                    else f" -- LOST {-d} CLAIMS, see FAIL above")
                statusTxt = ("FAILED validation -- see FAIL above" if not ok
                             else "checks OK (coverage recheck skipped as redundant)"
                             if skipRecheck
                             else f"verified OK at reach {int(_validationMax())}")
                _log(f"tau={tau}, k={k}: {nVerified} served, {statusTxt}"
                     f"{note} [{dt:.1f}s]")
                summaryRows.append({"tau": tau, "k": k, "claimed": len(claimedIds),
                                    "verified": len(verifiedResult['servicedIds']),
                                    "swaps": nSwaps, "seconds": dt})
                prevK = k

            _log(f"tau={tau} done ({time.perf_counter() - tauStart:.1f}s)")
        _printSummaryTable(summaryRows)

    else:
        # SLOW(ER) PATH -- VANTAGE_SEED_PERCENT > 0. Each subcase draws
        # int(VANTAGE_SEED_PERCENT * k) seeds from the front of
        # vantageSeedsPool. Different k's now need different SEED COUNTS,
        # and inserting a different number of forced seeds changes
        # greedyCornersCached's entire downstream trajectory from that
        # point on -- not just a suffix of it -- so subcases can no longer
        # share one computed-once-at-kMax run sliced per k. Each (tau, k)
        # pair gets its own separate greedy call and its own fresh
        # (non-incremental) coverage computation -- this is the real,
        # accepted cost of per-subcase-sized seeding, see the config
        # comment on VANTAGE_SEED_PERCENT.
        summaryRows = []
        for tau in TAUS:
            tauStart = time.perf_counter()
            # same search-at-margined-threshold reasoning as the fast path
            tauEff = tau + CLAIM_MARGIN
            for k in sorted(KS):
                if (tau, k) in doneKeys:
                    _log(f"tau={tau}, k={k}: resumed from file, skipped")
                    continue
                start_time = time.perf_counter()
                # Truncation (int()) can round a small percentage of a small
                # k down to 0, silently disabling seeding for that specific
                # subcase even though VANTAGE_SEED_PERCENT is nonzero (e.g.
                # 1% of k=50 truncates to 0). Since we're in this branch at
                # all, centrality picking is deliberately turned on -- floor
                # at 1 so every subcase gets at least one centrality-picked
                # seed, never silently zero.
                nSeeds = max(1, int(VANTAGE_SEED_PERCENT * k))
                seedsForThisK = vantageSeedsPool[:nSeeds]

                placed, served = _runGreedy(
                    polygons, k, tauEff, candidates, viewCache,
                    seed, convexity, seedsForThisK)
                _log(f"tau={tau}, k={k}: greedy served {served} "
                     f"({nSeeds} vantage seed(s))")

                nSwaps = 0
                effectiveVC = viewCache
                if REFINE_WITH_SWAPS:
                    placed, nSwaps, effectiveVC = _runRefine(
                        polygons, placed, tauEff, candidates, viewCache)

                claimedIds, _state = servicedIdsFromViewCache(
                    placed, effectiveVC, perim, ids, tauEff, state=None)

                verifiedResult, ok = _validateSubproblem(
                    tau, k, placed, claimedIds, polygons, ids, idToIndex,
                    onAnyBoundary, radius=SOLVE_RADIUS, maxRadius=_validationMax(),
                    skipCoverageRecheck=skipRecheck)
                allOk = allOk and ok
                results.append(verifiedResult)
                _flushSubmission()   # crash-safe: this subcase is on disk now

                dt = time.perf_counter() - start_time
                nVerified = len(verifiedResult['servicedIds'])
                d = nVerified - len(claimedIds)
                note = "" if d == 0 else (
                    f", +{d} extra harvested by the validation re-sweep" if d > 0
                    else f" -- LOST {-d} CLAIMS, see FAIL above")
                statusTxt = ("FAILED validation -- see FAIL above" if not ok
                             else "checks OK (coverage recheck skipped as redundant)"
                             if skipRecheck
                             else f"verified OK at reach {int(_validationMax())}")
                _log(f"tau={tau}, k={k}: {nVerified} served, {statusTxt}"
                     f"{note} [{dt:.1f}s]")
                summaryRows.append({"tau": tau, "k": k, "claimed": len(claimedIds),
                                    "verified": len(verifiedResult['servicedIds']),
                                    "swaps": nSwaps, "seconds": dt})

            _log(f"tau={tau} done ({time.perf_counter() - tauStart:.1f}s)")
        _printSummaryTable(summaryRows)

    _flushSubmission()   # already written per-subcase; final write for safety
    _log("All subcases passed their checks." if allOk
         else "SOME SUBCASES HAD PROBLEMS -- see FAIL lines above.")
    if skipRecheck:
        reach = max(SOLVE_RADIUS, _adaptiveMax() or 0)
        _log(f"Wrote {outPath} (coverage recheck SKIPPED -- the re-sweep "
             f"reach {int(_validationMax())} does not exceed the claim reach "
             f"{int(reach)}; raise VALIDATION_STEP above 1 to independently "
             f"re-check and harvest).")
    else:
        _log(f"Wrote {outPath} (independently re-swept at reach "
             f"{int(_validationMax())} per subcase).")
    _log(f"solve() total: {time.perf_counter() - solveStart:.2f}s")
 
# ===========================================================================
#  SECTION 2 — PLOT  (reads the VERIFIED file; independent of solving)
#  Colors each building by the verified serviced IDs from the file, so the
#  picture matches exactly what would be submitted.
# ===========================================================================

def averageBuildingDistance(sampleCap=None):
    """
    Nearest-neighbor distance statistics for the dataset: for every
    building, the distance from its centroid to the closest OTHER
    building's centroid, found via expanding SpatialGrid ring queries (no
    O(n^2) all-pairs scan). Prints and returns (mean, median, min, max).

    sampleCap: optionally restrict to a random sample of this many
    buildings for a quick estimate on huge datasets.
    """
    import statistics, math, random as _random
    polygons, ids = loadGeoJSON(GEOJSON_PATH)
    grid = SpatialGrid(polygons, cell=80.0)
    cents = grid.centroids
    idxs = list(range(len(polygons)))
    if sampleCap and len(idxs) > sampleCap:
        idxs = _random.Random(0).sample(idxs, sampleCap)
    dists = []
    for i in idxs:
        cx, cy = cents[i]
        r = 80.0
        best = None
        while best is None:
            for j in grid.neighborIndices(cx, cy, r):
                if j == i:
                    continue
                d = math.hypot(cents[j][0] - cx, cents[j][1] - cy)
                if best is None or d < best:
                    best = d
            # a neighbor found within r could still be beaten by one just
            # outside the queried ring; require best <= r before trusting it
            if best is not None and best <= r:
                break
            best = None if best is None or best > r else best
            r *= 2
        dists.append(best)
    mean = sum(dists) / len(dists)
    med = statistics.median(dists)
    print(f"nearest-neighbor centroid distance over {len(dists)} buildings: "
          f"mean {mean:.1f} m, median {med:.1f} m, "
          f"min {min(dists):.1f} m, max {max(dists):.1f} m")
    return mean, med, min(dists), max(dists)


def plotDataset(outPath=os.path.join(REPO_ROOT, "dataset.png")):
    """
    Plot the raw dataset (every building footprint) with a scale bar.
    Saves to outPath and returns it.
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    polygons, ids = loadGeoJSON(GEOJSON_PATH)
    allX = [v[0] for p in polygons for v in p]
    allY = [v[1] for p in polygons for v in p]
    pad = 60
    xlim = (min(allX) - pad, max(allX) + pad)
    ylim = (min(allY) - pad, max(allY) + pad)

    fig, ax = plt.subplots(figsize=(11, 11 * (ylim[1]-ylim[0]) / (xlim[1]-xlim[0])))
    ax.add_collection(PolyCollection(polygons, facecolor='#D7DCE1',
                                     edgecolor='#6B7480', linewidth=0.25))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{len(polygons)} buildings  "
                 f"({xlim[1]-xlim[0]-2*pad:.0f} m x {ylim[1]-ylim[0]-2*pad:.0f} m extent)",
                 fontsize=12, pad=10)

    # scale bar: nice round length ~1/5 of the width (100/200/500/1000 m)
    span = xlim[1] - xlim[0]
    target = span / 5
    barLen = min((100, 200, 500, 1000, 2000), key=lambda L: abs(L - target))
    bx0 = xlim[0] + span * 0.05
    by = ylim[0] + (ylim[1] - ylim[0]) * 0.04
    ax.plot([bx0, bx0 + barLen], [by, by], color='black', linewidth=3,
            solid_capstyle='butt')
    for x in (bx0, bx0 + barLen):
        ax.plot([x, x], [by - span*0.004, by + span*0.004], color='black', linewidth=1.5)
    ax.text(bx0 + barLen / 2, by + span * 0.008, f"{barLen} m",
            ha='center', va='bottom', fontsize=11)

    plt.tight_layout()
    plt.savefig(outPath, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"wrote {outPath}")
    return outPath


def plot(usePath=None):
    import matplotlib.pyplot as plt
    plotStart = time.perf_counter()
    if usePath is None:
        usePath = latestSubmissionPath()
        if usePath is None:
            raise FileNotFoundError(f"no submission_N.txt files in {SUBMISSION_DIR}")
    polygons, ids = loadGeoJSON(GEOJSON_PATH)
    subs = parseSubmission(usePath)
 
    allX = [v[0] for p in polygons for v in p]
    allY = [v[1] for p in polygons for v in p]
    pad = 50
    xlim = (min(allX) - pad, max(allX) + pad)
    ylim = (min(allY) - pad, max(allY) + pad)
 
    n = len(subs)
    cols = 3 if n >= 3 else n
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6.6 * rows))
    import numpy as np
    axes = np.atleast_1d(axes).flatten()
 
    for ax_i, sub in enumerate(subs):
        ax = axes[ax_i]
        tau, k, antennas = sub["tau"], sub["k"], sub["antennas"]
        servedSet = set(sub["claimedIds"])   # verified IDs straight from the file
 
        # one PolyCollection per class instead of one patch per polygon --
        # add_patch x 13K polygons x 9 subplots is a minutes-long
        # matplotlib pattern; collections render the same thing in seconds
        from matplotlib.collections import PolyCollection
        servedPolys = [polygons[i] for i in range(len(polygons)) if ids[i] in servedSet]
        otherPolys = [polygons[i] for i in range(len(polygons)) if ids[i] not in servedSet]
        if otherPolys:
            ax.add_collection(PolyCollection(otherPolys, facecolor='#F0F0F0',
                                             edgecolor='#BBBBBB', linewidth=0.2))
        if servedPolys:
            ax.add_collection(PolyCollection(servedPolys, facecolor='#C8E6C9',
                                             edgecolor='#1D9E75', linewidth=0.2))
 
        ax.scatter([a[0] for a in antennas], [a[1] for a in antennas],
                   c='red', s=4, zorder=5, edgecolors='white', linewidths=0.3)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect('equal')
        ax.set_title(f'\u03c4={tau},  k={k}\n{len(servedSet)} / {len(polygons)} served',
                     fontsize=11, pad=10)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
 
    for j in range(n, len(axes)):
        axes[j].axis('off')
 
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25, wspace=0.1, top=0.93)
    print(f"[plot() total] took {time.perf_counter() - plotStart:.2f}s")
    plt.show()

# ===========================================================================
#  Entry point — the __main__ guard is REQUIRED for multiprocessing.
# ===========================================================================

if __name__ == "__main__":
    mainStart = time.perf_counter()

    print(f"Start: {datetime.now()}")

    solve()
    print(f"Solved: {datetime.now()}")

    print(f"[__main__ total] took {time.perf_counter() - mainStart:.2f}s")