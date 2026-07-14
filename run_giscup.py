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
    greedyCornersCached, loadGeoJSON, loadOrBuildViewsheds, servicedIdsFromViewCache, writeSubmission, parseSubmission,
    localEvalCoverParallel, refineWithSwaps,
    normalizePolygon, pointOnEdge, computeConvexity,
    plt, MplPolygon,
)
from centrality_giscup import buildVisibilityGraphRadiusBVH, topKVantageVertices
from datetime import datetime
import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------



TAUS   = [0.25, 0.5, 0.75]
KS     = [50, 500, 1000]
 
SOLVE_RADIUS      = 160.0
SOLVE_GRIDCELL    = SOLVE_RADIUS
VALIDATION_RADIUS = SOLVE_RADIUS * 2   # wider = stricter re-check

# VANTAGE_RADIUS bounds vantage-seed-picking cost by LOCAL density instead
# of total dataset size (see buildVisibilityGraphRadiusBVH's docstring for
# the full reasoning -- an unrestricted V^2 all-pairs graph is a hard wall
# at real dataset scale, confirmed directly: ~1.8h extrapolated at ~48,000
# candidates). Every candidate within this radius gets tested -- nothing
# silently capped/truncated -- so this is the ONLY thing controlling how
# large a neighborhood each vertex searches. Tune to your actual dataset's
# density; too large re-introduces cost, too small may miss useful
# starting-vertex candidates in sparse areas.
VANTAGE_RADIUS    = 320.0

# Fraction (0.0-1.0) of EACH SUBCASE's own k to use as forced vantage-seed
# starting antennas -- e.g. 0.05 means a k=50 subcase gets 2 seeds and a
# k=1000 subcase gets 50, rather than every subcase getting the SAME fixed
# absolute count regardless of its own budget size. Defaults to 0.0 (OFF,
# plain greedy) -- NOT because vantage seeds don't work, but because they
# optimize a DIFFERENT objective than greedyCornersCached does.
# topKVantageVertices ranks corners by visibility-graph DEGREE (how many
# other corners it can see in a straight line, a pure geometric measure);
# greedyCornersCached optimizes marginal BUILDING-COVERAGE gain at
# threshold tau. There's no guarantee those agree, and when they don't,
# seeding actively HURTS: the greedy algorithm's strength comes from
# always picking the true best marginal-gain option at every step, and
# forcing the first few placements to be something else spends part of
# the fixed budget k on picks that weren't chosen for the real objective.
#
# Confirmed directly on the courtyard test scene: an absolute fixed count
# of 4-5 seeds caused real, reproducible coverage LOSSES (5 buildings
# served with plain greedy, down to 1-4 with those seeds forced) -- but a
# SMALL fraction of a subcase's own k (1-3 seeds out of a k=6 budget, and
# ALWAYS just the top-1 ranked seed across 36 tau/k/tiebreak combinations)
# matched plain greedy exactly, every time, with zero regressions. That's
# why this is percentage-of-k rather than one fixed absolute count for
# every subcase: a large absolute count can dominate a SMALL k subcase's
# entire budget even when it would be a harmless small fraction of a
# larger one. Still: matching plain greedy is the BEST result seen in any
# test so far, never an improvement over it -- there is no evidence yet
# that any amount of vantage seeding helps, only evidence about how much
# is safe. Only raise this above 0 if you've A/B tested it against
# VANTAGE_SEED_PERCENT=0 on YOUR actual data and confirmed it helps there.
#
# PERFORMANCE COST when > 0: different k's need different SEED COUNTS
# (percent * k), and inserting a different number of forced seeds changes
# greedyCornersCached's entire downstream trajectory from that point on --
# not just a suffix of it. So unlike the fixed-seed-list version, subcases
# can no longer share one computed-once-at-kMax run sliced per k; each
# (tau, k) pair needs its own separate greedy call and its own fresh
# (non-incremental) coverage computation. solve() below only pays this
# cost when VANTAGE_SEED_PERCENT > 0 -- at 0 it uses the original fast
# compute-once-per-tau path, identical to having no seeding code at all.
VANTAGE_SEED_PERCENT = 0.0

# STRtree leaf capacity (the BVH inside buildVisibilityGraphRadiusBVH).
# Shapely's own default is 10; 10-100 tested similarly at 12K buildings in
# earlier benchmarking, so this isn't especially sensitive -- not worth
# hand-tuning without a specific reason to believe your data behaves
# differently (see buildVisibilityGraphRadiusBVH's docstring).
VANTAGE_NODE_CAPACITY = 25

VANTAGE_PROCESSES = None   # None -> os.cpu_count()

# Post-process each subcase's greedy placement with a local-search swap
# refinement (see lib_giscup.refineWithSwaps): repeatedly try removing one
# placed antenna and adding one unplaced candidate, keep any swap that
# increases served-building count, stop at a local optimum or a budget.
# Off by default -- unlike vantage seeding, this has NEVER been observed to
# hurt (every swap it applies is individually verified to strictly
# increase served count before being applied), but the TIME cost is real:
# measured directly at k=200 on a 500-building scene, cost scales roughly
# LINEARLY at ~2.2s PER ITERATION now (every iteration does a full search,
# see REFINE_MAX_ITERATIONS below) -- 10 iterations took ~22s there. Also,
# REFINE_WITH_SWAPS=True forces fresh (non-incremental) coverage
# computation for every subcase instead of the fast
# compute-once-at-kMax-then-slice path, same tradeoff as
# VANTAGE_SEED_PERCENT > 0. Confirmed on the courtyard test scene: applied
# on top of an already-good plain-greedy solution it correctly finds zero
# improving swaps (true no-op, negligible cost); applied on top of a
# DELIBERATELY DEGRADED solution (bad vantage seeding) it fully recovered
# the lost coverage. Time your own real-scale case (k up to 1000, thousands
# of buildings) before enabling this in a real run -- don't assume the
# small-scene numbers above transfer directly.
REFINE_WITH_SWAPS = False

# Hard cap on swaps attempted per subcase -- THE (now only) lever
# controlling refinement's total cost. Every placed antenna is tried as a
# removal candidate every iteration (no partial-candidate cap -- see
# refineWithSwaps' docstring for why a full search each time is more
# CONSISTENT than ranking-and-truncating removal candidates by a proxy
# heuristic, which could silently exclude a genuinely great swap). Because
# every iteration does the same full amount of work, cost scales roughly
# LINEARLY with this number (measured: ~2.2s/iteration on a 500-building/
# k=200 test) -- there's no natural early-iteration discount the way a
# removal-candidate cap used to provide. 10 is a starting point, not a
# calibrated answer for your data; tune directly against your own
# real-scale timing.
REFINE_MAX_ITERATIONS = 10

# No separate improvement-threshold config here on purpose. An earlier
# version had a GRADIENT-based early stop (a threshold relative to the
# starting score that relaxed as coverage approached its ceiling), but
# confirmed directly: swap improvements are almost always exactly +1
# building (a real 6-swap run gained +1 every single time), so any such
# threshold behaves as an ALL-OR-NOTHING CLIFF rather than a useful dial --
# it either blocks every swap or none of them, since there's rarely more
# than one discrete step size to distinguish between. The only stopping
# condition that means anything for this problem is the natural one
# already built into refineWithSwaps: keep going as long as ANY improving
# swap exists (bestDelta > 0), bounded by REFINE_MAX_ITERATIONS for cost.

GEOJSON_PATH      = "GIS-cup-sample-dataset.geojson"
SUBMISSION_PATH   = f"unverified/submission_r{int(SOLVE_RADIUS)}_R{int(VALIDATION_RADIUS)}_V{int(VANTAGE_RADIUS)}.txt"

# ===========================================================================
#  SHARED VALIDATION HELPER  (used by validate() and solve() so both
#  independently re-check claims the exact same way — one definition of
#  "verified", not two copies that could drift apart)
# ===========================================================================

def _buildBoundaryChecker(polygons):
    """Returns onAnyBoundary(pt) -> bool, closed over this dataset's edges."""
    allBoundaryEdges = [(ea, eb) for poly in polygons
                        for ea, eb in normalizePolygon(poly)]
    def onAnyBoundary(pt):
        return any(pointOnEdge(pt[0], pt[1], ea, eb) for ea, eb in allBoundaryEdges)
    return onAnyBoundary


def _validateSubproblem(tau, k, antennas, claimedIds, polygons, ids, idToIndex,
                        onAnyBoundary, radius, gridCell=None, verbose=True):
    """
    Independently re-check ONE (tau, k) sub-problem's claims at `radius`
    (normally VALIDATION_RADIUS — wider than whatever radius produced
    `claimedIds`, so this never just re-confirms the same computation).
    Checks: antenna count matches k, every antenna sits on a building
    boundary, and every claimed building's re-swept coverage actually
    reaches tau. Uses localEvalCoverParallel (independent sweep, CPU-core
    parallel) rather than anything from the solve-side viewCache.

    Returns (verifiedResult, ok):
      verifiedResult : {"tau", "k", "antennas", "servicedIds"} using the
                        VERIFIED ids — safe to feed straight into
                        writeSubmission.
      ok              : False if any problem was found (antenna count,
                        off-boundary, or false claims).
    """
    if gridCell is None:
        gridCell = radius
    problems = []

    if len(antennas) != k:
        problems.append(f"antenna count {len(antennas)} != k={k}")

    offBoundary = sum(1 for a in antennas if not onAnyBoundary(a))
    if offBoundary:
        problems.append(f"{offBoundary} antennas off-boundary")

    # parallel coverage re-check (one sweep per antenna, across CPU cores)
    covDict = localEvalCoverParallel(antennas, polygons, radius=radius, gridCell=gridCell)

    falseClaims = []
    for bid in claimedIds:
        idx = idToIndex.get(bid)
        if idx is None:
            falseClaims.append((bid, "unknown id"))
        elif covDict.get(idx, 0.0) < tau:
            falseClaims.append((bid, f"{covDict.get(idx,0.0):.4f} < {tau}"))
    if falseClaims:
        problems.append(f"{len(falseClaims)} false claims")

    verifiedIds = [ids[i] for i in range(len(polygons)) if covDict.get(i, 0.0) >= tau]

    status = "OK" if not problems else "FAIL"
    if verbose:
        print(f"    verify (tau={tau}, k={k}) @ radius {radius}: {len(antennas)} antennas, "
              f"{len(claimedIds)} claimed / {len(verifiedIds)} verified -> {status}, "
              f"diff = {len(verifiedIds) - len(claimedIds)}")
        for p in problems:
            print(f"        - {p}")
        for bid, reason in falseClaims[:5]:
            print(f"          false: building {bid} ({reason})")

    verifiedResult = {"tau": tau, "k": k, "antennas": antennas, "servicedIds": verifiedIds}
    return verifiedResult, (not problems)

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
    from lib_giscup import candidateList
    candidates = candidateList(polygons)
    adjacency = buildVisibilityGraphRadiusBVH(polygons, candidates, radius=radius,
                                              node_capacity=nodeCapacity,
                                              processes=processes, verbose=verbose)
    top = topKVantageVertices(adjacency, k=n, convexity=convexity)
    if verbose:
        print(f"Vantage seeds ({len(top)}/{n} requested):")
        for v, deg in top:
            print(f"    {v}  (visibility-graph degree at selection: {deg})")
    return [v for v, _deg in top]

# ===========================================================================
#  SECTION 1 — SOLVE  (slow: the full precompute + nine sub-problems)
# ===========================================================================

def _timed(label, fn, *args, **kwargs):
    """Run fn(*args, **kwargs), print how long it took, return its result."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    print(f"[{label}] took {time.perf_counter() - t0:.2f}s")
    return result

def solve(seed=0):
    solveStart = time.perf_counter()

    polygons, ids = _timed("loadGeoJSON", loadGeoJSON, GEOJSON_PATH)
    idToIndex = {bid: i for i, bid in enumerate(ids)}
    onAnyBoundary = _timed("_buildBoundaryChecker", _buildBoundaryChecker, polygons)

    # build once or load from disk
    candidates, viewCache, perim = _timed(
        "loadOrBuildViewsheds", loadOrBuildViewsheds,
        polygons, radius=SOLVE_RADIUS, gridCell=SOLVE_GRIDCELL,
        path=f"cache/viewcache_r{int(SOLVE_RADIUS)}.pkl")

    results = []
    allOk = True
    kMax = max(KS)

    # Corner convexity: a cheap, purely geometric tiebreak (prefer sharper
    # outward corners over near-straight or reflex ones) applied ONLY when
    # candidates are otherwise tied on the real (gain, score) objective.
    # Computed once here and shared across every tau below.
    convexity = _timed("computeConvexity", computeConvexity, polygons)

    # Vantage seed POOL: the top kMax vantage-ranked vertices, computed ONCE
    # (tau-independent) -- each subcase below draws its own PERCENTAGE-sized
    # prefix of this same pool. Skipped entirely (empty pool, no visibility
    # graph built) when VANTAGE_SEED_PERCENT is 0 -- no point paying for a
    # pool nothing will draw from. See VANTAGE_SEED_PERCENT's config comment
    # for why percentage-of-k rather than one fixed count for every subcase.
    if VANTAGE_SEED_PERCENT > 0:
        vantageSeedsPool = _timed(
            "_pickVantageSeeds", _pickVantageSeeds,
            polygons, convexity, n=kMax, radius=VANTAGE_RADIUS,
            nodeCapacity=VANTAGE_NODE_CAPACITY, processes=VANTAGE_PROCESSES)
    else:
        vantageSeedsPool = []

    if VANTAGE_SEED_PERCENT <= 0:
        # FAST PATH -- identical to having no vantage-seed code at all.
        # Reuse across sub-problems: greedyCornersCached's antenna ORDER for
        # a fixed tau doesn't depend on the target k (every step's choice
        # only depends on what's already placed, not on how many more will
        # follow), so for each tau, run the search ONCE at the largest k,
        # then slice that ordered list for every smaller k -- cuts the
        # number of greedy calls from len(TAUS)*len(KS) down to len(TAUS).
        #
        # servicedIds are computed incrementally too: servicedIdsFromViewCache
        # reads coverage directly from the already-built viewCache (same
        # radius, same candidate set) instead of re-running a fresh grid
        # query + sweep from scratch, and only processes the NEW antennas
        # since the previous k (carrying the coverage state forward) --
        # measured ~250x faster than recomputing fresh per k on a test subset.
        for tau in TAUS:
            tauStart = time.perf_counter()
            start_time = time.perf_counter()
            placedFull, servedFull = greedyCornersCached(
                polygons, kMax, tau, candidates, viewCache,
                seed=seed, convexity=convexity, seedVertices=None)
            end_time = time.perf_counter()
            print(f"Full compute of tau={tau}, kMax={kMax}: served {servedFull} (took {end_time - start_time:.2f}s)")

            coverageState = None
            prevK = 0
            for k in sorted(KS):
                start_time = time.perf_counter()

                if REFINE_WITH_SWAPS:
                    # Refinement changes WHICH antennas are placed (not just
                    # appends), so the incremental "newAntennas since prevK"
                    # trick no longer applies for this k -- compute claimedIds
                    # fresh from the refined set instead. coverageState is
                    # deliberately NOT carried forward across k here for the
                    # same reason (each k's refined set isn't a superset of
                    # the previous k's).
                    antennasK, refinedServed, nSwaps = refineWithSwaps(
                        polygons, placedFull[:k], tau, candidates, viewCache,
                        maxIterations=REFINE_MAX_ITERATIONS,
                        verbose=False)
                    claimedIds, _state = servicedIdsFromViewCache(
                        antennasK, viewCache, perim, ids, tau, state=None)
                    swapNote = f", {nSwaps} swap(s) -> served {refinedServed}"
                    label = "fresh + refine + verify"
                else:
                    newAntennas = placedFull[prevK:k]
                    claimedIds, coverageState = servicedIdsFromViewCache(
                        newAntennas, viewCache, perim, ids, tau, state=coverageState)
                    antennasK = placedFull[:k]
                    swapNote = ""
                    label = "incremental + verify"

                verifiedResult, ok = _validateSubproblem(
                    tau, k, antennasK, claimedIds, polygons, ids, idToIndex,
                    onAnyBoundary, radius=VALIDATION_RADIUS)
                allOk = allOk and ok
                results.append(verifiedResult)

                end_time = time.perf_counter()
                print(f"tau={tau}, k={k}: {len(claimedIds)} claimed / "
                      f"{len(verifiedResult['servicedIds'])} verified{swapNote} "
                      f"({label}, took {end_time - start_time:.2f}s)")
                prevK = k

            print(f"[tau={tau} total] took {time.perf_counter() - tauStart:.2f}s")

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
        for tau in TAUS:
            tauStart = time.perf_counter()
            for k in sorted(KS):
                start_time = time.perf_counter()
                nSeeds = int(VANTAGE_SEED_PERCENT * k)
                seedsForThisK = vantageSeedsPool[:nSeeds]

                placed, served = greedyCornersCached(
                    polygons, k, tau, candidates, viewCache,
                    seed=seed, convexity=convexity, seedVertices=seedsForThisK)

                swapNote = ""
                if REFINE_WITH_SWAPS:
                    placed, refinedServed, nSwaps = refineWithSwaps(
                        polygons, placed, tau, candidates, viewCache,
                        maxIterations=REFINE_MAX_ITERATIONS,
                        verbose=False)
                    swapNote = f", {nSwaps} swap(s) -> served {refinedServed}"

                claimedIds, _state = servicedIdsFromViewCache(
                    placed, viewCache, perim, ids, tau, state=None)

                verifiedResult, ok = _validateSubproblem(
                    tau, k, placed, claimedIds, polygons, ids, idToIndex,
                    onAnyBoundary, radius=VALIDATION_RADIUS)
                allOk = allOk and ok
                results.append(verifiedResult)

                end_time = time.perf_counter()
                print(f"tau={tau}, k={k}: {nSeeds} seed(s) ({VANTAGE_SEED_PERCENT*100:.0f}% of k), "
                      f"served {served}{swapNote}, {len(claimedIds)} claimed / "
                      f"{len(verifiedResult['servicedIds'])} verified "
                      f"(fresh greedy + verify, took {end_time - start_time:.2f}s)")

            print(f"[tau={tau} total] took {time.perf_counter() - tauStart:.2f}s")

    writeSubmission(SUBMISSION_PATH, results)
    print(f"\n{'All subcases verified clean.' if allOk else 'Some subcases had problems -- see FAIL lines above.'}")
    print(f"Wrote {SUBMISSION_PATH} (verified at radius {VALIDATION_RADIUS} per subcase).")
    print(f"[solve() total] took {time.perf_counter() - solveStart:.2f}s")
 
# ===========================================================================
#  SECTION 2 — PLOT  (reads the VERIFIED file; independent of solving)
#  Colors each building by the verified serviced IDs from the file, so the
#  picture matches exactly what would be submitted.
# ===========================================================================

def plot(usePath=None):
    plotStart = time.perf_counter()
    if usePath is None:
        usePath = SUBMISSION_PATH
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
 
        for i, poly in enumerate(polygons):
            served = ids[i] in servedSet
            ax.add_patch(MplPolygon(poly, closed=True,
                facecolor=('#C8E6C9' if served else '#F0F0F0'),
                edgecolor=('#1D9E75' if served else '#BBBBBB'),
                linewidth=0.2))
 
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

    # plot(SUBMISSION_PATH)
    # print(f"Plotted: {datetime.now()}")

    print(f"[__main__ total] took {time.perf_counter() - mainStart:.2f}s")