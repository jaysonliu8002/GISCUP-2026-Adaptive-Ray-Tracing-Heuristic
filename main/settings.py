"""
settings.py — every adjustable option for the GIS Cup 2026 solver.

run_giscup.py is now purely a driver: it imports the names below and never
defines a tunable of its own. Change behaviour HERE, not there.

Every switch is documented in README.md, including the measurements behind
each default. Keep the comments in this file terse; put the prose there.
"""

import os

# Data files (dataset, cache/, submission/) live at the repo root, one level
# above this script's folder, so anchor paths there instead of relying on the
# current working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
#  PROCESSING UNITS  (see compute.py, which resolves and validates this)
# ---------------------------------------------------------------------------

# How many CPU worker processes every parallel stage uses.
#   None : every core on the machine.
#   int  : that many, always CAPPED at the number of cores that actually
#          exist, and floored at 1 -- asking for 64 on a 12-core machine
#          gets you 12, not an oversubscribed pool.
#
# This is the whole parallelism story: everything that can run in parallel
# does, on CPU cores, and there is no serial mode to fall back to. See
# README ("Why there is no GPU backend") before reaching for one -- the
# stages that dominate the wall clock are pure-python geometry that a GPU
# cannot run without rewriting the algorithm.
#
# solve() prints the resolved count before it does any work.
CPU_PROCESSES = None

# -- problem ----------------------------------------------------------------
TAUS = [0.25, 0.5, 0.75]
KS   = [100, 1000, 2500]
GEOJSON_PATH = os.path.join(REPO_ROOT, "giscup-dataset-practice.geojson")
# Just change this and re-run -- no cleanup needed. Each dataset gets its
# own subdirectory of CACHE_DIR and SUBMISSION_DIR (named <file>_<short
# fingerprint>), so datasets never share a cache or a submission series and
# switching back and forth reuses whatever was already built for each.
# See README ("Switching datasets").

# -- radii / viewshed reach --------------------------------------------------
SOLVE_RADIUS       = 80.0
ADAPTIVE_DOUBLINGS = 7                     # corridor reach = SOLVE_RADIUS * 2**n; 0/None = off
SOLVE_GRIDCELL     = SOLVE_RADIUS
VALIDATION_STEP    = 1.25                  # validation re-sweep reach = claim reach * this;
                                           # > 1 so the re-sweep can both re-check claims
                                           # independently AND harvest extra true ones

# -- main strategy ------------------------------------------------------------
LOOKAHEAD_GREEDY       = True             # chess-style placement search (see README)
LOOKAHEAD_BEAM         = 10                 # candidates given the deep evaluation per step
LOOKAHEAD_DEPTH_BY_TAU = {TAUS[0]: 1, TAUS[1]: 2, TAUS[2]: 3}   # tau -> ply gradient; None = fixed
LOOKAHEAD_DEPTH        = 2                 # used when the gradient is None

# -- refinement ---------------------------------------------------------------
REFINE_WITH_SWAPS     = True
REFINE_MAX_ITERATIONS = 99999               # cap on COMMITTED swaps
REFINE_TIME_BUDGET    = 3600                # seconds per subcase; None = unlimited
REFINE_VERBOSE        = True
                                           # the swap scan's worker count is NOT a
                                           # setting of its own any more -- it comes
                                           # from CPU_PROCESSES like every other
                                           # parallel stage, and the scan no longer
                                           # has a serial mode.

# -- crash safety --------------------------------------------------------------
RESUME_PARTIAL_SUBMISSION = True           # a partially-written submission_N.txt
                                           # (crash mid-solve) is continued, not
                                           # restarted; complete files stay
                                           # protected as before

# -- validation ---------------------------------------------------------------
SKIP_REDUNDANT_VALIDATION = True
CLAIM_MARGIN = 1e-6      # search, claim, and validate at tau + this, keeping
                         # every claim clear of the knife edge (see README)

# -- vantage seeding (centrality_giscup.py) ------------------------------------
VANTAGE_SEED_PERCENT  = 0.01               # 0 = off (enables the faster slicing path)
VANTAGE_RADIUS        = 10240.0
VANTAGE_NODE_CAPACITY = 25
                                           # worker count: see the REFINE note above --
                                           # CPU_PROCESSES.

# -- output --------------------------------------------------------------------
# Roots. Runs read and write a per-dataset subdirectory of each, keyed on
# the geometry in GEOJSON_PATH -- see lib_giscup.datasetDir.
SUBMISSION_DIR = os.path.join(REPO_ROOT, "submission")
CACHE_DIR      = os.path.join(REPO_ROOT, "cache")
