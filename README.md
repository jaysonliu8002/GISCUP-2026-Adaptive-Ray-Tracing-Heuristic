# GIS Cup 2026 solver — configuration reference

**IMPORTANT**

Program is currently in experimental phase.
Use the "\_test" files instead of the "\_giscup"

---

All switches live at the top of `run_giscup.py`. This file holds the prose:
what each one does, the measurements behind its default, and how the pieces
interact. Crop benchmarks below refer to two cutouts of the sample dataset:
**crop1** (605 buildings / 2,945 candidates) and **crop2** (1,855 / 10,168).
The full sample dataset is 12,860 buildings / 78,727 candidates.

Relavant switches to change:

| switch                  |
| ----------------------- |
| `SOLVE_RADIUS`          |
| `ADAPTIVE_DOUBLINGS`    |
| `VALIDATION_STEP`       |
| `LOOKAHEAD_BEAM`        |
| `LOOKAHEAD_DEPTH`       |
| `REFINE_MAX_ITERATIONS` |
| `REFINE_TIME_BUDGET`    |
| `VANTAGE_SEED_PERCENT`  |
| `VANTAGE_RADIUS`        |

---

## Problem

| switch            | meaning                                   |
| ----------------- | ----------------------------------------- |
| `TAUS`, `KS`      | the 9 subcases (τ × k) the contest scores |
| `GEOJSON_PATH`    | input dataset                             |
| `SUBMISSION_PATH` | output file                               |

Visit [GISCUP](https://sigspatial2026.sigspatial.org/giscup.html) for more information

---

## Radii / viewshed reach

**`SOLVE_RADIUS`** — the radius the per-candidate viewshed cache is
built at; everything the search "sees" comes from this cache. Measured on
crop1 (k=60): solving _at_ 160 beat solving at 80 + committing at 160
(dual-radius) by +32 served at τ=0.5 and +23 at τ=0.75 — the greedy chooses
better when it can actually see the full commit radius. Marginal visibility
falls off fast past 160 on this data (+18% going to 240, +9% to 320, at
~4×/~8× precompute cost). Precompute at 160 with the vectorized sweep:
~36 min single-core on the full dataset, divided by your process count.

**`ADAPTIVE_DOUBLINGS`** — long-street handling. After the base sweep,
any OPEN angular sector (nothing blocks within the radius — the candidate is
looking down a street or across a plaza) is re-swept against a
doubled-radius scene restricted to those corridor directions, repeating this
many times. `2` with `SOLVE_RADIUS=160` follows corridors 160 → 320 → 640,
so buildings far down a long street are detected and credited without paying
big-radius cost everywhere: enclosed candidates pay one base sweep, street
candidates pay only for the corridor's angular share (79% of candidates on
this data have ≥1 open corridor at 160 m, but the median corridor is ~5% of
the circle). Measured full-dataset costs: plain 160 ≈ 27.7 ms/candidate,
one doubling ≈ 36.5, two ≈ 58.6 — i.e. ~3/4/6.4 min on 12 processes for the
whole precompute. Matches a flat 640 m sweep (~2,360 ms/cand) to under a
centimeter of perimeter. Set `0`/`None` to disable. Cache files get a
`_d{ADAPTIVE_DOUBLINGS}` suffix so different settings coexist; the on-load
validity check keys on dataset + radius + corridor reach (`maxRadius` is
stored in the cache metadata), so a cache built with different doubling
settings — or one predating the metadata — is refused and rebuilt
automatically instead of silently loading wrong viewsheds.

Sweeps never credit a hit BEYOND the radius of the scene they ran against
(including the final adaptive pass): the local scene only contains
buildings within that radius, so a farther hit could be missing its true
occluder — crediting it risked a false claim the validation re-sweep
(same scene construction) could not catch. Beyond-reach visibility is
simply not claimed; the corridor doublings are the mechanism for
capturing it correctly.

**`VALIDATION_STEP`** — the validation re-sweep is the **same** adaptive
corridor method the solve cache was built with (same base radius, same
corridor doublings), just allowed to follow still-open corridors one extra
step, `VALIDATION_STEP ×` past the claim reach (e.g. 1.25: 640 → 800).
That extra reach is what makes the re-sweep worth running at all — two
properties, both consequences of the sweep never crediting a hit beyond
its scene radius:

- **It can only add, never subtract.** Any occluder of a point at distance
  d lies within d of the observer, so every claim within the claim reach is
  exact by construction; a wider sweep with the same rules re-derives all
  of them and can only _extend_ the set using the perimeter it newly sees
  past the claim reach. A same-reach re-sweep, by contrast, could only
  re-confirm (verified in testing: it reproduces the claims exactly).
- **The harvest is free score.** The contest scores visibility with **no**
  distance limit, so buildings the extra reach tips over τ are true claims,
  exactly as trustworthy as the re-checked originals — they come from the
  same sweep — and they get written to the submission.

Because validation runs on an independent code path
(`localEvalCoverParallel`, not the solve-side cache), it also cross-checks
for implementation bugs on either side. Cost is small: it sweeps only the
k chosen antennas, and street candidates pay only their corridors' angular
share for the extra step. Raise the step to harvest more of the long tail
if time allows; set it ≤ 1 to skip the re-sweep entirely (see
`SKIP_REDUNDANT_VALIDATION`).

(The old `USE_DUAL_RADIUS`/`COMMIT_RADIUS` mode — search a narrow cache,
re-solve each placed antenna wide at commit time — is deleted: the adaptive
corridor cache lets the search see the full commit reach directly, which
measured better than dual-radius on crop1 and left it with no role.)

`SOLVE_GRIDCELL` — spatial-grid cell size for scene lookups; the default
(= SOLVE_RADIUS) is fine.

---

## Main strategy

**`LOOKAHEAD_GREEDY`** — switches the placement engine from plain
greedy to `greedyLookahead`, a chess-engine-style search. Plain greedy
places whichever candidate tips the most buildings _right now_; the
lookahead scores each candidate as

```
totalScore(c) = gain(c) + best gain chain of the next (depth − 1) placements,
                each evaluated exactly against the hypothetical state
```

and places the argmax. This sees **pair synergies** plain greedy cannot: a
candidate with zero immediate gain whose partial coverage sets up a partner
to jointly tip several buildings wins the comparison — exactly the situation
at high τ, where single antennas rarely tip anything alone.

**`LOOKAHEAD_BEAM`** — how many candidates receive the expensive deep
evaluation each step (chess: move ordering — only the most promising moves
get searched). Candidates are ranked by (gain, convexity, coverage-sum
score); the coverage-sum tiebreak is what lets zero-gain-but-high-coverage
candidates into the beam at high τ. Bigger beam = fewer missed setups,
linearly more cost. Runtime is roughly `(beam + 1) ×` plain greedy at
depth 2 (measured ~2–2.3× in practice, since overlay evaluation
short-circuits heavily).

**`LOOKAHEAD_DEPTH_BY_TAU = {tau1: depth1, tau2: depth2, tau3: depth3}`** — the τ→depth
gradient: an incoming τ uses the depth of its _nearest_ key, so low τ gets
shallow search and high τ gets deep search. Rationale, all measured on the
crops at the cache in use:

| τ    | lookahead vs plain greedy                                      | conclusion                                                                                                          |
| ---- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------- |
| 0.25 | ±1 served                                                      | greedy is already near-optimal when everything tips easily → **depth 1** (delegates to plain greedy, zero overhead) |
| 0.5  | +5 (crop1), +20 (crop2)                                        | pair synergies matter → **depth 2**                                                                                 |
| 0.75 | +23 (crop1), +41 (crop2) at depth 2; depth 3 adds a further +2 | triples exist → **depth 3**                                                                                         |

Set it to `None` to use the fixed **`LOOKAHEAD_DEPTH`** everywhere.

Depth-3 caveat (why the inner beam matters): each extra ply multiplies cost
by the inner beam width, and a _truncated_ inner beam makes the deep
estimate noisy — measured, depth 3 with a half-width inner beam scored
_below_ depth 2 (283 vs 286 on crop1 τ=0.75); with a full-width inner beam
it scored above (288). `greedyLookahead` therefore auto-uses a full-width
inner beam whenever depth ≥ 3. Cost: depth 3 ran ~6.4× depth 2 on crop1.
The lookahead composes with refinement: crop1 τ=0.75 end-to-end went 272
(plain pipeline) → 291 (lookahead + refine), zero false claims.

---

## Refinement

**`REFINE_WITH_SWAPS`** — after greedy/lookahead, repeatedly find the
best (remove one placed antenna, add one unplaced candidate) swap over the
**complete** r × a space, verify its realized delta, and commit only if
strictly positive. Runs on an incremental engine (persistent baselines, tip
sets, pair caches, invalidated only in the swapped neighborhood) with a
`selfCheck=True` debug mode that recomputes all state from scratch every
pass and asserts equality — use it after editing the engine. Every stop
prints _why_ it stopped: local optimum, swap cap, or time budget.

**`REFINE_MAX_ITERATIONS`** — cap on committed swaps. The incremental
engine made per-swap cost scale with the changed neighborhood rather than a
full k × candidates rescan (measured 2.5–4.8× faster on crop1, more at
scale), so a high cap is affordable.

**`REFINE_TIME_BUDGET`** — hard wall-clock seconds per subcase
(`None` = unlimited). Swaps are committed as they're verified, so stopping
on budget keeps everything found so far; worst-case total refine time is
`9 × budget` regardless of τ/k regime.

`REFINE_VERBOSE` — per-swap log lines (predicted vs realized delta, running
total, timings). The per-subcase summary prints either way.

---

## Validation

**`SKIP_REDUNDANT_VALIDATION`** — skip the independent coverage re-sweep
when it would reach no further than the claims themselves
(`VALIDATION_STEP ≤ 1`), i.e. when it could neither falsify nor extend
anything and would only re-confirm. With `VALIDATION_STEP > 1` the re-sweep
always reaches past the claim reach, so it runs every subcase and the
per-subcase log says "verified OK at reach N" (plus any harvest); the
cheap antenna-count and on-boundary checks run in either case. The one
thing skipping gives up is the independent-code-path bug cross-check — the
extra radius is not needed for that, but the re-sweep is.

**`CLAIM_MARGIN = 1e-6`** — the search, the claims, and the validation all
run at `tau + CLAIM_MARGIN`. Rationale: the contest evaluator recomputes
visibility with its own code and its own floating-point path (and our sweep
observes from a point offset 1e-4 m outward from the vertex it submits), so
a claim sitting within float noise of the exact threshold could flip under
their implementation — and a false claim is the one error class worth
over-paying to avoid. This is not hypothetical on this data: **the dataset
is full of rectangles where an antenna sees exactly two of four sides**,
putting coverage at exactly ½ + O(1e-7) — measured on a 655-building crop
at τ=0.5, **14 of 66** greedy-served buildings sat on that knife edge. The
search must use the same margined threshold (not bare tau), or it counts
those as served and then silently loses them at claim time — at
`tau + margin` it spends its antenna budget tipping them robustly instead.
Set to `0.0` to gamble on the evaluator counting exact-half coverage.

---

## Vantage seeding

`VANTAGE_SEED_PERCENT` — fraction of k placed first from visibility-graph
centrality (`centrality_giscup.py`) before the main strategy runs. `0` disables
seeding **and** enables the faster solve path that computes one greedy per τ
at kMax and slices it per k; any nonzero value forces a fresh greedy per
(τ, k) and pays the graph build (~22 min at `VANTAGE_RADIUS=640` on the full
dataset in earlier runs). Historical note from testing: seeding with a
**fixed** seed count (same n regardless of k) lost to plain greedy; sizing
the draw as a **percentage of each subcase's k** is what made it work —
keep it proportional.
`VANTAGE_RADIUS`, `VANTAGE_NODE_CAPACITY`, `VANTAGE_PROCESSES` tune the
graph build.

---

## Utility functions in run_giscup.py

- `averageBuildingDistance(sampleCap=None)` — nearest-neighbor centroid
  distance stats via expanding grid-ring queries (full dataset: mean 20.9 m,
  median 18.8 m, min 2.1 m, max 129.4 m).
- `plotDataset(outPath)` — renders every footprint with an auto-sized scale
  bar (~1 s for the full dataset).
- `plot(usePath)` — per-subcase solution maps; uses `PolyCollection`
  rendering (the old per-polygon `add_patch` pattern took minutes).

## Performance quick reference (full sample dataset, measured)

| stage                           | cost                                                                        |
| ------------------------------- | --------------------------------------------------------------------------- |
| precompute, plain r=160         | ~36 min single-core (÷ processes)                                           |
| precompute, +1 doubling (→320)  | ~48 min single-core                                                         |
| precompute, +2 doublings (→640) | ~77 min single-core                                                         |
| shared `extraLen` table         | ~10 s, once per solve()                                                     |
| greedy (incremental engine)     | 4.7–11.2× faster than the old rescan version, bit-identical sequences       |
| lookahead depth 2 / 3           | ~2.3× / ~15× plain greedy per τ                                             |
| refine                          | neighborhood-bounded per swap + per-subcase time budget                     |
| validation                      | k antennas only; adaptive re-sweep one VALIDATION_STEP past the claim reach |
