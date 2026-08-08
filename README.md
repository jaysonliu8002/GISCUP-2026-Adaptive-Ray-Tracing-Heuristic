# GIS Cup 2026 solver — configuration reference

**Last updated 8/7/2026**

Sample testing instructions:

**PART 1**

1.In settings, set `ADAPTIVE_DOUBLINGS` = 2 and `GEOJSON_PATH = os.path.join(REPO_ROOT, "dataset1_sample.geojson")`

2.Look for a table like the following below and copy paste into a document

    tau     k    served   swaps   time
    ----------------------------------------
    0.25     50      1321       0  154.6s
    0.25    500      9157      27  885.3s
    0.25   1000     12368      30 1089.5s
    0.5      50       537       4  259.9s
    0.5     500      4884      30 1910.9s
    0.5    1000      8648      30 2286.3s
    0.75     50       209      10  433.6s
    0.75    500      2246      30 2887.6s
    0.75   1000      4351      30 8383.9s
    ----------------------------------------
    total served: 43721

3.Record the runtime it took and paste below the table, should be at the end like below

`[__main__ total] took xxxxx.xxs`

4.Repeat above steps with `ADAPTIVE_DOUBLINGS` = 5 and

`ADAPTIVE_DOUBLINGS` = 8

5.Repeat above steps (including 4) with `"dataset2_sample.geojson"` and `"dataset3_sample.geojson"`

`ADAPTIVE_DOUBLINGS` = 8

5.Repeat above steps (including 4) with `"dataset2_sample.geojson"` and `"dataset3_sample.geojson"`

**PART 2**

TBD

**IMPORTANT**

All switches live in `main/settings.py` (`test/` mirrors it — see below).
`run_giscup.py` is now purely a driver and defines no tunables of its own;
it imports them from there. This file holds the prose:
what each one does, the measurements behind its default, and how the pieces
interact. Crop benchmarks below refer to two cutouts of the sample dataset:
**crop1** (605 buildings / 2,945 candidates) and **crop2** (1,855 / 10,168).
The full sample dataset is 12,860 buildings / 78,727 candidates.

The 2026-07-26 performance work (parallelism, normalization, startup) was
measured on its own cutouts — **338 buildings / 2,335 candidates** and
**2,360 / 15,834** — which are _not_ crop1/crop2; those figures name their
sizes inline wherever they appear.

Use `run_giscup.py` to run the program.

`test/` holds the **submission** variant: the same solver plus
`packageSubmission()`, which builds the contest-ready
`submission_package.zip` (solution file + source folder + build/run
instructions). Its modules are otherwise a mirror of `main/` with the
imports renamed (`lib_giscup` → `lib_test`, etc.), so **anything changed in
`main/` has to be copied across** or the packaged submission ships stale
code — including the two modules added on 2026-08-01, `settings.py` and
`compute.py`. `test/` is gitignored, so git will not warn you about the
drift.

Relavant switches to change:

| switch                  |
| ----------------------- |
| `CPU_PROCESSES`         |
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

**The parallel swap scan** — sized by `CPU_PROCESSES`, with no serial
mode. Finding the best swap is the dominant cost of refinement, and it is
spread across workers in three stages per pass:

1. the non-interacting pairs (exact closed form, `gains[a] - loss(r)`) are
   resolved in the parent, establishing a delta floor;
2. that floor feeds a **vectorized upper bound** over every remaining
   `(r, a)` pair at once — the same `gains[a] + |lossHit| - nLoss` bound,
   evaluated as array math instead of a Python loop;
3. whatever survives the bound is evaluated **exactly** by worker processes.

Step 2 stays in the parent rather than being farmed out: these are
per-removal-candidate slices of a few thousand entries evaluated thousands of
times per pass, so shipping each to a worker would cost more in pickling than
the arithmetic saves. Only the survivors — the ones needing the expensive
exact interval math — are worth dispatching.

Workers read candidate intervals from the memory-mapped packed viewshed
(see below), never the multi-GB `viewCache`, which is what makes this fit in
memory at all. Verification and commits stay in the parent, unchanged, so
every committed swap is still exact-checked the same way.

Measured on a 2,360-building / 15,834-candidate crop at τ=0.5, 12 workers:
**2.7× at k=100, 1.9× at k=300**. (A k=800 run on that crop measured only
1.08×, but it had already served 2,359 of 2,360 buildings — saturated, with
no swaps left to find. It is not representative of the full dataset, where
k=1000 covers 12,860 buildings.) The speedup is well under 12× because the
bound-pruning stage and task hand-off stay serial; the win is on the
pass scan, which is where the time actually goes. Because refinement is
time-budgeted, a faster scan directly means **more committed swaps inside
the same budget**, not just an earlier finish.

The now-removed serial scan found the **same maximum delta every pass** as
this one (tested directly, 8 consecutive passes); the two could still land on
different local optima over a full run, because ties at that maximum broke in
a different order and the choice compounds. That equivalence is why deleting
it cost nothing but a second implementation of correctness-critical code to
keep in sync.

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

## Crash safety & resume

A full run is hours long, so nothing is held only in memory until the end.
There are three independent recovery points; each one is written **during**
the run, not after it.

**Precompute checkpoints.** The viewshed precompute appends finished
viewsheds to `cache/<name>.pkl.partial` every 2 minutes as one pickle frame
per checkpoint. (Short on purpose: each viewshed is written exactly once
regardless of interval, so checkpointing more often costs extra `fsync`
calls but not extra bytes — while a long interval risks never firing at
all, since the whole precompute is only ~6 min wall on 12 cores, and holds
more finished work in RAM waiting to flush.) A restart loads it and re-sweeps only the candidates
it does not already have. A frame torn by a crash mid-append is detected and
skipped on load, so the file is always usable; the header frame records
dataset fingerprint + radius + corridor reach, and a mismatch discards the
checkpoint rather than resuming with wrong viewsheds. The `.partial` is
deleted once the real cache is saved. The full cache itself is still written
**before any solving starts** (unchanged, and it already used
write-temp-then-`os.replace`), so the expensive stage is banked first.

**Vantage seed / visibility-graph checkpoints.** The centrality graph build
is the longest single stage in the pipeline (~22 min at radius 640), and it
now protects itself the same way, plus caches its results:

- finished vertices are appended to `cache/vantage_graph_*.pkl.partial` on
  the same 2-minute cadence; a restart restores them and tests only the
  vertices still missing (a resumed graph is byte-identical to one built in
  one run — edges are re-applied symmetrically exactly as the builder does);
- the finished graph is cached to `cache/vantage_graph_r{radius}.pkl`, keyed
  on dataset + radius, so later runs skip the build entirely and a settings
  change is refused rather than silently reused;
- the picked seeds are cached separately to `cache/vantage_seeds_*.pkl`,
  keyed additionally by the seed **count**. A re-run with identical settings
  loads that one small file and never touches the graph; a run that only
  changes `VANTAGE_SEED_PERCENT` still reuses the cached graph.

All writes are temp-file + `os.replace`, and the `.partial` is deleted once
the real graph is saved.

**What a cache is keyed on.** Every cache file's name and metadata carry
exactly the settings its _contents_ depend on, and nothing more — a knob
that cannot change the result must never cost a rebuild:

| cache                                    | keyed on                                              | deliberately **not** keyed on                           |
| ---------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------- |
| `viewcache_r{radius}[_d{doublings}].pkl` | dataset, `SOLVE_RADIUS`, corridor reach (`maxRadius`) | `SOLVE_GRIDCELL`, process/chunk counts                  |
| `viewcache_*_packed.*`                   | the viewshed cache it mirrors                         | as above                                                |
| `vantage_graph_r{radius}.pkl`            | dataset, `VANTAGE_RADIUS`                             | `VANTAGE_NODE_CAPACITY`, grid cell, `VANTAGE_PROCESSES` |
| `vantage_seeds_r{radius}_n{n}.pkl`       | the above + seed count                                | as above                                                |

The excluded ones are pure performance knobs. `VANTAGE_NODE_CAPACITY` is
the STRtree's leaf capacity: it changes the tree's shape, not which boxes
a query returns. The grid cell sizes a bucketing structure whose hits are
then filtered by an exact `distance² ≤ radius²` test. Process and chunk
counts only divide identical work up differently, and edges are applied
symmetrically, so the result is order-independent. Previously
`VANTAGE_NODE_CAPACITY` _was_ in both the graph's filename and its
metadata, which meant retuning a BVH constant threw away a 22-minute
build for no reason.

> **shapely is required for vantage seeding and is not currently installed.**
> With `VANTAGE_SEED_PERCENT > 0` (it defaults to `0.01`) a run needs it.
> Either `pip install shapely` or set `VANTAGE_SEED_PERCENT = 0`. The build
> now checks for it up front and raises one clear message naming both
> options — previously the import failed inside every worker, so the real
> cause arrived buried under tens of thousands of lines of repeated
> tracebacks. The check sits _after_ the resume/cache logic, so a cached
> graph or a fully-restored checkpoint still works without shapely at all.

**Per-subcase submission writes.** `solve()` rewrites the submission after
**every** verified subcase instead of once at the end. Writes go to a temp
file and are `os.replace`d into position, so an interruption mid-write
cannot truncate the file — the previous version survives intact.

**Resume from a partial submission** (`RESUME_PARTIAL_SUBMISSION`, default
on). On startup, if the highest-numbered `submission_N.txt` holds a strict
subset of this run's τ×k grid — the signature of an interrupted run — those
subcases are reused verbatim and the run continues into the same file,
skipping the greedy entirely for any τ whose subcases are all present.
Anything else starts a fresh numbered file, so **complete submissions stay
protected exactly as before**: a finished file is never appended to, never
overwritten, and a file that does not match the current `TAUS`/`KS` is left
alone rather than half-reused.

---

## Parallelism

What is spread across worker processes, what is merely vectorized in the
parent, and what deliberately is neither. Worker counts all come from
`CPU_PROCESSES`.

| stage                            | parallel?                                         |
| -------------------------------- | ------------------------------------------------- |
| viewshed precompute              | yes — `precomputeViewshedsParallel`, the big one  |
| validation coverage re-sweep     | yes — `localEvalCoverParallel`, one sweep/antenna |
| visibility graph (vantage seeds) | yes — `buildVisibilityGraphRadiusBVH`             |
| **refine swap scan**             | **yes — new, see `REFINE_PROCESSES`**             |
| greedy / lookahead placement     | no — inherently sequential (see below)            |

**Why placement is not parallelized.** Greedy and the lookahead are
strictly sequential by construction: step _n+1_'s gains depend on what step
_n_ committed, which is the whole point of the submodular argument. The
parallelizable work inside a step (scoring the beam) is already narrowed to
`LOOKAHEAD_BEAM` candidates, and shipping the coverage state to workers per
step would cost more than the scoring does. What made placement fast instead
was removing redundant work across the nine subcases — see below.

**Packed viewshed** (`cache/<name>_packed.*.npy`, built once, then
memory-mapped). The pickled `viewCache` is nested dicts of tuples, roughly
4× its ~1 GB disk size once unpickled — which is why refine workers could
never each hold a copy. The packed mirror stores the identical intervals as
flat CSR-style numpy arrays saved per-field, so every worker plus the parent
share **one physical copy** through the OS page cache. Two per-(candidate,
building) tables then come out of it as vectorized array reads instead of
Python loops:

- `extraLen` — the unmerged interval-length bound (same values
  `buildExtraLen` produced, verified pair-for-pair);
- `baseCov` — the **τ-independent** standalone merged coverage. This is the
  bigger win: `_CoverageState.__init__` used to re-merge every (candidate,
  building) pair on **every greedy call** — nine times per solve, for a
  table that never changes. It is now computed once per cache.

`buildingToCandidates` is likewise built once per solve and shared read-only,
instead of being rebuilt inside every greedy, lookahead, and refine call.

_Memory tradeoff, stated plainly:_ `baseCov` is a new persistent table
roughly the size of `extraLen`, and `buildingToCandidates` goes from
transient-per-call to persistent — together a few hundred MB on the full
dataset, on top of the unpickled `viewCache`. That is the price of the
52.9× on state construction and of not rebuilding the same index nine
times. Interval normalization gives most of it back (91% fewer interval
tuples), and the packed arrays are memory-mapped rather than heap-resident.
If you are ever memory-bound, lower `CPU_PROCESSES` — every pool is sized
from it, so fewer units means proportionally fewer resident workers. (The old
`REFINE_PROCESSES = 0` escape hatch, which dropped the refine workers
entirely by falling back to a serial scan, no longer exists.)

**Worker startup.** `lib_giscup` deliberately does **not** import matplotlib
at module scope. On Windows every multiprocessing worker re-imports the
module at spawn, and one solve creates ~19 pools (the precompute, plus a
validation and a refine pool per subcase) × `cpu_count()` workers — none of
which draw anything. Importing `matplotlib.pyplot` cost ~1.0 s per worker on
top of numpy's 0.49 s; making it lazy took module import from **1.47 s to
0.52 s (2.8×)**. `plt` and `MplPolygon` are still re-exported via PEP 562
`__getattr__`, so `from lib_giscup import plt` keeps working and pulls
matplotlib in only when a process actually touches it. `run_giscup` imports
it inside the two plotting functions for the same reason — a module-level
import there would be paid by every worker too, since spawn re-imports
`__main__`.

**Normalized intervals.** A sweep emits per-wedge interval _fragments_ —
averaging 11.6 per (candidate, building, edge), often adjacent or
overlapping, since the corridor passes deliberately overlap at window
boundaries. Every consumer already treats them as a union, so `normalizeView`
now sorts and unions each list **once, when the viewshed is built**, instead
of the union being re-derived inside each of the ~23 million coverage checks
a solve performs.

That invariant is what lets the hot path call `_addedLengthNorm`, which
returns just the added length via a two-pointer walk — no merged list
allocated, no baseline re-summed (`_covWithExtra` discarded that list
anyway). Profiled on a 2,360-building crop, `_mergeIntervalsIncremental` was
**53% of total lookahead time** (23.5M calls, 357 s of 669 s); the
replacement measured **3.7× faster** on real cache intervals. Normalization
also shrinks the cache, since fragments collapse into fewer tuples.

A cache built before this change is **upgraded in place on load** (a
union-preserving rewrite, then re-saved atomically) rather than rebuilt —
no re-sweep, and future runs skip the pass. Packed files carry a matching
flag so a stale one rebuilds. On the measured crop the collapse was
**8,870,296 → 763,506 intervals (91% fewer)**, which is also a large memory
saving.

**On exactness.** Per-building coverage is bit-identical after
normalization — verified, max difference exactly 0.0 across all 372,507
(candidate, building) pairs. Two things do change, and they are worth
keeping separate:

1. **`extraLen` gets much tighter — a speed win, not a result change.**
   That table is the _unmerged_ fragment-length sum, so before
   normalization it wildly overcounted overlap: the largest per-pair gap
   measured was **424 m**. It is only ever used as a sound upper bound to
   skip an exact coverage check (`baseVis + extraLen < tau·perim` ⇒ cannot
   tip), and it stays sound either way, so a tighter value skips strictly
   more work that could not have changed an answer. (`_covWithExtra`'s
   `tauHint` short-circuit, the one path that would return a bound _as_ a
   coverage value, is never used — every caller applies the bound itself
   and then asks for the exact figure.)

2. **Summation order shifts results by ~1e-15.** Adding ~11.6 fragments
   and adding their single merged union differ in the last bit, measured
   at most **7.1e-15** per merge, accumulating a little further in the
   coverage-sum tiebreak. This dataset is full of knife-edge cases (see
   `CLAIM_MARGIN` — antennas seeing exactly two of four sides land at
   exactly ½), so a 1e-15 nudge can flip one `>= tau` comparison or one
   exact-equality tiebreak and send greedy down a different trajectory.

So the pipeline is **not** bit-identical before and after — it is
equivalent within float noise. Measured across 6 (τ, k) subcases, per-case
served counts moved in both directions (−7 to +5) and the **totals came out
exactly equal, 10,038 either way** — different local optima of equal
quality, not a regression or a free win. Claim validity is unaffected:
`CLAIM_MARGIN` (1e-6) is nine orders of magnitude above the perturbation,
which is precisely the margin's purpose, and an independent re-sweep of the
normalized solutions found **zero false claims**.

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
`VANTAGE_RADIUS` and `VANTAGE_NODE_CAPACITY` tune the graph build; its worker
count comes from `CPU_PROCESSES` like every other stage.

The graph and the picked seeds are both cached to `cache/` and the build
checkpoints as it runs, so that ~22 min is paid once rather than on every
run or after every crash — see **Crash safety & resume**. Note that
seeding requires **shapely**, which is not installed here; that section
covers the options.

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
| shared per-pair tables          | built once per cache from the packed arrays, then reused by all 9 subcases  |
| `_CoverageState` construction   | **7.82 s → 0.15 s (52.9×)** via the shared `baseCov` table, per greedy call |
| coverage check (hot inner loop) | **3.7×** via normalized intervals + `_addedLengthNorm`                      |
| worker process startup          | **1.47 s → 0.52 s (2.8×)** by not importing matplotlib in workers           |
| greedy (incremental engine)     | 4.7–11.2× faster than the old rescan version, bit-identical sequences       |
| lookahead depth 2 / 3           | ~2.3× / ~15× plain greedy per τ                                             |
| refine swap scan                | **2.7× at k=100, 1.9× at k=300** on 12 workers; time-budgeted per subcase   |
| validation                      | k antennas only; adaptive re-sweep one VALIDATION_STEP past the claim reach |

Measured on a 2,360-building / 15,834-candidate crop unless noted. The three
bolded rows are new; they compound, since every subcase pays the first two
and the refine scan is where the wall-clock budget actually goes.
