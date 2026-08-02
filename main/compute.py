"""
compute.py — how many CPU worker processes the parallel stages use.

Single source of truth for settings.CPU_PROCESSES. Every parallel stage in
the pipeline asks this module for its workers instead of calling
os.cpu_count() itself, so that one setting actually governs the whole
program rather than one stage at a time -- and so the startup banner can
report what will really be used, resolved rather than requested.

Everything in this pipeline that can run in parallel does, across CPU
worker processes. There is no serial mode and no second backend: the work
that dominates the wall clock (the visibility sweeps, the refine swap
scan's exact evaluation) is pure-python geometry over dicts of tuples with
data-dependent control flow, which is the shape multiprocessing suits and
array/GPU execution does not. A GPU backend was built and removed on
2026-08-01 -- see README ("Why there is no GPU backend") for the
measurements behind that decision, so it does not get re-litigated from
scratch.

IMPORTANT (multiprocessing): worker processes re-import this module at
spawn, so importing it must stay cheap. Keep it dependency-free.
"""

import os

import settings


def availableUnits():
    """CPU cores on this machine."""
    return os.cpu_count() or 1


def unitCount():
    """
    How many worker processes to use: settings.CPU_PROCESSES, clamped to
    [1, availableUnits()]. None means "every core".

    The clamp is deliberate in both directions. Asking for more processes
    than there are cores buys nothing and costs context-switching plus one
    full worker's memory footprint each (workers mmap the packed viewshed,
    but still carry their own interpreter); asking for zero or a negative
    count would mean "no parallelism", which this pipeline no longer has a
    path for.
    """
    requested = getattr(settings, "CPU_PROCESSES", None)
    limit = availableUnits()
    if requested is None:
        return limit
    try:
        n = int(requested)
    except (TypeError, ValueError):
        raise ValueError(
            f"settings.CPU_PROCESSES is {requested!r}; expected None or an "
            "integer.")
    return max(1, min(n, limit))


def workerPool(initializer=None, initargs=()):
    """
    A multiprocessing pool sized by unitCount(). Callers use it exactly
    like mp.Pool -- as a context manager, or long-lived with an explicit
    close()/join().
    """
    import multiprocessing as mp
    return mp.Pool(processes=unitCount(), initializer=initializer,
                   initargs=initargs)


def describe():
    """
    The startup banner for solve(): what the parallel stages will actually
    run on and how many of them, resolved (post-clamp) rather than as
    requested. Printed before any work so a run that is quietly
    single-core is obvious from line one of the log instead of inferred
    from the timings.
    """
    n = unitCount()
    have = availableUnits()
    requested = getattr(settings, "CPU_PROCESSES", None)

    if requested is None:
        howMany = f"all {have} available"
    elif int(requested) > have:
        howMany = f"{n} (capped: {int(requested)} requested, only {have} exist)"
    elif int(requested) < 1:
        howMany = f"{n} (floored: {int(requested)} requested, minimum is 1)"
    else:
        howMany = f"{n} of {have}"

    return [
        f"Processing units: CPU cores -- using {howMany}.",
        f"  every parallel stage runs across {n} worker "
        f"process{'' if n == 1 else 'es'}.",
    ]
