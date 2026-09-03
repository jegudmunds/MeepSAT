"""
Step Functions for Animations and Field Data Extraction in MeepSAT

WHAT WAS THE MEMORY ISSUE?
----------------------
In the previous versions of stepfunctions, a large telescope run showed a flat
235 GB baseline (MEEP's own distributed FDTD grid, which is fine) with repeated
spikes to 375 GB and 490 GB.  All of the spike came due to the following reasons:

  1. ``sim.get_array()`` is COLLECTIVE and leaves an identical FULL-CELL array
     on EVERY MPI rank, plus MEEP's own internal reduction buffer.  With
     ``mpirun -np 8`` one call costs ~ ``2 * Nx * Ny * 16 B * nranks``, while
     the FDTD grid itself is only 1/8 per rank.  ``accumulate_efield_and_hfield``
     made six such calls back-to-back, every ``dt``.
  2. ``cmplx=True`` was used for the animation and the imaginary part was
     discarded on the very next line.
  3. ``np.abs(field.real, out=np.empty(...))`` allocated a second full-cell
     float64 while the complex one was still alive.
  4. ``eps_data`` was re-fetched every single frame for static geometry.
  5. All 8 ranks rendered the same ~75 Mpx matplotlib figure (~300 MB RGBA
     canvas each, before imshow's intermediate and the second render pass
     forced by ``bbox_inches='tight'``).
  6. All 8 ranks then ran ffmpeg against the SAME output path.
  7. ``downsample()`` materialised a full-size copy, because
     ``array[:n0*dx, :n1*dy]`` is non-contiguous once axis 1 is truncated so the
     following ``reshape`` cannot return a view.
  8. ``save_accumulated_fields`` held all six averages live at once.
  9. Three of the six accumulated components are identically zero for a 2D TM
     run (Ez source => Ex, Ey, Hz decouple and stay zero).

WHAT DID WE CHANGED IN THE LATEST VERSION (v2)?
------------------------------------------
1) Components are chosen automatically (TM / TE / all six), with an override.

2) Time-average fields: ``method="dft"`` (default) hands the job to MEEP's own
    ``add_dft_fields``, which accumulates DISTRIBUTED (each rank keeps only its
    own chunk) with NO per-step ``get_array`` at all.

    ``method="accumulate"``
    keeps the exact old arithmetic for field extraction but puts the
    accumulators in rank-0-only ``np.memmap`` files and reuses one scratch
    buffer per component.

3) Animation: in-loop waste is fixed (real-valued fetch, decimate before the dB
    maths, cached eps overlay, rank-0-only rendering, Agg canvas instead of
    pyplot, one MP4 instead of eight).  On top of that ``dump_field_frame`` can
    push frames to a single HDF5 file during the run and
    ``render_movie_from_h5`` renders the movie afterwards, keeping matplotlib
    out of the FDTD loop entirely.

A PHYSICS NOTE THAT MATTERS MORE THAN THE MEMORY IN DFT FIELD ACCUMULATION
--------------------------------------------------------------------------
``calculate_runtime_parameters`` in simulator.py deliberately makes the
extraction window an exact integer number of periods and samples uniformly at
``dt = period/points_per_period``.  With ``force_complex_fields=True`` and an
``mp.ContinuousSource`` the steady-state field is ``A(r) * exp(-i*2*pi*f*t)``.
Summing THAT uniformly over an exact integer number of periods cancels almost
completely: the geometric series is zero, and what survives is the one leftover
term from the endpoint asymmetry of ``at_every``.  So the old arithmetric ``Ez_global / count``
is ``A(r) * exp(-i*w*t_k) / count`` for a single arbitrary ``t_k``.

Measured on the 01_simple_single_lens_ARC case (see the memory-benchmark section
of ``examples/01_simple_single_lens_ARC_memory_testing.ipynb``), v1's
``max|Ez|`` comes out ~1e-3 of the DFT's.

What this does and does not mean:

  1) The SPATIAL PATTERN is fine.  That leftover term is proportional to ``A(r)``,
    so v1's normalised beam profiles and far-field patterns are correct -- the
    notebook's field cross-check shows v1 and the DFT lying exactly on top of
    each other.  Nothing previously published from v1 is invalidated by this.
  2) The MAGNITUDE is arbitrary.  It depends on where the sampling window happens
    to land, so it is not comparable between runs and means nothing absolutely.
  3) The PHASE is arbitrary, being ``exp(-i*w*t_k)`` for an accidental ``t_k``.
  4) There is NO AVERAGING.  Despite ``count`` samples going in, the answer is
    effectively built from one timestep, so it carries none of the transient
    rejection the averaging was there to provide.

``add_dft_fields`` computes ``sum_t E(t) * exp(+i*2*pi*f*t)``, the true phasor,
with every sample contributing.  So the recommended memory fix is a robustness
fix too -- just not a "v1 was wrong" fix.

ON MEMORY-MAPPED ARRAYS (the original question)
-----------------------------------------------
Yes, memmap works, and the pattern already exists in this repo at
``field_analysis.get_complex_field``.  But on its own it aims at the wrong
target:

* It FIXES long-lived buffers written once per step and read once at the end --
  exactly the accumulators and the frame store.  Both are handled below.
* It CANNOT fix the spikes, because those are arrays MEEP allocates *inside*
  ``get_array`` before your code ever sees them.  You cannot memmap what you did
  not allocate.
* The one hook is ``sim.get_array(..., arr=<preallocated>)``.  ``np.memmap`` is
  an ``ndarray`` subclass, so MEEP will write the slice straight into a
  disk-backed buffer.  ``_GET_ARRAY_ACCEPTS_ARR`` below detects support at
  import time.  Note this caps the PYTHON-side copy only; MEEP's internal MPI
  reduction buffer stays anonymous RAM.
* MPI caveat: eight ranks memmapping one file corrupts it.  Only rank 0 opens
  the memmap here.
* Filesystem caveat: a memmap on /cfs will be slow.  Point ``scratch_dir`` at
  node-local scratch ($TMPDIR) and let it flush periodically.
* RSS caveat: dirty memmap pages still count toward RSS until writeback.  It
  bounds peak far better than anonymous memory because the pages are evictable,
  but with ``--mem=0`` and a warm page cache the profile may not drop as much as
  you expect.  Not allocating 8 copies is the real win; memmap is the follow-up.
"""

import inspect
import json
import os
import shutil
import subprocess
import tempfile
import zipfile
from typing import Optional, Sequence, Tuple

import h5py
import matplotlib
import matplotlib.style
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

import meep as mp

import warnings
warnings.filterwarnings("ignore")


# =====================================================================
# 0.  Shared helpers: MPI, capability detection, array reduction
# =====================================================================

def _is_master() -> bool:
    """
    True on the single job-wide master rank.

    Prefer ``am_really_master`` over ``am_master``: with ``divide_parallel``
    subgroups ``am_master()`` is true once *per subgroup*, which would still
    give you N ffmpeg processes fighting over one output path.
    """
    try:
        return bool(mp.am_really_master())
    except AttributeError:
        try:
            return bool(mp.am_master())
        except AttributeError:
            return True


def _nprocs() -> int:
    try:
        return int(mp.count_processors())
    except Exception:
        return 1


def _mprint(*args, **kwargs):
    """print() that only fires on the master rank."""
    if _is_master():
        print(*args, **kwargs)


def _detect_get_array_arr_support() -> bool:
    """
    Does this MEEP build's ``Simulation.get_array`` accept ``arr=``?

    This is THE hook that lets us hand MEEP a preallocated (optionally
    memory-mapped) destination instead of letting it allocate a fresh full-cell
    array on every call.  Detected once, at import.
    """
    try:
        return "arr" in inspect.signature(mp.Simulation.get_array).parameters
    except (TypeError, ValueError):  # C-extension without introspectable sig
        return False


_GET_ARRAY_ACCEPTS_ARR = _detect_get_array_arr_support()
_HAS_ADD_DFT_FIELDS = hasattr(mp.Simulation, "add_dft_fields")


def print_capabilities():
    """Log what this MEEP build supports. Call once at the top of a run."""
    _mprint("=" * 60)
    _mprint("meepsat.stepfunctions_v2 capability report")
    _mprint(f"  MPI ranks                     : {_nprocs()}")
    _mprint(f"  get_array(arr=...) supported  : {_GET_ARRAY_ACCEPTS_ARR}")
    _mprint(f"  add_dft_fields supported      : {_HAS_ADD_DFT_FIELDS}")
    _mprint("=" * 60)


def _get_array_into(sim, component, vol=None, buf=None, cmplx=False):
    """
    Fetch a field slice, writing into ``buf`` when the MEEP build allows it.

    COLLECTIVE -- every rank must call this, even ranks that will throw the
    result away.  Skipping it on non-master ranks will deadlock the job.

    ``buf`` is only an optimisation (it avoids the malloc/free churn of one
    fresh full-cell array per call, and it is what makes an ``np.memmap``
    destination possible).  It never changes the returned values.
    """
    kwargs = {"component": component}
    if vol is not None:
        kwargs["vol"] = vol
    if cmplx:
        kwargs["cmplx"] = True

    if buf is not None and _GET_ARRAY_ACCEPTS_ARR:
        try:
            return sim.get_array(arr=buf, **kwargs)
        except (TypeError, ValueError):
            # Shape/dtype mismatch or an older signature -- fall through.
            pass
    return sim.get_array(**kwargs)


def _block_mean(src, dx, dy, out=None, row_chunk=512):
    """
    Block-average ``src`` by (dx, dy), in bounded-memory row chunks.

    Verified bit-identical to v1's ``downsample`` for every ``row_chunk``.

    --- v1 (stepfunctions.py, inner `downsample` of accumulate_efield_and_hfield) ---
        def downsample(array, downsample_x, downsample_y):
            original_shape = array.shape
            new_shape = (original_shape[0] // downsample_x,
                         original_shape[1] // downsample_y)
            truncated_array = array[:new_shape[0]*downsample_x, :new_shape[1]*downsample_y]
            downsampled_array = truncated_array.reshape(new_shape[0], downsample_x,
                                                        new_shape[1], downsample_y).mean(axis=(1, 3))
            return downsampled_array

    The v1 version allocates a FULL-SIZE temporary: once axis 1 is truncated the
    slice is no longer C-contiguous, so ``.reshape`` cannot return a view and
    numpy materialises a copy of the whole array.

    v2 does the same arithmetic but in row blocks, so the temporary is bounded
    by ``row_chunk * dx * n1 * dy`` elements instead of the whole grid.  When
    ``n1 * dy == src.shape[1]`` the reshape is a free view and there is no
    temporary at all.
    """
    if dx == 1 and dy == 1:
        if out is None:
            return src
        np.copyto(out, src)
        return out

    n0 = src.shape[0] // dx
    n1 = src.shape[1] // dy
    if out is None:
        out = np.empty((n0, n1), dtype=src.dtype)

    for i0 in range(0, n0, row_chunk):
        i1 = min(i0 + row_chunk, n0)
        block = src[i0 * dx:i1 * dx, :n1 * dy]
        out[i0:i1] = block.reshape(i1 - i0, dx, n1, dy).mean(axis=(1, 3))
    return out


def _block_centers(coords, factor):
    """
    Coordinates matching what ``_block_mean`` produces.

    --- v1 (stepfunctions.extract_xyzw) ---
        if downsampling_factor_x > 1:
            x = x[::downsampling_factor_x]
            y = y[::downsampling_factor_y]
            w = w[::downsampling_factor_x, ::downsampling_factor_y]

    Two problems with that:
      * ``x[::dx]`` yields ceil(len/dx) points while ``_block_mean`` yields
        len//dx, so the coordinate axis and the field array disagree in length
        whenever ``len % dx != 0``.  ``field_analysis.extract_box_contour``
        indexes ``ez[ixs, iys]`` using positions found in ``x``/``y``, so a
        mismatch silently picks the wrong grid points.
      * ``x[::dx]`` is the block's FIRST sample; a block mean lives at the
        block CENTRE.  v1 is off by half a block.
      * ``y`` was only downsampled when ``downsampling_factor_x > 1``, so an
        x-factor of 1 with a y-factor of 3 left y untouched.

    v2 returns exactly ``len//factor`` block-centre coordinates.
    """
    coords = np.asarray(coords)
    if factor == 1:
        return coords
    n = coords.shape[0] // factor
    return coords[:n * factor].reshape(n, factor).mean(axis=1)


def _decimate(arr, max_px):
    """
    Strided decimation down to roughly ``max_px`` along the longest axis.

    Returns a VIEW (zero allocation) -- the small contiguous copy only happens
    when the caller asks for one.  Used for animation frames only.

    Caveat: strided decimation can alias a fine fringe pattern.  For a movie
    that is normally fine and it is by far the cheapest option; pass
    ``frame_reduce="mean"`` to ``set_animation_params`` if you would rather pay
    for the anti-aliased block mean.
    """
    if not max_px:
        return arr, 1, 1
    longest = max(arr.shape)
    if longest <= max_px:
        return arr, 1, 1
    step = int(np.ceil(longest / float(max_px)))
    return arr[::step, ::step], step, step


def _savez_streaming(path, items, compress=True):
    """
    ``np.savez_compressed`` that never holds more than one array at a time.

    --- v1 (stepfunctions.save_accumulated_fields) ---
        np.savez_compressed(os.path.join(savepath, "efield_timeavg.npz"),
                            ex_real=np.real(Ex_avg), ex_imag=np.imag(Ex_avg),
                            ey_real=..., ..., count=count)

    numpy's savez takes every array as an argument, so all six averages (plus
    the real/imag views) must be resident simultaneously.  An .npz is just a ZIP
    of .npy members, so we can write them one at a time and free as we go.

    ``items`` is an iterable of ``(name, array_or_callable)``.  A callable is
    invoked only when it is that member's turn, which is what lets the DFT saver
    fetch one full-cell component at a time.
    """
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(path, "w", compression=mode, allowZip64=True) as zf:
        for name, value in items:
            arr = value() if callable(value) else value
            arr = np.asarray(arr)
            # NB: np.ascontiguousarray promotes 0-d to shape (1,), which would
            # silently change the `count` scalar. Only flatten-copy if needed.
            if arr.ndim and not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            with zf.open(name + ".npy", "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, arr, allow_pickle=False)
            del arr


# =====================================================================
# C.  Automatic component selection
# =====================================================================

_E_COMPONENTS = (mp.Ex, mp.Ey, mp.Ez)
_H_COMPONENTS = (mp.Hx, mp.Hy, mp.Hz)
_ALL_COMPONENTS = _E_COMPONENTS + _H_COMPONENTS

_COMPONENT_NAMES = {
    mp.Ex: "ex", mp.Ey: "ey", mp.Ez: "ez",
    mp.Hx: "hx", mp.Hy: "hy", mp.Hz: "hz",
}

# In 2D with isotropic, non-magnetic media the two polarisations decouple
# completely, so only one of these sets can be non-zero.
_TM_SET = (mp.Ez, mp.Hx, mp.Hy)   # Ez source
_TE_SET = (mp.Hz, mp.Ex, mp.Ey)   # Hz / Ex / Ey source


def _source_components(simulation):
    """
    Best-effort set of the components actually driven by the sources.

    ``meepsat.meep_geometry.set_prop_component`` converts the JSON string to a
    real MEEP constant and ``GaussianBeam.assemble`` passes it through as
    ``component=self.component``, so ``src.component`` is reliable here.  A
    plain ``mp.GaussianBeamSource`` built without an explicit component would
    report ``mp.ALL_COMPONENTS``; in that case fall back to the polarisation
    implied by ``beam_E0``.
    """
    driven = set()
    for src in getattr(simulation, "sources", []) or []:
        comp = getattr(src, "component", None)
        if comp in _ALL_COMPONENTS:
            driven.add(comp)
            continue
        e0 = getattr(src, "beam_E0", None)
        if e0 is not None:
            if abs(getattr(e0, "z", 0.0)) > 0:
                driven.add(mp.Ez)
            if abs(getattr(e0, "x", 0.0)) > 0:
                driven.add(mp.Ex)
            if abs(getattr(e0, "y", 0.0)) > 0:
                driven.add(mp.Ey)
            continue
        return None   # unknown source -> caller must assume all six
    return driven or None


def active_components(simulation, force_all=False):
    """
    Which of the six field components can be non-zero for this simulation.

    Returns a tuple in canonical (Ex, Ey, Ez, Hx, Hy, Hz) order.  Inactive
    components are still WRITTEN to the .npz as zero arrays, so
    ``field_analysis.load_case_fields`` and every notebook keep working
    unchanged -- we only skip the work of fetching and accumulating them.

    v1 unconditionally fetched, downsampled and accumulated all six.  For the
    Ez-driven 2D runs in this repo that is exactly 2x the necessary work, since
    Ex, Ey and Hz are identically zero.
    """
    if force_all:
        _mprint("[components] force_all_components=True -> all six components")
        return _ALL_COMPONENTS

    try:
        is_2d = float(simulation.cell_size.z) == 0.0
    except Exception:
        is_2d = False

    if not is_2d:
        _mprint("[components] 3D (or unknown) cell -> all six components")
        return _ALL_COMPONENTS

    driven = _source_components(simulation)
    if driven is None:
        _mprint("[components] could not classify the sources -> all six components")
        return _ALL_COMPONENTS

    if driven <= set(_TM_SET):
        chosen = tuple(c for c in _ALL_COMPONENTS if c in _TM_SET)
        _mprint("[components] 2D TM (Ez, Hx, Hy); Ex/Ey/Hz are identically zero "
                "and will be saved as zeros")
        return chosen
    if driven <= set(_TE_SET):
        chosen = tuple(c for c in _ALL_COMPONENTS if c in _TE_SET)
        _mprint("[components] 2D TE (Hz, Ex, Ey); Ez/Hx/Hy are identically zero "
                "and will be saved as zeros")
        return chosen

    _mprint("[components] sources span both polarisations -> all six components")
    return _ALL_COMPONENTS


# =====================================================================
# A.  Field extraction parameters
# =====================================================================

_FIELD = {
    "size_x": None,
    "size_y": None,
    "savepath": ".",
    "downsampling_factor_x": 1,
    "downsampling_factor_y": 1,
    # v2 additions
    "method": "dft",            # "dft" (default) | "accumulate"
    "dft_where": None,          # mp.Volume to restrict the DFT to
    "scratch_dir": None,        # node-local scratch for the memmaps
    "force_all_components": False,
    "buffer_mode": "auto",      # "auto" | "none" | "ram" | "memmap"
    "flush_every": 25,          # memmap flush cadence (accumulate path)
    "compress": True,
}


def set_field_params(field_params: dict):
    """
    Set the global field-extraction parameters.

    v1 keys are all accepted unchanged: ``size_x``, ``size_y``, ``savepath``,
    ``downsampling_factor_x``, ``downsampling_factor_y``.

    v2 adds:
      method                 "dft" (default) or "accumulate"
      dft_where              mp.Volume limiting where the DFT is stored.  This
                             is the knob that controls the DFT's (flat,
                             distributed) memory cost -- note that
                             ``field_analysis.extract_box_contour`` only ever
                             reads the four sides of a box, so the full cell is
                             often more than the analysis needs.
      scratch_dir            directory for the accumulate-path memmaps.  Use
                             node-local scratch ($TMPDIR), NOT /cfs.
      force_all_components   skip the TM/TE auto-detection.
      buffer_mode            destination for get_array on the accumulate path;
                             see ``_make_full_buffer``.
      flush_every            memmap flush cadence, in accumulation steps.
      compress               gzip the output .npz (True matches v1).
    """
    if "size_x" not in field_params:
        raise ValueError("size_x must be specified in field_params")
    if "size_y" not in field_params:
        raise ValueError("size_y must be specified in field_params")

    for key in _FIELD:
        if key in field_params:
            _FIELD[key] = field_params[key]
    _FIELD["size_x"] = field_params["size_x"]
    _FIELD["size_y"] = field_params["size_y"]
    _FIELD["savepath"] = field_params.get("savepath", ".")

    if _FIELD["method"] not in ("dft", "accumulate"):
        raise ValueError("method must be 'dft' or 'accumulate'")
    if _FIELD["method"] == "dft" and not _HAS_ADD_DFT_FIELDS:
        raise RuntimeError(
            "method='dft' requested but this MEEP build has no "
            "Simulation.add_dft_fields; use method='accumulate'")

    _mprint("Field extraction parameters set:")
    for key in ("size_x", "size_y", "savepath", "downsampling_factor_x",
                "downsampling_factor_y", "method", "scratch_dir",
                "force_all_components", "buffer_mode"):
        _mprint(f"  {key}: {_FIELD[key]}")

    # --- v1 (stepfunctions.set_field_params) ---
    #     global count
    #     count = 0
    #     global Ex_global, Ey_global, Ez_global, Hx_global, Hy_global, Hz_global
    #     Ex_global = None ... Hz_global = None
    # v2 keeps the same idea but in a dict, and the buffers themselves are
    # rank-0-only memmaps created lazily on the first accumulation step.
    _ACC["count"] = 0
    _ACC["maps"] = {}
    _ACC["scratch"] = {}
    _ACC["tmpdir"] = None
    _ACC["components"] = None
    _ACC["shape_ds"] = None
    _DFT["obj"] = None
    _DFT["components"] = None
    _DFT["nsamples"] = 0


def _full_volume():
    return mp.Volume(center=mp.Vector3(0, 0, 0),
                     size=mp.Vector3(_FIELD["size_x"], _FIELD["size_y"], 0))


def extract_xyzw(simulation):
    """
    Save the grid coordinates and integration weights matching the saved fields.

    --- v1 (stepfunctions.extract_xyzw) ---
        (x,y,z,w) = simulation.get_array_metadata(vol=box)
        if downsampling_factor_x > 1:
            x = x[::downsampling_factor_x]
            y = y[::downsampling_factor_y]
            w = w[::downsampling_factor_x, ::downsampling_factor_y]
        np.savez_compressed(os.path.join(savepath, "xyzw.npz"),
                            x_coords=x, y_coords=y, weights=w)

    v2 fixes the three defects described in ``_block_centers``: independent x/y
    factors, block-CENTRE coordinates instead of block-first-sample, and a
    length of exactly ``len//factor`` so the axes always match the field arrays.
    Weights are summed (not averaged) over each block, which is what keeps
    ``sum(w * f)`` a correct integral after downsampling.
    """
    dx = _FIELD["downsampling_factor_x"]
    dy = _FIELD["downsampling_factor_y"]

    x, y, z, w = simulation.get_array_metadata(vol=_full_volume())
    del z

    xc = _block_centers(x, dx)
    yc = _block_centers(y, dy)
    if dx > 1 or dy > 1:
        n0, n1 = xc.shape[0], yc.shape[0]
        w = np.asarray(w)[:n0 * dx, :n1 * dy]
        w = w.reshape(n0, dx, n1, dy).sum(axis=(1, 3))

    if _is_master():
        _savez_streaming(os.path.join(_FIELD["savepath"], "xyzw.npz"),
                         [("x_coords", xc), ("y_coords", yc), ("weights", w)],
                         compress=_FIELD["compress"])
        _mprint(f"[xyzw] saved: x={xc.shape}, y={yc.shape}, w={np.shape(w)}")
    return


# =====================================================================
# A1.  Time-averaged fields -- DFT path (default)
# =====================================================================

_DFT = {"obj": None, "components": None, "nsamples": 0, "freq": None}


def setup_dft_fields(simulation, freq, components=None, where=None):
    """
    Register MEEP's own DFT accumulator.  Call ONCE, before ``simulation.run()``.

    This replaces the entire v1 step-function pipeline:

    --- v1 (SPIDER2.py) ---
        simulation.run(...,
            mp.after_time(runtime_params["t0"],
                          mp.at_every(runtime_params["dt"],
                                      stepfunctions.accumulate_efield_and_hfield)),
            mp.at_end(stepfunctions.save_accumulated_fields), ...)

    --- v2 ---
        stepfunctions_v2.setup_dft_fields(simulation, freq=source_freq)
        simulation.run(..., mp.at_end(stepfunctions_v2.save_dft_fields), ...)

    Why this is the memory fix
    --------------------------
    ``add_dft_fields`` accumulates inside MEEP, in C++, DISTRIBUTED: each rank
    holds only its own chunk.  There is no per-step ``get_array``, so the
    full-cell-array-on-every-rank spike disappears entirely.  What replaces it
    is a FLAT allocation of roughly

        n_components * local_grid_points * 16 bytes

    per rank -- flat rather than spiky, and 1/nranks rather than replicated.

    If that flat cost is still too large for the full cell, pass ``where`` (or
    set ``dft_where`` in ``set_field_params``) to restrict it.
    ``field_analysis.extract_box_contour`` only ever reads the four sides of a
    rectangular box, so the full cell is often far more than the analysis needs.

    Why this is also the correctness fix
    ------------------------------------
    See the module docstring: v1 summed ``A(r)*exp(-i*w*t)`` uniformly over an
    exact integer number of periods, which sums to ~0.  ``add_dft_fields``
    computes ``sum_t E(t)*exp(+i*w*t)``, the phasor the near-to-far code
    actually wants.  The absolute scale differs from v1 (MEEP folds in its own
    dt factor); every downstream quantity in ``field_analysis`` is either
    normalised or a ratio, so this does not matter -- but do not compare raw
    magnitudes between v1 and v2 outputs.
    """
    if not _HAS_ADD_DFT_FIELDS:
        raise RuntimeError("this MEEP build has no Simulation.add_dft_fields")

    if components is None:
        components = active_components(
            simulation, force_all=_FIELD["force_all_components"])
    if where is None:
        where = _FIELD["dft_where"] or _full_volume()

    comps = list(components)

    # MEEP's frequency argument has changed shape across versions; try the
    # modern list form first, then the (fcen, df, nfreq) form.
    obj = None
    errors = []
    for attempt in (
        lambda: simulation.add_dft_fields(comps, [freq], where=where, yee_grid=False),
        lambda: simulation.add_dft_fields(comps, freq, 0, 1, where=where, yee_grid=False),
        lambda: simulation.add_dft_fields(comps, freq, 0, 1, where=where),
    ):
        try:
            obj = attempt()
            break
        except (TypeError, ValueError) as exc:
            errors.append(repr(exc))
    if obj is None:
        raise RuntimeError("could not call add_dft_fields on this MEEP build:\n  "
                           + "\n  ".join(errors))

    _DFT["obj"] = obj
    _DFT["components"] = tuple(comps)
    _DFT["freq"] = freq
    _DFT["nsamples"] = 0

    _mprint(f"[dft] registered add_dft_fields at f={freq} for "
            f"{[_COMPONENT_NAMES[c] for c in comps]}")
    _mprint(f"[dft] where: center={where.center}, size={where.size}")
    return obj


def count_dft_sample(simulation):
    """
    Optional ``mp.at_every`` companion that only bumps a counter.

    MEEP does not expose how many timesteps went into a DFT, and the ``count``
    key is written into the .npz for parity with v1.  Registering this costs
    nothing (it touches no field data) and makes ``count`` meaningful:

        mp.at_every(runtime_params["dt"], stepfunctions_v2.count_dft_sample)

    If you do not register it, ``count`` is written as -1.
    """
    _DFT["nsamples"] += 1


def save_dft_fields(simulation):
    """
    ``mp.at_end`` companion to :func:`setup_dft_fields`.

    Writes ``efield_timeavg.npz`` / ``hfield_timeavg.npz`` with exactly the v1
    key names (``ex_real``, ``ex_imag``, ..., ``count``), so
    ``field_analysis.load_case_fields`` and every notebook keep working.

    Memory discipline: ``get_dft_array`` is COLLECTIVE and, like ``get_array``,
    leaves a full-cell array on every rank.  We therefore fetch ONE component
    at a time, downsample it, free the full array, and write it into the open
    zip before moving to the next.  So the single unavoidable full-cell array
    appears once per component at the very end of the run, instead of six times
    per timestep throughout it.
    """
    obj = _DFT["obj"]
    if obj is None:
        raise RuntimeError("save_dft_fields called without setup_dft_fields")

    comps = set(_DFT["components"])
    dx = _FIELD["downsampling_factor_x"]
    dy = _FIELD["downsampling_factor_y"]
    savepath = _FIELD["savepath"]
    count = _DFT["nsamples"] if _DFT["nsamples"] else -1

    _mprint("[dft] extracting DFT arrays (one component at a time)...")

    shape_ds = [None]   # filled from the first fetched component

    def _fetch_downsampled(comp):
        """Collective on all ranks; returns the downsampled array on master."""
        if comp in comps:
            full = simulation.get_dft_array(obj, comp, 0)   # COLLECTIVE
        else:
            full = None                                     # provably zero
        if not _is_master():
            del full
            return None
        if full is None:
            if shape_ds[0] is None:
                return None
            return np.zeros(shape_ds[0], dtype=np.complex64)
        small = _block_mean(np.asarray(full), dx, dy).astype(np.complex64)
        del full
        shape_ds[0] = small.shape
        return small

    def _write_group(filename, group):
        path = os.path.join(savepath, filename)
        mode = zipfile.ZIP_DEFLATED if _FIELD["compress"] else zipfile.ZIP_STORED
        zf = zipfile.ZipFile(path, "w", compression=mode,
                             allowZip64=True) if _is_master() else None
        try:
            for comp in group:
                small = _fetch_downsampled(comp)     # collective -- all ranks
                if zf is None:
                    continue
                name = _COMPONENT_NAMES[comp]
                if small is None:
                    # An inactive component fetched before any active one; we do
                    # not know the shape yet, so defer it to a second pass.
                    continue
                for suffix, part in (("real", small.real), ("imag", small.imag)):
                    with zf.open(f"{name}_{suffix}.npy", "w", force_zip64=True) as fh:
                        np.lib.format.write_array(
                            fh, np.ascontiguousarray(part, dtype=np.float32),
                            allow_pickle=False)
                del small
            if zf is not None:
                # Second pass for any component whose zeros could not be sized
                # on the first pass, plus the count key.
                written = set(n.rsplit("_", 1)[0] for n in zf.namelist())
                for comp in group:
                    name = _COMPONENT_NAMES[comp]
                    if name in written:
                        continue
                    zeros = np.zeros(shape_ds[0] or (1, 1), dtype=np.float32)
                    for suffix in ("real", "imag"):
                        with zf.open(f"{name}_{suffix}.npy", "w",
                                     force_zip64=True) as fh:
                            np.lib.format.write_array(fh, zeros, allow_pickle=False)
                with zf.open("count.npy", "w") as fh:
                    np.lib.format.write_array(fh, np.array(count), allow_pickle=False)
        finally:
            if zf is not None:
                zf.close()
        _mprint(f"[dft] wrote {path}")

    _write_group("efield_timeavg.npz", _E_COMPONENTS)
    _write_group("hfield_timeavg.npz", _H_COMPONENTS)
    _mprint(f"[dft] done (count={count}, downsampled shape={shape_ds[0]})")
    return


# =====================================================================
# A2.  Time-averaged fields -- memmap accumulator path (opt-in flag)
# =====================================================================

_ACC = {"count": 0, "maps": {}, "scratch": {}, "tmpdir": None,
        "components": None, "shape_ds": None}


def _acc_scratch_dir():
    if _ACC["tmpdir"] is None:
        base = _FIELD["scratch_dir"] or os.environ.get("TMPDIR") or None
        _ACC["tmpdir"] = tempfile.mkdtemp(prefix="meepsat_acc_", dir=base)
        _mprint(f"[accumulate] memmap scratch: {_ACC['tmpdir']}")
    return _ACC["tmpdir"]


def _make_full_buffer(shape):
    """
    Persistent destination for ``get_array(arr=...)``, if this build supports it.

    Only worth allocating when ``arr=`` is honoured -- otherwise MEEP allocates
    a fresh array anyway and we would be holding a full-cell buffer for nothing.

    buffer_mode:
      "auto"   -> "ram" when arr= is supported, else "none"
      "ram"    -> ordinary array; trades a spiky alloc/free for a flat one and
                  removes the malloc churn that drives the sawtooth
      "memmap" -> disk-backed, one file PER RANK (never share one file across
                  ranks).  This is the only way to keep the full-cell transient
                  off the heap, at the cost of MEEP writing the slice to disk
                  every call -- only sane on node-local scratch.
      "none"   -> let MEEP allocate each time (v1 behaviour)
    """
    mode = _FIELD.get("buffer_mode", "auto")
    if mode == "auto":
        mode = "ram" if _GET_ARRAY_ACCEPTS_ARR else "none"
    if mode == "none" or not _GET_ARRAY_ACCEPTS_ARR:
        return None
    if mode == "ram":
        return np.zeros(shape, dtype=np.complex128)
    if mode == "memmap":
        rank = 0
        try:
            rank = int(mp.my_rank())
        except Exception:
            pass
        path = os.path.join(_acc_scratch_dir(), f"getarray_buf_rank{rank}.dat")
        return np.memmap(path, dtype=np.complex128, mode="w+", shape=shape)
    raise ValueError(f"unknown buffer_mode: {mode}")


def accumulate_efield_and_hfield(simulation):
    """
    We implement accumulation arithmetic used in the previous version (v1),
    with the memory behaviour removed in this version (v2).

    --- v1 (stepfunctions.accumulate_efield_and_hfield) ---
        ex = simulation.get_array(vol=full_volume, component=mp.Ex, cmplx=True)
        ex_down = downsample(ex, downsampling_factor_x, downsampling_factor_y)
        if Ex_global is None:
            Ex_global = np.zeros_like(ex_down, dtype=np.complex64)
            ... six of these, on EVERY rank ...
        np.add(Ex_global, ex_down, out=Ex_global)
        del ex, ex_down
        ... repeated verbatim for ey, ez, hx, hy, hz ...

    What did we changed, and why?
      * only the components that can be non-zero are fetched (see
        ``active_components``) -- for a 2D Ez run that halves the work;
      * the accumulators live on rank 0 ONLY, as float32 ``np.memmap`` pairs,
        so they are neither replicated nranks times nor resident in RAM;
      * ``_block_mean`` replaces ``downsample``, bounding the temporary instead
        of copying the whole grid;
      * one reusable scratch buffer per shape instead of a fresh allocation per
        component per step.

    Parity with v1, measured rather than assumed: the accumulated SUMS are
    bit-identical (``_block_mean`` reproduces ``downsample`` exactly, and
    float32-real-pair accumulation rounds the same way as complex64
    accumulation).  The saved values differ from v1 by at most 1 float32 ULP,
    entirely because of the final division: v1 computes ``complex64 / count``,
    which numpy evaluates with the complex division algorithm, while v2 divides
    the real and imaginary parts separately.  v2's is the more accurate of the
    two.

    NOTE: ``get_array`` is collective, so it is called on EVERY rank even though
    only rank 0 keeps the result.  Guarding it would deadlock the job.
    """
    dx = _FIELD["downsampling_factor_x"]
    dy = _FIELD["downsampling_factor_y"]
    vol = _full_volume()

    if _ACC["components"] is None:
        _ACC["components"] = active_components(
            simulation, force_all=_FIELD["force_all_components"])
        _mprint(f"[accumulate] downsampling by ({dx}, {dy})")

    for comp in _ACC["components"]:
        buf = _ACC["scratch"].get("full")
        raw = _get_array_into(simulation, comp, vol=vol, buf=buf, cmplx=True)

        if not _ACC["scratch"].get("full_checked"):
            _ACC["scratch"]["full"] = _make_full_buffer(raw.shape)
            _ACC["scratch"]["full_checked"] = True

        if not _is_master():
            del raw
            continue

        small = _block_mean(raw, dx, dy, out=_ACC["scratch"].get("small"))
        if _ACC["scratch"].get("small") is None and (dx > 1 or dy > 1):
            _ACC["scratch"]["small"] = small

        if _ACC["shape_ds"] is None:
            _ACC["shape_ds"] = small.shape
            _mprint(f"[accumulate] full {raw.shape} -> downsampled {small.shape}; "
                    f"accumulators are rank-0 float32 memmaps")

        mr, mi = _acc_maps(comp, small.shape)
        mr += small.real
        mi += small.imag
        del raw

    _ACC["count"] += 1
    if _ACC["count"] % _FIELD["flush_every"] == 0:
        for mr, mi in _ACC["maps"].values():
            mr.flush()
            mi.flush()
        _mprint(f"[accumulate] t={simulation.meep_time():.2f} "
                f"count={_ACC['count']} (memmaps flushed)")


def _acc_maps(comp, shape):
    """Lazily create the rank-0 float32 memmap pair for one component."""
    if comp not in _ACC["maps"]:
        name = _COMPONENT_NAMES[comp]
        d = _acc_scratch_dir()
        mr = np.memmap(os.path.join(d, f"{name}_real.dat"),
                       dtype=np.float32, mode="w+", shape=shape)
        mi = np.memmap(os.path.join(d, f"{name}_imag.dat"),
                       dtype=np.float32, mode="w+", shape=shape)
        mr[:] = 0.0
        mi[:] = 0.0
        _ACC["maps"][comp] = (mr, mi)
    return _ACC["maps"][comp]


def save_accumulated_fields(simulation=None):
    """
    ``mp.at_end`` companion to :func:`accumulate_efield_and_hfield`.

    --- v1 (stepfunctions.save_accumulated_fields) ---
        Ex_avg = calculate_average_fields(Ex_global, count); del Ex_global
        ... all six ...
        np.savez_compressed("efield_timeavg.npz",
                            ex_real=np.real(Ex_avg), ex_imag=np.imag(Ex_avg), ...)

    v1 held all six averages live simultaneously (``array / count`` allocates a
    new array per component, and savez needs every argument resident).  v2
    divides straight off the memmap and streams each member into the zip one at
    a time via ``_savez_streaming``.
    """
    if not _is_master():
        return

    count = max(_ACC["count"], 1)
    shape = _ACC["shape_ds"] or (1, 1)
    savepath = _FIELD["savepath"]
    _mprint(f"[accumulate] averaging over count={_ACC['count']} and saving...")

    def member(comp, part):
        def _get():
            if comp not in _ACC["maps"]:
                return np.zeros(shape, dtype=np.float32)   # provably-zero component
            mr, mi = _ACC["maps"][comp]
            return np.asarray(mr if part == "real" else mi) / count
        return _get

    for filename, group in (("efield_timeavg.npz", _E_COMPONENTS),
                            ("hfield_timeavg.npz", _H_COMPONENTS)):
        items = []
        for comp in group:
            n = _COMPONENT_NAMES[comp]
            items.append((f"{n}_real", member(comp, "real")))
            items.append((f"{n}_imag", member(comp, "imag")))
        items.append(("count", np.array(_ACC["count"])))
        _savez_streaming(os.path.join(savepath, filename), items,
                         compress=_FIELD["compress"])
        _mprint(f"[accumulate] wrote {os.path.join(savepath, filename)}")

    cleanup_accumulators()


def cleanup_accumulators():
    """Release the memmaps and delete the scratch directory."""
    for mr, mi in _ACC["maps"].values():
        try:
            mr.flush()
            mi.flush()
        except Exception:
            pass
    _ACC["maps"] = {}
    _ACC["scratch"] = {}
    if _ACC["tmpdir"] and os.path.isdir(_ACC["tmpdir"]):
        shutil.rmtree(_ACC["tmpdir"], ignore_errors=True)
        _mprint(f"[accumulate] removed scratch {_ACC['tmpdir']}")
    _ACC["tmpdir"] = None


# --- v1 kept verbatim for reference / A-B testing -------------------------
def calculate_average_fields(array, count):
    """Unchanged from v1: ``array / count``."""
    return array / count


# =====================================================================
# B.  Animation
# =====================================================================

_ANIM = {
    "Nfps": 12,
    "image_every": 25,
    "anim_file_name": None,
    "plotting_params": None,
    # v2 additions
    "render_mode": "dump",   # "dump" (default) | "inline"
    "max_frame_px": 2000,    # decimate frames down to ~this many px on the long axis
    "dpi": 150,              # v1 used 300, i.e. ~4x the pixels of this
    "base_factor": 8,        # v1 used 12
    "frame_reduce": "decimate",   # "decimate" (cheap) | "mean" (anti-aliased)
    "field_mode": "real",    # "real"/"raw" -> Re(E)^2 | "complex" -> |E|^2
    "frame_h5": None,        # defaults to <anim_file_name>_frames.h5
}

# Per-component caches, keyed by the generated function name.
_FRAME_CACHE = {}
_PLT_READY = False

# Accepted spellings for the animation field mode.  "real" plots the
# INSTANTANEOUS field Re(E(t)); "complex" plots the ENVELOPE |E(t)|.
_FIELD_MODE_ALIASES = {
    "real": "real", "raw": "real", "instantaneous": "real", "re": "real",
    "complex": "complex", "cmplx": "complex", "envelope": "complex",
    "abs": "complex", "magnitude": "complex",
}

# One log line per component, not one per frame.
_FIELD_MODE_LOGGED = set()


def _normalise_field_mode(mode):
    """Canonicalise a user-supplied field_mode, failing loudly on a typo."""
    key = str(mode).strip().lower()
    if key not in _FIELD_MODE_ALIASES:
        raise ValueError(
            f"field_mode must be 'real'/'raw' or 'complex' (got {mode!r})")
    return _FIELD_MODE_ALIASES[key]


def _field_mode_label(mode):
    """Default colourbar / title text for a mode."""
    return "|E|^2 (dB)" if mode == "complex" else "Power (dB)"


def _resolve_field_mode(sim, plotting, key):
    """
    Which fetch this component uses; a per-component ``plotting_params`` entry
    beats the global ``_ANIM["field_mode"]``.

    MUST be evaluated on EVERY rank.  ``get_array`` is collective, so a rank
    that disagrees about ``cmplx`` changes the shape and dtype of the call and
    hangs the job.  That is why ``_frame_context`` calls this OUTSIDE its
    ``_is_master()`` guard.

    ``cmplx=True`` only carries information when the Simulation was built with
    ``force_complex_fields=True``.  Without it Im(E) is identically zero, so
    ``|E|^2 == Re(E)^2`` and the complex fetch would cost twice the memory to
    tell you nothing new -- fall back rather than pay for that.
    """
    mode = _normalise_field_mode(plotting.get("field_mode", _ANIM["field_mode"]))
    note = ""
    if mode == "complex" and not getattr(sim, "force_complex_fields", False):
        mode = "real"
        note = ("  (forced: Simulation has no force_complex_fields=True, "
                "so Im(E) is zero)")
    if key not in _FIELD_MODE_LOGGED:
        _FIELD_MODE_LOGGED.add(key)
        _mprint(f"[anim] {key}: field_mode={mode}{note}")
    return mode


def set_animation_params(anim_params: dict):
    """
    v1 keys (``Nfps``, ``image_every``, ``anim_file_name``, ``plotting_params``)
    are accepted unchanged.  v2 adds:

      render_mode    "dump"   -- write decimated frames to one HDF5 file during
                                the run and render the movie afterwards with
                                ``render_movie_from_h5``.  Matplotlib never runs
                                inside the FDTD loop.  Also leaves you the raw
                                frame data, so the movie can be re-styled
                                without re-simulating.
                     "inline" -- render during the run as v1 did, but with the
                                waste removed (see ``E_field_power_dB``).
      max_frame_px   decimate to about this many pixels on the long axis before
                     any dB arithmetic.  v1 fed the FULL grid to imshow.
      dpi            figure dpi (v1: 300)
      base_factor    figure size factor (v1: 12).  With a 5.8:1 cell, v1's
                     12 @ 300 dpi is a ~70x12 inch, ~75 Mpx figure -- a ~300 MB
                     RGBA canvas per frame, on every rank.
      frame_reduce   "decimate" (strided view, zero allocation, can alias a fine
                     fringe pattern) or "mean" (block average, anti-aliased,
                     costs the bounded temporary in ``_block_mean``)
      field_mode     which field the animation plots.  "real" (default, also
                     spelled "raw") fetches the INSTANTANEOUS field and plots
                     10*log10(Re(E)^2) -- the movie shows the wave propagating.
                     "complex" fetches with ``cmplx=True`` and plots
                     10*log10(|E|^2) -- the ENVELOPE, which is nearly static
                     once a ContinuousSource run reaches steady state, and is
                     what you want for reading focal spots and sidelobes.
                     Requires ``force_complex_fields=True`` on the Simulation;
                     without it the imaginary part is zero and this silently
                     falls back to "real" (logged once per component).
                     Cost: the complex fetch is complex128 rather than float64
                     over the full cell, on every rank -- roughly 2x the
                     per-frame fetch.  It is still ONE fetch, and the dB maths
                     still runs after decimation, so this is about half of v1's
                     animation cost, not a return to it.
                     Settable per component via ``plotting_params[func_name]``.
      frame_h5       explicit path for the frame store
    """
    for key in _ANIM:
        if key in anim_params:
            _ANIM[key] = anim_params[key]
    _ANIM["plotting_params"] = anim_params.get("plotting_params", None)
    if _ANIM["render_mode"] not in ("dump", "inline"):
        raise ValueError("render_mode must be 'dump' or 'inline'")
    # Validate here, not at the first frame: a typo like "imag" should not
    # surface several hundred timesteps into an MPI run.
    _ANIM["field_mode"] = _normalise_field_mode(_ANIM["field_mode"])
    for _k, _pp in (_ANIM["plotting_params"] or {}).items():
        if isinstance(_pp, dict) and "field_mode" in _pp:
            _normalise_field_mode(_pp["field_mode"])
    _FRAME_CACHE.clear()
    _FIELD_MODE_LOGGED.clear()
    _mprint(f"[anim] render_mode={_ANIM['render_mode']} "
            f"field_mode={_ANIM['field_mode']} "
            f"max_frame_px={_ANIM['max_frame_px']} dpi={_ANIM['dpi']}")
    return


def set_plt_params(base_factor=None):
    """
    One-time matplotlib setup and custom colormaps.

    --- v1 (stepfunctions.E_field_power_dB) ---
        set_plt_params(plt, len(field_arr[1]), len(field_arr[0]), base_factor=12)

    v1 called this on EVERY frame, which re-ran ``plt.style.use`` and rebuilt
    both LinearSegmentedColormaps each time, and set a global
    ``figure.figsize`` derived from the full grid aspect.  v2 runs it once and
    sizes each figure explicitly instead of via rcParams.
    """
    global _PLT_READY, cmap_alpha, cmap_blue
    if _PLT_READY:
        return
    matplotlib.rcParams["figure.dpi"] = _ANIM["dpi"]
    matplotlib.rcParams["savefig.dpi"] = _ANIM["dpi"]
    matplotlib.rcParams["axes.labelsize"] = "medium"
    matplotlib.rcParams["axes.titlesize"] = "large"
    matplotlib.rcParams["xtick.labelsize"] = "medium"
    matplotlib.rcParams["ytick.labelsize"] = "medium"
    matplotlib.rcParams["legend.fontsize"] = "medium"
    matplotlib.rcParams["font.size"] = 14
    matplotlib.style.use("dark_background")

    cmap_alpha = LinearSegmentedColormap.from_list(
        "custom_alpha", [[1, 1, 1, 0], [1, 1, 1, 1]])
    cmap_blue = LinearSegmentedColormap.from_list(
        "custom_blue", [[0, 0, 0], [0, 0.66, 1], [1, 1, 1]])
    _PLT_READY = True


def set_figsize(x, y, base_factor=8):
    """Unchanged from v1."""
    factor = x / y
    if factor > 1:
        return [base_factor * factor, base_factor]
    if factor < 1:
        return [base_factor, base_factor / factor]
    return [base_factor, base_factor]


def label_plot(ax, title=None, xlabel=None, ylabel=None, elapsed=None):
    """Unchanged from v1."""
    if title is not None:
        ax.set_title(f"{title} at MEEP Timestep:{elapsed:0.1f}")
    if xlabel is not None:
        ax.set_xlabel("x (mm)" if xlabel is None else xlabel)
    if ylabel is not None:
        ax.set_ylabel("y (mm)" if ylabel is None else ylabel)


class Animate2DArray:
    """
    Frame renderer.  Same public surface as v1 (``create_frame``,
    ``plot_2d_array``, ``grab_frame``, ``to_mp4``, ``to_gif_simple``).

    Differences from v1:
      * every entry point is rank-0-only.  In v1 all 8 ranks rendered the same
        figure and all 8 then ran ffmpeg against the SAME ``anim_file_name``
        (only ``temp_dir`` differed) -- a write race as well as 8x the memory.
      * figures are built with ``matplotlib.figure.Figure`` + ``FigureCanvasAgg``
        instead of ``plt.subplots()``, so nothing is registered in pyplot's
        global figure manager and there is no leak to close around.
      * ``bbox_inches='tight'`` is gone; it forces a second full render pass.
        A fixed ``constrained_layout`` gives the same result for one pass.
      * the in-memory frame list is gone; frames always go to disk.
    """

    def __init__(self, fps, use_disk_cache=True, temp_dir=None, dpi=None):
        self.fps = fps
        self.dpi = dpi or _ANIM["dpi"]
        self.use_disk_cache = True          # v2 never buffers PNGs in RAM
        self._saved_frames = []             # kept for API compatibility only
        self.frame_count = 0
        self.temp_dir = None
        if _is_master():
            if temp_dir is None:
                self.temp_dir = tempfile.mkdtemp(prefix="meep_anim_")
            else:
                self.temp_dir = temp_dir
                os.makedirs(self.temp_dir, exist_ok=True)
            _mprint(f"[anim] frame cache: {self.temp_dir}")

    def plot_2d_array(self, array, eps_data=None, title=None, xlabel=None,
                      ylabel=None, extent=None, x_ticks=None, x_tick_labels=None,
                      y_ticks=None, y_tick_labels=None, elapsed=None,
                      cmap="viridis", cbar_label_=None, invert=False,
                      scale="linear", vmin=None, vmax=None):
        if not _is_master():
            return
        set_plt_params()

        ny, nx = array.shape
        fig = Figure(figsize=set_figsize(nx, ny, _ANIM["base_factor"]),
                     dpi=self.dpi, constrained_layout=True)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)

        im = ax.imshow(array, cmap=cmap, extent=extent, vmin=vmin, vmax=vmax)
        if eps_data is not None:
            ax.imshow(eps_data, cmap=cmap_alpha, origin="lower", alpha=0.2)

        label_plot(ax, title, xlabel, ylabel, elapsed)

        if cbar_label_ is not None:
            ax_divider = make_axes_locatable(ax)
            cax = ax_divider.append_axes("top", size="5%", pad="20%")
            cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
            cax.xaxis.set_ticks_position("top")
            cbar.set_label(label=cbar_label_, labelpad=0.9)

        for setter, values in ((ax.set_xticks, x_ticks),
                               (ax.set_xticklabels, x_tick_labels),
                               (ax.set_yticks, y_ticks),
                               (ax.set_yticklabels, y_tick_labels)):
            if values is not None:
                setter(values)

        self.grab_frame(fig, ax, elapsed)

    def create_frame(self, plot_func: str = "plot_2d_array", kwargs: dict = None):
        """Unchanged dispatch behaviour from v1."""
        kwargs = kwargs or {}
        if not _is_master():
            return
        getattr(self, plot_func)(
            array=kwargs.get("array"), eps_data=kwargs.get("eps_data"),
            title=kwargs.get("title"), xlabel=kwargs.get("xlabel", "x"),
            ylabel=kwargs.get("ylabel", "y"), extent=kwargs.get("extent"),
            x_ticks=kwargs.get("x_ticks"), x_tick_labels=kwargs.get("x_tick_labels"),
            y_ticks=kwargs.get("y_ticks"), y_tick_labels=kwargs.get("y_tick_labels"),
            elapsed=kwargs.get("elapsed"), cmap=kwargs.get("cmap", "viridis"),
            cbar_label_=kwargs.get("cbar_label_"), invert=kwargs.get("invert", False),
            scale=kwargs.get("scale", "linear"), vmin=kwargs.get("vmin"),
            vmax=kwargs.get("vmax"))

    def frame_size(self, fig) -> Tuple[int, int]:
        """Unchanged from v1."""
        w, h = fig.get_size_inches()
        return int(w * fig.dpi), int(h * fig.dpi)

    def grab_frame(self, fig=None, ax=None, elapsed=0, frame_format="png"):
        """
        --- v1 ---
            fig.savefig(frame_file, format='png', bbox_inches='tight',
                        facecolor=fig.get_facecolor(), edgecolor='none')
            ... else branch buffered the PNG bytes in self._saved_frames ...

        v2 drops both ``bbox_inches='tight'`` (a second render pass) and the
        in-memory branch.  ``Figure`` has no pyplot registration, so the object
        is freed as soon as it goes out of scope.
        """
        if fig is None or not _is_master():
            return
        frame_file = os.path.join(self.temp_dir, f"frame_{self.frame_count:06d}.png")
        try:
            fig.savefig(frame_file, format="png",
                        facecolor=fig.get_facecolor(), edgecolor="none")
            self.frame_count += 1
        except Exception as exc:
            print(f"ERROR saving frame at timestep {elapsed}: {exc}")
        finally:
            fig.clear()

    def _ffmpeg(self, filename, extra_vf=None, codec=None):
        command = ["ffmpeg", "-framerate", str(self.fps),
                   "-i", os.path.join(self.temp_dir, "frame_%06d.png"),
                   "-vf", "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",
                   "-r", str(self.fps)]
        if codec:
            command += ["-vcodec", codec, "-pix_fmt", "yuv420p", "-crf", "18"]
        command += ["-y", filename]
        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=1800)
            if result.returncode == 0:
                size = os.path.getsize(filename) / 1024 / 1024
                print(f"Saved {filename} ({size:.2f} MB)")
            else:
                print(f"FFmpeg error: {result.stderr[-2000:]}")
        except Exception as exc:
            print(f"Error running FFmpeg: {exc}")

    def to_mp4(self, filename: str, frame_format: str = "png", codec: str = "h264"):
        if not _is_master():
            return
        if self.frame_count == 0:
            print("ERROR: No frames to save!")
            return
        print(f"Creating MP4 from {self.frame_count} frames on disk")
        try:
            self._ffmpeg(filename, codec=codec)
        finally:
            self.cleanup()

    def to_gif_simple(self, filename: str, frame_format: str = "png"):
        if not _is_master() or self.frame_count == 0:
            return
        self._ffmpeg(filename)

    def cleanup(self):
        if self.temp_dir and os.path.isdir(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            print(f"Cleaned up temporary directory: {self.temp_dir}")
            self.temp_dir = None


FIELD_COMPONENTS = [
    {"func_name": "Ez2_dB", "component": mp.Ez, "display_name": "Ez",
     "suffix": "electric_field_power_anim"},
    {"func_name": "Ey2_dB", "component": mp.Ey, "display_name": "Ey",
     "suffix": "electric_field_power_anim"},
    {"func_name": "Ex2_dB", "component": mp.Ex, "display_name": "Ex",
     "suffix": "electric_field_power_anim"},
]


def _frame_context(sim, component, key):
    """
    Everything about a frame that does not change from step to step, computed
    once and cached: the decimation step, the tick arrays, and the epsilon
    overlay.

    --- v1 (stepfunctions.E_field_power_dB, EVERY frame) ---
        eps_data = sim.get_array(component=mp.Dielectric, size=sim.cell_size,
                                 center=mp.Vector3()).transpose()
        x, y, z, w = sim.get_array_metadata()
        ...
        x_ticks = np.linspace(0, len(x), num_ticks).astype(int)

    The geometry is static, so v1 fetched a third full-cell float64 array every
    single frame for data that never changes.
    """
    if key in _FRAME_CACHE:
        return _FRAME_CACHE[key]

    eps_full = sim.get_array(component=mp.Dielectric, size=sim.cell_size,
                             center=mp.Vector3())        # COLLECTIVE
    eps_t = np.asarray(eps_full).transpose()
    _, step_y, step_x = _decimate(eps_t, _ANIM["max_frame_px"])

    ctx = {"step_y": step_y, "step_x": step_x, "eps": None,
           "x_ticks": None, "x_tick_labels": None,
           "y_ticks": None, "y_tick_labels": None,
           "x": None, "y": None, "field_mode": "real"}

    x, y, z, w = sim.get_array_metadata()                # COLLECTIVE
    del z, w

    # Resolved on EVERY rank, deliberately outside the master guard below:
    # _fetch_power_dB turns this into a collective get_array(cmplx=...) call,
    # and ranks that disagree about it deadlock the job.
    ctx["field_mode"] = _resolve_field_mode(
        sim, (_ANIM["plotting_params"] or {}).get(key, {}) or {}, key)

    if _is_master():
        ctx["eps"] = np.ascontiguousarray(
            eps_t[::step_y, ::step_x] > 1.0)
        plotting = (_ANIM["plotting_params"] or {}).get(key, {}) or {}
        num_ticks = plotting.get("num_ticks", 5)
        ny, nx = ctx["eps"].shape
        ctx["x_ticks"] = np.linspace(0, nx, num_ticks).astype(int)
        ctx["x_tick_labels"] = np.round(np.linspace(min(x), max(x), num_ticks), 1)
        ctx["y_ticks"] = np.linspace(0, ny, num_ticks).astype(int)
        ctx["y_tick_labels"] = np.round(np.linspace(min(y), max(y), num_ticks), 1)
        ctx["x"] = np.asarray(x)[::step_x]
        ctx["y"] = np.asarray(y)[::step_y]
    del eps_full, eps_t, x, y

    _FRAME_CACHE[key] = ctx
    return ctx


def _fetch_power_dB(sim, component, ctx):
    """
    Fetch one animation frame and reduce it to a small float32 dB array.

    --- v1 (stepfunctions.E_field_power_dB) ---
        field_arr = sim.get_array(component=component, size=sim.cell_size,
                                  center=mp.Vector3(), cmplx=True).transpose()
        if np.iscomplexobj(field_arr):
            field_arr = np.abs(field_arr.real, out=np.empty(field_arr.shape,
                                                            dtype=np.float64))
            field_arr **= 2
        np.log10(field_arr, out=field_arr)
        field_arr *= 10

    Three separate costs removed:
      * ``cmplx=True`` fetched complex128 and the very next line took ``.real``,
        so half of every fetch was discarded -> v2 fetches real by DEFAULT.
        ``field_mode="complex"`` opts back into the complex fetch on purpose,
        to plot the envelope |E|^2 rather than the instantaneous Re(E)^2 --
        it costs ~2x per frame, which is why it is opt-in;
      * ``np.abs(..., out=np.empty(...))`` allocated a SECOND full-cell float64
        while the complex one was still alive -> v2 never has two;
      * the squaring and log10 ran over the FULL grid before matplotlib
        decimated it for display -> v2 decimates first, so the arithmetic runs
        on a ~2000 px array instead of the whole cell.

    In the real path ``abs(x)**2 == x**2``, so dropping the ``np.abs`` changes
    nothing.  In the complex path |E|^2 = Re^2 + Im^2 is computed directly,
    which skips the sqrt that ``np.abs`` would take only to be squared again.

    COLLECTIVE: the fetch runs on every rank; only master keeps the result.
    ``ctx["field_mode"]`` is resolved in ``_frame_context`` on every rank, so
    the ``cmplx`` flag below is guaranteed identical across the job.
    """
    cmplx = ctx.get("field_mode", "real") == "complex"

    raw = sim.get_array(component=component, size=sim.cell_size,
                        center=mp.Vector3(), cmplx=cmplx)
    if not _is_master():
        del raw
        return None

    view = np.asarray(raw).transpose()
    sy, sx = ctx["step_y"], ctx["step_x"]
    if _ANIM["frame_reduce"] == "mean" and (sy > 1 or sx > 1):
        # Coherent block average, THEN modulus.  Averaging the complex field
        # and then taking |.| is not the same as averaging |.|, and this order
        # is what the real path already does (average, then square).
        small = _block_mean(np.ascontiguousarray(view), sy, sx)
        if not cmplx:
            small = small.astype(np.float32, copy=False)
    else:
        small = np.ascontiguousarray(
            view[::sy, ::sx], dtype=(np.complex128 if cmplx else np.float32))
    del raw, view

    if cmplx:
        # |E|^2 = Re^2 + Im^2, on the already-decimated frame.
        small = (small.real ** 2 + small.imag ** 2).astype(np.float32)
    else:
        small *= small                                    # Re(E)^2

    with np.errstate(divide="ignore", invalid="ignore"):
        np.log10(small, out=small)
    small *= 10
    return small


def E_field_power_dB(sim, component, component_name, func_name=None):
    """
    In-loop rendering path (``render_mode="inline"``): v1's behaviour with the
    waste removed.  See ``_fetch_power_dB`` and ``_frame_context`` for the
    per-item v1 comparison.

    The one structural change: everything from ``set_plt_params`` onward is
    rank-0 only.  v1 had no ``am_master`` guard anywhere, so all 8 ranks
    rendered the same ~75 Mpx figure.  ``get_array`` stays OUTSIDE the guard --
    it is collective and skipping it on non-master ranks deadlocks the job.
    """
    key = func_name or f"{component_name}2_dB"
    caller_func = globals()[key]
    set_plt_params()          # cmap_blue is referenced below, so define it first

    plotting = (_ANIM["plotting_params"] or {}).get(key, {}) or {}
    ctx = _frame_context(sim, component, key)             # collective, cached
    small = _fetch_power_dB(sim, component, ctx)          # collective
    if not _is_master():
        return

    _default_label = _field_mode_label(ctx["field_mode"])

    if getattr(caller_func, "anim", None) is None:
        caller_func.anim = Animate2DArray(fps=_ANIM["Nfps"])
        _mprint(f"[anim] initialised {component_name}^2 animation "
                f"({small.shape[1]}x{small.shape[0]} px frames)")

    caller_func.anim.create_frame(
        plot_func="plot_2d_array",
        kwargs={"array": small,
                "eps_data": ctx["eps"],
                "title": plotting.get("title", _default_label),
                "xlabel": plotting.get("xlabel", "X (mm)"),
                "ylabel": plotting.get("ylabel", "Y (mm)"),
                "x_ticks": ctx["x_ticks"], "x_tick_labels": ctx["x_tick_labels"],
                "y_ticks": ctx["y_ticks"], "y_tick_labels": ctx["y_tick_labels"],
                "elapsed": sim.meep_time(),
                "cmap": cmap_blue if plotting.get("cmap", "custom_blue") == "custom_blue"
                        else plotting.get("cmap"),
                "cbar_label_": plotting.get("cbar_label", _default_label),
                "invert": plotting.get("invert", False),
                "scale": plotting.get("scale", "log"),
                "vmin": plotting.get("vmin", -50),
                "vmax": plotting.get("vmax", 0)})
    del small
    return


# ---------------------------------------------------------------------
# B2.  Disk-dump path: frames to HDF5 during the run, movie afterwards
# ---------------------------------------------------------------------

_FRAME_STORE = {}


def _frame_h5_path(key):
    base = _ANIM["frame_h5"] or _ANIM["anim_file_name"] or "animation"
    if base.endswith(".mp4"):
        base = base[:-4]
    return f"{base}_{key}_frames.h5"


def dump_field_frame(sim, component, component_name, func_name=None):
    """
    Step function for ``render_mode="dump"``: append one decimated float32 dB
    frame to a single HDF5 file and return.  Matplotlib is never touched.

    Per frame this costs a few MB written by ONE rank, versus v1's ~300 MB RGBA
    canvas rendered on EVERY rank.  The movie is produced after
    ``simulation.run()`` returns, by :func:`render_movie_from_h5`.

    Side benefit: the raw frame data outlives the simulation, so colour scale,
    colormap, labels and fps can all be changed without re-running the FDTD.
    """
    key = func_name or f"{component_name}2_dB"
    ctx = _frame_context(sim, component, key)             # collective, cached
    small = _fetch_power_dB(sim, component, ctx)          # collective
    if not _is_master():
        return

    store = _FRAME_STORE.get(key)
    if store is None:
        path = _frame_h5_path(key)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fh = h5py.File(path, "w")
        ny, nx = small.shape
        dset = fh.create_dataset("frames", shape=(0, ny, nx),
                                 maxshape=(None, ny, nx),
                                 chunks=(1, ny, nx), dtype=np.float32,
                                 compression="lzf")
        times = fh.create_dataset("times", shape=(0,), maxshape=(None,),
                                  dtype=np.float64)
        fh.create_dataset("eps_mask", data=ctx["eps"], compression="lzf")
        if ctx["x"] is not None:
            fh.create_dataset("x_coords", data=ctx["x"])
            fh.create_dataset("y_coords", data=ctx["y"])
        plotting = (_ANIM["plotting_params"] or {}).get(key, {}) or {}
        fh.attrs["plotting_params"] = json.dumps(plotting)
        fh.attrs["component_name"] = component_name
        # Self-describing: the whole point of dump mode is re-styling the movie
        # later without re-simulating, which needs to know what was plotted.
        fh.attrs["field_mode"] = ctx["field_mode"]
        fh.attrs["fps"] = _ANIM["Nfps"]
        store = {"file": fh, "frames": dset, "times": times, "path": path}
        _FRAME_STORE[key] = store
        _mprint(f"[anim] frame store: {path} ({nx}x{ny} px/frame)")

    dset, times = store["frames"], store["times"]
    n = dset.shape[0]
    dset.resize(n + 1, axis=0)
    dset[n] = small
    times.resize(n + 1, axis=0)
    times[n] = sim.meep_time()
    del small
    return


def close_frame_stores():
    """Flush and close every open frame store."""
    for key, store in list(_FRAME_STORE.items()):
        try:
            store["file"].flush()
            store["file"].close()
        except Exception:
            pass
    return {k: v["path"] for k, v in _FRAME_STORE.items()}


def render_movie_from_h5(h5path, out_mp4, fps=None, plotting_params=None,
                         dpi=None, codec="h264"):
    """
    Render an MP4 from a frame store.  Call AFTER ``simulation.run()`` returns.

    Reads one frame at a time -- h5py slices lazily, so memory is constant in
    the number of frames regardless of how long the run was.

    Runs on rank 0 only, so exactly one MP4 is produced.  (v1 let all 8 ranks
    call ``to_mp4`` with the same ``anim_file_name``, so eight ffmpeg processes
    wrote one path concurrently.)
    """
    if not _is_master():
        return
    set_plt_params()

    with h5py.File(h5path, "r") as fh:
        frames = fh["frames"]
        times = fh["times"]
        eps = np.asarray(fh["eps_mask"]) if "eps_mask" in fh else None
        x = np.asarray(fh["x_coords"]) if "x_coords" in fh else None
        y = np.asarray(fh["y_coords"]) if "y_coords" in fh else None
        plotting = plotting_params or json.loads(
            fh.attrs.get("plotting_params", "{}"))
        fps = fps or int(fh.attrs.get("fps", _ANIM["Nfps"]))
        field_mode = fh.attrs.get("field_mode", "real")
        if isinstance(field_mode, bytes):
            field_mode = field_mode.decode()
        default_label = _field_mode_label(field_mode)
        nframes = frames.shape[0]
        if nframes == 0:
            print(f"ERROR: {h5path} contains no frames")
            return

        num_ticks = plotting.get("num_ticks", 5)
        ny, nx = frames.shape[1], frames.shape[2]
        x_ticks = np.linspace(0, nx, num_ticks).astype(int)
        y_ticks = np.linspace(0, ny, num_ticks).astype(int)
        x_labels = (np.round(np.linspace(x.min(), x.max(), num_ticks), 1)
                    if x is not None else None)
        y_labels = (np.round(np.linspace(y.min(), y.max(), num_ticks), 1)
                    if y is not None else None)
        cmap_name = plotting.get("cmap", "custom_blue")

        anim = Animate2DArray(fps=fps, dpi=dpi)
        print(f"[anim] rendering {nframes} frames from {h5path}")
        for i in range(nframes):
            anim.create_frame(
                plot_func="plot_2d_array",
                kwargs={"array": frames[i],          # lazy read, one frame
                        "eps_data": eps,
                        "title": plotting.get("title", default_label),
                        "xlabel": plotting.get("xlabel", "X (mm)"),
                        "ylabel": plotting.get("ylabel", "Y (mm)"),
                        "x_ticks": x_ticks, "x_tick_labels": x_labels,
                        "y_ticks": y_ticks, "y_tick_labels": y_labels,
                        "elapsed": float(times[i]),
                        "cmap": cmap_blue if cmap_name == "custom_blue" else cmap_name,
                        "cbar_label_": plotting.get("cbar_label", default_label),
                        "scale": plotting.get("scale", "log"),
                        "vmin": plotting.get("vmin", -50),
                        "vmax": plotting.get("vmax", 0)})
    anim.to_mp4(out_mp4, codec=codec)
    return out_mp4


# ---------------------------------------------------------------------
# Generated per-component step functions (Ez2_dB, Ey2_dB, Ex2_dB)
# ---------------------------------------------------------------------

def create_field_func(component, display_name, func_name):
    """
    Same factory as v1, but the generated function dispatches on
    ``render_mode``: ``dump_field_frame`` (default) or ``E_field_power_dB``.
    """
    def field_func(sim):
        if _ANIM["render_mode"] == "dump":
            dump_field_frame(sim, component, display_name, func_name)
        else:
            E_field_power_dB(sim, component, display_name, func_name)
    return field_func


for _info in FIELD_COMPONENTS:
    _func = create_field_func(_info["component"], _info["display_name"],
                              _info["func_name"])
    _func.__name__ = _info["func_name"]
    globals()[_info["func_name"]] = _func
    globals()[_info["func_name"]].anim = None
del _info, _func


def save_animation(sim=None):
    """
    ``mp.at_end`` hook.  Handles both render modes.

    --- v1 (stepfunctions.save_animation) ---
        component_func.anim.to_mp4(f"{anim_file_name}_{display_name}2_{suffix}.mp4")

    In dump mode this closes the HDF5 stores and renders each movie once, on
    rank 0.  In inline mode it finalises the already-rendered frames.

    In dump mode you may prefer to call this AFTER ``simulation.run()`` returns
    rather than via ``mp.at_end``, so that rendering happens with the FDTD grid
    already torn down and the whole node free -- ``close_frame_stores()`` plus
    ``render_movie_from_h5()`` does exactly that.
    """
    base = _ANIM["anim_file_name"]
    if base is None:
        raise ValueError("Animation parameters not set. Call set_animation_params() first.")
    if base.endswith(".mp4"):
        base = base[:-4]      # v1 produced "...mp4_Ez2_...mp4"; drop the stray suffix

    if _ANIM["render_mode"] == "dump":
        paths = close_frame_stores()
        if not _is_master():
            return
        for info in FIELD_COMPONENTS:
            key = info["func_name"]
            if key not in paths:
                continue
            out = f"{base}_{info['display_name']}2_{info['suffix']}.mp4"
            render_movie_from_h5(paths[key], out)
        _mprint("All animations rendered from frame stores.")
        return

    if not _is_master():
        return
    for info in FIELD_COMPONENTS:
        func = globals()[info["func_name"]]
        anim = getattr(func, "anim", None)
        if anim is None:
            continue
        out = f"{base}_{info['display_name']}2_{info['suffix']}.mp4"
        try:
            anim.to_mp4(out)
        except Exception as exc:
            print(f"Error saving {info['display_name']}^2 animation: {exc}")
        finally:
            func.anim = None
    _mprint("All animations saved and memory cleaned up.")
    return


# =====================================================================
# HOW TO WIRE THIS INTO A RUN SCRIPT
# =====================================================================
#
# --- v1 (examples/.../SPIDER2.py) -----------------------------------
#
#   import meepsat.stepfunctions as stepfunctions
#
#   stepfunctions.set_animation_params(anim_params={
#       'image_every': ..., 'Nfps': ..., 'anim_file_name': savepath + name})
#   stepfunctions.set_field_params(field_params={
#       'size_x': size_x, 'size_y': size_y, 'savepath': savepath,
#       'downsampling_factor_x': dsx, 'downsampling_factor_y': dsy})
#
#   simulation.run(
#       mp.at_every(rp["animation_timestep"], stepfunctions.Ez2_dB),
#       mp.after_time(rp["t0"], mp.at_every(rp["dt"],
#                     stepfunctions.accumulate_efield_and_hfield)),
#       mp.at_end(stepfunctions.save_animation),
#       mp.at_end(stepfunctions.save_accumulated_fields),
#       mp.at_end(stepfunctions.extract_xyzw),
#       until=rp["total_time"])
#
# --- v2, DFT + dump (recommended) -----------------------------------
#
#   import meepsat.stepfunctions_v2 as stepfunctions
#
#   stepfunctions.print_capabilities()
#
#   stepfunctions.set_animation_params(anim_params={
#       'image_every': ..., 'Nfps': ..., 'anim_file_name': savepath + name,
#       'plotting_params': ...,
#       'render_mode': 'dump', 'max_frame_px': 2000, 'dpi': 150,
#       'field_mode': 'real'})        # or 'complex' for the |E| envelope
#   stepfunctions.set_field_params(field_params={
#       'size_x': size_x, 'size_y': size_y, 'savepath': savepath,
#       'downsampling_factor_x': dsx, 'downsampling_factor_y': dsy,
#       'method': 'dft'})
#
#   # Registered ONCE, before the run. No per-step get_array at all.
#   stepfunctions.setup_dft_fields(simulation, freq=source_freq)
#
#   simulation.run(
#       mp.at_every(rp["animation_timestep"], stepfunctions.Ez2_dB),
#       mp.after_time(rp["t0"], mp.at_every(rp["dt"],
#                     stepfunctions.count_dft_sample)),   # optional, for `count`
#       mp.at_end(stepfunctions.save_dft_fields),
#       mp.at_end(stepfunctions.extract_xyzw),
#       until=rp["total_time"])
#
#   # Render with the FDTD grid already torn down and the node free:
#   for key, path in stepfunctions.close_frame_stores().items():
#       stepfunctions.render_movie_from_h5(
#           path, f"{savepath}{name}_{key}.mp4")
#
# --- v2, keeping v1's accumulation arithmetic -----------------------
#
#   stepfunctions.set_field_params(field_params={
#       ..., 'method': 'accumulate',
#       'scratch_dir': os.environ.get('TMPDIR'),   # node-local, NOT /cfs
#       'buffer_mode': 'auto'})
#
#   simulation.run(
#       mp.after_time(rp["t0"], mp.at_every(rp["dt"],
#                     stepfunctions.accumulate_efield_and_hfield)),
#       mp.at_end(stepfunctions.save_accumulated_fields),
#       mp.at_end(stepfunctions.extract_xyzw),
#       until=rp["total_time"])
#
# --- restricting the DFT to what the analysis actually reads --------
#
#   # field_analysis.extract_box_contour only ever reads the four sides of a
#   # box, so the DFT does not need the whole cell:
#   stepfunctions.set_field_params(field_params={..., 'method': 'dft',
#       'dft_where': mp.Volume(center=mp.Vector3(0, 0, 0),
#                              size=mp.Vector3(520, 250, 0))})
