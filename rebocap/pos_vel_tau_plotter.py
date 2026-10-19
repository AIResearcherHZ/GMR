from __future__ import annotations

import gc
import math
import multiprocessing as mp
import os
import signal
from collections.abc import Sequence
from multiprocessing import shared_memory

import numpy as np
from libs.drivers.rate_limiter import perf_counter

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

_SIG_PER_JOINT = 6
_DES_COLOR = "#FF9100"
_MEAS_COLOR = "#00E5FF"
_SIG_TITLE = ("pos", "vel", "τ")
_TARGET_POINTS = 1500
_FULL_REDRAW_MIN_S = 0.5


def _short_label(lbl: str) -> str:
    for suf in ("_link_motor", "_motor", "_link"):
        lbl = lbl.replace(suf, "")
    return lbl


def _shm_array(
    name: str, shape: tuple, dtype, create: bool = False
) -> tuple[shared_memory.SharedMemory, np.ndarray]:
    dtype = np.dtype(dtype)
    size = math.prod(shape) * dtype.itemsize
    try:
        shm = shared_memory.SharedMemory(
            name=name, create=create, size=size, track=create
        )
    except TypeError:
        if create:
            shm = shared_memory.SharedMemory(name=name, create=True, size=size)
        else:
            from multiprocessing import resource_tracker

            orig = resource_tracker.register
            resource_tracker.register = lambda *a, **k: None
            try:
                shm = shared_memory.SharedMemory(name=name)
            finally:
                resource_tracker.register = orig
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    if create:
        arr[:] = 0
    return shm, arr


def _plotter_main(shm_name, head_name, labels, capacity, window_sec, freq):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    gc.collect(2)
    gc.freeze()
    gc.disable()

    import matplotlib

    matplotlib.use("TkAgg", force=True)
    matplotlib.rcParams.update(
        {
            "font.family": ["Noto Sans CJK JP", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "path.simplify": True,
            "path.simplify_threshold": 1.0,
            "agg.path.chunksize": 10000,
        }
    )
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    n = len(labels)
    n_cols_total = 1 + _SIG_PER_JOINT * n + 1
    sig_slice = slice(1, 1 + _SIG_PER_JOINT * n)
    stamp_col = n_cols_total - 1

    ring_shm, ring = _shm_array(shm_name, (capacity, n_cols_total), np.float64)
    head_shm, head_arr = _shm_array(head_name, (2,), np.int64)

    view_cap = max(64, int(window_sec * freq))
    ring_t = np.zeros(view_cap, dtype=np.float64)
    ring_sig = np.zeros((view_cap, _SIG_PER_JOINT * n), dtype=np.float64)
    view_t = np.empty(view_cap, dtype=np.float64)
    view_sig = np.empty((view_cap, _SIG_PER_JOINT * n), dtype=np.float64)
    ring_head = 0
    ring_count = 0
    last_seen = 0

    fig, axes = plt.subplots(
        n,
        3,
        figsize=(10.2, min(0.92 * n + 0.9, 10.0)),
        sharex=True,
        squeeze=False,
        constrained_layout=True,
    )
    fig.canvas.manager.set_window_title("目标 vs 真机 · pos/vel/tau（实时）")
    fig.patch.set_facecolor("#101418")
    fig.suptitle(
        "目标 (橙·虚线)    vs    真机 (青·实线)        pos / vel / τ",
        color="#eceff1",
        fontsize=11,
    )
    try:
        win = fig.canvas.manager.window
        dpi = fig.dpi
        sw, sh = win.winfo_screenwidth() / dpi, win.winfo_screenheight() / dpi
        fig.set_size_inches(
            max(8.0, min(3 * 3.4, sw * 0.92)),
            max(3.0, min(0.92 * n + 0.9, sh * 0.90)),
        )
        win.wm_geometry("+70+40")
    except Exception:
        pass

    def _style_ax(ax):
        ax.set_facecolor("#181c22")
        ax.tick_params(colors="#cfd8dc", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#37474f")
        ax.grid(True, color="#263238", linewidth=0.6)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))

    des_lines = [[None] * 3 for _ in range(n)]
    meas_lines = [[None] * 3 for _ in range(n)]
    for j in range(n):
        for r in range(3):
            ax = axes[j][r]
            _style_ax(ax)
            if j == 0:
                ax.set_title(_SIG_TITLE[r], color="#b0bec5", fontsize=10)
            if j == n - 1:
                ax.set_xlabel("t (s)", color="#b0bec5", fontsize=9)
            ax.axhline(0.0, color="#37474f", linewidth=0.8, linestyle=":")
            des_lines[j][r] = ax.plot(
                [], [], color=_DES_COLOR, lw=1.4, ls="--", animated=True
            )[0]
            meas_lines[j][r] = ax.plot(
                [], [], color=_MEAS_COLOR, lw=1.5, animated=True
            )[0]
        axes[j][0].set_ylabel(
            _short_label(labels[j]),
            color="#eceff1",
            fontsize=8,
            rotation=0,
            ha="right",
            va="center",
            labelpad=8,
        )

    axes[0][0].set_xlim(-window_sec, 0.02 * window_sec)
    all_lines = [ln for row in des_lines for ln in row]
    all_lines += [ln for row in meas_lines for ln in row]

    ylim_state = [[None] * 3 for _ in range(n)]
    shrink_t = [[0.0] * 3 for _ in range(n)]
    last_full = 0.0
    frozen = False
    closed = False

    def _autoscale(ax, j, r, lo, hi, now):
        if not (np.isfinite(lo) and np.isfinite(hi)):
            return False
        if hi - lo < 1e-9:
            mid = 0.5 * (lo + hi)
            lo, hi = mid - 0.5, mid + 0.5
        rng = hi - lo
        cur = ylim_state[j][r]
        if cur is None:
            nlim = (lo - 0.4 * rng, hi + 0.4 * rng)
        else:
            clo, chi = cur
            crng = chi - clo
            if lo < clo + 0.02 * crng or hi > chi - 0.02 * crng:
                nlim = (min(clo, lo - 0.5 * rng), max(chi, hi + 0.5 * rng))
            elif crng > 4.0 * rng and now - shrink_t[j][r] > 3.0:
                nlim = (lo - 0.3 * rng, hi + 0.3 * rng)
            else:
                return False
        ax.set_ylim(nlim)
        ylim_state[j][r] = nlim
        shrink_t[j][r] = now
        return True

    canvas = fig.canvas
    bg = None

    def _on_draw(_evt):
        nonlocal bg
        bg = canvas.copy_from_bbox(fig.bbox)
        for ln in all_lines:
            fig.draw_artist(ln)

    canvas.mpl_connect("draw_event", _on_draw)

    def _render():
        if bg is None:
            canvas.draw()
            return
        canvas.restore_region(bg)
        for ln in all_lines:
            fig.draw_artist(ln)
        canvas.blit(fig.bbox)

    def drain():
        nonlocal ring_head, ring_count, last_seen
        cur = int(head_arr[0])
        stop = bool(head_arr[1])
        if cur == last_seen:
            return stop
        if cur - last_seen > capacity:
            last_seen = cur - capacity
        backlog = cur - last_seen
        start = last_seen % capacity
        end = (last_seen + backlog) % capacity
        chunk = (
            ring[start:end].copy()
            if start < end
            else np.vstack([ring[start:], ring[:end]])
        )
        valid = chunk[:, stamp_col] == last_seen + np.arange(backlog)
        chunk = chunk[valid]
        last_seen = cur
        m = len(chunk)
        if m > view_cap:
            chunk = chunk[-view_cap:]
            m = view_cap
        if m:
            w_end = (ring_head + m) % view_cap
            if ring_head < w_end:
                ring_t[ring_head:w_end] = chunk[:, 0]
                ring_sig[ring_head:w_end] = chunk[:, sig_slice]
            else:
                split = view_cap - ring_head
                ring_t[ring_head:] = chunk[:split, 0]
                ring_t[:w_end] = chunk[split:, 0]
                ring_sig[ring_head:] = chunk[:split, sig_slice]
                ring_sig[:w_end] = chunk[split:, sig_slice]
            ring_head = w_end
            ring_count = min(ring_count + m, view_cap)
        return stop

    def snapshot():
        if ring_count < view_cap:
            return ring_t[:ring_count], ring_sig[:ring_count]
        k = view_cap - ring_head
        view_t[:k] = ring_t[ring_head:]
        view_t[k:] = ring_t[:ring_head]
        view_sig[:k] = ring_sig[ring_head:]
        view_sig[k:] = ring_sig[:ring_head]
        return view_t, view_sig

    def _close(_evt=None):
        nonlocal closed
        if closed:
            return
        closed = True
        try:
            timer.stop()
        except Exception:
            pass
        try:
            plt.close(fig)
        except Exception:
            pass

    def _on_timer():
        nonlocal last_full, frozen
        if closed:
            return
        if drain():
            _close()
            return
        if ring_count == 0:
            return

        ts, sg = snapshot()
        sig = sg.reshape(-1, n, _SIG_PER_JOINT)
        x = ts - ts[-1]
        step = max(1, len(x) // _TARGET_POINTS)
        xd = x[::step]
        sd = sig[::step]

        for j in range(n):
            for r in range(3):
                des_lines[j][r].set_data(xd, sd[:, j, 2 * r])
                meas_lines[j][r].set_data(xd, sd[:, j, 2 * r + 1])

        now = perf_counter()
        need_full = False
        if now - last_full >= _FULL_REDRAW_MIN_S:
            smin, smax = sig.min(axis=0), sig.max(axis=0)
            for j in range(n):
                for r in range(3):
                    lo = min(smin[j, 2 * r], smin[j, 2 * r + 1])
                    hi = max(smax[j, 2 * r], smax[j, 2 * r + 1])
                    if _autoscale(axes[j][r], j, r, float(lo), float(hi), now):
                        need_full = True

        if need_full:
            last_full = now
            if not frozen:
                frozen = True
                try:
                    fig.set_layout_engine("none")
                except Exception:
                    pass
            canvas.draw()
        else:
            _render()
        try:
            canvas.flush_events()
        except Exception:
            pass

    timer = canvas.new_timer(interval=50)
    timer.add_callback(_on_timer)
    canvas.mpl_connect("close_event", _close)
    timer.start()
    try:
        plt.show()
    finally:
        _close()
        ring_shm.close()
        head_shm.close()


class PosVelTauPlotter:
    __slots__ = (
        "_capacity",
        "_head",
        "_head_shm",
        "_n",
        "_proc",
        "_ring",
        "_ring_shm",
        "_sig",
        "_stage",
        "_stamp",
        "_t0",
    )

    def __init__(
        self,
        labels: Sequence[str],
        freq: float = 200.0,
        window_sec: float = 10.0,
        ring_capacity: int = 4096,
    ) -> None:
        n = len(labels)
        self._n = n
        cols = 1 + _SIG_PER_JOINT * n + 1
        capacity = max(1024, int(ring_capacity))
        suffix = f"{os.getpid()}_{int(perf_counter() * 1e6) & 0xFFFFFF:x}"
        self._ring_shm, self._ring = _shm_array(
            f"taks_pvt_ring_{suffix}", (capacity, cols), np.float64, create=True
        )
        self._head_shm, self._head = _shm_array(
            f"taks_pvt_head_{suffix}", (2,), np.int64, create=True
        )
        self._capacity = capacity
        self._t0 = perf_counter()
        self._sig = slice(1, 1 + _SIG_PER_JOINT * n)
        self._stamp = cols - 1
        self._stage = np.empty(_SIG_PER_JOINT * n, dtype=np.float64)

        ctx = mp.get_context("spawn")
        self._proc = ctx.Process(
            target=_plotter_main,
            args=(
                self._ring_shm.name,
                self._head_shm.name,
                list(labels),
                capacity,
                window_sec,
                freq,
            ),
            daemon=False,
        )
        self._proc.start()

    def push(
        self,
        pos_des: np.ndarray,
        pos_meas: np.ndarray,
        vel_des: np.ndarray,
        vel_meas: np.ndarray,
        tau_des: np.ndarray,
        tau_meas: np.ndarray,
    ) -> None:
        if self._proc is None:
            return
        n = self._n
        s = self._stage
        s[0::_SIG_PER_JOINT] = pos_des[:n]
        s[1::_SIG_PER_JOINT] = pos_meas[:n]
        s[2::_SIG_PER_JOINT] = vel_des[:n]
        s[3::_SIG_PER_JOINT] = vel_meas[:n]
        s[4::_SIG_PER_JOINT] = tau_des[:n]
        s[5::_SIG_PER_JOINT] = tau_meas[:n]
        head = int(self._head[0])
        row = self._ring[head % self._capacity]
        row[0] = perf_counter() - self._t0
        row[self._sig] = s
        row[self._stamp] = head
        self._head[0] = head + 1

    def close(self, timeout: float = 10.0) -> None:
        proc = self._proc
        if proc is None:
            return
        try:
            self._head[1] = 1
        except Exception:
            pass
        proc.join(timeout=timeout)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2.0)
        if proc.is_alive() and proc.pid is not None:
            try:
                os.kill(proc.pid, 9)
            except Exception:
                pass
            proc.join(timeout=1.0)
        self._proc = None
        for shm in (self._ring_shm, self._head_shm):
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
