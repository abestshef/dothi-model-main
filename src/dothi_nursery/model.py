from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

# --- Fixed model geometry / timing (from the notebook) -----------------------
WIDTH = 30       # width of the field, in plants
LENGTH = 26      # length of the field in plants, including the external row 0
TRUN = 30        # max run time of the simulation, in months
TR_START = 3     # start of the transmission season (in "month within year")
TR_END = 11      # end of the transmission season
ALPHA = 0.1      # dispersal-kernel scale parameter
STRIPES = True   # leave every 5th row empty

# Season boundaries (absolute months) at which the rates change and the
# Gillespie step must be re-drawn.
_SEASON_CUTS = (11, 15, 23, 27)

# --- Mapping the friendly UI choices onto model parameters -------------------
BETA_MAP = {"Low": 0.6, "Moderate": 1.0, "High": 1.4}     # within-field rate
BETAP_MAP = {"Low": 0.1, "Moderate": 0.2, "High": 0.3}    # external rate
GAMMA_MONTHS = {"2 months": 2, "4 months": 4, "6 months": 6}  # incubation period

# Human-readable labels for the four control strategies.
CONTROL_COLORS = ["r", "b", "plum", "k"]


@dataclass
class SimParams:
    """All the inputs a single simulation needs."""

    beta0: float          # within-field transmission rate
    betap0: float         # external transmission rate
    gamma0: float         # E -> I transition rate (1 / incubation months)
    control_month: int    # month at which control is applied
    radius_m: float       # removal radius for controls 3 & 4, in metres
    alpha: float = ALPHA

    @classmethod
    def from_choices(
        cls,
        beta_choice: str,
        betap_choice: str,
        gamma_choice: str,
        control_month: int,
        radius_m: float,
    ) -> "SimParams":
        """Build params from the same categorical choices the notebook offered."""
        return cls(
            beta0=BETA_MAP[beta_choice],
            betap0=BETAP_MAP[betap_choice],
            gamma0=1.0 / GAMMA_MONTHS[gamma_choice],
            control_month=int(control_month),
            radius_m=float(radius_m),
        )

    def control_labels(self) -> List[str]:
        r = self.radius_m
        # Trim a trailing ".0" so "2.0m" reads as "2m".
        rtxt = f"{r:g}"
        return [
            "Inf only & replace",
            "Inf only & empty",
            f"Radius {rtxt}m & replace",
            f"Radius {rtxt}m & empty",
        ]


def _set_params(t_in_year: float, beta0: float, betap0: float, gamma0: float):
    """Seasonal transmission / transition rates for a given month-within-year."""
    if t_in_year < TR_START or t_in_year > TR_END:
        sine_wave = 0.0
        gamma = gamma0 / 10.0
    else:
        sine_wave = np.sin(2 * np.pi * (t_in_year - TR_START) / 16) ** 0.2
        gamma = gamma0
    return beta0 * sine_wave, betap0 * sine_wave, gamma


def _dispersal(grid: np.ndarray, beta: float, betap: float, alpha: float):
    """Vectorised dispersal kernel.

    Returns ``(per_susceptible_pressure, susceptible_coords, phi_total)`` where
    ``phi_total`` is the summed force of infection over every
    infected -> susceptible pair, exactly as ``findscale`` computed it in the
    notebook, and ``per_susceptible_pressure`` is that force summed per
    susceptible site (used to pick which site gets infected).
    """
    inf = np.argwhere(grid == 2)   # rows/cols of infectious plants (incl. row 0)
    sus = np.argwhere(grid == 0)   # rows/cols of susceptibles
    if inf.size == 0 or sus.size == 0:
        return None, sus, 0.0

    ir = inf[:, 0][:, None]
    ic = inf[:, 1][:, None]
    sr = sus[:, 0][None, :]
    sc = sus[:, 1][None, :]

    # 0.5m grid spacing -> halve the index differences before taking distance.
    d2 = (0.5 * (ir - sr)) ** 2 + (0.5 * (ic - sc)) ** 2
    kernel = alpha / (2 * np.pi * (d2 + alpha ** 2) ** 1.5)

    # Infectives in row 0 are the *external* source and use the external rate.
    beta_used = np.where(ir > 0, beta, betap)  # shape (n_inf, 1)
    weights = beta_used * kernel               # shape (n_inf, n_sus)

    per_sus = weights.sum(axis=0)
    return per_sus, sus, float(per_sus.sum())


def _choose_event(grid, per_sus, sus, phi, gamma, scale, rng):
    """Pick and apply the next event, matching the notebook's probabilities."""
    n_exposed = int((grid == 1).sum())
    if rng.random() < gamma * n_exposed / scale:
        # E -> I: a uniformly-chosen asymptomatic plant becomes symptomatic.
        exposed = np.argwhere(grid == 1)
        pick = rng.integers(0, len(exposed))
        grid[exposed[pick, 0], exposed[pick, 1]] = 2
    else:
        # S -> E: a susceptible is infected with probability proportional to
        # the force of infection acting on it.
        threshold = rng.random() * phi
        cum = np.cumsum(per_sus)
        j = int(np.searchsorted(cum, threshold, side="right"))
        if j >= len(sus):
            j = len(sus) - 1
        grid[sus[j, 0], sus[j, 1]] = 1


def _apply_control(grid, radius, replace, empty_baseline):
    """Survey the field, remove symptomatic plants (and a neighbourhood).

    ``radius == 1`` removes only the symptomatic plants themselves; a larger
    radius also clears the surrounding diamond of sites. ``replace`` decides
    whether cleared sites become fresh susceptibles (0) or are left empty (4).
    The exact index arithmetic (including the small left/right asymmetry) is
    preserved from the notebook. Returns the number of plants removed.
    """
    choice = np.where(grid[1:, :] == 2)   # symptomatic plants, excluding row 0
    rows, cols = choice[0], choice[1]

    if replace:
        grid[rows + 1, cols] = 0
        for f in range(len(rows)):
            sx = rows[f] + 1
            sy = cols[f]
            for r in range(1, radius):
                for k in range(radius - r):
                    if sx > r:
                        if sy < WIDTH - k and grid[sx - r, sy + k] < 4:
                            grid[sx - r, sy + k] = 0
                        if sy > k - 1 and grid[sx - r, sy - k] < 4:
                            grid[sx - r, sy - k] = 0
                    if sx < LENGTH - 1 - r:
                        if sy < WIDTH - k and grid[sx + r, sy + k] < 4:
                            grid[sx + r, sy + k] = 0
                        if sy > k - 1 and grid[sx + r, sy - k] < 4:
                            grid[sx + r, sy - k] = 0
                    if sy > r and grid[sx, sy - r] < 4:
                        grid[sx, sy - r] = 0
                    if sy < WIDTH - r and grid[sx, sy + r] < 4:
                        grid[sx, sy + r] = 0
    else:
        grid[rows + 1, cols] = 4
        for f in range(len(rows)):
            sx = rows[f] + 1
            sy = cols[f]
            for r in range(1, radius):
                for k in range(radius - r):
                    if sx > r:
                        if sy < WIDTH - k:
                            grid[sx - r, sy + k] = 4
                        if sy > k - 1:
                            grid[sx - r, sy - k] = 4
                    if sx < LENGTH - 1 - r:
                        if sy < WIDTH - k:
                            grid[sx + r, sy + k] = 4
                        if sy > k - 1:
                            grid[sx + r, sy - k] = 4
                    if sy > r - 1:
                        grid[sx, sy - r] = 4
                    if sy < WIDTH - r:
                        grid[sx, sy + r] = 4

    return int((grid == 4).sum()) - empty_baseline


def _run_to_control(grid, control, p, rng):
    """Simulate from planting up to the control month; return time-series lists."""
    tsteps = [0.0]
    infecteds = [WIDTH]
    exposeds = [0]
    current_t = 3.0001  # nothing can happen before month 3

    while current_t < control:
        marker = 0
        tnow = current_t % 12

        # No asymptomatics and it's winter -> jump forward, nothing happens.
        if (grid == 1).sum() == 0 and (tnow < TR_START or tnow > TR_END):
            current_t = 15.001 if current_t < 15 else 27.001
            tnow = current_t % 12

        beta, betap, gamma = _set_params(tnow, p.beta0, p.betap0, p.gamma0)
        per_sus, sus, phi = _dispersal(grid, beta, betap, p.alpha)
        n_exposed = int((grid == 1).sum())
        scale = gamma * n_exposed + phi

        if scale != 0:
            dt = -np.log(rng.random()) / scale
        else:
            dt = 1
            current_t = current_t + dt
            marker = 1

        endrun = False
        for cut in _SEASON_CUTS:
            if current_t < cut and current_t + dt > cut:
                current_t = cut + 0.01
                tsteps.append(cut + 0.01)
                infecteds.append(int((grid == 2).sum()))
                exposeds.append(int((grid == 1).sum()))
                marker = 1
                break
            elif current_t < control and current_t + dt > control:
                endrun = True
                break
        if endrun:
            break

        if marker == 0:
            _choose_event(grid, per_sus, sus, phi, gamma, scale, rng)
            tsteps.append(dt + current_t)
            current_t = tsteps[-1]
            infecteds.append(int((grid == 2).sum()))
            exposeds.append(int((grid == 1).sum()))

    return tsteps, infecteds, exposeds


def _run_after_control(grid, current_t, tsteps, infecteds, exposeds, p, rng):
    """Continue the simulation from the control month to the end of the run."""
    while current_t < TRUN:
        marker = 0
        tnow = current_t % 12

        beta, betap, gamma = _set_params(tnow, p.beta0, p.betap0, p.gamma0)
        per_sus, sus, phi = _dispersal(grid, beta, betap, p.alpha)
        n_exposed = int((grid == 1).sum())
        scale = gamma * n_exposed + phi

        if scale != 0:
            dt = -np.log(rng.random()) / scale
        else:
            dt = 1
            current_t = current_t + dt
            marker = 1

        endrun = False
        for cut in _SEASON_CUTS:
            if current_t < cut and current_t + dt > cut:
                current_t = cut + 0.01
                tsteps.append(cut + 0.01)
                infecteds.append(int((grid == 2).sum()))
                exposeds.append(int((grid == 1).sum()))
                marker = 1
                break
            elif current_t > 27 and current_t + dt > TRUN:
                tsteps.append(TRUN)
                infecteds.append(int((grid == 2).sum()))
                exposeds.append(int((grid == 1).sum()))
                endrun = True
                break
        if endrun:
            break

        if marker == 0:
            _choose_event(grid, per_sus, sus, phi, gamma, scale, rng)
            tsteps.append(dt + current_t)
            current_t = tsteps[-1]
            infecteds.append(int((grid == 2).sum()))
            exposeds.append(int((grid == 1).sum()))


def run_simulation(
    p: SimParams,
    rng: Optional[np.random.Generator] = None,
    progress: Optional[Callable[[int, str], None]] = None,
) -> dict:
    """Run the full model: spread to the control month, then four strategies.

    Parameters
    ----------
    p : SimParams
        The model inputs.
    rng : numpy Generator, optional
        Supply a seeded generator for reproducible runs (defaults to fresh
        entropy, matching the notebook's "different every time" behaviour).
    progress : callable(stage:int, message:str), optional
        Called as the run proceeds so a UI can show a progress bar. ``stage``
        runs 0..5 (up-to-control, then one per control strategy).

    Returns
    -------
    dict with keys:
        ``control``   the control month,
        ``radius_m``  the chosen removal radius,
        ``labels``    the four strategy labels,
        ``series``    list of dicts ``{"t", "symptomatic", "yield_loss"}``,
        ``gridsaves`` (12, LENGTH, WIDTH) field snapshots,
        ``max_symptomatic`` / ``max_yield`` for axis scaling.
    """
    if rng is None:
        rng = np.random.default_rng()
    if progress is None:
        progress = lambda stage, msg: None  # noqa: E731

    control = p.control_month

    # --- shared state up to the control point --------------------------------
    grid = np.zeros((LENGTH, WIDTH))
    grid[0, :] = 2                       # external source of infection along south edge
    if STRIPES:
        grid[1:, 2::5] = 4               # every 5th row left empty
    empty_baseline = int((grid == 4).sum())

    progress(0, "Running model up to control point")
    tsteps_b, infecteds_b, exposeds_b = _run_to_control(grid, control, p, rng)

    gridsaves = np.zeros((12, LENGTH, WIDTH))
    gridsaves[0] = grid.copy()           # snapshot: just before control

    tsteps_b.append(control)
    infecteds_b.append(int((grid == 2).sum()))
    exposeds_b.append(int((grid == 1).sum()))
    n_pre = len(tsteps_b)

    labels = p.control_labels()
    series = []
    max_symptomatic = 0
    max_yield = 0

    # --- four control strategies, each restarting from the saved state -------
    for case in range(4):
        replace = case in (0, 2)
        radius = 1 if case in (0, 1) else 1 + int(2 * p.radius_m)
        progress(case + 1, f"Implementing control {case + 1}: {labels[case]}")

        tsteps = tsteps_b.copy()
        infecteds = infecteds_b.copy()
        exposeds = exposeds_b.copy()
        grid = gridsaves[0].copy()
        current_t = control

        removed = _apply_control(grid, radius, replace, empty_baseline)

        gridsaves[1 + case * 3] = grid.copy()   # snapshot: just after control
        tsteps.append(control)
        infecteds.append(int((grid == 2).sum()))
        exposeds.append(int((grid == 1).sum()))

        _run_after_control(grid, current_t, tsteps, infecteds, exposeds, p, rng)

        gridsaves[2 + case * 3] = grid.copy()   # snapshot: at 30 months

        # Symptomatic in-field = total infectious minus the 30 external row-0 cells.
        symptomatic = [inf - WIDTH for inf in infecteds]
        # "Yield loss" = symptomatic, plus everything removed once control kicks in.
        yield_loss = (
            [inf - WIDTH for inf in infecteds[:n_pre]]
            + [inf - WIDTH + removed for inf in infecteds[n_pre:]]
        )

        max_symptomatic = max(max_symptomatic, max(symptomatic))
        max_yield = max(max_yield, max(yield_loss))

        series.append({"t": tsteps, "symptomatic": symptomatic, "yield_loss": yield_loss})

    return {
        "control": control,
        "radius_m": p.radius_m,
        "labels": labels,
        "series": series,
        "gridsaves": gridsaves,
        "max_symptomatic": max_symptomatic,
        "max_yield": max_yield,
        "width": WIDTH,
        "length": LENGTH,
    }
