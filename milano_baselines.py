"""Milano-task adapters for three baselines lifted from the user's
sensor-network harness:
  * non-uniform ergodic   (radial sweep, α biased by density)
  * optimal-transport     (anchor allocation; not Brenier OT)
  * SAC harvesting        (Soft Actor-Critic trained on a sensor model)

Plus the Zeng-2019 rotary-wing energy model used to compute E_total (kJ).

All generators output (N, 2) cartesian positions in [-1, 1]^2, so they
plug into the existing eval_prefix_stop.py via the canonical npz format.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import gaussian_filter


# ---------------------------------------------------------------------------
# Zeng et al. (IEEE TWC 2019) rotary-wing energy model
# ---------------------------------------------------------------------------
class RotaryWingEnergyModel:
    """P(v) = blade + induced + parasitic ; plus an accel-jump surcharge
    and a hover term, identical to the user's compute_energy_metrics()."""

    def __init__(self, P_0=79.86, P_i=88.63, U_tip=120.0, v_0=4.03,
                 d_0=0.6, rho_air=1.225, s=0.05, A=0.503, mass=3.3,
                 domain_size=8.0, time_step=0.1):
        self.P_0, self.P_i, self.U_tip, self.v_0 = P_0, P_i, U_tip, v_0
        self.parasitic_coeff = 0.5 * d_0 * rho_air * s * A
        self.P_hover = P_0 + P_i
        self.domain_size = domain_size  # metres (1 unit in [-1,1] = domain_size/2 m)
        self.time_step = time_step
        self.mass = mass
        self.accel_coeff = 0.5 * mass
        # max-range velocity (used for OT-jump surcharge)
        vv = np.linspace(1, 40, 1000)
        self.V_mr = float(vv[np.argmin([self.power(v) / v for v in vv])])

    def power(self, v):
        v = np.maximum(np.asarray(v, dtype=float), 0.01)
        P_blade = self.P_0 * (1 + 3 * v**2 / self.U_tip**2)
        ratio = v**2 / (2 * self.v_0**2)
        induced = np.sqrt(np.maximum(1 + v**4 / (4 * self.v_0**4) - ratio, 0.01))
        P_induced = self.P_i * induced
        P_par = self.parasitic_coeff * v**3
        return P_blade + P_induced + P_par

    def energy_for_acceleration(self, dv):
        return self.accel_coeff * dv**2 * 1.5

    def E_total_kJ(self, pos_xy, dt, method_type="ergodic"):
        """Same definition as the user's compute_energy_metrics."""
        pos_xy = np.asarray(pos_xy, dtype=np.float64)
        if pos_xy.shape[0] < 2:
            return 0.0
        dx = np.diff(pos_xy, axis=0)
        # pos_xy is in [-1,1]; multiply by domain_size/2 to get metres.
        # (the user's helper used domain_size only — we follow that convention
        # and treat 1 normalised unit as domain_size metres so numbers are
        # directly comparable to their reported kJ values.)
        step_m = np.linalg.norm(dx, axis=1) * self.domain_size
        v = step_m / dt
        v_clip = np.clip(v, 0.1, 35.0)

        # path-power term (with hover for stationary opt-transport steps)
        if method_type == "optimal_transport":
            hover_mask = v <= 0.5
            E_path = (self.power(v_clip) * dt).sum() * 1.0  # placeholder
            # redo per-step to apply hover where slow
            E_path = 0.0
            for vi, vc in zip(v, v_clip):
                if vi <= 0.5:
                    E_path += self.P_hover * dt
                else:
                    E_path += float(self.power(vc)) * dt
        else:
            E_path = float((self.power(v_clip) * dt).sum())

        # acceleration surcharge for big velocity jumps
        E_accel = 0.0
        if len(v) >= 2:
            dv = np.abs(np.diff(v))
            big = dv > 2.0
            E_accel += float(self.energy_for_acceleration(dv[big]).sum())
            if method_type == "optimal_transport":
                jumps = step_m[step_m > 0.5]
                if jumps.size:
                    cv = np.minimum(self.V_mr * 0.5, jumps / dt)
                    E_accel += float(2 * self.energy_for_acceleration(cv).sum())
            elif method_type in ("ddqn", "sac"):
                step_n = np.linalg.norm(dx, axis=1)
                with np.errstate(invalid="ignore", divide="ignore"):
                    dnx = dx[:, 0] / (step_n + 1e-10)
                    dny = dx[:, 1] / (step_n + 1e-10)
                if len(dnx) > 1:
                    dots = dnx[:-1] * dnx[1:] + dny[:-1] * dny[1:]
                    sharp = int(np.sum(dots < 0.5))
                    tv = min(float(np.mean(v)) * 0.7, self.V_mr * 0.5)
                    E_accel += sharp * 2 * self.energy_for_acceleration(tv)

        # hover term for near-zero steps
        E_hover = 0.0
        if method_type in ("optimal_transport", "ddqn", "sac"):
            E_hover = float(np.sum(step_m < 0.05) * self.P_hover * dt)

        return (E_path + E_accel + E_hover) / 1000.0  # kJ


# ---------------------------------------------------------------------------
# Milano density viewed as a polar rate-map on the unit disc
# ---------------------------------------------------------------------------
class MilanoRateMap:
    """Smoothed Milano grid as a DataRateMap-style object (rho=1, delta=0).

    The unit disc is contained in [-1,1]^2 so rate_cartesian works
    everywhere a polar generator can reach. Density outside the disc is
    just unreachable for these generators — fourier_erg / corr will
    naturally penalise that.
    """
    rho = 1.0
    delta = 0.0

    def __init__(self, grid_path, smooth_sigma=1.5, eps_floor=1e-12,
                 xlim=(-1.0, 1.0), ylim=(-1.0, 1.0),
                 nfz_discs=None,
                 nfz_center=None, nfz_radius=None):
        from target_density import load_milano_target
        t = load_milano_target(grid_path, smooth_sigma=smooth_sigma,
                               eps_floor=eps_floor, xlim=xlim, ylim=ylim,
                               nfz_discs=nfz_discs,
                               nfz_center=nfz_center, nfz_radius=nfz_radius)
        g = t["grid"]
        self.xlim, self.ylim = xlim, ylim
        self.nfz_discs = nfz_discs
        self.nfz_center = nfz_center
        self.nfz_radius = nfz_radius
        h, w = g.shape
        xs = t["xs"]
        ys = t["ys"]
        cell_area = ((xlim[1] - xlim[0]) / max(w - 1, 1)
                     * (ylim[1] - ylim[0]) / max(h - 1, 1))
        mass = g / g.sum()
        # interpolant returns *density* (mass per unit area)
        self._interp = RegularGridInterpolator(
            (ys, xs), mass / max(cell_area, 1e-16),
            bounds_error=False, fill_value=0.0, method="linear")
        self._rate_lut = None
        self._r_grid = None
        self._theta_grid = None

    def rate_cartesian(self, x, y):
        if x * x + y * y > self.rho * self.rho:
            return 0.0
        return float(np.asarray(self._interp([[y, x]])).reshape(-1)[0])

    def rate(self, r, theta):
        if r > self.rho or r < self.delta:
            return 0.0
        return self.rate_cartesian(r * np.cos(theta), r * np.sin(theta))

    def build_rate_lut(self, n_r=100, n_theta=200):
        if self._rate_lut is not None:
            return
        rr = np.linspace(self.delta, self.rho, n_r)
        tt = np.linspace(0, 2 * np.pi, n_theta)
        R, T = np.meshgrid(rr, tt, indexing="ij")
        X = R * np.cos(T)
        Y = R * np.sin(T)
        pts = np.column_stack([Y.ravel(), X.ravel()])
        rates = np.maximum(self._interp(pts), 0.0).reshape(R.shape)
        # normalise so ∫∫ rate(r,θ) r dr dθ over disc = 1
        dr = rr[1] - rr[0]
        dtheta = tt[1] - tt[0]
        norm = float((rates * R * dr * dtheta).sum())
        if norm > 0:
            rates = rates / norm
        self._rate_lut = rates
        self._r_grid = rr
        self._theta_grid = tt
        self._lut_norm = norm

    def normalized_rate(self, r, theta):
        if self._rate_lut is None:
            self.build_rate_lut()
        n_r = len(self._r_grid)
        n_t = len(self._theta_grid)
        dr = self._r_grid[1] - self._r_grid[0]
        dt = self._theta_grid[1] - self._theta_grid[0]
        theta = theta % (2 * np.pi)
        i_r = int(np.clip((r - self._r_grid[0]) / dr, 0, n_r - 2))
        i_t = int(np.clip(theta / dt, 0, n_t - 2))
        fr = (r - self._r_grid[i_r]) / dr
        ft = (theta - self._theta_grid[i_t]) / dt
        L = self._rate_lut
        return float(
            (1 - fr) * (1 - ft) * L[i_r, i_t]
            + (1 - fr) * ft * L[i_r, i_t + 1]
            + fr * (1 - ft) * L[i_r + 1, i_t]
            + fr * ft * L[i_r + 1, i_t + 1])


# ---------------------------------------------------------------------------
# Non-Uniform Ergodic
# ---------------------------------------------------------------------------
def _build_alpha_pdf_and_H(rate_lut, r_grid, theta_grid, alpha_grid,
                           s_grid, rho, delta, phi_max):
    """Vectorised version of the user's numba kernel.
    Returns alpha_pdf (n_alpha,) and H_lut (n_s, n_alpha)."""
    n_alpha = len(alpha_grid)
    n_s = len(s_grid)
    n_r, n_theta = rate_lut.shape
    ds = s_grid[1] - s_grid[0]
    dr = r_grid[1] - r_grid[0]
    dtheta = theta_grid[1] - theta_grid[0]

    s_arr = s_grid[None, :]            # (1, n_s)
    alpha_arr = alpha_grid[:, None]    # (n_alpha, 1)
    r = (rho - delta) * s_arr**2 + delta  # G(s) quadratic, (1, n_s)
    theta = (phi_max * s_arr + alpha_arr) % (2 * np.pi)  # (n_alpha, n_s)
    r_b = np.broadcast_to(r, theta.shape)

    i_r = np.clip(((r_b - r_grid[0]) / dr).astype(int), 0, n_r - 2)
    i_t = np.clip((theta / dtheta).astype(int), 0, n_theta - 2)
    fr = (r_b - r_grid[i_r]) / dr
    ft = (theta - theta_grid[i_t]) / dtheta
    L = rate_lut
    val = ((1 - fr) * (1 - ft) * L[i_r, i_t]
           + (1 - fr) * ft * L[i_r, i_t + 1]
           + fr * (1 - ft) * L[i_r + 1, i_t]
           + fr * ft * L[i_r + 1, i_t + 1])
    # Jacobian: dr/ds = 2(rho-delta)*s for G(s) = (rho-delta)*s^2 + delta
    integrand = val * r_b * 2.0 * (rho - delta) * s_arr  # (n_alpha, n_s)

    f_A = np.trapezoid(integrand, dx=ds, axis=1) if hasattr(np, "trapezoid") \
        else np.trapz(integrand, dx=ds, axis=1)
    alpha_pdf = f_A.copy()

    # H_lut[i_s, alpha] = (1 / f_A) * trapz_0^s(integrand)
    seg = 0.5 * (integrand[:, :-1] + integrand[:, 1:]) * ds  # (n_alpha, n_s-1)
    cum = np.zeros_like(integrand)
    cum[:, 1:] = np.cumsum(seg, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        H = np.where(f_A[:, None] > 1e-10, cum / f_A[:, None],
                     np.broadcast_to(s_grid[None, :], cum.shape))
    return alpha_pdf, H.T  # transpose to (n_s, n_alpha) like upstream


def _inv_cdf_sample(u, cdf, x_grid):
    idx = int(np.searchsorted(cdf, u, side="right"))
    idx = max(min(idx, len(cdf) - 1), 1)
    denom = cdf[idx] - cdf[idx - 1]
    if abs(denom) < 1e-10:
        return float(x_grid[idx])
    return float(x_grid[idx - 1] + (u - cdf[idx - 1]) / denom
                 * (x_grid[idx] - x_grid[idx - 1]))


def _binsearch_inv(y, H_col, s_grid):
    idx = int(np.searchsorted(H_col, y, side="right"))
    idx = max(min(idx, len(H_col) - 1), 1)
    denom = H_col[idx] - H_col[idx - 1]
    if abs(denom) < 1e-10:
        return float(s_grid[idx])
    return float(s_grid[idx - 1] + (y - H_col[idx - 1]) / denom
                 * (s_grid[idx] - s_grid[idx - 1]))


class NonUniformErgodic:
    name = "nonuniform_erg"

    def __init__(self, rate_map, phi_max=np.pi / 5,
                 alpha_res=200, s_res=100):
        self.rate_map = rate_map
        self.rho, self.delta = rate_map.rho, rate_map.delta
        self.phi_max = phi_max
        rate_map.build_rate_lut()
        self._alpha_grid = np.linspace(-0.1 * np.pi, 2 * np.pi, alpha_res)
        self._s_grid = np.linspace(0, 1, s_res)
        self._alpha_pdf, self._H_lut = _build_alpha_pdf_and_H(
            rate_map._rate_lut, rate_map._r_grid, rate_map._theta_grid,
            self._alpha_grid, self._s_grid,
            self.rho, self.delta, self.phi_max)
        # normalise α-pdf and build α-cdf
        norm = (np.trapezoid(self._alpha_pdf, self._alpha_grid)
                if hasattr(np, "trapezoid")
                else np.trapz(self._alpha_pdf, self._alpha_grid))
        if norm > 0:
            self._alpha_pdf = self._alpha_pdf / norm
        dx = self._alpha_grid[1] - self._alpha_grid[0]
        cdf = np.zeros_like(self._alpha_pdf)
        for i in range(1, len(cdf)):
            cdf[i] = cdf[i - 1] + 0.5 * (self._alpha_pdf[i - 1]
                                         + self._alpha_pdf[i]) * dx
        if cdf[-1] > 0:
            cdf = cdf / cdf[-1]
        self._alpha_cdf = cdf

    def generate(self, num_intervals, points_per_interval, seed=None):
        rng = np.random.default_rng(seed)
        n_alpha = len(self._alpha_grid)
        out_r = []
        out_t = []
        for _ in range(num_intervals):
            u = rng.random()
            alpha = _inv_cdf_sample(u, self._alpha_cdf, self._alpha_grid)
            alpha_min = self._alpha_grid[0]
            alpha_span = self._alpha_grid[-1] - alpha_min
            ai = max(min(int((alpha - alpha_min) / alpha_span * (n_alpha - 1)), n_alpha - 1), 0)
            H_col = self._H_lut[:, ai]
            for i in range(points_per_interval):
                s = _binsearch_inv(i / points_per_interval, H_col, self._s_grid)
                out_r.append((self.rho - self.delta) * s**2 + self.delta)
                out_t.append((self.phi_max * s + alpha) % (2 * np.pi))
            for i in range(points_per_interval):
                s = _binsearch_inv(1.0 - i / points_per_interval,
                                   H_col, self._s_grid)
                out_r.append((self.rho - self.delta) * s**2 + self.delta)
                out_t.append((self.phi_max * s + alpha) % (2 * np.pi))
        r = np.asarray(out_r); t = np.asarray(out_t)
        return np.column_stack([r * np.cos(t), r * np.sin(t)])


# ---------------------------------------------------------------------------
# Optimal Transport (anchor-allocation, not Brenier OT)
# ---------------------------------------------------------------------------
class OptimalTransportCoverage:
    name = "opt_transport"

    def __init__(self, rate_map, grid_resolution=15):
        self.rate_map = rate_map
        self.rho = rate_map.rho
        rate_map.build_rate_lut()
        self.uav_positions = np.array([
            [x, y]
            for x in np.linspace(-self.rho, self.rho, grid_resolution)
            for y in np.linspace(-self.rho, self.rho, grid_resolution)
            if x * x + y * y <= self.rho * self.rho])

    def generate(self, num_points, seed=None):
        rng = np.random.default_rng(seed)
        # rejection-sample 1000 target points from the density
        samples = []
        while len(samples) < 1000:
            r = np.sqrt(rng.uniform(0, 1)) * self.rho
            theta = rng.uniform(0, 2 * np.pi)
            if rng.uniform(0, 10) < self.rate_map.normalized_rate(r, theta) * 10:
                samples.append([r * np.cos(theta), r * np.sin(theta)])
        samples = np.array(samples)
        # nearest-anchor assignment
        d = np.linalg.norm(samples[:, None, :] - self.uav_positions[None, :, :],
                           axis=2)
        best = np.argmin(d, axis=1)
        counts = np.bincount(best, minlength=len(self.uav_positions))

        traj = []
        for j, c in enumerate(counts):
            if c == 0:
                continue
            xa, ya = self.uav_positions[j]
            r0 = np.sqrt(xa * xa + ya * ya)
            t0 = np.arctan2(ya, xa) % (2 * np.pi)
            visits = max(1, int(c / len(samples) * num_points))
            for _ in range(visits):
                rj = np.clip(r0 + rng.uniform(-0.02, 0.02),
                             0.01, self.rho - 0.01)
                tj = (t0 + rng.uniform(-0.05, 0.05)) % (2 * np.pi)
                traj.append([rj * np.cos(tj), rj * np.sin(tj)])
        # pad / truncate
        while len(traj) < num_points:
            n = min(1000, num_points - len(traj))
            traj.extend(traj[:n] if traj else [[0.5, 0.0]])
        return np.asarray(traj[:num_points])


# ---------------------------------------------------------------------------
# SAC harvesting
# ---------------------------------------------------------------------------
def _channel_rate(d2, alt, alpha_los=2.27, alpha_nlos=3.64):
    if d2 < 1e-6:
        d2 = 0.01
    d3 = np.sqrt(d2 * d2 + alt * alt)
    elev = np.arctan(alt / d2) * 180 / np.pi
    p_los = 1.0 / (1.0 + 9.61 * np.exp(-0.16 * (elev - 9.61)))
    ep = p_los * d3 ** (-alpha_los) + (1 - p_los) * d3 ** (-alpha_nlos)
    return max(np.log2(1 + 100 * ep), 0.01)


def _make_sac_env_class():
    """Defer the gym import so the module is importable without SB3."""
    import gymnasium as gym
    from gymnasium import spaces

    class SACUAVEnv(gym.Env):
        metadata = {"render_modes": []}
        TOP_K = 10
        MAX_STEP = 3
        GAMMA_S = 0.99

        def __init__(self, rate_map, grid_size=50, max_steps=10000,
                     altitude=0.1, comm_range=0.18, n_devices=25,
                     buf_cap=20.0, vel_penalty=0.0):
            super().__init__()
            self.rate_map = rate_map
            self.grid_size = grid_size
            self.max_steps = max_steps
            self.altitude = altitude
            self.comm_range = comm_range
            self.rho = rate_map.rho
            self.n_devices_target = n_devices
            self.buf_cap = buf_cap
            # Quadratic action-norm penalty in the reward, in spirit of the
            # OT-CFM +E penalty: discourages always-max-step actions so the
            # policy doesn't trivially fly fast in circles. Set to 0 to
            # recover the original (no kinematic regularisation) behaviour.
            self.vel_penalty = float(vel_penalty)
            self.action_space = spaces.Box(-1.0, 1.0, (2,), dtype=np.float32)
            self.observation_space = spaces.Box(
                -1.0, 1.0, (self.TOP_K * 4 + 5,), dtype=np.float32)
            self._build_valid_mask()
            self._build_sensors()

        def _build_valid_mask(self):
            self.vm = np.zeros((self.grid_size, self.grid_size), dtype=bool)
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    x, y = self._g2c(i, j)
                    if x * x + y * y <= self.rho * self.rho:
                        self.vm[i, j] = True

        def _build_sensors(self):
            rng = np.random.default_rng(0)
            dpos = []; dinit = []
            n = 0; att = 0
            while n < self.n_devices_target and att < 10000:
                i = rng.integers(0, self.grid_size)
                j = rng.integers(0, self.grid_size)
                x, y = self._g2c(i, j)
                if x * x + y * y <= self.rho * self.rho:
                    rate = self.rate_map.rate_cartesian(x, y)
                    if rng.random() < rate * 5:
                        dpos.append([x, y])
                        dinit.append(rate * rng.uniform(5, 25))
                        n += 1
                att += 1
            if n == 0:
                # fallback: put one sensor at the densest grid cell
                dpos.append([0.0, 0.0])
                dinit.append(1.0)
            self.sensor_xy = np.asarray(dpos, dtype=np.float32)
            rates = np.asarray(dinit, dtype=np.float32)
            self.target_freq = rates / (rates.sum() + 1e-6)
            self.nd = len(self.sensor_xy)
            self._buf_init = np.clip(self.target_freq * self.buf_cap * 5,
                                     0.5, self.buf_cap)

        def _g2c(self, i, j):
            return (-self.rho + 2 * self.rho * i / (self.grid_size - 1),
                    -self.rho + 2 * self.rho * j / (self.grid_size - 1))

        def _uav_xy(self):
            return self._g2c(int(self.pos[0]), int(self.pos[1]))

        def _distances(self):
            px, py = self._uav_xy()
            d = self.sensor_xy - np.array([px, py], dtype=np.float32)
            return np.sqrt((d ** 2).sum(axis=1))

        def _density(self):
            px, py = self._uav_xy()
            return float(self.rate_map.rate_cartesian(px, py))

        def _get_obs(self):
            px, py = self._uav_xy()
            dists = self._distances()
            buf_n = self.buf / (self._buf_init + 1e-6)
            under = self.target_freq / (self.visit_freq + 1e-6)
            under = np.clip(under / (under.max() + 1e-6), 0, 1)
            scores = (buf_n + 0.1) * under / (dists + 0.1)
            top_k = min(self.TOP_K, self.nd)
            idx = np.argpartition(scores, -top_k)[-top_k:]
            idx = idx[np.argsort(scores[idx])[::-1]]
            obs = np.zeros(self.TOP_K * 4 + 5, dtype=np.float32)
            for rank, i in enumerate(idx):
                obs[rank * 4 + 0] = (self.sensor_xy[i, 0] - px) / (2 * self.rho)
                obs[rank * 4 + 1] = (self.sensor_xy[i, 1] - py) / (2 * self.rho)
                obs[rank * 4 + 2] = float(np.clip(buf_n[i], 0, 1))
                obs[rank * 4 + 3] = float(np.clip(under[i], 0, 1))
            base = self.TOP_K * 4
            obs[base + 0] = self.pos[0] / (self.grid_size - 1) * 2 - 1
            obs[base + 1] = self.pos[1] / (self.grid_size - 1) * 2 - 1
            obs[base + 2] = float(np.clip(self._density() * 5, -1, 1))
            obs[base + 3] = float(self._last_action[0])
            obs[base + 4] = float(self._last_action[1])
            return np.clip(obs, -1.0, 1.0)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            vp = np.argwhere(self.vm)
            self.pos = vp[np.random.randint(len(vp))].astype(float)
            self.buf = self._buf_init.copy()
            self.steps = 0
            self.traj = [self.pos.copy()]
            self.visit_freq = np.ones(self.nd, dtype=np.float32) / self.nd
            self._last_action = np.zeros(2, dtype=np.float32)
            self._prev_density = self._density()
            self._prev_cell = (int(self.pos[0]), int(self.pos[1]))
            self._stay = 0
            return self._get_obs(), {}

        def step(self, action):
            self._last_action = np.asarray(action, dtype=np.float32)
            vx, vy = float(action[0]), float(action[1])
            dx = int(np.round(np.clip(vx * self.MAX_STEP,
                                       -self.MAX_STEP, self.MAX_STEP)))
            dy = int(np.round(np.clip(vy * self.MAX_STEP,
                                       -self.MAX_STEP, self.MAX_STEP)))
            ni = int(self.pos[0] + dx); nj = int(self.pos[1] + dy)
            moved = False
            if (0 <= ni < self.grid_size and 0 <= nj < self.grid_size
                    and self.vm[ni, nj]):
                self.pos = np.array([ni, nj], dtype=float)
                self.traj.append(self.pos.copy())
                moved = True
            self.buf = np.minimum(self.buf + self.target_freq * 0.15, self.buf_cap)
            dists = self._distances()
            mask = (dists <= self.comm_range) & (self.buf > 0)
            col = 0.0
            if mask.any():
                cidx = np.where(mask)[0]
                rates = np.array([_channel_rate(float(dists[i]), self.altitude)
                                  for i in cidx])
                ca = np.minimum(rates * 0.5, self.buf[mask])
                self.buf[mask] -= ca
                col = float(ca.sum())
                self.visit_freq[mask] += ca
                tot = self.visit_freq.sum()
                if tot > 0:
                    self.visit_freq /= tot
            d = self._density()
            shaping = self.GAMMA_S * d - self._prev_density
            self._prev_density = d
            cell = (int(self.pos[0]), int(self.pos[1]))
            if cell == self._prev_cell:
                self._stay += 1
            else:
                self._stay = 0
            self._prev_cell = cell
            stay_p = 0.02 * max(0, self._stay - 10)
            # Quadratic velocity (action-norm) penalty — matched in spirit
            # to the +E term in OT-CFM. Drives the policy toward smaller
            # actions when they are sufficient, suppressing the L≈18 km
            # path-length pathology of the unregularised reward.
            vel_pen = self.vel_penalty * (vx * vx + vy * vy)
            r = float(np.clip(0.3 * col + 0.5 * shaping - stay_p
                              - 0.001 * float(moved) - vel_pen,
                              -2.0, 5.0))
            self.steps += 1
            done = bool(self.steps >= self.max_steps
                        or float(self.buf.sum())
                        < 0.01 * float(self._buf_init.sum()))
            return self._get_obs(), r, done, False, {}

        def get_cartesian(self):
            out = []
            for p in self.traj:
                x, y = self._g2c(int(p[0]), int(p[1]))
                if x * x + y * y <= self.rho * self.rho:
                    out.append([x, y])
            return np.asarray(out)

    return SACUAVEnv


class SACTrajectoryGenerator:
    name = "sac"

    def __init__(self, rate_map, grid_size=50, comm_range=0.18,
                 train_steps=200_000, vel_penalty=0.0):
        self.rate_map = rate_map
        self.grid_size = grid_size
        self.comm_range = comm_range
        self.train_steps = train_steps
        self.vel_penalty = float(vel_penalty)
        self.model = None
        self._EnvCls = _make_sac_env_class()

    def _make_env(self):
        return self._EnvCls(self.rate_map, self.grid_size,
                            max_steps=10_000, comm_range=self.comm_range,
                            vel_penalty=self.vel_penalty)

    def train(self, seed=42):
        from stable_baselines3 import SAC
        from stable_baselines3.common.env_checker import check_env
        env = self._make_env()
        check_env(env, warn=True, skip_render_check=True)
        self.model = SAC(
            "MlpPolicy", env, verbose=0, seed=seed,
            device="cpu", learning_rate=3e-4, batch_size=256,
            buffer_size=200_000, learning_starts=2_000,
            train_freq=1, gradient_steps=1, ent_coef="auto",
            policy_kwargs=dict(net_arch=[256, 256]))
        self.model.learn(total_timesteps=self.train_steps)

    def generate(self, num_points, seed=None):
        if self.model is None:
            raise RuntimeError("Call .train() before .generate()")
        traj = []
        s = seed
        while len(traj) < num_points:
            env = self._EnvCls(self.rate_map, self.grid_size,
                               max_steps=2000, comm_range=self.comm_range)
            obs, _ = env.reset(seed=s)
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _, _ = env.step(action)
            traj.extend(env.get_cartesian().tolist())
            s = (s + 1) if s is not None else None
        return np.asarray(traj[:num_points])
