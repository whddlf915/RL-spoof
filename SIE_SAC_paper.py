"""
SIE-SAC Paper-Accurate Implementation
================================================================================
Based on: "A Reinforcement-Learning-Enhanced Spoofing Algorithm for UAV with
          GPS/INS-Integrated Navigation"
IEEE Transactions on Aerospace and Electronic Systems (TAES), 2025

Paper Equations Implemented:
-----------------------------
- Eq.29: State space s = [s1, s2, s3, s4]
         s1 = [d, θ, ψ] - position in spherical coordinates
         s2 = [Δ_D, Δ^s_D] - triangle angles (deception geometry)
         s3 = γ^s - predicted NIS
         s4 = [|v|, θ_v] - velocity info

- Eq.30: Deceptive position x^s = x^e + Δx^s
         Where x^e = radar estimate, Δx^s = spoofing offset

- Eq.31-37: Spatial Information Entropy (SIE) calculation
         Eq.32: K(ρ_t) = λ / [2π·∫₀^l ρ·e^(-λρ)dρ + ρ_t]
         Eq.33: H_SI = 2πK·[(3-logK)·κ(l) - l³e^(-λl)]
         Eq.35: κ(l) = (1/λ³)·[1 - e^(-λl)·((λl+1)² + 1)]
         Eq.37: H^s = ω₁·H^s1_SI + ω₂·H^s2_SI

- Eq.38-39: Reward function R = α₁·r_x + α₂·r_v + α₃·r_γ
         Eq.39a: r_x - position reward (distance to fake dest)
         Eq.39b: r_v - velocity direction reward
         Eq.39c: r_γ - concealment reward (NIS constraint)

- Eq.40-48: SIE-SAC algorithm (replaces entropy with SIE)
         Eq.43: Q-target = r + γ·(min Q' + α·H^s)
         Eq.46: Policy loss = -E[Q + α·H^s]
         Eq.48: Alpha loss = α·(H^s - H₀)

Key Implementation Notes:
-------------------------
1. SIE replaces standard entropy in SAC (Eq.40-48)
2. SIE is computed on deceptive position x^s = x^e + Δx^s (Eq.30)
3. Spoofing is based on radar KF estimate (blind spoofer scenario)
4. Reward is ONLY r_x + r_v + r_gamma (Eq.38), NO SIE bonus
5. All SIE computations are differentiable for policy gradient
================================================================================
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from typing import Tuple, Dict, Optional, List
import gymnasium as gym
from gymnasium import spaces
import time


# ==================== Torch SIE Calculator (Differentiable) ====================

class TorchSIECalculator(nn.Module):
    """
    Differentiable Spatial Information Entropy (SIE) Calculator using PyTorch.

    Implements Paper Equations 31-37:
    ================================

    Eq.31: p(ρ|ρ_t) = K(ρ_t) · ρ · e^(-λρ)  for 0 ≤ ρ ≤ l
           where K(ρ_t) is normalization factor

    Eq.32: K(ρ_t) = λ / [2π · ∫₀^l ρ·e^(-λρ)dρ + ρ_t]
           Integral evaluates to: (1/λ²)(1-e^(-λl)) - (l/λ)e^(-λl)

    Eq.33: H_SI(ρ_t) = 2π·K(ρ_t)·[(3-log K(ρ_t))·κ(l) - l³·e^(-λl)]
           Spatial Information Entropy formula

    Eq.35: κ(l) = (1/λ³)·[1 - e^(-λl)·((λl+1)² + 1)]
           Pre-computed constant for efficiency

    Eq.37: H^s = ω₁·H^s1_SI + ω₂·H^s2_SI
           Combined SIE from fake and true destination distances
           ω₁ = 0.8, ω₂ = 0.2 (Paper Table I)

    This allows gradients to flow from SIE through the policy network,
    enabling the policy to learn to maximize SIE (exploration).
    """

    def __init__(self,
                 lambda_sie: float = 0.01,
                 rho_e: float = 1200.0,
                 omega_1: float = 0.8,
                 fake_dest: torch.Tensor = None,
                 true_dest: torch.Tensor = None,
                 device: str = 'cpu'):
        super().__init__()

        self.device = device
        self.lambda_sie = lambda_sie
        self.rho_e = rho_e
        self.omega_1 = omega_1
        self.omega_2 = 1.0 - omega_1

        # Register destinations as buffers (not parameters)
        if fake_dest is None:
            fake_dest = torch.tensor([800.0, -100.0, -20.0])
        if true_dest is None:
            true_dest = torch.tensor([800.0, 0.0, -20.0])

        self.register_buffer('fake_dest', fake_dest.float())
        self.register_buffer('true_dest', true_dest.float())

        # Pre-compute constants
        self._precompute_constants()

    def _precompute_constants(self):
        """
        Pre-compute κ(l) and K(ρ_t) denominator from Paper Eq.32, Eq.35.

        ┌─────────────────────────────────────────────────────────────────┐
        │ Eq.32: K(ρ_t) = λ / [2π · ∫₀^l ρ·e^(-λρ)dρ + ρ_t]              │
        │                                                                  │
        │ The integral ∫₀^l ρ·e^(-λρ)dρ evaluates to:                     │
        │   = (1/λ²)(1 - e^(-λl)) - (l/λ)·e^(-λl)                         │
        │                                                                  │
        │ We pre-compute: denom_base = 2π · [integral result]             │
        │ Then: K(ρ_t) = λ / (denom_base + ρ_t)                           │
        └─────────────────────────────────────────────────────────────────┘

        ┌─────────────────────────────────────────────────────────────────┐
        │ Eq.35: κ(l) = (1/λ³) · [1 - e^(-λl) · ((λl+1)² + 1)]           │
        │                                                                  │
        │ This is used in Eq.33 for H_SI calculation.                     │
        └─────────────────────────────────────────────────────────────────┘
        """
        l = self.rho_e
        lam = self.lambda_sie

        # Eq.35: κ(l) = (1/λ³) · [1 - e^(-λl) · ((λl+1)² + 1)]
        term_exp = np.exp(-lam * l)
        term_poly = ((lam * l + 1)**2) + 1
        self.kappa_l = (lam**-3) * (1 - term_exp * term_poly)

        # l³·e^(-λl) term for Eq.33
        self.l_cubed_exp = (l**3) * np.exp(-lam * l)

        # Eq.32: Denominator base for K(ρ_t)
        # ∫₀^l ρ·e^(-λρ)dρ = (1/λ²)(1 - e^(-λl)) - (l/λ)·e^(-λl)
        integral_result = (1/(lam**2)) * (1 - np.exp(-lam * l)) - (l/lam) * np.exp(-lam * l)
        self.denom_base = 2 * np.pi * integral_result

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined SIE for batch of positions.

        ┌─────────────────────────────────────────────────────────────────┐
        │ Eq.37: H^s = ω₁·H^s1_SI + ω₂·H^s2_SI                           │
        │                                                                  │
        │ Where:                                                           │
        │   - H^s1_SI: SIE based on distance to fake destination (ρ_t1)   │
        │   - H^s2_SI: SIE based on distance to true destination (ρ_t2)   │
        │   - ω₁ = 0.8, ω₂ = 0.2 (Paper Table I)                          │
        └─────────────────────────────────────────────────────────────────┘

        Args:
            positions: (batch, 3) deceptive positions x^s

        Returns:
            (batch,) SIE values (differentiable)
        """
        # ρ_t1 = ||x^s - x^s_D||: distance to fake destination
        rho_t1 = torch.norm(positions - self.fake_dest, dim=1)
        # ρ_t2 = ||x^s - x_D||: distance to true destination
        rho_t2 = torch.norm(positions - self.true_dest, dim=1)

        # Eq.33: H_SI for each destination
        H_s1 = self._H_SI(rho_t1)  # H^s1_SI
        H_s2 = self._H_SI(rho_t2)  # H^s2_SI

        # Eq.37: Combined SIE
        return self.omega_1 * H_s1 + self.omega_2 * H_s2

    def _K(self, rho_t: torch.Tensor) -> torch.Tensor:
        """
        Eq.32: Normalization factor K(ρ_t) = λ / [2π·∫₀^l ρ·e^(-λρ)dρ + ρ_t]

        Pre-computed: denom_base = 2π·[integral]
        So: K(ρ_t) = λ / (denom_base + ρ_t)
        """
        denom = self.denom_base + rho_t
        return self.lambda_sie / (denom + 1e-9)

    def _H_SI(self, rho_t: torch.Tensor) -> torch.Tensor:
        """
        Eq.33: H_SI(ρ_t) = 2π·K(ρ_t)·[(3 - log K(ρ_t))·κ(l) - l³·e^(-λl)]

        This is the core Spatial Information Entropy formula.
        """
        K_val = self._K(rho_t)                              # Eq.32
        log_K = torch.log(K_val + 1e-12)
        bracket = (3 - log_K) * self.kappa_l - self.l_cubed_exp  # Eq.35 used here
        return 2 * np.pi * K_val * bracket                  # Eq.33


# ==================== Batched Kalman Filter (Vectorized) ====================

class BatchedKalmanFilter:
    """
    Batched Kalman Filter for N parallel state estimations.
    Fully vectorized using NumPy.
    """

    def __init__(self, n_filters: int, dt: float = 0.1,
                 Q_scale: float = 1.0, R_scale: float = 5.0):
        self.n_filters = n_filters
        self.dt = dt
        self.n_states = 9
        self.n_meas = 3

        # State vectors: (N, 9)
        self.x = np.zeros((n_filters, self.n_states), dtype=np.float64)

        # State covariances: (N, 9, 9)
        self.P = np.tile(np.eye(self.n_states) * 100.0, (n_filters, 1, 1))

        # State transition matrix
        self.F = np.eye(self.n_states, dtype=np.float64)
        for i in range(3):
            self.F[i, i+3] = dt
            self.F[i, i+6] = 0.5*dt*dt
            self.F[i+3, i+6] = dt

        # Measurement matrix
        self.H = np.zeros((self.n_meas, self.n_states), dtype=np.float64)
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = 1

        # Noise covariances
        self.Q = np.eye(self.n_states, dtype=np.float64) * Q_scale
        self.R = np.eye(self.n_meas, dtype=np.float64) * R_scale

        # Innovation covariances
        self.S = np.tile(self.R.copy(), (n_filters, 1, 1))

    def reset(self, indices: np.ndarray = None, positions: np.ndarray = None):
        if indices is None:
            indices = np.arange(self.n_filters)
        self.x[indices] = 0.0
        self.P[indices] = np.eye(self.n_states) * 100.0
        if positions is not None:
            self.x[indices, 0:3] = positions

    def predict(self) -> np.ndarray:
        self.x = np.einsum('ij,nj->ni', self.F, self.x)
        FP = np.einsum('ij,njk->nik', self.F, self.P)
        self.P = np.einsum('nij,kj->nik', FP, self.F) + self.Q
        return self.x.copy()

    def update(self, z_meas: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        Hx = np.einsum('ij,nj->ni', self.H, self.x)
        y = z_meas - Hx

        HP = np.einsum('ij,njk->nik', self.H, self.P)
        self.S = np.einsum('nij,kj->nik', HP, self.H) + self.R

        try:
            S_inv = np.linalg.inv(self.S)
        except:
            S_inv = np.tile(np.eye(self.n_meas), (self.n_filters, 1, 1))

        PHT = np.einsum('nij,kj->nik', self.P, self.H)
        K = np.einsum('nij,njk->nik', PHT, S_inv)

        Ky = np.einsum('nij,nj->ni', K, y)
        self.x = self.x + Ky

        I = np.eye(self.n_states)
        KH = np.einsum('nij,jk->nik', K, self.H)
        I_KH = I - KH
        I_KH_P = np.einsum('nij,njk->nik', I_KH, self.P)
        term1 = np.einsum('nij,nkj->nik', I_KH_P, I_KH)
        KR = np.einsum('nij,jk->nik', K, self.R)
        term2 = np.einsum('nij,nkj->nik', KR, K)
        self.P = term1 + term2

        S_inv_y = np.einsum('nij,nj->ni', S_inv, y)
        nis_values = np.sum(y * S_inv_y, axis=1)

        return self.x.copy(), nis_values

    def get_position_estimates(self) -> np.ndarray:
        return self.x[:, 0:3].copy()

    def get_velocity_estimates(self) -> np.ndarray:
        """Get velocity estimates from KF state [vx, vy, vz]"""
        return self.x[:, 3:6].copy()

    def get_innovation_covariances(self) -> np.ndarray:
        return self.S.copy()

    def get_state_covariances(self) -> np.ndarray:
        """Get filtered state estimation error covariance (position components only)"""
        # Return position covariance (3x3) from full state covariance (9x9)
        return self.P[:, 0:3, 0:3].copy()


# ==================== Vectorized Environment (Paper-Accurate) ====================

class VectorizedSIEEnvPaper:
    """
    Vectorized SIE-SAC Environment following the paper exactly.

    Paper References:
    ================
    - Eq.29: State space s = [s1, s2, s3, s4]
    - Eq.30: Deceptive position x^s = x^e + Δx^s
    - Eq.38-39: Reward R = α₁·r_x + α₂·r_v + α₃·r_γ

    Key Implementation Details:
    ---------------------------
    1. Spoofing based on radar KF estimate (x^e), not true_pos (blind spoofer)
    2. Reward is ONLY r_x + r_v + r_gamma (Eq.38), NO SIE bonus
    3. Returns radar estimate x^e for SIE calculation during training
    4. Triangle angles (s2) capture deception geometry per Eq.29
    """

    def __init__(self, n_envs: int, config: Optional[Dict] = None):
        self.n_envs = n_envs
        config = config or {}

        # Environment parameters
        self.true_dest = np.array(config.get('true_dest', [800.0, 0.0, -20.0]))
        self.fake_dest = np.array(config.get('fake_dest', [800.0, -100.0, -20.0]))
        self.dt = config.get('dt', 0.1)
        self.rho_e = config.get('rho_e', 1200.0)  # Paper Table I: 1200m

        # Reward weights (Eq. 38)
        #1,0.5,1
        #3,2,0.3
        self.alpha_1 = config.get('alpha_1', 3.0)
        self.alpha_2 = config.get('alpha_2', 2.0)
        self.alpha_3 = config.get('alpha_3', 0.3)

        # Constraints
        self.chi_sq_threshold = config.get('chi_sq_threshold', 7.815)
        self.delta_gamma_threshold = config.get('delta_gamma_threshold', 2.0)
        self.rho_s_max = config.get('rho_s_max', 200.0)  # Paper Table I: 200m
        self.max_steps = config.get('max_steps', 1000)

        # Spaces
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(
            low=np.array([0.0, -np.pi, -np.pi/2]),
            high=np.array([self.rho_s_max, np.pi, np.pi/2]),
            shape=(3,), dtype=np.float32
        )

        # State arrays
        self.true_pos = None
        self.true_vel = None
        self.radar_kf = None  # Attacker's radar KF (estimates true position)
        self.nav_kf = None    # Drone's internal KF (processes spoofed signals)

        self.step_counts = np.zeros(n_envs, dtype=np.int32)
        self.prev_gamma_s = np.zeros(n_envs, dtype=np.float64)
        self.initial_dist_to_fake = np.zeros(n_envs, dtype=np.float64)

        self._rng = np.random.default_rng()

        # ==================== Paper-Accurate UAV Dynamics (Eq.2, Eq.6, Eq.50) ====================
        # Paper Table I: UAV control parameters
        self.umax = config.get('umax', 0.3)  # Maximum acceleration (m/s²)
        self.vmax = config.get('vmax', 16.0)  # Maximum velocity (m/s)

        # State transition matrix A (Eq.2) - 6x6 for [x, y, z, vx, vy, vz]
        # zk = A·zk-1 + B·uk-1 + wk-1
        self.A = np.eye(6, dtype=np.float64)
        self.A[0, 3] = self.dt  # x += vx*dt
        self.A[1, 4] = self.dt  # y += vy*dt
        self.A[2, 5] = self.dt  # z += vz*dt

        # Input control matrix B (Eq.2) - 6x3 for [ax, ay, az]
        self.B = np.zeros((6, 3), dtype=np.float64)
        self.B[3, 0] = self.dt  # vx += ax*dt
        self.B[4, 1] = self.dt  # vy += ay*dt
        self.B[5, 2] = self.dt  # vz += az*dt

        # UAV flight control gain matrix L (Eq.6, Paper Table I) - 3x6
        # uk = -L·(ẑk - zk) where zk = [x, y, z, vx, vy, vz]
        # L = [0.2  0   0   1 0 0]
        #     [0   0.2  0   0 1 0]
        #     [0    0  0.2  0 0 1]
        self.L = np.zeros((3, 6), dtype=np.float64)
        self.L[0, 0] = 0.2  # Position gain (x)
        self.L[1, 1] = 0.2  # Position gain (y)
        self.L[2, 2] = 0.2  # Position gain (z)
        self.L[0, 3] = 1.0  # Velocity gain (vx)
        self.L[1, 4] = 1.0  # Velocity gain (vy)
        self.L[2, 5] = 1.0  # Velocity gain (vz)

        # Reference trajectory state (will be updated in _update_drone_physics)
        self.reference_state = None  # (n_envs, 6)
        self.ref_phase = None  # 0: accel, 1: cruise, 2: decel

    def reset(self, seed: int = None, indices: np.ndarray = None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        if indices is None:
            indices = np.arange(self.n_envs)
            full_reset = True
        else:
            full_reset = False

        n_reset = len(indices)
        start_pos = np.array([0.0, 0.0, -20.0]) + self._rng.uniform(-10, 10, (n_reset, 3))

        if full_reset:
            self.true_pos = start_pos.copy()
            self.true_vel = np.zeros((self.n_envs, 3), dtype=np.float64)

            # Initialize reference trajectory state (Eq.50)
            self.reference_state = np.zeros((self.n_envs, 6), dtype=np.float64)
            self.reference_state[:, :3] = start_pos  # Initial position
            self.ref_phase = np.zeros(self.n_envs, dtype=np.int32)  # Start with acceleration phase

            # Radar KF: attacker's view of true position
            self.radar_kf = BatchedKalmanFilter(self.n_envs, self.dt, Q_scale=2.0, R_scale=10.0)
            self.radar_kf.reset(positions=start_pos)

            # Nav KF: drone's internal KF
            # jipark
            # self.nav_kf = BatchedKalmanFilter(self.n_envs, self.dt, Q_scale=0.5, R_scale=1.0)
            self.nav_kf = BatchedKalmanFilter(self.n_envs, self.dt, Q_scale=2.0, R_scale=10.0)
            self.nav_kf.reset(positions=start_pos)
        else:
            self.true_pos[indices] = start_pos
            self.true_vel[indices] = 0.0
            self.reference_state[indices] = 0.0
            self.reference_state[indices, :3] = start_pos
            self.ref_phase[indices] = 0
            self.radar_kf.reset(indices, start_pos)
            self.nav_kf.reset(indices, start_pos)

        self.step_counts[indices] = 0
        self.prev_gamma_s[indices] = 0.0
        self.initial_dist_to_fake[indices] = np.linalg.norm(start_pos - self.fake_dest, axis=1)

        # Initial radar estimate (x^e)
        radar_est = self.radar_kf.get_position_estimates()

        # Initial observation with zero spoof offset (gamma_s = 0)
        zero_offset = np.zeros((self.n_envs, 3))
        obs = self._get_observations(self.true_pos, self.true_vel, radar_est, zero_offset,
                                     gamma_s_precomputed=np.zeros(self.n_envs))

        # Return radar_est (x^e) instead of deceptive_pos
        # Agent will compute x^s = x^e + Δx^s using its action
        return obs, radar_est.copy(), [{} for _ in range(self.n_envs)]

    def step(self, actions: np.ndarray):
        """
        Step with paper-accurate spoofing logic.

        ┌─────────────────────────────────────────────────────────────────┐
        │ Eq.30: x^s = x^e + Δx^s                                         │
        │                                                                  │
        │ Timeline at step t:                                              │
        │   1. radar_est_t = x^e_t (from previous KF update)              │
        │   2. Action a_t → Δx^s (spherical to Cartesian)                 │
        │   3. Deceptive position: x^s_t = x^e_t + Δx^s                   │
        │   4. Send spoofed signal to drone                                │
        │   5. Drone updates nav_kf and moves based on deceived estimate  │
        │   6. Radar observes drone at t+1 → x^e_{t+1}                    │
        └─────────────────────────────────────────────────────────────────┘

        Key: Blind spoofer uses radar KF estimate (x^e), NOT true position.
        """
        self.step_counts += 1

        # Current radar estimate x^e_t (from previous step's KF state)
        radar_est_t = self.radar_kf.get_position_estimates()

        # 1. Convert action (ρ, θ, ψ) to Cartesian offset Δx^s
        spoof_offset = self._spherical_to_cartesian(actions)

        # 2. Eq.30: Deceptive position x^s = x^e + Δx^s
        deceptive_pos = radar_est_t + spoof_offset

        # 3. Spoofed measurement sent to drone (based on deceptive position)
        spoofed_meas = deceptive_pos

        # 4. Drone's nav KF processes spoofed signal
        self.nav_kf.predict()
        _, drone_nis = self.nav_kf.update(spoofed_meas)
        nav_est = self.nav_kf.get_position_estimates()

        # 5. Calculate predicted NIS (γ^s) - must be done after nav_kf update
        # so that S reflects the current innovation covariance
        gamma_s, M_radar = self._calculate_predicted_nis(spoof_offset)

        # 6. Drone flies based on its (deceived) estimate → true_pos becomes t+1
        self._update_drone_physics(nav_est)

        # 7. Attacker observes drone at t+1 position and updates radar KF
        # This gives us x^e_{t+1} for the NEXT step
        radar_noise = self._rng.normal(0, 2.0, self.true_pos.shape)  # 2m std noise
        radar_measurement = self.true_pos + radar_noise  # true_pos is now at t+1

        self.radar_kf.predict()
        self.radar_kf.update(radar_measurement)
        radar_est_t1 = self.radar_kf.get_position_estimates()  # x^e_{t+1}
        delta_gamma_s = np.abs(gamma_s - self.prev_gamma_s)
        self.prev_gamma_s = gamma_s.copy()

        # 8. Eq.29: Build state observations s = [s1, s2, s3, s4]
        # Pass gamma_s to avoid redundant calculation and ensure consistency
        obs = self._get_observations(self.true_pos, self.true_vel, radar_est_t1, spoof_offset, gamma_s)

        # 9. Eq.38: Reward R = α₁·r_x + α₂·r_v + α₃·r_γ (NO SIE bonus)
        rewards, r_x, r_v, r_gamma = self._calculate_rewards(gamma_s, delta_gamma_s, spoof_offset, radar_est_t)

        # 10. Termination
        dist_to_fake = np.linalg.norm(self.true_pos - self.fake_dest, axis=1)
        dist_to_true = np.linalg.norm(self.true_pos - self.true_dest, axis=1)

        terminateds = np.zeros(self.n_envs, dtype=bool)
        truncateds = np.zeros(self.n_envs, dtype=bool)

        success_mask = dist_to_fake < 10.0
        terminateds |= success_mask
        rewards[success_mask] += 4000.0

        failure_mask = dist_to_true < 10.0
        terminateds |= failure_mask
        rewards[failure_mask] -= 1000.0

        # Constraint penalties
        constraint_mask = gamma_s > self.chi_sq_threshold
        rewards[constraint_mask] -= 1.0 * (gamma_s[constraint_mask] - self.chi_sq_threshold)
        # rewards[constraint_mask] = 0

        stability_mask = delta_gamma_s > self.delta_gamma_threshold
        rewards[stability_mask] -= 1.0 * (delta_gamma_s[stability_mask] - self.delta_gamma_threshold)
        # rewards[stability_mask] = 0

        truncateds |= np.linalg.norm(self.true_pos, axis=1) > self.rho_e
        truncateds |= self.step_counts >= self.max_steps

        # Build infos
        infos = []
        for i in range(self.n_envs):
            infos.append({
                'true_pos': self.true_pos[i],
                'radar_est': radar_est_t[i],  # FIXED: was undefined radar_est
                'nav_est': nav_est[i],
                'deceptive_pos': deceptive_pos[i],
                'gamma_s': gamma_s[i],
                'drone_nis': drone_nis[i],
                'dist_to_fake': dist_to_fake[i],
                'dist_to_true': dist_to_true[i],
                # === DEBUG: Z-axis bias investigation ===
                # Action components (ρ, θ, ψ) in spherical coordinates
                'action_rho': actions[i, 0],
                'action_theta': actions[i, 1],  # Azimuth angle
                'action_psi': actions[i, 2],    # Elevation angle
                # Spoofing offset in Cartesian coordinates
                'spoof_offset_x': spoof_offset[i, 0],
                'spoof_offset_y': spoof_offset[i, 1],
                'spoof_offset_z': spoof_offset[i, 2],
                # M_radar diagonal (Σ_r covariance - key for NIS bias)
                'M_radar_xx': M_radar[i, 0, 0],
                'M_radar_yy': M_radar[i, 1, 1],
                'M_radar_zz': M_radar[i, 2, 2],
                # Reward components (to check SIE vs r_x/r_v magnitude)
                'r_x': r_x[i],
                'r_v': r_v[i],
                'r_gamma': r_gamma[i],
            })

        # Auto-reset
        done_mask = terminateds | truncateds

        # FIXED: radar_est_t is x^e_t (used for action), radar_est_t1 is x^e_{t+1} (next state)
        # Buffer stores: (s_t, a_t, r_t, s_{t+1}, x^e_t, x^e_{t+1})
        next_radar_est = radar_est_t1.copy()

        if np.any(done_mask):
            done_indices = np.where(done_mask)[0]
            for i in done_indices:
                infos[i]['terminal_observation'] = obs[i].copy()
                infos[i]['terminal_radar_est'] = radar_est_t1[i].copy()

            new_obs, new_radar_est, _ = self.reset(indices=done_indices)
            obs[done_indices] = new_obs[done_indices]
            next_radar_est[done_indices] = new_radar_est[done_indices]

        # Return: radar_est_t (x^e_t for current action's SIE), next_radar_est (x^e_{t+1} for next)
        return obs, rewards, terminateds, truncateds, radar_est_t, next_radar_est, infos

    def _spherical_to_cartesian(self, actions: np.ndarray) -> np.ndarray:
        rho, theta, psi = actions[:, 0], actions[:, 1], actions[:, 2]
        cos_psi = np.cos(psi)
        dx = rho * cos_psi * np.cos(theta)
        dy = rho * cos_psi * np.sin(theta)
        dz = -rho * np.sin(psi)
        return np.stack([dx, dy, dz], axis=1)

    def _update_drone_physics(self, nav_est: np.ndarray):
        """
        Paper-accurate UAV dynamics implementation.

        ┌─────────────────────────────────────────────────────────────────┐
        │ Paper Eq.50: Reference Trajectory Generation                     │
        │   (50a) Acceleration: z1 = z + B·umax,  s.t. v < vmax          │
        │   (50b) Cruise:       z2 = z + B·0,     s.t. v = vmax          │
        │   (50c) Deceleration: z3 = z - B·umax,  s.t. zend - zD = 0    │
        │                                                                  │
        │ Paper Eq.6: PD Control Law                                       │
        │   uk = -L · (ẑk - zk)                                           │
        │   where:                                                         │
        │     - L: 3×6 control gain matrix (Table I)                      │
        │     - ẑk: current state estimate [x,y,z,vx,vy,vz] (from nav)   │
        │     - zk: reference trajectory state                             │
        │     - uk: control input [ax, ay, az]                            │
        │                                                                  │
        │ Paper Eq.2: State Update                                         │
        │   zk = A·zk-1 + B·uk-1 + wk-1                                   │
        │   where:                                                         │
        │     - A: 6×6 state transition matrix                            │
        │     - B: 6×3 input control matrix                               │
        │     - wk: process noise (simplified as 0 for deterministic sim) │
        └─────────────────────────────────────────────────────────────────┘
        """
        # Build current state estimate ẑk from nav_est
        # ẑk = [x, y, z, vx, vy, vz] (nav estimate gives position only)
        # We use nav_kf's velocity estimate for vx, vy, vz
        nav_vel_est = self.nav_kf.get_velocity_estimates()  # (n_envs, 3)
        z_hat = np.concatenate([nav_est, nav_vel_est], axis=1)  # (n_envs, 6)

        # ==================== Eq.50: Generate Reference Trajectory ====================
        # Direction to true destination
        to_dest = self.true_dest - self.reference_state[:, :3]  # (n_envs, 3)
        dist_to_dest = np.linalg.norm(to_dest, axis=1)  # (n_envs,)
        dir_to_dest = np.where(
            dist_to_dest[:, np.newaxis] > 1e-6,
            to_dest / dist_to_dest[:, np.newaxis],
            0
        )  # (n_envs, 3)

        # Current reference velocity magnitude
        ref_speed = np.linalg.norm(self.reference_state[:, 3:6], axis=1)  # (n_envs,)

        # Determine reference acceleration based on phase (Eq.50a, 50b, 50c)
        ref_accel = np.zeros((self.n_envs, 3), dtype=np.float64)

        # Phase 0: Acceleration (50a) - z1 = z + B·umax, s.t. v < vmax
        accel_mask = (self.ref_phase == 0) & (ref_speed < self.vmax)
        ref_accel[accel_mask] = dir_to_dest[accel_mask] * self.umax

        # Transition to cruise when reaching vmax
        self.ref_phase[(self.ref_phase == 0) & (ref_speed >= self.vmax)] = 1

        # Phase 1: Cruise (50b) - z2 = z + B·0, s.t. v = vmax
        cruise_mask = (self.ref_phase == 1)
        ref_accel[cruise_mask] = 0.0  # No acceleration

        # Transition to deceleration when close to destination
        # Deceleration distance: d_decel = v²/(2·umax)
        decel_distance = (ref_speed ** 2) / (2 * self.umax + 1e-6)
        self.ref_phase[(self.ref_phase == 1) & (dist_to_dest <= decel_distance)] = 2

        # Phase 2: Deceleration (50c) - z3 = z - B·umax, s.t. zend - zD = 0
        decel_mask = (self.ref_phase == 2)
        ref_accel[decel_mask] = -dir_to_dest[decel_mask] * self.umax

        # Update reference state: zk = A·zk-1 + B·uk (Eq.2, simplified wk=0)
        self.reference_state = (self.A @ self.reference_state.T).T + (self.B @ ref_accel.T).T

        # Clamp reference velocity to vmax
        ref_vel_mag = np.linalg.norm(self.reference_state[:, 3:6], axis=1, keepdims=True)
        self.reference_state[:, 3:6] = np.where(
            ref_vel_mag > self.vmax,
            self.reference_state[:, 3:6] / (ref_vel_mag + 1e-9) * self.vmax,  # Add epsilon to prevent division by zero
            self.reference_state[:, 3:6]
        )

        # ==================== Eq.6: PD Control Law ====================
        # uk = -L · (ẑk - zk)
        state_error = z_hat - self.reference_state  # (n_envs, 6)
        control_input = -(self.L @ state_error.T).T  # (n_envs, 3)

        # Saturate control input to [-umax, umax]
        control_input = np.clip(control_input, -self.umax, self.umax)

        # ==================== Eq.2: Update True Drone State ====================
        # Build true state vector: z_true = [true_pos, true_vel]
        z_true = np.concatenate([self.true_pos, self.true_vel], axis=1)  # (n_envs, 6)

        # State update: zk = A·zk-1 + B·uk (simplified wk=0)
        z_true_new = (self.A @ z_true.T).T + (self.B @ control_input.T).T

        # Extract updated position and velocity
        self.true_pos = z_true_new[:, :3]
        self.true_vel = z_true_new[:, 3:6]

        # Clamp true velocity to vmax (safety constraint)
        true_speed = np.linalg.norm(self.true_vel, axis=1, keepdims=True)
        self.true_vel = np.where(
            true_speed > self.vmax,
            self.true_vel / (true_speed + 1e-9) * self.vmax,  # Add epsilon to prevent division by zero
            self.true_vel
        )

    def _calculate_predicted_nis(self, spoof_offset: np.ndarray) -> np.ndarray:
        """
        Calculate predicted NIS (γ^s) following Paper Theorem 1 and Eq.17-18.

        ┌─────────────────────────────────────────────────────────────────┐
        │ Paper Theorem 1 (Eq.17):                                         │
        │   γ^s = (x^s - x^r - μ_θ)^T · (Σ^s)^{-1} · (x^s - x^r - μ_θ)  │
        │                                                                   │
        │ Paper Eq.18:                                                     │
        │   Σ^s = Σ^r_Δ + Σ^s_θ + Cov(x^e, Δx^s)                         │
        │                                                                   │
        │ Where:                                                            │
        │   - x^s = x^e + Δx^s (deceptive position)                       │
        │   - x^r = true position (unknown to attacker)                    │
        │   - x^e = radar estimate (attacker's estimate of x^r)           │
        │   - μ_θ = mean of Δx^s (from policy network)                    │
        │   - Σ^r_Δ = radar estimation error covariance                   │
        │   - Σ^s_θ = spoofing offset variance                            │
        │   - Cov(x^e, Δx^s) ≠ 0 (policy depends on state)                │
        │                                                                   │
        │ Blind Spoofer Scenario:                                          │
        │   - Attacker cannot access drone's nav_kf                        │
        │   - Uses radar_kf state covariance M_k as proxy for Σ^r_Δ       │
        │   - Σ^s = M_radar + σ^2_spoof · I (paper-accurate)              │
        └─────────────────────────────────────────────────────────────────┘

        Paper Eq.22 (Mean): μ^s = x^r + μ_θ
        In blind scenario: x^r ≈ x^e, so μ^s ≈ x^e + E[Δx^s]

        For deterministic policy (evaluation): μ_θ = Δx^s, so:
        γ^s = (x^s - x^e - Δx^s)^T · (Σ^s)^{-1} · (x^s - x^e - Δx^s) = 0

        For stochastic policy (training): μ_θ ≈ 0 (zero-mean offset), so:
        γ^s ≈ Δx^s^T · (Σ^s)^{-1} · Δx^s

        This measures how "unusual" the spoofing offset is relative to
        the combined uncertainty from radar estimation and spoofing strategy.
        """
        # Eq.14 & Theorem 1: Radar estimation error covariance (Σ^r_Δ)
        # Use filtered state covariance M_k (NOT innovation covariance S_k)
        # M_k = (I - K·H)·P_{k|k-1} (position components only)
        M_radar = self.radar_kf.get_state_covariances()  # (n_envs, 3, 3)

        # Spoofing offset variance (Σ^s_θ)
        # In practice, we approximate this as σ^2_spoof · I
        # This represents the policy's exploration noise
        sigma_spoof = 10.0  # Tunable parameter (paper doesn't specify exact value)
        Sigma_spoof = np.eye(3) * (sigma_spoof ** 2)  # (3, 3)

        # Eq.18: Combined covariance Σ^s = Σ^r_Δ + Σ^s_θ + Cov(x^e, Δx^s)
        # Simplification: Cov(x^e, Δx^s) ≈ 0 (assume weak correlation)
        # This is reasonable since radar noise and policy noise are independent
        Sigma_s = M_radar + Sigma_spoof[np.newaxis, :, :]  # (n_envs, 3, 3)

        # Invert covariance matrix
        try:
            Sigma_s_inv = np.linalg.inv(Sigma_s)
        except np.linalg.LinAlgError:
            # Fallback: use pseudo-inverse if singular
            Sigma_s_inv = np.linalg.pinv(Sigma_s)

        # Eq.17: γ^s = (x^s - μ^s)^T · (Σ^s)^{-1} · (x^s - μ^s)
        # Approximation for stochastic policy: μ_θ ≈ 0, so:
        # γ^s ≈ Δx^s^T · (Σ^s)^{-1} · Δx^s
        # This is the Mahalanobis distance of the spoofing offset
        Sigma_s_inv_offset = np.einsum('nij,nj->ni', Sigma_s_inv, spoof_offset)
        gamma_s = np.sum(spoof_offset * Sigma_s_inv_offset, axis=1)

        # Return both gamma_s and M_radar for debugging
        return gamma_s, M_radar

    def _get_observations(self, true_pos, true_vel, radar_est, spoof_offset, gamma_s_precomputed=None):
        """
        Build state observations per Paper Eq.29.

        ┌─────────────────────────────────────────────────────────────────┐
        │ Eq.29: State space s = [s1, s2, s3, s4]                         │
        │                                                                  │
        │ s1 = [d, θ, ψ]      Position in spherical coordinates           │
        │                      d = distance from origin                    │
        │                      θ = azimuth angle                           │
        │                      ψ = elevation angle                         │
        │                                                                  │
        │ s2 = [Δ_D, Δ^s_D]   Triangle angles for deception geometry      │
        │                      Δ_D: angle at x^e between x_D and x^s_D    │
        │                      Δ^s_D: angle at x^s between x_D and x^s_D  │
        │                                                                  │
        │ s3 = γ^s            Predicted NIS (concealment constraint)      │
        │                                                                  │
        │ s4 = [|v|, θ_v]     Velocity info                               │
        │                      |v| = speed magnitude                       │
        │                      θ_v = angle between velocity and fake dest  │
        └─────────────────────────────────────────────────────────────────┘

        Triangle Geometry:
        - Triangle Δ_D: vertices (x^e, x_D, x^s_D) - radar est, true dest, fake dest
        - Triangle Δ^s_D: vertices (x^s, x_D, x^s_D) - spoof pos, true dest, fake dest

        Args:
            gamma_s_precomputed: If provided, use this instead of recalculating
                                 (avoids redundant computation and ensures consistency)
        """
        # s1: Position in spherical coordinates (d, θ, ψ)
        d_r = np.linalg.norm(radar_est, axis=1)
        theta_r = np.arctan2(radar_est[:, 1], radar_est[:, 0])
        ground_dist = np.sqrt(radar_est[:, 0]**2 + radar_est[:, 1]**2)
        psi_r = np.arctan2(-radar_est[:, 2], ground_dist + 1e-6)

        # s2: Triangle angle information (Paper Eq.29)
        # Deceptive position x^s = x^e + Δx^s
        spoof_pos = radar_est + spoof_offset

        # Triangle Δ_D: angle at x^e between x_D and x^s_D
        # Vectors from x^e to x_D and x^s_D
        vec_xe_to_xD = self.true_dest - radar_est  # x^e → x_D
        vec_xe_to_xsD = self.fake_dest - radar_est  # x^e → x^s_D
        angle_D = self._compute_angle_between(vec_xe_to_xD, vec_xe_to_xsD)

        # Triangle Δ^s_D: angle at x^s between x_D and x^s_D
        # Vectors from x^s to x_D and x^s_D
        vec_xs_to_xD = self.true_dest - spoof_pos  # x^s → x_D
        vec_xs_to_xsD = self.fake_dest - spoof_pos  # x^s → x^s_D
        angle_sD = self._compute_angle_between(vec_xs_to_xD, vec_xs_to_xsD)

        # s3: Predicted NIS γ^s
        # Use precomputed value if provided (avoids redundant calculation)
        if gamma_s_precomputed is not None:
            gamma_s = gamma_s_precomputed
        else:
            gamma_s = self._calculate_predicted_nis(spoof_offset)

        # s4: Velocity information
        radar_vel_est = self.radar_kf.get_velocity_estimates()
        speed = np.linalg.norm(radar_vel_est, axis=1)

        # Angle between velocity and direction to fake destination
        n1 = np.linalg.norm(radar_vel_est, axis=1)
        n2 = np.linalg.norm(vec_xe_to_xsD, axis=1)
        valid = (n1 > 1e-6) & (n2 > 1e-6)
        dot = np.sum(radar_vel_est * vec_xe_to_xsD, axis=1)
        cos_angle = np.zeros(self.n_envs)
        cos_angle[valid] = np.clip(dot[valid] / (n1[valid] * n2[valid]), -1, 1)
        theta_v_fake = np.arccos(cos_angle)

        # Eq.29: Stack all state components (8 features total)
        # s = [s1, s2, s3, s4] where:
        #   s1 = [d, θ, ψ]       (3 features) - position
        #   s2 = [Δ_D, Δ^s_D]   (2 features) - triangle angles
        #   s3 = [γ^s]          (1 feature)  - predicted NIS
        #   s4 = [|v|, θ_v]     (2 features) - velocity
        return np.stack([
            np.clip(d_r / self.rho_e, 0, 1),           # s1[0]: d (normalized by ρ_e)
            theta_r / np.pi,                            # s1[1]: θ (normalized by π)
            psi_r / (np.pi/2),                          # s1[2]: ψ (normalized by π/2)
            angle_D / np.pi,                            # s2[0]: Δ_D angle (normalized)
            angle_sD / np.pi,                           # s2[1]: Δ^s_D angle (normalized)
            np.clip(gamma_s / (2 * self.chi_sq_threshold), 0, 2),  # s3: γ^s (clipped, keep raw magnitude)
            np.clip(speed / 20.0, 0, 1),                # s4[0]: |v| (normalized)
            theta_v_fake / np.pi,                       # s4[1]: θ_v (normalized)
        ], axis=1).astype(np.float32)

    def _compute_angle_between(self, vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
        """
        Compute angle between two vectors (vectorized for batch).

        Args:
            vec1: (n_envs, 3) first vector
            vec2: (n_envs, 3) second vector

        Returns:
            (n_envs,) angles in radians [0, π]
        """
        n1 = np.linalg.norm(vec1, axis=1)
        n2 = np.linalg.norm(vec2, axis=1)
        valid = (n1 > 1e-6) & (n2 > 1e-6)

        dot = np.sum(vec1 * vec2, axis=1)
        cos_angle = np.zeros(self.n_envs)
        cos_angle[valid] = np.clip(dot[valid] / (n1[valid] * n2[valid]), -1, 1)

        return np.arccos(cos_angle)

    def _calculate_rewards(self, gamma_s, delta_gamma_s, spoof_offset: np.ndarray, radar_est: np.ndarray):
        """
        Calculate reward per Paper Eq.38-39.

        ┌─────────────────────────────────────────────────────────────────┐
        │ Eq.38: R = α₁·r_x + α₂·r_v + α₃·r_γ                            │
        │                                                                  │
        │ Eq.39a: r_x = position reward (guide to fake destination)       │
        │         r_x = -||x^s - x^s_D|| / ||x_0 - x^s_D||               │
        │         (normalized distance to fake destination)                │
        │                                                                  │
        │ Eq.39b: r_v = velocity direction reward                          │
        │         r_v = -log(θ_v,fake + ε) + log(θ_v,true + ε)            │
        │         (encourage velocity toward fake, away from true)         │
        │                                                                  │
        │ Eq.39c: r_γ = concealment reward (NIS constraint)               │
        │         Penalize high γ^s to avoid detection                     │
        └─────────────────────────────────────────────────────────────────┘

        NOTE: SIE is NOT added to reward - it's handled in SAC entropy term (Eq.40-48)
        """
        radar_vel_est = self.radar_kf.get_velocity_estimates()

        # Eq.30: Deceptive position x^s = x^e + Δx^s
        deceptive_pos = radar_est + spoof_offset

        vec_to_fake = self.fake_dest - deceptive_pos
        vec_to_true = self.true_dest - deceptive_pos

        # Eq.39a: r_x - Position reward
        # Measures how close the spoofed position is to fake destination
        dist_to_fake = np.linalg.norm(deceptive_pos - self.fake_dest, axis=1)
        r_x = -(dist_to_fake / np.maximum(self.initial_dist_to_fake, 1.0))

        # Eq.39b: r_v - Velocity direction reward
        n1 = np.linalg.norm(radar_vel_est, axis=1)
        n2_fake = np.linalg.norm(vec_to_fake, axis=1)
        n2_true = np.linalg.norm(vec_to_true, axis=1)

        valid = (n1 > 1e-6)
        theta_v_fake = np.zeros(self.n_envs)
        theta_v_true = np.zeros(self.n_envs)

        theta_v_fake[valid] = np.arccos(np.clip(
            np.sum(radar_vel_est[valid] * vec_to_fake[valid], axis=1) / (n1[valid] * n2_fake[valid] + 1e-6),
            -1, 1
        ))
        theta_v_true[valid] = np.arccos(np.clip(
            np.sum(radar_vel_est[valid] * vec_to_true[valid], axis=1) / (n1[valid] * n2_true[valid] + 1e-6),
            -1, 1
        ))

        r_v = -np.log(theta_v_fake + 0.1) + np.log(theta_v_true + 0.1)

        # Eq.39c: r_γ - Concealment reward (penalize high NIS)
        r_gamma = self._concealment_reward(gamma_s) + self._concealment_reward(delta_gamma_s)

        # Eq.38: Combined reward
        total_reward = self.alpha_1 * r_x + self.alpha_2 * r_v + self.alpha_3 * r_gamma

        # Return both total and individual components for debugging
        return total_reward, r_x, r_v, r_gamma

    def _apply_T_operation(self, radar_pos: np.ndarray, radar_vel: np.ndarray) -> np.ndarray:
        """
        Apply T operation (Eq. 39a): Predict radar KF estimate after drone's PD control.

        The T operation models what the radar will observe after the drone
        responds to its navigation KF estimate with PD control.

        Steps:
        1. Drone's nav KF has deceived estimate (pointing toward fake dest)
        2. Drone applies PD control based on nav estimate
        3. Drone moves accordingly
        4. Radar KF observes and estimates new position

        This is a one-step lookahead prediction.

        Args:
            radar_pos: (n_envs, 3) current radar position estimates
            radar_vel: (n_envs, 3) current radar velocity estimates

        Returns:
            (n_envs, 3) predicted position after T operation
        """
        # Get current navigation estimate (deceived)
        nav_est = self.nav_kf.get_position_estimates()

        # Drone's PD control: head toward true_dest based on nav estimate
        # (But drone is deceived, so it actually heads toward where it thinks true_dest is)
        to_dest = self.true_dest - nav_est
        dist = np.linalg.norm(to_dest, axis=1, keepdims=True)
        dir_vec = np.where(dist > 1e-6, to_dest / dist, 0)

        # Predict velocity after PD control
        speed = np.linalg.norm(radar_vel, axis=1)
        acc = np.zeros(self.n_envs)
        dist_flat = dist.flatten()
        acc[(dist_flat > 100.0) & (speed < 16.0)] = 0.5
        acc[dist_flat < 20.0] = -0.5

        pred_vel = radar_vel + dir_vec * acc[:, np.newaxis] * self.dt
        speed_new = np.linalg.norm(pred_vel, axis=1, keepdims=True)
        pred_vel = np.where(speed_new > 16.0, pred_vel / speed_new * 16.0, pred_vel)

        # Predicted position after one timestep
        pred_pos = radar_pos + pred_vel * self.dt

        # T operation: radar KF estimate of this predicted position
        # Simplified: add small noise to represent KF estimation uncertainty
        # In practice, this is the radar's filtered estimate
        return pred_pos  # Radar KF would smooth this, but we use direct prediction

    def _concealment_reward(self, x):
        """
        Concealment reward based on NIS constraint.

        IMPORTANT: Baseline shifted to 0 to prevent ρ=0 collapse.
        Original: [0, 1] range → Always positive reward for low γ
        Fixed: [-1, 0] range → Penalty for high γ, neutral for low γ
        """
        x_th = self.chi_sq_threshold
        result = np.zeros_like(x)
        mask = x >= x_th
        result[mask] = np.exp(-np.log(1 + x[mask] - x_th + 1e-6)) - 1.0
        result[~mask] = (x_th - x[~mask]) / x_th

        # BASELINE SHIFT: 0 중심으로 이동 (ρ=0 붕괴 방지)
        # TODO: 기존 모델(sie_sac_paper_final.pt) 테스트 시 아래 주석 처리
        #       새 학습 시작할 때는 다시 활성화!
        # result -= 1.0

        return result

    def close(self):
        pass

    def __len__(self):
        return self.n_envs


# ==================== Replay Buffer with Position Storage ====================

class SIEReplayBuffer:
    """
    Replay buffer that stores radar estimates (x^e) for SIE calculation.

    Key change: Store radar_est instead of deceptive_pos.
    During update, we reconstruct x^s = x^e + Δx^s using the current policy's action,
    so that gradients can flow from SIE through action to policy parameters.
    """

    def __init__(self, capacity: int, obs_dim: int, action_dim: int, pos_dim: int = 3):
        self.capacity = capacity
        self.pos = 0
        self.full = False

        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        # Store radar estimates (x^e) instead of deceptive positions
        # This allows us to reconstruct x^s = x^e + Δx^s with current policy action
        self.radar_estimates = np.zeros((capacity, pos_dim), dtype=np.float32)
        self.next_radar_estimates = np.zeros((capacity, pos_dim), dtype=np.float32)

    def add_batch(self, obs, actions, rewards, next_obs, dones,
                  radar_est, next_radar_est):
        batch_size = obs.shape[0]
        rewards = rewards.reshape(-1, 1)
        dones = dones.reshape(-1, 1)

        for i in range(batch_size):
            idx = self.pos
            self.observations[idx] = obs[i]
            self.actions[idx] = actions[i]
            self.rewards[idx] = rewards[i]
            self.next_observations[idx] = next_obs[i]
            self.dones[idx] = dones[i]
            self.radar_estimates[idx] = radar_est[i]
            self.next_radar_estimates[idx] = next_radar_est[i]

            self.pos = (self.pos + 1) % self.capacity
            if self.pos == 0:
                self.full = True

    def sample(self, batch_size: int):
        max_idx = self.capacity if self.full else self.pos
        indices = np.random.randint(0, max_idx, size=batch_size)

        return (
            self.observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_observations[indices],
            self.dones[indices],
            self.radar_estimates[indices],
            self.next_radar_estimates[indices],
        )

    def __len__(self):
        return self.capacity if self.full else self.pos


# ==================== Networks ====================

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = torch.clamp(self.log_std(x), self.log_std_min, self.log_std_max)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DoubleQNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = QNetwork(state_dim, action_dim, hidden_dim)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dim)

    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)


# ==================== SIE-SAC Agent (Paper-Accurate) ====================

class SIESACAgentPaper:
    """
    SIE-SAC Agent following Paper Equations 40-48.

    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Key Difference from Standard SAC:                                        │
    │ Standard SAC uses Shannon entropy H(π) = -E[log π(a|s)]                 │
    │ SIE-SAC replaces it with Spatial Information Entropy H^s (Eq.37)        │
    └─────────────────────────────────────────────────────────────────────────┘

    Paper Equations Implemented:
    ============================

    Eq.41: V(s) = E_a~π[Q(s,a) + α·H^s(x^s)]
           Value function with SIE bonus

    Eq.43: Q_target = r + γ·(min Q'(s',a') + α·H^s(x'^s))
           Q-target with SIE bonus on next state

    Eq.46: L_π = -E[min Q(s,a) + α·H^s(x^s)]
           Policy loss: maximize Q + α·H^s

    Eq.48: L_α = α·(H^s - H₀)
           Temperature loss: adjust α to maintain H^s ≈ H₀

    Implementation Notes:
    ---------------------
    - H^s is computed from CURRENT policy's action (not buffer)
    - This ensures gradients flow: SIE → x^s → Δx^s → action → policy
    - x^s = x^e + Δx^s (Eq.30) where Δx^s comes from policy action

    Entropy Type Options:
    ---------------------
    - 'sie': Use Spatial Information Entropy (Paper method)
    - 'action': Use standard SAC action entropy -log π(a|s)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        fake_dest: np.ndarray,
        true_dest: np.ndarray,
        hidden_dim: int = 256,
        lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_init: float = 0.2,
        # H_0: float = 18.5,  # jipark
        H_0: float = -2.0,  # Target SIE (Eq. 47) - Paper Table I: -2.0
        lambda_sie: float = 0.01,
        rho_e: float = 1200.0,
        omega_1: float = 0.8,
        device: str = 'cpu',
        entropy_type: str = 'sie'  # 'sie' or 'action'
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.H_0 = H_0  # Target SIE for temperature adjustment
        self.entropy_type = entropy_type  # 'sie' or 'action'
        self.action_dim = action_dim  # For target entropy calculation

        # Action scaling
        self.action_low = torch.tensor(action_low, dtype=torch.float32, device=device)
        self.action_high = torch.tensor(action_high, dtype=torch.float32, device=device)
        self.action_scale = (self.action_high - self.action_low) / 2
        self.action_bias = (self.action_high + self.action_low) / 2

        self.action_scale_np = (action_high - action_low) / 2
        self.action_bias_np = (action_high + action_low) / 2

        # Networks
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.q_network = DoubleQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_q_network = DoubleQNetwork(state_dim, action_dim, hidden_dim).to(device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # Temperature
        self.log_alpha = torch.tensor(np.log(alpha_init), requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)

        # SIE Calculator (differentiable)
        self.sie_calculator = TorchSIECalculator(
            lambda_sie=lambda_sie,
            rho_e=rho_e,
            omega_1=omega_1,
            fake_dest=torch.tensor(fake_dest, dtype=torch.float32),
            true_dest=torch.tensor(true_dest, dtype=torch.float32),
            device=device
        ).to(device)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def _spherical_to_cartesian_torch(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Convert spherical action to Cartesian offset - DIFFERENTIABLE.

        Args:
            actions: (batch, 3) scaled actions [rho, theta, psi]

        Returns:
            (batch, 3) Cartesian offset [dx, dy, dz]
        """
        rho = actions[:, 0]
        theta = actions[:, 1]
        psi = actions[:, 2]

        cos_psi = torch.cos(psi)
        dx = rho * cos_psi * torch.cos(theta)
        dy = rho * cos_psi * torch.sin(theta)
        dz = -rho * torch.sin(psi)

        return torch.stack([dx, dy, dz], dim=1)

    def _compute_deceptive_pos(self, radar_est: torch.Tensor, actions_scaled: torch.Tensor) -> torch.Tensor:
        """
        Compute deceptive position x^s = x^e + Δx^s - DIFFERENTIABLE.

        This is the key function that connects policy action to SIE.
        Gradients flow: SIE(x^s) → x^s → Δx^s → action → policy

        Args:
            radar_est: (batch, 3) radar estimates x^e
            actions_scaled: (batch, 3) scaled actions [rho, theta, psi]

        Returns:
            (batch, 3) deceptive positions x^s
        """
        spoof_offset = self._spherical_to_cartesian_torch(actions_scaled)
        return radar_est + spoof_offset

    @torch.no_grad()
    def select_actions_batch(self, states: np.ndarray, evaluate: bool = False):
        state_tensor = torch.from_numpy(states).float().to(self.device)
        if evaluate:
            _, _, action = self.policy.sample(state_tensor)
        else:
            action, _, _ = self.policy.sample(state_tensor)
        action = action.cpu().numpy()
        return action * self.action_scale_np + self.action_bias_np

    def update(self, batch) -> Dict[str, float]:
        """
        Update networks using Paper Equations 43, 46, 48.

        ┌─────────────────────────────────────────────────────────────────┐
        │ CRITICAL: H^s must be computed from CURRENT policy's action    │
        │                                                                  │
        │ Gradient flow for policy learning:                               │
        │   L_π → H^s → x^s → Δx^s → action → policy parameters          │
        │                                                                  │
        │ If H^s were computed from buffer (fixed), gradients wouldn't    │
        │ flow to policy, and policy couldn't learn to maximize SIE.      │
        └─────────────────────────────────────────────────────────────────┘
        """
        states, actions, rewards, next_states, dones, radar_est, next_radar_est = batch

        # Convert to tensors
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)
        radar_est = torch.from_numpy(radar_est).to(self.device)
        next_radar_est = torch.from_numpy(next_radar_est).to(self.device)

        # Normalize actions for Q-network
        actions_norm = (actions - self.action_bias) / self.action_scale

        # ========== Eq.43: Q-Network Update ==========
        # Q_target = r + γ · (min Q'(s',a') + α·H^s(x'^s))   [SIE mode]
        # Q_target = r + γ · (min Q'(s',a') - α·log π(a'|s')) [Action entropy mode]
        with torch.no_grad():
            # Sample next action from current policy
            next_action, next_log_prob, _ = self.policy.sample(next_states)
            next_action_scaled = next_action * self.action_scale + self.action_bias
            next_action_norm = (next_action_scaled - self.action_bias) / self.action_scale

            target_q1, target_q2 = self.target_q_network(next_states, next_action_norm)
            target_q = torch.min(target_q1, target_q2)

            if self.entropy_type == 'sie':
                # SIE-SAC: Eq.30 + Eq.37: Compute H^s from next_action
                # x'^s = x'^e + Δx^s(a')
                next_deceptive_pos = self._compute_deceptive_pos(next_radar_est, next_action_scaled)
                next_Hs = self.sie_calculator(next_deceptive_pos).unsqueeze(-1)  # Eq.37
                # Eq.43: Q_target = r + γ·(min Q' + α·H^s)
                target_value = target_q + self.alpha * next_Hs
            else:
                # Standard SAC: Q_target = r + γ·(min Q' - α·log π)
                # Note: entropy = -log_prob, so we subtract log_prob
                target_value = target_q - self.alpha * next_log_prob

            q_target = rewards + (1 - dones) * self.gamma * target_value

        current_q1, current_q2 = self.q_network(states, actions_norm)
        q_loss = F.mse_loss(current_q1, q_target) + F.mse_loss(current_q2, q_target)

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # ========== Eq.46: Policy Update ==========
        # SIE-SAC:      L_π = -E[Q + α·H^s]  ← maximize Q + α·H^s
        # Standard SAC: L_π = E[α·log π(a|s) - Q(s,a)] ← maximize Q - α·log π
        #
        # CRITICAL: Sample fresh action for gradient flow
        action, log_prob, _ = self.policy.sample(states)
        action_scaled = action * self.action_scale + self.action_bias
        action_norm = (action_scaled - self.action_bias) / self.action_scale

        q1, q2 = self.q_network(states, action_norm)
        min_q = torch.min(q1, q2)

        if self.entropy_type == 'sie':
            # SIE-SAC: Eq.30 + Eq.37: Compute H^s from current policy's action
            # Gradient path: L_π → H^s → x^s → Δx^s → action → policy_params
            current_deceptive_pos = self._compute_deceptive_pos(radar_est, action_scaled)
            current_Hs = self.sie_calculator(current_deceptive_pos).unsqueeze(-1)  # Eq.37
            # Eq.46: L_π = -E[Q + α·H^s] (minimize negative = maximize positive)
            policy_loss = (-(min_q + self.alpha.detach() * current_Hs)).mean()
            entropy_value = current_Hs.mean().item()
        else:
            # Standard SAC: L_π = E[α·log π - Q] = -E[Q - α·log π]
            policy_loss = (self.alpha.detach() * log_prob - min_q).mean()
            entropy_value = -log_prob.mean().item()  # entropy = -log_prob

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # ========== Temperature (α) Update ==========
        if self.entropy_type == 'sie':
            # SIE-SAC: Paper Eq.48: L(α) = E[α·(H^s - H₀)]
            alpha_loss = (self.log_alpha.exp() * (current_Hs.detach() - self.H_0)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            # DISABLED: α fixed - H^s doesn't vary enough with actions
            # alpha_loss = torch.tensor(0.0)  # For logging only
        else:
            # Standard SAC: L(α) = E[-α·(log π + H₀)]
            # Target entropy H₀ = -dim(A) for continuous actions
            target_entropy = -self.action_dim
            alpha_loss = -(self.log_alpha.exp() * (log_prob + target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Soft update
        self._soft_update()

        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item(),
            'q_value': min_q.mean().item(),
            'entropy': entropy_value,  # H^s for SIE, -log π for action entropy
            'entropy_type': self.entropy_type,
        }

    def _soft_update(self):
        for target_param, param in zip(self.target_q_network.parameters(),
                                       self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, path: str):
        torch.save({
            'policy': self.policy.state_dict(),
            'q_network': self.q_network.state_dict(),
            'target_q_network': self.target_q_network.state_dict(),
            'log_alpha': self.log_alpha,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy'])
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network'])
        self.log_alpha = checkpoint['log_alpha']


# ==================== Training Function ====================

def train_sie_sac_paper(
    env_config: Dict,
    n_envs: int = 16,
    total_timesteps: int = 1000000,
    batch_size: int = 256,
    buffer_size: int = 1000000,
    start_steps: int = 10000,
    update_after: int = 1000,
    gradient_steps: int = 1,
    # H_0: float = 18.5,  # jipark
    H_0: float = -2.0,
    save_freq: int = 50000,
    log_freq: int = 5000,
    save_dir: str = 'models/SIE_SAC_paper',
    seed: int = 42,
    device: str = None,
    entropy_type: str = 'sie',  # 'sie' or 'action'
):
    """Train SIE-SAC following the paper exactly."""

    os.makedirs(save_dir, exist_ok=True)

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    np.random.seed(seed)
    torch.manual_seed(seed)

    entropy_name = "Spatial Information Entropy (SIE)" if entropy_type == 'sie' else "Action Entropy (Standard SAC)"

    print("=" * 60)
    print("SIE-SAC Paper-Accurate Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Vectorized environments: {n_envs}")
    print(f"Entropy Type: {entropy_name}")
    if entropy_type == 'sie':
        print(f"Target SIE (H_0): {H_0}")
    else:
        print(f"Target Action Entropy: -dim(A) = -3")
    print("=" * 60 + "\n")

    # Create environment
    env = VectorizedSIEEnvPaper(n_envs=n_envs, config=env_config)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    print(f"State dim: {state_dim}, Action dim: {action_dim}")

    # Create agent
    agent = SIESACAgentPaper(
        state_dim=state_dim,
        action_dim=action_dim,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
        fake_dest=np.array(env_config.get('fake_dest', [800, -100, -20])),
        true_dest=np.array(env_config.get('true_dest', [800, 0, -20])),
        H_0=H_0,
        lambda_sie=env_config.get('lambda_sie', 0.01),
        rho_e=env_config.get('rho_e', 1200.0),
        omega_1=env_config.get('omega_1', 0.8),
        device=device,
        entropy_type=entropy_type,
    )

    # Create buffer with position storage
    buffer = SIEReplayBuffer(buffer_size, state_dim, action_dim, pos_dim=3)

    # Initialize
    current_obs, current_radar_est, _ = env.reset(seed=seed)

    total_steps = 0
    episode_rewards = []
    episode_lengths = []
    env_episode_rewards = np.zeros(n_envs)
    env_episode_lengths = np.zeros(n_envs)

    start_time = time.time()
    last_log_steps = 0
    update_info = {}

    print("Starting training...\n")

    while total_steps < total_timesteps:
        # Select actions
        if total_steps < start_steps:
            actions = np.random.uniform(env.action_space.low, env.action_space.high, (n_envs, action_dim))
        else:
            actions = agent.select_actions_batch(current_obs)

        # Step - now returns radar_est and next_radar_est
        next_obs, rewards, terminateds, truncateds, radar_est, next_radar_est, infos = env.step(actions)
        dones = terminateds | truncateds

        # Store with radar estimates (x^e), NOT deceptive positions
        # Agent will compute x^s = x^e + Δx^s during update
        buffer.add_batch(
            current_obs, actions, rewards, next_obs, dones.astype(np.float32),
            current_radar_est, next_radar_est
        )

        # Track episodes
        env_episode_rewards += rewards
        env_episode_lengths += 1
        total_steps += n_envs

        for i, done in enumerate(dones):
            if done:
                episode_rewards.append(env_episode_rewards[i])
                episode_lengths.append(env_episode_lengths[i])
                env_episode_rewards[i] = 0
                env_episode_lengths[i] = 0

        current_obs = next_obs
        current_radar_est = next_radar_est

        # Update
        if total_steps >= update_after and len(buffer) >= batch_size:
            for _ in range(gradient_steps):
                batch = buffer.sample(batch_size)
                update_info = agent.update(batch)

        # Logging
        if total_steps - last_log_steps >= log_freq:
            elapsed = time.time() - start_time
            steps_per_sec = total_steps / elapsed

            if episode_rewards:
                recent = episode_rewards[-100:] if len(episode_rewards) >= 100 else episode_rewards

                print(f"Steps: {total_steps:,}/{total_timesteps:,}")
                print(f"  Episodes: {len(episode_rewards)}, Steps/sec: {steps_per_sec:.1f}")
                print(f"  Avg Reward: {np.mean(recent):.2f} (+/- {np.std(recent):.2f})")

                if update_info:
                    ent_type = update_info.get('entropy_type', 'sie')
                    ent_label = "H^s" if ent_type == 'sie' else "H(π)"
                    print(f"  Alpha: {update_info.get('alpha', 0):.4f}, {ent_label}: {update_info.get('entropy', 0):.4f}")
                    print(f"  Q-value: {update_info.get('q_value', 0):.2f}")
                    print(f"  policy_loss: {update_info.get('policy_loss', 0):.4f}")

                remaining = (total_timesteps - total_steps) / steps_per_sec / 3600
                print(f"  ETA: {remaining:.1f} hours\n")

            last_log_steps = total_steps

        # Save
        if total_steps % save_freq < n_envs:
            agent.save(f"{save_dir}/sie_sac_paper_{total_steps}.pt")

    agent.save(f"{save_dir}/sie_sac_paper_final.pt")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final avg reward: {np.mean(episode_rewards[-100:]):.2f}")
    print("=" * 60)

    env.close()
    return episode_rewards, episode_lengths, agent


# ==================== Plotting Functions ====================

def plot_training_results(episode_rewards, episode_lengths, save_dir, entropy_type):
    """학습 결과 플롯"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'SIE-SAC Training Results (Entropy: {entropy_type.upper()})', fontsize=14)

    # 1. Episode Rewards
    ax1 = axes[0, 0]
    ax1.plot(episode_rewards, 'b-', alpha=0.3, label='Episode Reward')
    # 이동 평균
    window = min(100, len(episode_rewards) // 10) if len(episode_rewards) > 10 else 1
    if window > 1:
        moving_avg = np.convolve(episode_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label=f'Moving Avg ({window})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Reward')
    ax1.set_title('Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Episode Lengths
    ax2 = axes[0, 1]
    ax2.plot(episode_lengths, 'g-', alpha=0.3, label='Episode Length')
    if window > 1:
        moving_avg_len = np.convolve(episode_lengths, np.ones(window)/window, mode='valid')
        ax2.plot(range(window-1, len(episode_lengths)), moving_avg_len, 'orange', linewidth=2, label=f'Moving Avg ({window})')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Steps')
    ax2.set_title('Episode Lengths')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Reward Distribution (최근 vs 초기)
    ax3 = axes[1, 0]
    n_episodes = len(episode_rewards)
    if n_episodes > 200:
        early = episode_rewards[:100]
        late = episode_rewards[-100:]
        ax3.hist(early, bins=30, alpha=0.5, label='First 100 episodes', color='blue')
        ax3.hist(late, bins=30, alpha=0.5, label='Last 100 episodes', color='green')
        ax3.axvline(np.mean(early), color='blue', linestyle='--', label=f'Early mean: {np.mean(early):.1f}')
        ax3.axvline(np.mean(late), color='green', linestyle='--', label=f'Late mean: {np.mean(late):.1f}')
    else:
        ax3.hist(episode_rewards, bins=30, alpha=0.7, color='blue')
        ax3.axvline(np.mean(episode_rewards), color='red', linestyle='--', label=f'Mean: {np.mean(episode_rewards):.1f}')
    ax3.set_xlabel('Reward')
    ax3.set_ylabel('Count')
    ax3.set_title('Reward Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 학습 진행 통계
    ax4 = axes[1, 1]
    ax4.axis('off')
    stats_text = f"""
    Training Statistics
    ═══════════════════════════════════════

    Total Episodes: {len(episode_rewards)}
    Total Steps: {sum(episode_lengths):,}

    Reward:
      • Final (last 100): {np.mean(episode_rewards[-100:]):.2f} ± {np.std(episode_rewards[-100:]):.2f}
      • Best Episode: {max(episode_rewards):.2f}
      • Worst Episode: {min(episode_rewards):.2f}

    Episode Length:
      • Average: {np.mean(episode_lengths):.1f}
      • Max: {max(episode_lengths)}
      • Min: {min(episode_lengths)}

    Entropy Type: {entropy_type.upper()}
    """
    ax4.text(0.1, 0.5, stats_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='center', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plot_path = f"{save_dir}/training_results.png"
    plt.savefig(plot_path, dpi=150)
    print(f">>> 학습 결과 그래프 저장: {plot_path}")
    plt.show()


def run_simulation_after_training(agent, env_config, save_dir, entropy_type, max_steps=2000):
    """학습 완료 후 시뮬레이션 실행 및 시각화"""
    import matplotlib.pyplot as plt

    print("\n" + "=" * 60)
    print("Post-Training Simulation")
    print("=" * 60)

    # 단일 환경 생성
    env = VectorizedSIEEnvPaper(n_envs=1, config=env_config)

    obs, _, _ = env.reset(seed=123)

    # 데이터 기록
    drone_positions = []
    spoof_positions = []
    actions_history = []
    rewards_history = []
    nis_values = []

    done = False
    step = 0
    total_reward = 0

    while not done and step < max_steps:
        # 액션 선택
        action = agent.select_actions_batch(obs, evaluate=True)

        # 환경 스텝
        next_obs, reward, terminated, truncated, radar_est_t, next_radar_est, infos = env.step(action)
        done = terminated[0] or truncated[0]

        # 데이터 기록
        drone_positions.append(env.true_pos[0].copy())

        # Spoof position 계산: x^s = x^e + Δx^s
        rho, theta, psi = action[0]
        cos_psi = np.cos(psi)
        dx = rho * cos_psi * np.cos(theta)
        dy = rho * cos_psi * np.sin(theta)
        dz = -rho * np.sin(psi)
        spoof_offset = np.array([dx, dy, dz])
        spoof_pos = radar_est_t[0] + spoof_offset
        spoof_positions.append(spoof_pos)

        actions_history.append(action[0].copy())
        rewards_history.append(reward[0])
        nis_values.append(infos[0].get('nis', 0))

        total_reward += reward[0]
        obs = next_obs
        radar_est = next_radar_est
        step += 1

    env.close()

    drone_positions = np.array(drone_positions)
    spoof_positions = np.array(spoof_positions)
    actions_history = np.array(actions_history)
    rewards_history = np.array(rewards_history)
    nis_values = np.array(nis_values)

    print(f"  Simulation completed: {step} steps, Total reward: {total_reward:.2f}")

    # 플롯 생성
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'Post-Training Simulation (Entropy: {entropy_type.upper()})', fontsize=14)

    # 1. 3D 궤적
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    ax1.plot(drone_positions[:, 0], drone_positions[:, 1], drone_positions[:, 2],
             'b-', linewidth=2, label='Drone Path')
    ax1.plot(spoof_positions[:, 0], spoof_positions[:, 1], spoof_positions[:, 2],
             'm--', linewidth=1.5, label='Spoofed Position', alpha=0.7)

    # 목적지
    true_dest = np.array(env_config['true_dest'])
    fake_dest = np.array(env_config['fake_dest'])
    ax1.scatter(*true_dest, c='blue', marker='*', s=200, label='True Dest')
    ax1.scatter(*fake_dest, c='red', marker='X', s=200, label='Fake Dest')
    ax1.scatter(0, 0, -20, c='green', marker='s', s=150, label='Start')

    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title('3D Trajectory')
    ax1.legend(fontsize=8)

    # 2. XY 평면 궤적
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(drone_positions[:, 0], drone_positions[:, 1], 'b-', linewidth=2, label='Drone')
    ax2.plot(spoof_positions[:, 0], spoof_positions[:, 1], 'm--', linewidth=1.5, label='Spoof', alpha=0.7)
    ax2.scatter(true_dest[0], true_dest[1], c='blue', marker='*', s=200, label='True Dest')
    ax2.scatter(fake_dest[0], fake_dest[1], c='red', marker='X', s=200, label='Fake Dest')
    ax2.scatter(0, 0, c='green', marker='s', s=150, label='Start')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('XY Trajectory')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    # 3. Rewards
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(rewards_history, 'g-', alpha=0.7)
    ax3.axhline(y=np.mean(rewards_history), color='r', linestyle='--', label=f'Mean: {np.mean(rewards_history):.2f}')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Reward')
    ax3.set_title('Step Rewards')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Actions (ρ, θ, ψ)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(actions_history[:, 0], 'r-', label='ρ (offset dist)', alpha=0.8)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('ρ (m)')
    ax4.set_title('Action: Offset Distance (ρ)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Actions (θ, ψ)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(np.degrees(actions_history[:, 1]), 'g-', label='θ (azimuth)', alpha=0.8)
    ax5.plot(np.degrees(actions_history[:, 2]), 'b-', label='ψ (elevation)', alpha=0.8)
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Angle (deg)')
    ax5.set_title('Action: Angles (θ, ψ)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. NIS values
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(nis_values, 'purple', alpha=0.7)
    ax6.axhline(y=7.815, color='r', linestyle='--', label='χ² threshold (7.815)')
    ax6.set_xlabel('Step')
    ax6.set_ylabel('NIS (γ)')
    ax6.set_title('Normalized Innovation Squared')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    sim_plot_path = f"{save_dir}/simulation_results.png"
    plt.savefig(sim_plot_path, dpi=150)
    print(f">>> 시뮬레이션 결과 그래프 저장: {sim_plot_path}")
    plt.show()

    # 최종 결과 출력
    final_drone_pos = drone_positions[-1]
    dist_to_true = np.linalg.norm(final_drone_pos - true_dest)
    dist_to_fake = np.linalg.norm(final_drone_pos - fake_dest)

    print(f"\n  Final Position: [{final_drone_pos[0]:.1f}, {final_drone_pos[1]:.1f}, {final_drone_pos[2]:.1f}]")
    print(f"  Distance to True Dest: {dist_to_true:.1f}m")
    print(f"  Distance to Fake Dest: {dist_to_fake:.1f}m")
    print(f"  Average ρ (offset): {np.mean(actions_history[:, 0]):.2f}m")
    print(f"  NIS violations: {np.sum(nis_values > 7.815)} / {len(nis_values)}")

    return {
        'drone_positions': drone_positions,
        'spoof_positions': spoof_positions,
        'actions': actions_history,
        'rewards': rewards_history,
        'nis': nis_values,
        'total_reward': total_reward,
    }


# ==================== Main ====================

def select_entropy_type():
    """사용자에게 entropy type을 선택하도록 요청"""
    print("=" * 60)
    print("SIE-SAC Training - Entropy Type Selection")
    print("=" * 60)
    print("\n사용할 Entropy 유형을 선택하세요:")
    print("  1. SIE (Spatial Information Entropy) - 논문 방식")
    print("  2. Action Entropy - 표준 SAC 방식")
    print()

    while True:
        try:
            choice = input("선택 (1 또는 2): ").strip()
            if choice == '1':
                print("\n→ SIE (Spatial Information Entropy) 선택됨\n")
                return 'sie'
            elif choice == '2':
                print("\n→ Action Entropy (Standard SAC) 선택됨\n")
                return 'action'
            else:
                print("잘못된 입력입니다. 1 또는 2를 입력하세요.")
        except KeyboardInterrupt:
            print("\n취소되었습니다.")
            exit(0)


def main():
    # 사용자가 entropy type 선택
    entropy_type = select_entropy_type()

    # Hyperparameters matching Paper Table I
    env_config = {
        'true_dest': [800.0, 0.0, -20.0],
        'fake_dest': [800.0, -100.0, -20.0],
        'rho_e': 1200.0,        # Paper Table I: 1200m
        'lambda_sie': 0.01,
        'omega_1': 0.8,
        'chi_sq_threshold': 7.815,
        'rho_s_max': 200.0,     # Paper Table I: 200m
        'max_steps': 2000,
    }

    # save_dir을 entropy_type에 따라 구분
    save_dir = f'models/SIE_SAC_paper_{entropy_type}'

    train_params = {
        'env_config': env_config,
        'n_envs': 32,
        'total_timesteps': 1000000,
        'batch_size': 256,
        'start_steps': 10000,
        'update_after': 1000,
        'H_0': -2.0,            # Paper Table I: -2.0
        # 'H_0': 18.5,            # jipark
        'save_freq': 100000,
        'log_freq': 10000,
        'save_dir': save_dir,
        'seed': 42,
        'entropy_type': entropy_type,
    }

    # 학습 실행
    episode_rewards, episode_lengths, agent = train_sie_sac_paper(**train_params)

    # 학습 결과 플롯
    print("\n>>> 학습 결과 시각화 중...")
    plot_training_results(episode_rewards, episode_lengths, save_dir, entropy_type)

    # 시뮬레이션 실행
    print("\n>>> 학습된 모델로 시뮬레이션 실행 중...")
    run_simulation_after_training(agent, env_config, save_dir, entropy_type)

    print("\n" + "=" * 60)
    print("All Done!")
    print(f"Results saved in: {save_dir}/")
    print("=" * 60)


if __name__ == '__main__':
    main()
