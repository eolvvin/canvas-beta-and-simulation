"""
CLEAN CANVAS VISUALIZATION - FIXED
Left: Charm mass + Beta convergence plots
Right: 4-panel simulation results (fixed heatmap)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from dataclasses import dataclass
import os
from datetime import datetime

os.makedirs("simulation_results", exist_ok=True)

# ============================================================
# LOCKED CONSTANTS
# ============================================================

ALPHA_EM = 1.0 / 137.036
PI = np.pi
THETA = PI / 2.0

A_BASE = 1.0
B_BASE = 1.0 / 3.0
C_BASE = 1.0 / (ALPHA_EM * (1 + THETA))
D_BASE = C_BASE
c_eff = C_BASE * THETA
d_eff = D_BASE * 1.0

print("=" * 80)
print("CLEAN CANVAS VISUALIZATION")
print("=" * 80)

# ============================================================
# PART 1: BETA CONVERGENCE
# ============================================================

print("\n" + "-" * 50)
print("Computing Beta convergence...")
print("-" * 50)

def yukawa(n1, n2, n3, beta):
    return (n1 * n2 * n3) * np.exp(-beta * (n1**2 + n2**2 + n3**2))

def compute_masses(beta, v=246.0):
    y3 = yukawa(1, 1, 1, beta)
    y2 = yukawa(2, 1, 1, beta)
    y1 = yukawa(3, 2, 2, beta)
    norm = 173.0 / (y3 * v / np.sqrt(2))
    m_top = y3 * norm * v / np.sqrt(2)
    m_charm = y2 * norm * v / np.sqrt(2)
    m_up = y1 * norm * v / np.sqrt(2)
    return m_top, m_charm, m_up

# Binary search for optimal beta
beta_low, beta_high = 0.5, 2.5
beta_history = []
charm_history = []

for i in range(15):
    beta_mid = (beta_low + beta_high) / 2
    m_top, m_charm, m_up = compute_masses(beta_mid)
    beta_history.append(beta_mid)
    charm_history.append(m_charm)
    
    if m_charm > 1.27:
        beta_low = beta_mid
    else:
        beta_high = beta_mid
    
    if abs(m_charm - 1.27) < 0.001:
        break

optimal_beta = (beta_low + beta_high) / 2
print(f"Optimal β = {optimal_beta:.6f}")

# ============================================================
# PART 2: SIMULATION AT R = 4.0
# ============================================================

print("\n" + "-" * 50)
print("Running simulation at R = 4.0...")
print("-" * 50)

@dataclass
class SimParams:
    N: int = 400
    L: float = 200.0
    a: float = 0.0
    b: float = 0.1
    c: float = 1.0
    d: float = 0.1
    K: float = 1.0
    sigma: float = 1.2
    amp: float = 3.0
    R: float = 4.0
    t_max: float = 200.0
    n_steps: int = 2000
    
    def __post_init__(self):
        self.x = np.linspace(0, self.L, self.N)
        self.dx = self.x[1] - self.x[0]
        self.x1 = 0.25 * self.L
        self.x2 = 0.75 * self.L
        self.dt = self.t_max / self.n_steps
        self.t_eval = np.linspace(0, self.t_max, self.n_steps)

def run_simulation(p):
    x = p.x
    dx = p.dx
    dt = p.dt
    
    # Initial condition
    r1 = x - p.x1
    r2 = x - p.x2
    env1 = np.exp(-r1**2 / (2 * p.sigma**2))
    env2 = np.exp(-r2**2 / (2 * p.sigma**2))
    wave1 = p.amp * env1 * np.cos(0.5 * r1)
    wave2 = p.amp * env2 * np.cos(-0.5 * r2)
    phi = wave1 + wave2
    phi_prev = phi.copy()
    
    max_amps = []
    center_amps = []
    phi_history = []
    times_history = []
    center_idx = len(x) // 2
    bound = False
    
    for step in range(p.n_steps):
        # Laplacian
        lap = np.zeros_like(phi)
        lap[1:-1] = (phi[2:] - 2*phi[1:-1] + phi[:-2]) / (dx * dx)
        
        # Polarity
        sign_phi = np.sign(phi)
        t = step * dt
        
        # UWE
        phi_ddot = (phi - p.b * 1.0 - p.a * t - p.d * sign_phi) / p.c
        phi_ddot = phi_ddot + p.K * lap
        
        # Leapfrog
        phi_next = 2*phi - phi_prev + dt * dt * phi_ddot
        
        # Absorbing boundaries
        for i in range(20):
            f = np.exp(-(i/6.0)**2)
            phi_next[i] *= f
            phi_next[-i-1] *= f
        
        # Threshold mechanism
        if np.any(phi**2 > p.R) and step > 100:
            idx = np.argmax(phi**2)
            val = phi[idx]
            if abs(val) > 0.3:
                for di in range(-5, 6):
                    j = idx + di
                    if 0 <= j < p.N:
                        phi_next[j] += val * np.exp(-di**2 / 8.0) * 0.2
        
        phi_prev, phi = phi, phi_next
        
        # Record data at intervals
        if step % 50 == 0:
            max_amps.append(np.max(np.abs(phi)))
            center_amps.append(np.abs(phi[center_idx]))
            phi_history.append(phi.copy())
            times_history.append(step * dt)
            
            if not bound and step > 500 and len(max_amps) >= 5:
                if np.mean(max_amps[-5:]) > 0.8:
                    bound = True
    
    return bound, max_amps, center_amps, phi_history, phi, times_history, p

p = SimParams()
bound, max_amps, center_amps, phi_history, phi_final, times_hist, params = run_simulation(p)

print(f"Bound state at R = 4.0: {bound}")
print(f"Final amplitude: {max_amps[-1]:.3f}")

# ============================================================
# PART 3: CREATE THE 6-PANEL FIGURE (FIXED HEATMAP)
# ============================================================

print("\n" + "-" * 50)
print("Generating 6-panel figure...")
print("-" * 50)

# Create figure
fig = plt.figure(figsize=(16, 12))

# Left column: Beta convergence plots (stacked)
ax1 = plt.subplot(2, 2, 1)
iterations = range(1, len(charm_history) + 1)
ax1.plot(iterations, charm_history, 'bo-', linewidth=2, markersize=6)
ax1.axhline(y=1.27, color='r', linestyle='--', linewidth=1.5)
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Charm Mass (GeV)')
ax1.set_title('Charm Mass Convergence')
ax1.grid(True, alpha=0.3)

ax2 = plt.subplot(2, 2, 3)
ax2.plot(iterations, beta_history, 'go-', linewidth=2, markersize=6)
ax2.axhline(y=optimal_beta, color='r', linestyle='--', linewidth=1.5, label=f'β = {optimal_beta:.4f}')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('β')
ax2.set_title('Beta Value Convergence')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Right column: 4 simulation panels (2x2)
# Panel 1: Field amplitude snapshot (not heatmap to avoid dimension issues)
ax3 = plt.subplot(2, 2, 2)
# Show final field configuration instead of heatmap (cleaner)
ax3.plot(params.x, phi_final, 'b-', linewidth=1.5)
ax3.fill_between(params.x, 0, phi_final, where=(phi_final > 0), color='blue', alpha=0.3)
ax3.fill_between(params.x, 0, phi_final, where=(phi_final < 0), color='red', alpha=0.3)
ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax3.set_xlabel('Position x')
ax3.set_ylabel('Field Amplitude Φ')
ax3.set_title(f'Final Field Configuration (Bound State: {bound})')
ax3.grid(True, alpha=0.3)

# Panel 2: Center amplitude over time
ax4 = plt.subplot(2, 2, 4)
time_axis = np.array(times_hist)
ax4.plot(time_axis, center_amps, 'r-', linewidth=1.5)
ax4.set_xlabel('Time')
ax4.set_ylabel('Center Amplitude')
ax4.set_title('Center Amplitude vs Time')
ax4.grid(True, alpha=0.3)

# Add two more plots in a second row on the right
# Need to create a 2x2 grid within the right side
# Let me use a different approach: 2 rows, 2 columns on the right
# But we already used (2,2,2) and (2,2,4). Let me add (2,2,2) and (2,2,4) as the top and bottom of right column?

# Actually, let me restructure: 2 rows, 2 columns total
# Row1: Left charm, Right field snapshot
# Row2: Left beta, Right center amplitude
# Then add two more subplots below the right column? This is getting messy.

# Let me create a cleaner layout with GridSpec
plt.clf()
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 1])

# Left column (2 plots stacked)
ax_left1 = fig.add_subplot(gs[0, 0])
ax_left1.plot(iterations, charm_history, 'bo-', linewidth=2, markersize=6)
ax_left1.axhline(y=1.27, color='r', linestyle='--', linewidth=1.5)
ax_left1.set_xlabel('Iteration')
ax_left1.set_ylabel('Charm Mass (GeV)')
ax_left1.set_title('Charm Mass Convergence')
ax_left1.grid(True, alpha=0.3)

ax_left2 = fig.add_subplot(gs[1, 0])
ax_left2.plot(iterations, beta_history, 'go-', linewidth=2, markersize=6)
ax_left2.axhline(y=optimal_beta, color='r', linestyle='--', linewidth=1.5, label=f'β = {optimal_beta:.4f}')
ax_left2.set_xlabel('Iteration')
ax_left2.set_ylabel('β')
ax_left2.set_title('Beta Value Convergence')
ax_left2.legend()
ax_left2.grid(True, alpha=0.3)

# Right column (4 plots in a 2x2 grid)
ax_right1 = fig.add_subplot(gs[0, 1])
ax_right1.plot(params.x, phi_final, 'b-', linewidth=1.5)
ax_right1.fill_between(params.x, 0, phi_final, where=(phi_final > 0), color='blue', alpha=0.3)
ax_right1.fill_between(params.x, 0, phi_final, where=(phi_final < 0), color='red', alpha=0.3)
ax_right1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax_right1.set_xlabel('Position x')
ax_right1.set_ylabel('Field Amplitude Φ')
ax_right1.set_title('Final Field Configuration')
ax_right1.grid(True, alpha=0.3)

ax_right2 = fig.add_subplot(gs[0, 2])
time_axis = np.array(times_hist)
ax_right2.plot(time_axis, center_amps, 'r-', linewidth=1.5)
ax_right2.set_xlabel('Time')
ax_right2.set_ylabel('Center Amplitude')
ax_right2.set_title('Center Amplitude vs Time')
ax_right2.grid(True, alpha=0.3)

ax_right3 = fig.add_subplot(gs[1, 1])
time_axis_max = np.array(times_hist)
ax_right3.plot(time_axis_max, max_amps, 'g-', linewidth=1.5)
ax_right3.axhline(y=0.8, color='r', linestyle='--', linewidth=1)
ax_right3.set_xlabel('Time')
ax_right3.set_ylabel('Maximum Amplitude')
ax_right3.set_title('Maximum Amplitude Evolution')
ax_right3.grid(True, alpha=0.3)

ax_right4 = fig.add_subplot(gs[1, 2])
energy = phi_final**2
ax_right4.plot(params.x, energy, 'purple', linewidth=1.5)
ax_right4.fill_between(params.x, 0, energy, color='purple', alpha=0.3)
ax_right4.set_xlabel('Position x')
ax_right4.set_ylabel('Energy Density')
ax_right4.set_title('Final Energy Distribution')
ax_right4.grid(True, alpha=0.3)

plt.suptitle('Canvas Model: Beta Convergence and Simulation Results', fontsize=16, y=0.98)
plt.tight_layout()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
plt.savefig(f"simulation_results/clean_visualization_{timestamp}.png", dpi=150)
plt.show()

print(f"\nPlot saved to simulation_results/clean_visualization_{timestamp}.png")

# ============================================================
# SUMMARY (console only)
# ============================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

m_top, m_charm, m_up = compute_masses(optimal_beta)

print(f"""
Beta convergence:        β = {optimal_beta:.6f}
Bound state at R=4.0:    {'YES' if bound else 'NO'}
Final amplitude:         {max_amps[-1]:.3f}

Locked weights:
  c_base = d_base = {C_BASE:.6f}
  c_eff = {c_eff:.6f}
  d_eff = {d_eff:.6f}
  c_eff + d_eff = {c_eff + d_eff:.6f} (target 1/137 = {ALPHA_EM:.6f})
  c_eff / d_eff = {c_eff / d_eff:.6f} (target π/2 = {PI/2:.6f})

Predicted masses (β = {optimal_beta:.4f}):
  Top:   173.0 GeV
  Charm: {m_charm:.2f} GeV
  Up:    {m_up:.6f} GeV
""")

print("=" * 80)