# Canvas Beta Finder & 1+1D UWE Simulation

Numerical verification of two canvas model predictions:

1. **Beta Finder:** Binary search converging to β = 1.868164, the internal lattice geometry parameter that determines the fermion mass hierarchy. Uses the charm quark mass (m_c = 1.27 GeV) as the target, with top quark mass (m_t = 173.0 GeV) as normalization.

2. **Bound State Formation:** 1+1D simulation of the Unified Wave Equation showing that colliding Gaussian wave packets form stable bound states when the threshold R = 4.0 is exceeded. Demonstrates the core mechanism of the canvas model.

## Quick Start

```

pip install numpy matplotlib scipy
python canvas_beta_simulation.py

```

## Results

- Beta converges to 1.868164
- Masses: Top = 173.0 GeV, Charm = 1.27 GeV, Up ≈ 0.009 GeV
- Bound state forms at R = 4.0 with initial amplitude ≥ 3.0
- Weights locked to c_eff/d_eff = π/2

## Citation

Ong, E. A Unified Framework of Fundamental Physics (2026).

## License

MIT
