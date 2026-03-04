# Hierarchical Inference and Closure Learning via Bilevel Optimization

Code accompanying the paper on bilevel optimization for simultaneous hierarchical Bayesian inference and closure learning in physics-based models.

## Overview

This repository implements a framework that jointly:
1. **Infers system parameters** across multiple related physical systems via a hierarchical Bayesian model, using ensemble MALA (Metropolis-Adjusted Langevin Algorithm) for posterior sampling.
2. **Learns a closure model** $`\alpha`$ (an MLP) representing unknown physics (e.g., nonlinear damping law).
3. **Trains a forward surrogate model** $`\beta`$ (an FNO or PINN) that maps system parameters to PDE/ODE solutions conditioned on the current closure $`\alpha`$.

At each epoch, an **ensemble MALA** step first updates the hierarchical posterior over unknown parameters $`\boldsymbol{\theta}`$ and population-level hyperparameters $`\boldsymbol{\phi} = (\boldsymbol{\mu}_{\boldsymbol{\phi}}, \boldsymbol{\tau}_{\boldsymbol{\phi}})`$, using the current $`\alpha`$ and $`\beta`$. The resulting chain samples are then passed to a **bilevel optimization** step:
- **Inner loop (lower level):** Optimize $\beta$ to minimize a physics residual or supervised loss given $\alpha$.
- **Outer loop (upper level):** Optimize $\alpha$ to maximize the marginal log-likelihood of observations, with $\beta$ treated as having converged from the inner loop.

## Benchmark Problems

### 1. Mass-Damper (ODE) — `mass-damper/`

A nonlinear oscillator with an unknown damping law:

$$
\ddot{u} + f(\dot{u}) + ku = F(t), \quad F(t) = 10\sin(t),
$$

with initial position $x_0$ and initial velocity $\dot{x}_0$. The true  closure is $f(\dot{u}) = 0.08\dot{u} + 0.08\dot{u}^3$. Unknown parameters are $\boldsymbol{\theta} = (\log k, x_0, \dot{x}_0)$.

| Script | Forward model $\beta$ | Inner loss |
|---|---|---|
| `train_solver.py` | Leapfrog/RK4 ODE solver | — |
| `train_PINNs.py` | PINN | Physics residual |
| `train_FNO_physics.py` | FNO (1D) | Physics residual |
| `train_FNO_supervised.py` | FNO (1D) | Supervised |
| `train_PINNs_nonhierarchy.py` | PINN | Physics residual |

### 2. Darcy Flow (2D PDE) — `darcy/`

A nonlinear elliptic PDE with a learnable permeability field:

$$
-\nabla \cdot \bigl(a(u, \mathbf{x}) \nabla u\bigr) = s(\mathbf{x})
$$

where $a(u, \mathbf{x}) = \exp \left(\sum_j z_j \phi_j(\mathbf{x})\right) \cdot \boldsymbol{\sigma}(f(u))$ and $f(u) = u^2/2$. Unknown parameters are $\boldsymbol{\theta} = (z_1, z_2, z_3)$. The grid is $[0,1]^2$ with $50 \times 50$ nodes.

| Script | Forward model $\beta$ | Inner loss |
|---|---|---|
| `train_solver.py` | FEM (Fixed Point via JaxOpt) | — |
| `train_PINNs.py` | PINN | Physics residual |
| `train_FNO_supervised.py` | FNO (2D) | Supervised |
| `train_FNO_physics.py` | FNO (2D) | Physics residual |
| `train_PINNs_nonhierarchy.py` | PINN | Physics residual |


### 3. Generalized Burgers Equation (PDE) — `burgers/`

A generalized Burgers equation with an unknown convective term:

$$
u_t + f(u) u_x = \nu u_{xx}, \quad u(x,0) = z \sin(2\pi x) \sin(\pi x).
$$

Nonlinear closure is $f(u) = 7\bigl(\boldsymbol{\sigma}(3u) - 0.5\bigr)$ and unknown parameters are $\boldsymbol{\theta} = (\log \nu, z)$. The grid spans $x \in [-1, 1]$, $t \in [0, 0.5]$.

| Script | Forward model $\beta$ | Inner loss |
|---|---|---|
| `train_FNO_supervised.py` | FNO (2D) | Supervised |

## Dependencies

Install all dependencies from the provided `environment.yml`:

```bash
conda env create -f environment.yml
conda activate jax
```

GPU support requires the appropriate `jaxlib` build; please see the [JAX installation guide](https://github.com/google/jax#installation).

## Running Experiments

Each script reads its configuration from a YAML file in the same directory. Edit the relevant `config_*.yml` before running.

```bash
# Mass-damper: bilevel with PINN surrogate
cd mass-damper
python train_PINNs.py

# Mass-damper: bilevel with FNO surrogate (supervised inner loss)
python train_FNO_supervised.py

# Mass-damper: alternating inference with exact ODE solver
python train_solver.py

# Darcy Flow: bilevel with PINN surrogate
cd ../darcy
python train_PINNs.py

# Burgers equation
cd ../burgers
python train_FNO_supervised.py
```

All runs log metrics and plots to [Weights & Biases](https://wandb.ai). Set your entity in the `wandb:` section of the config file.

## Repository Structure

```
.
├── mass-damper/
│   ├── data_generation.py              # RK4/leapfrog ODE solver, observation generation
│   ├── fno.py                          # FNO architecture (1D/2D)
│   ├── mlp.py                          # MLP (closure alpha) and PINN (surrogate beta)
│   ├── utils.py                        # Shared utilities
│   ├── utils_nonhierarchy.py           # Utilities for non-hierarchical ablation
│   ├── langevin_FNO.py                 # Ensemble MALA for FNO-based beta
│   ├── langevin_PINNs.py               # Ensemble MALA for PINN-based beta
│   ├── langevin_PINNs_nonhierarchy.py  # Ensemble MALA for non-hierarchical ablation
│   ├── losses_FNO.py                   # Losses for FNO variant
│   ├── losses_PINNs.py                 # Physics (ODE) residual and likelihood losses
│   ├── losses_PINNs_nonhierarchy.py    # Losses for non-hierarchical ablation
│   ├── constant_FNO_physics.py         # Grid constants for FNO physics variant
│   ├── constant_FNO_supervised.py      # Grid constants for FNO supervised variant
│   ├── constant_PINNs.py               # Grid constants for PINN variant
│   ├── constant_PINNs_nonhierarchy.py  # Grid constants for non-hierarchical ablation
│   ├── constant_solver.py              # Grid constants for solver variant
│   ├── train_solver.py                 # Alternating inference with exact ODE solver
│   ├── train_PINNs.py                  # Bilevel: PINN surrogate
│   ├── train_FNO_physics.py            # Bilevel: FNO surrogate, physics inner loss
│   ├── train_FNO_supervised.py         # Bilevel: FNO surrogate, supervised inner loss
│   ├── train_PINNs_nonhierarchy.py     # Ablation: non-hierarchical
│   ├── config_FNO_physics.yml
│   ├── config_FNO_supervised.yml
│   ├── config_PINNs.yml
│   ├── config_PINNs_nonhierarchy.yml
│   └── config_solver.yml
│
├── darcy/                            # Darcy Flow
│   ├── data_generation.py              # FEM solver (JaxOpt fixed-point), observation generation
│   ├── fno.py                          # FNO architecture (2D)
│   ├── mlp.py                          # MLP closure alpha and PINN surrogate beta
│   ├── utils.py                        # Shared utilities
│   ├── langevin_FNO.py                 # Ensemble MALA for FNO-based beta
│   ├── langevin_PINNs.py               # Ensemble MALA for PINN-based beta
│   ├── langevin_PINNs_nonhierarchy.py  # Ensemble MALA for non-hierarchical ablation
│   ├── losses_FNO.py                   # Losses for FNO variant
│   ├── losses_PINNs.py                 # Physics (PDE) residual and likelihood losses
│   ├── losses_PINNs_nonhierarchy.py    # Losses for non-hierarchical ablation
│   ├── constant_FNO_physics.py
│   ├── constant_FNO_supervised.py
│   ├── constant_PINNs.py
│   ├── constant_PINNs_nonhierarchy.py
│   ├── constant_solver.py
│   ├── train_solver.py                 # Alternating inference with FEM solver
│   ├── train_PINNs.py                  # Bilevel: PINN surrogate
│   ├── train_FNO_physics.py            # Bilevel: FNO surrogate, physics inner loss
│   ├── train_FNO_supervised.py         # Bilevel: FNO surrogate, supervised inner loss
│   ├── train_PINNs_nonhierarchy.py     # Ablation: non-hierarchical
│   ├── config_FNO_physics.yml
│   ├── config_FNO_supervised.yml
│   ├── config_PINNs.yml
│   ├── config_PINNs_nonhierarchy.yml
│   └── config_solver.yml
│
└── burgers/
    ├── data_generation.py          # Finite-difference Burgers solver, observation generation
    ├── fno.py                      # FNO architecture (2D)
    ├── mlp.py                      # MLP closure alpha
    ├── utils.py                    # Shared utilities
    ├── langevin_FNO.py             # Ensemble MALA
    ├── losses_FNO.py               # Supervised loss and log-likelihood
    ├── constant_FNO_supervised.py  # Grid constants
    ├── train_FNO_supervised.py     # Bilevel: FNO surrogate, supervised inner loss
    └── config_FNO_supervised.yml
```