# Hierarchical Inference and Closure Learning via Bilevel Optimization

Code accompanying the paper on bilevel optimization for simultaneous hierarchical Bayesian inference and closure learning in physics-based models.

## Overview

This repository implements a framework that jointly:
1. **Infers system parameters** across multiple related physical systems via a hierarchical Bayesian model, using ensemble MALA (Metropolis-Adjusted Langevin Algorithm) for posterior sampling.
2. **Learns a closure term** `α` (an MLP) representing unknown or misspecified physics (e.g., a nonlinear damping force or nonlinear wave speed).
3. **Trains a forward surrogate model** `β` (an FNO or PINN) that maps system parameters to PDE/ODE solutions conditioned on the current closure `α`.

The closure and surrogate are trained jointly through a **bilevel optimization** scheme:
- **Inner loop (lower level):** Optimize `β` to minimize a physics residual or supervised loss given `α`.
- **Outer loop (upper level):** Optimize `α` to maximize the marginal log-likelihood of observations, with `β` treated as having converged from the inner loop.

After each bilevel step, an ensemble MALA sweep updates the hierarchical posterior over per-system parameters `{θ_i}` and population-level hyperparameters `φ = (μ_φ, τ_φ)`.

## Benchmark Problems

### 1. Mass-Damper (ODE) — `mass-damper/`

A nonlinear oscillator with an unknown damping term:

```
m ẍ + α(ẋ) + k x = F(t),    F(t) = 10 sin(t)
```

The true (hidden) closure is `α(ẋ) = a ẋ + b ẋ³`. Per-system parameters are `θ_i = (log k, x₀, ẋ₀)`.

| Script | Forward model `β` | Inner loss | Notes |
|---|---|---|---|
| `train_solver.py` | Leapfrog/RK4 ODE solver | — | Alternating inference, no bilevel |
| `train_PINNs.py` | PINN | Physics (ODE residual + IC) | Full bilevel |
| `train_FNO_physics.py` | FNO (1D) | Physics residual | Full bilevel |
| `train_FNO_supervised.py` | FNO (1D) | Supervised (data) | Full bilevel |
| `train_PINNs_nonhierarchy.py` | PINN | Physics | Ablation: no hierarchy |

### 2. Burgers Equation (1D PDE) — `burgers/`

A modified Burgers equation with an unknown nonlinear wave speed:

```
u_t + c(u) u_x = ν u_xx,    c(u) = 7(σ(3u) − 0.5)
```

Per-system parameters are `θ_i = (log ν, amplitude)`. The grid spans `x ∈ [−1, 1]`, `t ∈ [0, 0.5]`.

| Script | Forward model `β` | Inner loss |
|---|---|---|
| `train_FNO_supervised.py` | FNO (2D) | Supervised (data) |

### 3. Poisson Equation (2D PDE) — `poisson/`

A nonlinear elliptic PDE with a learnable coefficient field:

```
∇ · (a(u; z) ∇u) = f(x, y)
```

where `a(u; z) = softplus(Σ_j z_j φ_j(x)) · σ(g(u))` and `g(u) = u²/2`. Per-system parameters `θ_i = (z₁, z₂, z₃)` control the coefficient expansion. The grid is `[0,1]² ` with `50×50` nodes.

| Script | Forward model `β` | Inner loss | Notes |
|---|---|---|---|
| `train_PINNs.py` | PINN | Physics (PDE residual) | Full bilevel |
| `train_FNO_supervised.py` | FNO (2D) | Supervised (data) | Full bilevel |
| `train_FNO_physics.py` | FNO (2D) | Physics residual | Full bilevel |
| `train_PINNs_nonhierarchy.py` | PINN | Physics | Ablation: no hierarchy |
| `train_solver.py` | FEM (FPI via JaxOpt) | — | Alternating inference |

## Repository Structure

```
.
├── mass-damper/
│   ├── data_generation.py          # RK4/leapfrog ODE solver, observation generation
│   ├── fno.py                      # FNO architecture (1D/2D)
│   ├── mlp.py                      # MLP (closure α) and PINN (surrogate β)
│   ├── langevin_FNO.py             # Ensemble MALA for FNO-based β
│   ├── langevin_PINNs.py           # Ensemble MALA for PINN-based β
│   ├── langevin_PINNs_nonhierarchy.py
│   ├── losses_FNO.py               # Outer/inner losses for FNO variant
│   ├── losses_PINNs.py             # Physics (ODE) residual and likelihood losses
│   ├── losses_PINNs_nonhierarchy.py
│   ├── train_solver.py             # Alternating inference with exact solver
│   ├── train_PINNs.py              # Bilevel: PINN surrogate
│   ├── train_FNO_physics.py        # Bilevel: FNO surrogate, physics inner loss
│   ├── train_FNO_supervised.py     # Bilevel: FNO surrogate, supervised inner loss
│   ├── train_PINNs_nonhierarchy.py # Ablation: non-hierarchical
│   ├── constant_*.py               # Grid constants loaded from config
│   ├── utils.py / utils_nonhierarchy.py
│   └── config_*.yml                # Hyperparameter configs per method
│
├── burgers/
│   ├── data_generation.py          # Finite-difference Burgers solver, obs generation
│   ├── fno.py                      # FNO architecture (2D)
│   ├── mlp.py                      # MLP closure α
│   ├── langevin_FNO.py             # Ensemble MALA
│   ├── losses_FNO.py               # Supervised loss and log-likelihood
│   ├── train_FNO_supervised.py     # Bilevel training entry point
│   ├── constant_FNO_supervised.py  # Grid constants
│   ├── utils.py
│   └── config_FNO_supervised.yml
│
├── poisson/
│   ├── data_generation.py          # FEM solver (JaxOpt fixed-point), obs generation
│   ├── fno.py                      # FNO architecture (2D)
│   ├── mlp.py                      # MLP closure α and PINN surrogate β
│   ├── langevin_FNO.py             # Ensemble MALA for FNO-based β
│   ├── langevin_PINNs.py           # Ensemble MALA for PINN-based β
│   ├── langevin_PINNs_nonhierarchy.py
│   ├── losses_FNO.py
│   ├── losses_PINNs.py
│   ├── losses_PINNs_nonhierarchy.py
│   ├── train_solver.py
│   ├── train_PINNs.py
│   ├── train_FNO_supervised.py
│   ├── train_FNO_physics.py
│   ├── train_PINNs_nonhierarchy.py
│   ├── constant_*.py
│   ├── utils.py
│   └── config_*.yml
│
└── Bilevel (1).pdf                 # Paper
```

## Method Details

### Hierarchical Model

Parameters are drawn from a two-level hierarchy:

```
φ = (μ_φ, log τ_φ)   ~  hyperprior (Normal-LogNormal)
θ_i | φ              ~  N(μ_φ, τ_φ)    for i = 1, …, N
y_i | θ_i            ~  N(H_i β(θ_i; α), σ²_obs)
```

The joint state sampled by MALA is the concatenation `[θ_1, …, θ_N, μ_φ, log τ_φ]`.

### Bilevel Optimization

At each outer epoch:

1. **MALA sweep** (inner inference): Run `n_samples` steps of ensemble MALA to update the chain over `[θ_1, …, θ_N, μ_φ, log τ_φ]` using the current `α` and `β`.
2. **Bilevel step**:
   - *Inner loop*: Take `fno_steps` gradient steps on `β` to minimize the physics residual (or supervised loss) under the current `α`, using a randomly sampled mini-batch of chain states.
   - *Outer loop*: Differentiate through the inner loop (via `jax.checkpoint` rematerialization) to compute `∂L_outer/∂α` and update `α` with Adam.
3. **Logging**: L2 error of learned closure vs. ground truth, log-likelihood loss, and FNO/PINN predictions are logged to WandB.

### Models

- **Closure `α`**: MLP with SiLU activations (typically 2–4 hidden layers of width 64). Maps scalar velocity / field value to scalar force / coefficient.
- **Surrogate `β`** (FNO variant): Fourier Neural Operator with P-layer lifting, `n_layers` FNO blocks, and Q-layer projection. Supports 1D (time series) and 2D (space-time / space-space) inputs.
- **Surrogate `β`** (PINN variant): Dense MLP with `tanh` activations; takes `(t, θ_i)` as input and outputs the solution at that point.

### Sampler

Ensemble MALA with adaptive step size:
- Proposal: `θ* = θ + ½ε² C ∇ log p + ε R W`,  where `C = Cov(chain)` and `R = chol(C)`.
- Metropolis–Hastings correction step.
- Step size adaptation: `ε ← ε √(1 + lr · (accept_prob − target_ratio))`.

## Dependencies

```
jax
jaxlib
flax
optax
jaxopt
ml_collections
wandb
numpy
matplotlib
```

Install with:
```bash
pip install jax jaxlib flax optax jaxopt ml-collections wandb numpy matplotlib
```

GPU support requires the appropriate `jaxlib` build; see the [JAX installation guide](https://github.com/google/jax#installation).

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

# Burgers equation
cd ../burgers
python train_FNO_supervised.py

# Poisson equation: bilevel with PINN surrogate
cd ../poisson
python train_PINNs.py
```

All runs log metrics and plots to [Weights & Biases](https://wandb.ai). Set your entity in the `wandb:` section of the config file.

### Key Config Parameters

| Parameter | Description |
|---|---|
| `data_systems.n_systems` | Number of independent physical systems |
| `data_systems.obser_noise` | Observation noise standard deviation |
| `langevin_sampler.n_chains` | Number of MALA chains |
| `langevin_sampler.n_samples` | MALA steps per outer epoch |
| `langevin_sampler.step_size` | Initial MALA step size |
| `models.mlp_alpha.*` | Closure MLP architecture and optimizer settings |
| `models.fno_beta.*` | FNO surrogate architecture and optimizer settings |
| `models.pinn_beta.*` | PINN surrogate architecture and optimizer settings |

## Outputs

Checkpoints are saved inside the WandB run directory under `checkpoints/`:
- `best_model_1` — model with lowest closure L2 error seen so far
- `current_model_1` — periodic snapshot every 200 epochs
- `parameters.npy`, `H_mats.npy`, `y_obser.npy` — ground truth data
- `whole_chain.npy`, `*_loss.npy` — sampler chain and loss histories
