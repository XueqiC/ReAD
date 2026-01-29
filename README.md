# ReAD: Reinforcement-guided Capability Distillation under Budget Constraints
Offical Code Repo for **ReAD** (*Re*inforcement-guided c*A*pability *D*istillation), an adaptive capability distillation framework that (i) infers **task-essential capabilities**, (ii) performs **capability-targeted data generation on the fly**, and (iii) uses an **uncertainty-aware contextual bandit** to allocate a fixed distillation budget across **interdependent** capabilities.

## TL;DR
We make two empirical observations under a fixed token budget:
1) **Cross-capability transfer**: improving one capability can systematically shift others.  
2) **Budget inefficiency**: additional distillation often yields diminishing task gains and increasing harmful spillover.  
ReAD adapts allocation online using low-cost monitoring and improves downstream utility under the same budget.

---

## Repository Contents

- `configs/`  
  YAML configs for models, tasks, budgets, bandit, and data generation.
- `scripts/`  
  End-to-end entry points (data generation, training, evaluation, profiling).
- `read/`  
  Utilities to compute capability profiles, normalization, and logs.
- `src/read/`  
  Core ReAD implementation (requirement estimator, generator, bandit policy).
- `probes/`  
  Probe suite definitions (lightweight monitoring prompts + scoring).
- `benchmarks/`  
  Task evaluation wrappers and adapters.
- `data/`  
  Local cache for prompts, manifests, and generated teacher outputs (not included).
- `outputs/`  
  Training logs, profiles, and final reports.
- `checkpoint/`  
  Saved models and intermediate student checkpoints.

> **Note:** Capability scores are normalized to a common **[0, 100]** scale throughout monitoring and transfer computation.

---

## Installation

### 1) Create environment
```bash
conda create -n read python=3.10 -y
conda activate read
