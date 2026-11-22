# Quantumplify: quantum amplitude estimation for ADB joint-loss targets

Lightweight prototype for the ADB Quantum Innovation Challenge, "Joint-Loss Modelling for
Disaster Risk Financing" (Phase 1, November 2026). Applicant: Jins Thomas (individual), team name Quantumplify.

The prototype tests, on the ADB Disaster Relief Bonds benchmark (Kyrgyz Republic and Tajikistan,
USD 80 million each, three-year term), whether quantum amplitude estimation (QAE) reaches higher
accuracy per query than classical sampling, and what it would cost on real hardware.

**Result in one paragraph.** QAE beats the four baselines named in the brief on the rare joint
tail, and its error falls as 1/M against 1/sqrt(N) for sampling. An importance sampler tuned to
the target beats QAE on the rarest target down to about 0.5% relative error, and the exact solver
takes a fraction of a millisecond, so there is no quantum advantage on the two-country benchmark.
A shared-shock loader makes the circuit scale linearly with the number of countries.

## What it contains

| Part | File | What it does |
|---|---|---|
| Exact model | `jointloss/model.py` | Rebuilds the benchmark from the brief: 25-state earthquake Markov chain with shared events, shared precipitation triggers, 100-state annual distribution, three-year capped convolution. Reproduces all 25 reference values of Table 5. |
| Baselines of the brief | `jointloss/baselines.py` | Monte Carlo, scrambled Sobol (8 scrambles), stratified sampling by first payout year, importance sampling (half true, half loss-tilted). |
| Tuned importance sampling | `jointloss/tuned_is.py`, `run_classical_exact.py` | Exact variance of any annual proposal (sum over all 100^3 paths), a proposal optimised per target, and the accuracy at which QAE overtakes it. |
| QAE, exact loading | `jointloss/qae.py` | 9-qubit circuits (state preparation A, Grover operator Q) for all five targets, and maximum-likelihood amplitude estimation (Suzuki et al. 2020). |
| QAE, shared-shock loader | `jointloss/loader.py`, `run_scaling.py` | State preparation from per-country tables on 3 + 4n qubits (conditional independence given shared events), for n countries; exact n-country references; gate counts for 2 to 10 countries. |
| Resources | `run_resources.py` | Fault-tolerant estimate with the Microsoft resource estimator and noise limits. |
| Organisers' conventions | `run_evaluation.py` | Pass or fail against the `evaluation-v1` tolerances at 8,192 samples; result files in the organisers' JSON layout. |
| Payout versus need | `jointloss/basis_risk.py`, `run_basis_risk.py` | Two illustrative basis-risk targets on the same circuit and baselines. The link between payout and need is an uncalibrated assumption. |
| Experiments, figures | `run_experiments.py`, `make_figures.py` | Error against query budget for every method, target and scenario; charts. |
| Tests | `tests/` | 45 tests: Table 5 and intermediate values of the brief, unbiasedness of each sampler, exact-variance formulas, circuit amplitudes, loader against exact two- and three-country distributions, estimator accuracy. |

## Setup and execution

Python 3.12 is required.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests          # 45 tests, about 20 seconds
.venv/bin/python -m jointloss.model          # the five targets for the five scenarios
.venv/bin/python run_experiments.py          # error curves, about 16 minutes (--quick: 2 minutes)
.venv/bin/python run_classical_exact.py      # exact classical requirements and QAE crossover
.venv/bin/python run_scaling.py              # shared-shock loader, 2 to 10 countries
.venv/bin/python run_evaluation.py           # organisers' tolerances and JSON result files
.venv/bin/python run_basis_risk.py           # payout-versus-need targets
.venv/bin/python make_figures.py             # charts in results/figures/
.venv/bin/pip install "qdk[qre,qiskit]"      # optional, for the next line
.venv/bin/python run_resources.py            # fault-tolerant estimate and noise limits
```

Random seeds are fixed, so the CSV files in `results/` are reproducible.

## Hardware and simulator details

- Machine: Apple M3 Max laptop, macOS 26.5; CPU only, no GPU and no quantum hardware.
- Quantum simulation: exact statevector simulation with Qiskit 2.6.2, 9 to 20 qubits.
- Circuit check: for every target, the probability of measuring 1 on the objective qubit after
  `Q^m A` equals `sin^2((2m + 1) theta)` to within 1e-10. The estimator experiments then draw
  measurement shots from this law, which is what an ideal shot-based simulation of the same
  circuits returns. No hardware noise is modelled in the estimator runs.
- Gate counts: transpiled to `cx, rz, sx, x`, all-to-all connectivity, optimisation level 3.
- Resource estimate: Microsoft Azure Quantum Resource Estimator, package `qdk` 1.34.0, run offline.

## Sample results (`results/`)

Exact reference values (`exact_targets.csv`) match Table 5 of the brief for all five scenarios.

Queries needed in the base scenario for 10% and 1% relative error (`queries_to_target.csv`,
`classical_exact.csv`). One query is one simulated term path for a classical method and one
application of A for QAE. Monte Carlo and importance sampling are exact variances; Sobol,
stratified and QAE are simulated (200 repetitions); * is extrapolated beyond 262,144 paths.

| Target, 10% error | Monte Carlo | Sobol | Stratified | Importance (brief) | Importance (tuned) | QAE |
|---|---|---|---|---|---|---|
| Both bonds exhausted (3.2e-5) | 3,161,151 | 2,621,514* | 764,061* | 120,754 | 1,495 | 24,382 |
| Expected loss above USD 80m | 14,313 | 5,583 | 3,439 | 1,419 | 269 | 2,350 |
| Portfolio loss above USD 80m | 8,201 | 3,268 | 2,117 | 1,125 | 354 | 957 |
| Both bonds pay | 2,166 | 1,105 | 509 | 1,061 | 237 | 633 |
| Expected principal loss | 460 | 128 or fewer | 128 or fewer | 111 | 67 | 472 |

| Target, 1% error | Monte Carlo | Sobol | Stratified | Importance (brief) | Importance (tuned) | QAE |
|---|---|---|---|---|---|---|
| Both bonds exhausted (3.2e-5) | 316,115,102 | 217,595,505* | 53,361,454* | 12,075,367 | 149,462 | 259,125 |
| Expected loss above USD 80m | 1,431,319 | 199,975 | 375,635* | 141,888 | 26,939 | 25,281 |
| Portfolio loss above USD 80m | 820,101 | 123,851 | 213,015 | 112,508 | 35,351 | 12,106 |
| Both bonds pay | 216,563 | 39,757 | 58,577 | 106,123 | 23,678 | 5,849 |
| Expected principal loss | 46,004 | 1,338 | 5,347 | 11,092 | 6,717 | 5,366 |

QAE needs fewer queries than tuned importance sampling below a relative error of about 4.3% (both
bonds pay), 2.4% (loss above USD 80m), 1.1% (expected loss above USD 80m), 0.9% (expected loss) and
0.5% (both bonds exhausted).

Shared-shock loader (`scaling.csv`, all bonds exhausted):

| Countries | Qubits in A | Two-qubit gates in A | Two-qubit gates in Q | Verified on statevector |
|---|---|---|---|---|
| 2 | 12 | 81 | 600 | yes |
| 3 | 16 | 127 | 1,088 | yes |
| 4 | 20 | 173 | 1,626 | yes |
| 5 | 24 | 219 | 2,162 | no |
| 10 | 44 | 449 | 5,992 | no |

The exact-loading circuit for two countries has 9 qubits, 503 two-qubit gates in A and 1,229 in Q.

Fault-tolerant estimate for the rarest target with the shared-shock circuit (`fault_tolerant.csv`):
a 10%-class estimate needs about 290,000 physical qubits and 20 minutes at 50 ns gates and a
physical error of 1e-3 (48,000 qubits at 1e-4), and 21 days at 100 microsecond gates. Noise limits
(`noise_limits.csv`): at a two-qubit error of 1e-3 about two Grover iterations are useful; the
signal for the rarest target first peaks at a Grover power of about 139.

Organisers' tolerances at 8,192 samples (`evaluation_v1_check.csv`): no sampling method passes all
five targets; QAE passes all five with 284,128 queries per target.

Payout-versus-need targets (`basis_risk.csv`, base scenario, illustrative assumption): need high
while the bond pays nothing 4.9%; bond pays while need is low 7.8%.

## Limits of this version

- **No quantum advantage on this benchmark.** The exact solver is instant, and tuned importance
  sampling needs fewer queries than QAE at 10% and 1% error on the rarest target.
- **Query counts are not run time.** One quantum query is a coherent circuit of 600 to 1,200
  two-qubit gates per Grover iterate; the rare target needs Grover powers in the hundreds to
  thousands. This needs fault-tolerant hardware.
- **The same structure helps classical methods.** The conditional independence that makes the
  quantum loader cheap also makes exact classical computation linear in the number of countries.
  A quantum estimator could matter only for models without that structure.
- **Loader coverage.** The shared-shock loader is implemented for the two flag targets (all bonds
  pay, all bonds exhausted); the three loss targets use exact loading.
- **Synthetic multi-country model.** The brief defines no model beyond two countries; the scaling
  study reuses the two calibrated countries and assumes one regional shock source.
- **Organisers' package.** The official benchmark package was not available to us. The JSON result
  files follow its public documentation and are not validated with it; `problem_id` is filled only
  for `adb-base` (from the organisers' sample file). The two World Bank scenarios are not covered.
- **Importance sampling detail.** The brief names the half-true, half-loss-tilted proposal; the
  tilt used here is proportional to p(x) times the annual portfolio loss, applied per year.
- **Schedule.** The doubling schedule of Grover powers gave occasional large errors on the rare
  target (see Callison and Browne 2022); a growth factor of 1.5 is used (`schedule_study.csv`).
- No noise model in the estimator runs and no hardware run.

## Disclosure of AI tools used

The code, tests, experiment scripts and this README were written with an AI coding assistant
(Claude, by Anthropic, used through Claude Code), directed and reviewed by the applicant. The AI
assistant was also used to research the challenge documents and literature and to draft the
proposal text. All numerical results come from running the code in this repository.

## Third-party software and references

Open-source libraries: NumPy, SciPy, Matplotlib, Qiskit, pytest; optional: Microsoft `qdk`.

- ADB / QAI Ventures, "Quantum Approaches to Joint-Loss Modelling for Natural Catastrophe Risk: Challenge Description" (2026): model, reference values and baselines.
- Suzuki et al., "Amplitude estimation without phase estimation", Quantum Information Processing 19, 75 (2020).
- Egger et al., "Credit risk analysis using quantum computers", IEEE Transactions on Computers 70(12) (2021): conditional-independence loading.
- Glasserman and Li, "Importance sampling for portfolio credit risk", Management Science 51(11) (2005).
- Callison and Browne, "Improved maximum-likelihood quantum amplitude estimation", arXiv:2209.03321 (2022).
- Tanaka et al., "Amplitude estimation via maximum likelihood on noisy quantum computer", Quantum Information Processing 20, 293 (2021).
- Beverland et al., "Assessing requirements to scale to practical quantum advantage", arXiv:2211.07629 (2022).
- Brassard et al., "Quantum amplitude amplification and estimation" (2002); Woerner and Egger, "Quantum risk analysis", npj Quantum Information 5, 15 (2019).

## Availability

This repository is public from the submission of the Phase 1 application (22 November 2026).
A ZIP copy is part of the submission package.
