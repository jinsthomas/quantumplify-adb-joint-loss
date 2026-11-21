# Quantumplify: quantum amplitude estimation for ADB joint-loss targets

Lightweight prototype for the ADB Quantum Innovation Challenge, "Joint-Loss Modelling for
Disaster Risk Financing" (Phase 1, November 2026). Applicant: Jins Thomas (individual), team name Quantumplify.

The prototype answers the test set in the challenge brief on the ADB Disaster Relief Bonds
benchmark (Kyrgyz Republic and Tajikistan, USD 80 million each, three-year term): does a quantum
method reach higher accuracy per query than the four classical sampling baselines?

## What it contains

| Part | File | What it does |
|---|---|---|
| Exact model | `jointloss/model.py` | Rebuilds the benchmark from the brief: 25-state earthquake Markov chain with shared events, shared precipitation triggers, 100-state annual distribution, three-year capped convolution. Reproduces all 25 reference values of Table 5. |
| Classical baselines | `jointloss/baselines.py` | Monte Carlo, scrambled Sobol (8 scrambles), stratified sampling by first payout year, importance sampling (half true, half loss-tilted). |
| Quantum estimator | `jointloss/qae.py` | 9-qubit amplitude-estimation circuits (state preparation A, Grover operator Q) and maximum-likelihood amplitude estimation (Suzuki et al. 2020). Gate counts by transpilation. |
| Experiments | `run_experiments.py` | Error against query budget for every method, target and scenario; queries needed for 10% and 1% error; schedule study; gate counts. |
| Figures | `make_figures.py` | Charts from the CSV results. |
| Tests | `tests/` | 31 tests: Table 5 and intermediate values of the brief, unbiasedness of each baseline, circuit amplitudes, estimator accuracy. |

## Setup and execution

Python 3.12 is required.

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q tests          # 31 tests, about 25 seconds
.venv/bin/python -m jointloss.model          # prints the five targets for the five scenarios
.venv/bin/python run_experiments.py          # full benchmark, about 16 minutes (add --quick for 2 minutes)
.venv/bin/python make_figures.py             # charts in results/figures/
```

The random seed is fixed (`20261121`), so the CSV files in `results/` are reproducible.

## Hardware and simulator details

- Machine: Apple M3 Max laptop, macOS 26.5; CPU only, no GPU and no quantum hardware.
- Quantum simulation: exact statevector simulation with Qiskit 2.6.2 (`qiskit.quantum_info.Statevector`), 9 qubits.
- Circuit check: for every target, the probability of measuring 1 on the objective qubit after
  `Q^m A` equals `sin^2((2m + 1) theta)` to within 1e-10 (tested for m = 0, 1, 2, 5). The
  estimator experiments then draw measurement shots from this law, which is what an ideal
  shot-based simulation of the same circuits returns. No hardware noise is modelled in this version.
- Gate counts: transpiled to the basis `cx, rz, sx, x` with all-to-all connectivity, optimisation level 3.

## Sample results (`results/`)

Exact reference values (`exact_targets.csv`) match Table 5 of the brief for all five scenarios.

Queries needed in the base scenario (`queries_to_target.csv`; one query = one simulated term path
for the classical methods, one application of A for the quantum method; * = extrapolated beyond
the largest budget run, 262,144 paths):

| Target, 10% relative error | Monte Carlo | Sobol | Stratified | Importance | QAE |
|---|---|---|---|---|---|
| Both bonds exhausted (3.2e-5) | 2,584,679* | 2,621,514* | 764,061* | 115,461 | 24,382 |
| Expected loss above USD 80m | 13,901 | 5,583 | 3,439 | 1,595 | 2,350 |
| Portfolio loss above USD 80m | 8,617 | 3,268 | 2,117 | 1,366 | 957 |
| Both bonds pay | 2,164 | 1,105 | 509 | 1,146 | 633 |
| Expected principal loss | 486 | 128 | 128 | 128 | 472 |

| Target, 1% relative error | Monte Carlo | Sobol | Stratified | Importance | QAE |
|---|---|---|---|---|---|
| Both bonds exhausted (3.2e-5) | 192,712,828* | 217,595,505* | 53,361,454* | 19,287,265* | 259,125 |
| Expected loss above USD 80m | 1,441,060* | 199,975 | 375,635* | 132,596 | 25,281 |
| Portfolio loss above USD 80m | 950,196* | 123,851 | 213,015 | 120,973 | 12,106 |
| Both bonds pay | 200,410 | 39,757 | 58,577 | 103,826 | 5,849 |
| Expected principal loss | 41,091 | 1,338 | 5,347 | 11,704 | 5,366 |

Circuit size (`resources.csv`): 9 qubits; A has 503 two-qubit gates; one Grover iterate Q has
1,229 two-qubit gates (two-qubit depth 1,107).

## Limits of this version

- **Query counts are not run time.** One quantum query is a coherent circuit of about 1,200
  two-qubit gates per Grover iterate, and the rare target needs Grover powers in the thousands.
  No current quantum computer can run this; it needs fault-tolerant hardware.
- **The distribution is loaded, not computed, by the circuit.** The term distribution is computed
  classically (which already gives the exact answer in under a millisecond) and loaded exactly.
  This measures query complexity only. A loader that performs the three-year convolution inside
  the circuit, and scales with the number of countries, is Phase 2 work.
- **Classical methods win in places.** Sobol beats QAE on the expected principal loss at both
  accuracy levels; stratified and importance sampling win on two targets at 10% error.
- **Importance sampling detail.** The brief names the half-true, half-loss-tilted proposal; the
  tilt used here is proportional to p(x) times the annual portfolio loss, applied per year.
  A better-tuned proposal would narrow the gap to QAE.
- **Schedule.** The usual doubling schedule of Grover powers gave occasional large errors on the
  rare target; a growth factor of 1.5 is used instead (`schedule_study.csv`).
- No noise model and no hardware run.

## Disclosure of AI tools used

The code, tests, experiment scripts and this README were written with an AI coding assistant
(Claude, by Anthropic, used through Claude Code), directed and reviewed by the applicant. The AI
assistant was also used to research the challenge documents and literature and to draft the
proposal text. All numerical results come from running the code in this repository.

## Third-party software and references

Open-source libraries: NumPy, SciPy, Matplotlib, Qiskit, pytest.

- ADB / QAI Ventures, "Quantum Approaches to Joint-Loss Modelling for Natural Catastrophe Risk: Challenge Description" (2026): model, reference values and baselines.
- Suzuki et al., "Amplitude estimation without phase estimation", Quantum Information Processing 19, 75 (2020).
- Brassard et al., "Quantum amplitude amplification and estimation" (2002).
- Woerner and Egger, "Quantum risk analysis", npj Quantum Information 5, 15 (2019).

## Confidentiality

This repository is private, in line with the challenge rules on confidentiality of submissions.
Access for the jury is available on request.
