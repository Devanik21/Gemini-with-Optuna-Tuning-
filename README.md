# Gemini With Optuna Tuning

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square) ![Stars](https://img.shields.io/github/stars/Devanik21/Gemini-with-Optuna-Tuning-?style=flat-square&color=yellow) ![Forks](https://img.shields.io/github/forks/Devanik21/Gemini-with-Optuna-Tuning-?style=flat-square&color=blue) ![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github) ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> Systematic hyperparameter optimisation for Gemini-powered applications — Optuna-driven search over prompt parameters, generation settings, and model selection.

---

**Topics:** `deep-learning` · `bayesian-optimization` · `gemini` · `generative-ai` · `google-ai` · `hyperparameter-optimization` · `large-language-models` · `llm` · `model-tuning` · `optuna`

## Overview

This project demonstrates a rigorous hyperparameter optimisation workflow for LLM-powered applications
using Optuna — Bayesian optimisation framework — to systematically search over the parameter space of
Gemini API calls. Unlike trial-and-error prompt engineering, this approach treats each parameter
combination (temperature, top_p, top_k, max_tokens, system prompt variant, few-shot example count)
as a hyperparameter in a measurable optimisation problem and uses Optuna's Tree-structured Parzen
Estimator (TPE) algorithm to efficiently navigate the search space.

The framework defines a custom objective function that evaluates each Gemini configuration on a
held-out evaluation set with a task-specific quality metric: ROUGE-L for summarisation tasks,
exact match accuracy for classification, semantic similarity for question answering, or a
custom LLM-as-judge score for subjective quality assessment. The objective function is differentiable
with respect to discrete parameter choices through the TPE surrogate model, enabling intelligent
exploration/exploitation trade-off beyond random or grid search.

An Optuna Study persists all trial results to a SQLite database, providing a complete record
of the hyperparameter search history with parameter importance analysis, parallel coordinate plots
of the Pareto front for multi-objective tuning, and automatic best-trial selection.

---

## Motivation

Most LLM applications are deployed with default generation parameters (temperature 0.7, default top_p)
that may be far from optimal for the specific task. A few percentage points of quality improvement
from systematic hyperparameter tuning can have significant impact at production scale. This project
provides a reusable, principled optimisation framework that makes that improvement systematic
rather than accidental.

---

## Architecture

```
Task Definition: eval_set + quality_metric
        │
  Optuna Study (TPE sampler)
        │
  Trial Loop:
  ├── Sample: temperature, top_p, top_k, max_tokens
  │           system_prompt_variant, n_shots
  │
  ├── Execute: Gemini API call per eval sample
  │
  └── Score: ROUGE / exact_match / semantic_sim / LLM-judge
        │
  Best trial → production configuration
        │
  Optuna dashboard: parameter importance, Pareto front
```

---

## Features

### Optuna TPE Hyperparameter Search
Tree-structured Parzen Estimator Bayesian optimisation over continuous parameters (temperature, top_p, top_k) and discrete parameters (system prompt variant, few-shot count) simultaneously.

### Multi-Objective Optimisation
Simultaneously optimise quality and latency (or cost) using Optuna's NSGAIISampler for Pareto front identification — trading off response quality against token cost.

### Task-Specific Quality Metrics
Pluggable evaluation functions: ROUGE-L (summarisation), exact match (classification), cosine semantic similarity (Q&A), and LLM-as-judge 1–10 scoring (open-ended tasks).

### Prompt Template Search
Include discrete prompt structure choices as hyperparameters: instruction phrasing variants, output format specifications (JSON vs. prose), and few-shot example selection from a pool.

### Persistent Study Database
All Optuna trials stored in SQLite — resume interrupted search, share studies across team members, and maintain complete audit trail of explored configurations.

### Parameter Importance Analysis
Optuna's fANOVA-based parameter importance ranking reveals which hyperparameters most strongly influence the objective, enabling focused manual refinement.

### Parallel Coordinate Visualisation
Interactive parallel coordinate plot of all completed trials coloured by objective value — visually identifying optimal parameter regions.

### Auto-Export Best Configuration
Automatic export of the best-found configuration as a Python dict and JSON file ready for production deployment.

---

## Tech Stack

| Library / Tool | Role | Why This Choice |
|---|---|---|
| **Google Gemini API** | LLM backend | gemini-2.0-flash and gemini-1.5-pro generation endpoints |
| **Optuna** | Hyperparameter optimisation | TPE sampler, NSGAIISampler, SQLite persistence |
| **ROUGE score** | Evaluation metric | ROUGE-L for generation quality measurement |
| **sentence-transformers** | Semantic similarity | Embedding-based semantic similarity metric |
| **Streamlit** | Dashboard | Trial results visualisation and configuration export |
| **pandas** | Results analysis | Trial data aggregation and statistical summary |
| **python-dotenv** | Config | API key management |

---

## Getting Started

### Prerequisites

- Python 3.9+ (or Node.js 18+ for TypeScript/JavaScript projects)
- A virtual environment manager (`venv`, `conda`, or equivalent)
- API keys as listed in the Configuration section

### Installation

```bash
git clone https://github.com/Devanik21/Gemini-with-Optuna-Tuning-.git
cd Gemini-with-Optuna-Tuning-
python -m venv venv && source venv/bin/activate
pip install optuna google-generativeai rouge-score sentence-transformers \
            streamlit pandas python-dotenv
echo 'GOOGLE_API_KEY=your_key' > .env
python run_study.py --task summarisation --trials 100
```

---

## Usage

```bash
# Run optimisation study
python run_study.py --task summarisation --trials 100 --study_name gemini_sum_v1

# Resume an interrupted study
python run_study.py --study_name gemini_sum_v1 --trials 50 --resume

# View best configuration
python best_config.py --study_name gemini_sum_v1

# Launch results dashboard
streamlit run dashboard.py -- --study_name gemini_sum_v1

# Export best config for production
python export.py --study_name gemini_sum_v1 --output prod_config.json
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `GOOGLE_API_KEY` | `(required)` | Google Gemini API key |
| `STUDY_DB` | `studies/optuna_study.db` | SQLite database for Optuna study persistence |
| `DEFAULT_TASK` | `summarisation` | Task type: summarisation, classification, qa, open_ended |
| `N_TRIALS` | `100` | Number of Optuna optimisation trials |
| `PARALLEL_JOBS` | `1` | Parallel trial execution (1 = sequential, N = concurrent) |

> Copy `.env.example` to `.env` and populate required values before running.

---

## Project Structure

```
Gemini-with-Optuna-Tuning/
├── README.md
├── requirements.txt
├── app.py
└── ...
```

---

## Roadmap

- [ ] Multi-model joint optimisation: include model choice (Flash vs Pro) as a discrete hyperparameter
- [ ] Automatic evaluation set generation: use an LLM to create task-specific evaluation samples
- [ ] Cost-constrained optimisation: hard budget constraint on total API cost during the search
- [ ] Continuous learning: re-run optimisation when new Gemini model versions are released
- [ ] Team dashboard: shared Optuna study with experiment tracking and team-wide best config registry

---

## Contributing

Contributions, issues, and suggestions are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-idea`
3. Commit your changes: `git commit -m 'feat: add your idea'`
4. Push to your branch: `git push origin feature/your-idea`
5. Open a Pull Request with a clear description

Please follow conventional commit messages and add documentation for new features.

---

## Notes

Each Optuna trial makes multiple Gemini API calls (one per evaluation sample). A study of 100 trials on an evaluation set of 50 samples requires 5,000 API calls — budget accordingly. Use gemini-2.0-flash for cost-efficient optimisation and validate the best configuration with gemini-1.5-pro if higher quality is required.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with curiosity, depth, and care — because good projects deserve good documentation.*
