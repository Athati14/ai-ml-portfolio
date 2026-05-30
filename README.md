# AI/ML Portfolio Projects

**Prepared for Anjani Thati | AI/ML Engineer | Generative AI · MLOps · Real-Time Inference**

10 portfolio-building projects with runnable starter code. Each project lives in its own folder and is designed to be pushed to a public GitHub repository as a verifiable portfolio artifact.

---

## Projects

| # | Project | Stack | Folder |
|---|---------|-------|--------|
| 1 | [Open-Source RAG Knowledge Assistant](#1-rag-knowledge-assistant) | Python, FAISS, sentence-transformers, LangChain, RAGAS | `01_rag_pipeline/` |
| 2 | [Agentic Multi-Step Workflow Agent](#2-agentic-workflow-agent) | Python, OpenAI tool-calling, JSON tracing | `02_agentic_workflow/` |
| 3 | [Fine-Tuning with LoRA / QLoRA](#3-lora-fine-tuning) | transformers, peft, bitsandbytes, trl | `03_lora_finetuning/` |
| 4 | [End-to-End MLOps Template](#4-mlops-template) | scikit-learn, MLflow, FastAPI, Docker, GitHub Actions | `04_mlops_template/` |
| 5 | [LLM Evaluation & Observability Dashboard](#5-llm-eval-dashboard) | Python, pandas, Streamlit, any LLM API | `05_llm_eval_dashboard/` |
| 6 | [Real-Time Fraud / Anomaly Detection](#6-fraud-detection) | LightGBM, scikit-learn, Kafka (in-memory), FastAPI | `06_fraud_detection/` |
| 7 | [Diffusion / Generative Vision Mini-Project](#7-diffusion-model) | PyTorch, torchvision | `07_diffusion_model/` |
| 8 | [Distributed Training Benchmark (FSDP)](#8-distributed-training) | PyTorch, FSDP, AMP | `08_distributed_training/` |
| 9 | [AI Cost-Optimization Toolkit](#9-cost-optimization) | Python, sentence-transformers, any LLM API | `09_cost_optimization/` |
| 10 | [Full-Stack GenAI Product (Capstone)](#10-fullstack-genai-capstone) | FastAPI, RAG, Agent, HTML/JS, Docker | `10_fullstack_genai_capstone/` |

---

## 1. RAG Knowledge Assistant

Chunk + embed a document corpus into FAISS, retrieve top-k context, generate grounded answers, and score retrieval quality with RAGAS-style metrics.

**Files:** `rag_pipeline.py`, `evaluate.py`

---

## 2. Agentic Workflow Agent

A tool-using agent with a plan-act-observe loop, typed tool registry, retry/guardrail logic, and full step tracing.

**Files:** `agent.py`, `tools.py`

---

## 3. LoRA Fine-Tuning

Parameter-efficient fine-tuning of Mistral-7B on instruction data using 4-bit quantization + LoRA adapters via TRL's SFTTrainer. Runs on a single mid-range GPU.

**Files:** `train_lora.py`, `evaluate_finetune.py`

---

## 4. MLOps Template

Train + log a model to MLflow, serve it behind FastAPI in Docker, deploy via GitHub Actions CI/CD, and run PSI drift checks on incoming features.

**Files:** `train.py`, `serve.py`, `drift_check.py`, `Dockerfile`, `.github/workflows/deploy.yml`

---

## 5. LLM Eval Dashboard

Run a prompt/model matrix over a test set, score each output on multiple axes (accuracy, latency, cost, groundedness), and explore results in a Streamlit dashboard.

**Files:** `eval_runner.py`, `dashboard.py`

---

## 6. Real-Time Fraud Detection

A producer streams synthetic transactions; a consumer scores each with a trained LightGBM model in real time and flags anomalies. Uses in-memory queue (swap for Kafka).

**Files:** `train_fraud_model.py`, `stream.py`

---

## 7. Diffusion Model

A compact DDPM (denoising diffusion) on MNIST, with a from-scratch noise schedule and a UNet-lite backbone. Demonstrates diffusion fundamentals end to end.

**Files:** `diffusion.py`

---

## 8. Distributed Training Benchmark

Benchmark single-GPU vs. AMP vs. FSDP-sharded training. Records throughput, peak memory, and wall-clock time across regimes.

**Files:** `bench_fsdp.py`

---

## 9. AI Cost-Optimization Toolkit

A wrapper that cuts LLM inference cost via a semantic response cache and a small-model-first router that escalates to a large model only on low confidence. Reports measured savings.

**Files:** `cost_optimizer.py`

---

## 10. Full-Stack GenAI Capstone

A deployable domain assistant: FastAPI backend combining RAG retrieval with agentic tool actions, exposing a clean API with a minimal HTML/JS web client. The portfolio centerpiece tying all other projects together.

**Files:** `app.py`, `index.html`

---

## Getting Started

```bash
git clone https://github.com/Athati14/ai-ml-portfolio.git
cd ai-ml-portfolio
pip install -r requirements.txt  # add per-project requirements
```

Replace stubbed LLM calls and sample datasets with real ones, add tests and a per-project README, and each becomes a verifiable portfolio artifact.
