# RS Change-Review Agent

An **LLM-driven multi-tool agent** for remote-sensing change detection. Given two
bi-temporal images, the agent **autonomously selects** among **6 change-detection
algorithms** — instead of running a fixed, hand-coded pipeline.

> **LLM as the brain, specialized models as the eyes.**
> ReAct loop + 6 tools, orchestrated by DeepSeek.

---

## ✨ Why this project (Agent Engineering Highlights)

| Highlight | What it demonstrates |
|-----------|---------------------|
| **Multi-tool agent** | 6 algorithms exposed as independent tools (`BaseAction`), orchestrated by a ReAct loop (lagent) with DeepSeek as the LLM |
| **Tool contract & separation of concerns** | Every algorithm is split into *algorithm implementation* (standalone inference, framework-agnostic) and *tool adapter* (agent-facing). The algorithm never depends on the agent framework → swappable to MCP / LangChain without touching models |
| **Multi-algorithm selection** | Capability profiles (method family, output encoding, trained dataset, speed) let the agent choose the right algorithm per task — groundwork for "best-choice" scheduling |
| **Anti-hallucination safeguard** | Post-loop verification detects when the LLM claims a result without actually invoking any tool |
| **Real multi-tool integration bug, fixed** | Bare module-name collisions (`inference` / `models`) across 5 vendored models — solved with package-based imports + unique model namespaces |

## 🧠 Architecture

```
User: "detect building changes between T1 and T2"
        │
        ▼
┌──────────────────────────────────────────────┐
│  Brain = LLM (DeepSeek via GPTAPI)           │
│  ReAct loop: Thought → Action → Observation  │
└──────────────┬───────────────────────────────┘
               │  ActionExecutor picks a tool
               ▼
┌──────────────────────────────────────────────┐
│  6 change-detection tools (BaseAction)       │
│  MCI · ChangeFormer · BIT · TinyCD           │
│  SNUNet · DDPM-CD                            │
└──────────────┬───────────────────────────────┘
               │  each tool → algorithm + adapter
               ▼
┌──────────────────────────────────────────────┐
│  Algorithm layer (standalone, framework-free)│
│  inference.py → mask (0=unchanged, 1=change) │
└──────────────────────────────────────────────┘
```

**Two-layer design per algorithm** (see `docs/tool-contract.md`):

```
algorithms/<name>/inference.py   # algorithm: model loading + inference, runs standalone
tools/<name>_tool.py             # adapter: BaseAction, input validation, contract result
```

## 🗂️ The 6 Algorithms

| Algorithm | Architecture | Output | Weights |
|-----------|-------------|--------|---------|
| MCI (paper baseline) | SegFormer + Transformer | 3-class + caption | 913 MB |
| ChangeFormer | Transformer (Siamese) | binary | 492 MB |
| BIT | Transformer (bi-temporal) | binary | 60 MB |
| TinyCD | Lightweight CNN (EfficientNet) | binary | 1.3 MB |
| SNUNet | CNN (dense Siamese) | binary | 40 MB |
| DDPM-CD | Diffusion + change head | binary | 1.75 GB |

## 🛠️ Tech Stack

`Python` · `PyTorch` · `lagent` (agent framework) · `DeepSeek API` · `HuggingFace` · `torchvision`

## 🚀 Quick Start

```bash
# 1. environment (Python 3.9 + CUDA torch)
conda create -n change_agent python=3.9 -y
conda activate change_agent
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
pip install einops timm opencv-python numpy scikit-image imageio tqdm pandas
cd lagent-main && pip install -e . && cd ..

# 2. download each algorithm's weights (see algorithms/<name>/WEIGHTS.md)

# 3. set the LLM API key
set DEEPSEEK_API_KEY=sk-...

# 4. run the multi-algorithm agent
python tools/demo_multi_agent.py
```

## 📁 Directory Structure

```
├── algorithms/            # 5 change-detection algorithms (algorithm layer)
│   ├── changeformer/      #   + inference.py + WEIGHTS.md
│   ├── bit/
│   ├── tinycd/
│   ├── snunet/
│   └── ddpm_cd/
├── tools/                 # tool adapters (BaseAction) + demo scripts
│   ├── *_tool.py          #   one adapter per algorithm
│   └── demo_multi_agent.py#   5-tool agent + anti-hallucination check
├── docs/
│   ├── tool-contract.md   #   tool integration contract (algorithm/adapter split)
│   └── capability-profile.md # capability metadata for algorithm selection
├── Multi_change/          # MCI (the original paper's model)
└── lagent-main/           # the lagent agent framework (vendored)
```

## 📝 Notes

- This is a **research project** (remote-sensing intelligent agent) jointly developed by a team; the multi-algorithm agent, tool contract, capability profiles and the collision fix are my primary contributions.
- Weights are **not committed** (large); each algorithm ships a `WEIGHTS.md` with download + SHA-256 verification.
- Model selection is currently prompt-driven (LLM reads tool descriptions); a programmatic nearest-neighbor (data-similarity) selector is planned.
