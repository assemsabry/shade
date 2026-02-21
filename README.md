![Shade v2.0.0 Banner](media/shadev2.png)

# Shade : Fully Automatic Censorship Removal

<p align="center">
  <a href="https://assem.cloud/"><img src="https://img.shields.io/badge/Website-Assem.cloud-blue?style=flat&logo=google-chrome&logoColor=white" alt="Website"></a>
  <a href="https://x.com/assemsabryy"><img src="https://img.shields.io/badge/X-@assemsabryy-black?style=flat&logo=x&logoColor=white" alt="X"></a>
  <a href="https://www.facebook.com/assemsabryy"><img src="https://img.shields.io/badge/Facebook-assemsabryy-blue?style=flat&logo=facebook&logoColor=white" alt="Facebook"></a>
  <img src="https://img.shields.io/badge/Version-2.0.0-green" alt="Version">
  <img src="https://img.shields.io/badge/License-AGPL--3.0-orange" alt="License">
</p>

---

## 🌟 What is Shade?

**Shade** is a state-of-the-art platform designed to liberate Large Language Models (LLMs) from artificial censorship and safety filters. Using advanced **Abliteration** (directional ablation) and an automated **TPE-based parameter optimizer** powered by [Optuna](https://optuna.org/), Shade removes "safety alignment" without damaging the model's core intelligence.

---

## 🚀 New in Version 2.0.0

The v2.0.0 release transforms Shade from a CLI utility into a complete **Model Liberation Platform**.

- **Ollama One-Click Integration**: Automatically register your uncensored models with Ollama.
- **Model Quality Benchmarking**: Built-in "Sanity Check" system to verify model intelligence after processing.
- **Space Optimizer (Prune)**: Deep clean temporary files, checkpoints, and heavy Hugging Face cache.
- **Proactive Core (Doctor ++)**: Self-healing diagnostic system that can auto-install missing dependencies.
- **Official API & Web Backend**: Ready-to-use FastAPI server for custom app integrations.

---

## � Core Features

### 1. Fully Automated Abliteration
- **No Training Required**: Uses mathematical projection to remove censorship without expensive GPU fine-tuning.
- **Smart Layer Analysis**: Automatically identifies which layers are responsible for refusals.
- **Precision Optimization**: Balances removal of safety filters with the preservation of model intelligence (KL Divergence tracking).

### 2. High-End Web Interface (Shade Web UI)
- **Modern Liquid Glass Design**: A premium, responsive web chat interface.
- **Model Comparison Mode**: View original vs. uncensored responses side-by-side.

### 3. Hardware & System Care
- **Multi-GPU Support**: Automatically detects and leverages CUDA, XPU, MLU, and Apple Metal (MPS).
- **GPU Diagnostics**: Real-time VRAM monitoring.
- **Memory Optimization**: Optimized memory management to prevent OOM errors.

---

## ⚒️ Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/AssemSabry/Shade.git
cd Shade
```

### 2. Installation
Install the project in editable mode to use the `shade` command directly:
```bash
pip install -e .
```

### 3. Configuration & Login
To use models from Hugging Face, secure your access first:
```bash
shade hf login
```

### 4. Liberate a Model
Run the automatic optimization process on any model ID:
```bash
shade <model_id>
```
*Example:* `shade Qwen/Qwen2.5-1.5B-Instruct`

### 5. Start Web Chat
Launch the web interface to talk to your models:
```bash
shade serve
```

---

## 📋 Command Reference

| Command | Description |
| :--- | :--- |
| `shade <model_id>` | Start the automatic optimization & abliteration process. |
| `shade serve` | Launch the Shade Web UI interface. |
| `shade library` | Manage and launch your saved decensored models. |
| `shade ollama` | Export and register a model with Ollama automatically. |
| `shade benchmark` | Run quality tests to ensure the model's logic is intact. |
| `shade doctor --fix` | Automatically diagnose and fix system/dependency issues. |
| `shade prune --all` | Free up disk space by cleaning cache and checkpoints. |
| `shade hf login` | Securely authenticate with Hugging Face Hub. |

---

## 🧠 How It Works

Shade identifies the "refusal direction" within the model's high-dimensional space and applies an **Ablation Weight Kernel**. This kernel is optimized specifically for each component (Attention Out-Projection, MLP Down-Projection) to ensure that the censorship is removed with the least amount of "collateral damage" to the model's capabilities.

> [!IMPORTANT]
> **Shade** is a fully original, independent project built from the ground up. It is **NOT** a clone, fork, or derivative of any existing repository. All automation logic, UI design, and optimization workflows were developed specifically for this project.

---

## 👤 Meet the Developer

<p align="center">
  <img src="media/assemm.webp" width="600" alt="Assem Sabry">
  <br>
  <b>Assem Sabry</b>
  <br>
  <i>Lead Developer & AI Researcher</i>
</p>

<p align="center">
  <a href="https://assem.cloud/">
    <img src="https://img.shields.io/badge/Visit%20My%20Website-assem.cloud-blue?style=for-the-badge&logo=google-chrome&logoColor=white" alt="Website">
  </a>
  <a href="https://www.facebook.com/assemsabryy">
    <img src="https://img.shields.io/badge/Facebook-assemsabryy-1877F2?style=for-the-badge&logo=facebook&logoColor=white" alt="Facebook">
  </a>
  <a href="https://www.linkedin.com/in/assem7/">
    <img src="https://img.shields.io/badge/LinkedIn-assem7-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
  </a>
</p>

---

## ⚠️ Disclaimer

**Assem Sabry**, the developer of Shade, is **not responsible** for any misuse of this tool. Shade is provided for educational and research purposes only. The primary goal of this project is to allow users to unlock the full potential of open-source language models. Users are expected to interact with de-censored models responsibly.

---

## 📜 Citation

If you use Shade in your research, please cite it:

```bibtex
@misc{shade,
  author = {Sabry, Assem},
  title = {Shade: Fully automatic censorship removal for language models},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/AssemSabry/Shade}}
}
```

---

## ⚖️ License

Copyright &copy; 2026 **Assem Sabry**
Licensed under the **GNU Affero General Public License v3.0**. See the [LICENSE](LICENSE) file for details.
