# Shade CLI Commands Manual

This document provides a comprehensive list of all available commands for the **Shade AI** command-line interface.

## Core Commands

| Command | Description |
|---------|-------------|
| `shade` | Run without any subcommand to start the interactive model optimization. |
| `shade version` | Show Shade version information. |
| `shade hf login` | Login to Hugging Face Hub to access gated models. |
| `shade doctor` | Check system requirements and diagnose issues. |
| `shade info` | Show detailed information about Shade and its environment. |
| `shade config` | Show configuration information and paths. |

## Model Management

| Command | Description |
|---------|-------------|
| `shade download <model_id>` | Download a model from Hugging Face (e.g., `meta-llama/Llama-3.1-8B-Instruct`). |

## CUDA & GPU Management

| Command | Description |
|---------|-------------|
| `shade cuda check` | Check if CUDA is installed and available on your system. |
| `shade cuda get` | Help install CUDA for GPU acceleration (opens download page and shows steps). |
| `shade get cuda` | Alias for `shade cuda get`. |

## Options

| Option | Description |
|---------|-------------|
| `-v, --show-ver` | Show version and exit. |
| `--help` | Show the help message and exit. |

---
*For more information, visit the [Shade GitHub Repository](https://github.com/AssemSabry/Shade).*
