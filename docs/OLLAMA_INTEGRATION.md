# Ollama Integration Guide

This guide explains how to export your uncensored models from **Shade AI** to **Ollama** for seamless use in your favorite local AI tools.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
1. **Ollama**: Download and install it from [ollama.com](https://ollama.com/).
2. **Shade AI**: Ensure you have a processed and saved model in your library.

---

## 🚀 Step-by-Step Integration

### 1. Process and Save your Model
First, you must run the Shade optimization process on a model and save it.
```bash
shade Qwen/Qwen2.5-1.5B-Instruct
```
After the process completes, use the **Save** option to store the model in your local `models/` directory.

### 2. Find the Model Path
To integrate with Ollama, you need the absolute path of the saved model. Use the following command to list your saved models:
```bash
shade library
```
This will display a table containing the **Model Name** and its **Path**.

### 3. Run the Ollama Export Command
Use the `shade ollama` command followed by the path to the model you want to export.

**Basic Usage:**
```bash
shade ollama "./models/Qwen2.5-1.5B-Instruct-shade"
```

**Custom Name:**
By default, the model will be named `[folder-name]-shade` in Ollama. You can specify a custom name using the `--name` option:
```bash
shade ollama "./models/Qwen2.5-1.5B-Instruct-shade" --name my-custom-model
```

### 4. Direct Registration (How it works)
When you run the command, Shade performs the following:
1. Generates a `Modelfile` inside the model folder.
2. Points the `FROM` instruction to the absolute path of your uncensored weights.
3. Automatically executes `ollama create` to register the model.

---

## 🛠️ Usage in Ollama

Once the registration is complete, you can use the model like any other Ollama model.

**In the Terminal:**
```bash
ollama run qwen2.5-1.5b-instruct-shade
```

**In Third-Party Apps:**
Your new model will now appear in the model selection list of apps like:
- AnythingLLM
- Open WebUI
- Enchanted (iOS/macOS)
- Msty

---

## 🔍 Troubleshooting

- **"Ollama command not found"**: Ensure Ollama is installed and added to your system's PATH.
- **"Permission Denied"**: Run the terminal as Administrator (Windows) or use `sudo` (Linux/macOS) if required.
- **Model not showing up**: Restart the Ollama server and run `ollama list` to verify registration.

---
*For more help, visit the [Shade GitHub Repository](https://github.com/AssemSabry/Shade).*
