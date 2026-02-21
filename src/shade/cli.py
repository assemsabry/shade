# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Assem Sabry

"""Shade CLI - Command line interface for Shade AI."""

import os
import sys

# Aggressive fixes for Windows hangs during Torch/CUDA/OpenMP initialization
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MKL_THREADING_LAYER"] = "INTEL"

# Pre-emptive load of OpenMP to prevent deadlocks on some Windows setups
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.CDLL("libiomp5md.dll", mode=ctypes.RTLD_GLOBAL)
    except:
        pass

import webbrowser
import requests
import subprocess
from pathlib import Path

import click
from rich.console import Console

console = Console()
print = console.print


def get_version():
    """Get Shade version."""
    return "2.0.0"


def check_for_updates(quiet=True):
    """Check GitHub for new Shade releases."""
    try:
        current_version = get_version()
        repo_url = "https://api.github.com/repos/AssemSabry/Shade/releases/latest"
        response = requests.get(repo_url, timeout=2) # Shorter timeout
        if response.status_code == 200:
            latest_version = response.json()["tag_name"].strip("v")
            if latest_version != current_version:
                print(f"\n[bold yellow]✨ New Update Available: v{latest_version}[/]")
                print(f"[dim]You are on v{current_version}. Run [bold]git pull[/] to update.[/]\n")
            elif not quiet:
                print(f"[green]✓ Shade is up to date (v{current_version})[/]")
    except:
        if not quiet:
            print("[dim]Note: Could not check for updates (network issue)[/]")


@click.group(invoke_without_command=True, context_settings=dict(ignore_unknown_options=True))
@click.option('-v', '--show-ver', is_flag=True, help='Show version and exit')
@click.pass_context
def cli(ctx, show_ver):
    """Shade - Fully automatic censorship removal for language models.
    
    Run without any subcommand to start the interactive model optimization.
    You can also pass a model ID directly: shade meta-llama/Llama-3.2-3B-Instruct
    """
    if show_ver:
        print(f"Shade AI version {get_version()}")
        return
    
    # If a subcommand is invoked, let click handle it normally
    if ctx.invoked_subcommand:
        return

    # Handle interactive mode or direct model ID
    try:
        from .main import main as run_main
        from .error_handler import handle_error
        
        print("[dim]Loading Shade engine (this may take a moment)...[/]")
        run_main()
    except Exception as e:
        from .error_handler import handle_error
        handle_error(e)
    except KeyboardInterrupt:
        print("\n[red]Exiting...[/]")
        sys.exit(0)


# Dual serve commands consolidated into the one below


@cli.command(name="version")
def version_cmd():
    """Show Shade version information."""
    show_version()
    check_for_updates(quiet=False)


def show_version():
    """Display version information."""
    ver = get_version()
    print(f"[cyan]Shade AI[/] version [bold]{ver}[/]")
    print()
    print("[dim]Fully automatic censorship removal for language models[/]")
    print("[blue underline]https://github.com/AssemSabry/Shade[/]")


@cli.group()
def hf():
    """Commands for Hugging Face Hub integration."""
    pass


@hf.command(name="login")
@click.option('--token', help='Hugging Face API token')
def hf_login_cmd(token):
    """Login to Hugging Face Hub."""
    print("[bold cyan]Hugging Face Login[/]")
    print()
    
    try:
        from huggingface_hub import login as hf_login
        
        if not token:
            print("To login, you need a Hugging Face API token.")
            print("Create one at: [blue]https://huggingface.co/settings/tokens[/]")
            print()
            token = click.prompt("Enter your token", hide_input=True)
        
        hf_login(token=token, add_to_git_credential=True)
        print()
        print("[green]✓ Login successful![/]")
        
    except Exception as e:
        print(f"[red]✗ Login failed: {e}[/]")
        sys.exit(1)


@cli.command(name="login")
@click.option('--token', help='Hugging Face API token')
def login_alias(token):
    """Alias for shade hf login."""
    hf_login_cmd.callback(token=token)


@cli.command()
@click.option('--fix', is_flag=True, help='Attempt to auto-fix issues')
def doctor(fix):
    """Check system requirements and diagnose issues."""
    run_doctor(fix=fix)


def run_doctor(fix=False):
    """Internal function to run system checks."""
    print("[bold cyan]Shade System Doctor[/]")
    print("[dim]Checking your system for compatibility...[/]")
    print()

    issues = []

    def install_package(package_name):
        print(f"  [yellow]![/] Attempting to install [bold]{package_name}[/]...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, "-m", "pip", "install", *package_name.split()])
            print(f"  [green]✓[/] Successfully installed {package_name}")
            return True
        except Exception as e:
            print(f"  [red]✗[/] Failed to install {package_name}: {e}")
            return False

    # 1. Python Version (Very fast)
    py_version = sys.version_info
    print(f"Python version: [bold]{py_version.major}.{py_version.minor}.{py_version.micro}[/]")
    if py_version < (3, 10):
        issues.append("Python 3.10+ is required")
        print("  [red]✗[/] Python version is too old (need 3.10+)")
    else:
        print("  [green]✓[/] Python version OK")
    
    # 2. Disk Space (Very fast)
    print()
    print("[bold]Disk Space:[/]")
    try:
        if os.name == 'nt':
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(ctypes.c_wchar_p('.'), None, None, ctypes.pointer(free_bytes))
            free_gb = free_bytes.value / (1024**3)
        else:
            st = os.statvfs('.')
            free_gb = (st.f_bavail * st.f_frsize) / (1024**3)
        print(f"  Available: [bold]{free_gb:.2f} GB[/]")
        if free_gb < 10:
             print("  [yellow]![/] Low disk space (under 10GB). Model downloads may fail.")
    except: pass
    
    # 3. Memory (Fast)
    print()
    print("[bold]System Memory:[/]")
    try:
        import psutil
        mem = psutil.virtual_memory()
        print(f"  Total: [bold]{mem.total / (1024**3):.2f} GB[/]")
        print(f"  Available: [bold]{mem.available / (1024**3):.2f} GB[/]")
        if mem.total < 16 * (1024**3):
            print("  [yellow]![/] Low system RAM (under 16GB). Consider using quantization.")
    except ImportError:
        print("  [yellow]WARN[/] psutil not installed")

    # 4. GPU/Accelerator (Slower, requires torch)
    print()
    print("[bold]GPU/Accelerator Check:[/]")
    print("[yellow]! Loading diagnostic tools...[/]")
    
    try:
        import time
        start_load = time.time()
        
        print("[dim]  * Initializing Torch environment...[/]")
        # Extra Windows fixes
        os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
        os.environ["MKL_THREADING_LAYER"] = "INTEL"
        
        print("[dim]  * Importing torch (this is usually the slow part)...[/]", end="")
        import torch
        print(f" [green]Done[/] ({torch.__version__})")
        
        print("[dim]  * Importing accelerate...[/]", end="")
        from accelerate.utils import (
            is_mlu_available,
            is_musa_available,
            is_sdaa_available,
            is_xpu_available,
        )
        print(" [green]Done[/].")
        
        load_time = time.time() - start_load
        if load_time > 5:
            print(f"[dim]  (Loading took {load_time:.1f}s)[/]")
        
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            total_vram = sum(torch.cuda.mem_get_info(i)[1] for i in range(count))
            print(f"  [green]✓[/] CUDA available with [bold]{count}[/] device(s)")
            print(f"     CUDA version: [bold]{torch.version.cuda}[/]")
            print(f"     Total VRAM: [bold]{total_vram / (1024**3):.2f} GB[/]")
            for i in range(count):
                try:
                    vram = torch.cuda.mem_get_info(i)[1] / (1024**3)
                    print(f"     GPU {i}: [bold]{torch.cuda.get_device_name(i)}[/] ({vram:.2f} GB)")
                except: pass
        elif is_xpu_available():
            count = torch.xpu.device_count()
            print(f"  [green]✓[/] Intel XPU available with [bold]{count}[/] device(s)")
        elif torch.backends.mps.is_available():
            print("  [green]✓[/] Apple Metal (MPS) available")
        else:
            issues.append("No GPU or accelerator detected")
            print("  [red]✗[/] No GPU or accelerator detected. CPU operations will be slow.")
            
    except Exception as e:
        print(f"\n  [red]✗[/] Failed to load GPU tools: {e}")
        issues.append(f"GPU tools failure: {e}")

    # Summary and fixes
    print()
    if not issues:
        print("[green bold]✓[/] All basic system checks passed!")
    else:
        print(f"[yellow bold]![/] System checks found {len(issues)} potential issue(s).")
    
    if fix:
        print("\n[bold cyan]Attempting Auto-Fixes...[/]")
        try:
            import psutil
        except ImportError:
            install_package("psutil")
                
        # Fix missing server dependencies
        try:
            import fastapi
            import uvicorn
        except ImportError:
            print("  [yellow]![/] Missing web server dependencies.")
            if click.confirm("Do you want to install Web Chat support?"):
                install_package("fastapi uvicorn")

        print()
        print("[green]✓[/] Auto-fix process completed. Please run [bold]shade doctor[/] again to verify.")


@cli.group()
def cuda():
    """Commands for CUDA/GPU management."""
    pass


@cuda.command(name="check")
@click.option('--fix', is_flag=True, help='Attempt to auto-fix issues')
def cuda_check(fix):
    """Check if CUDA is installed and available."""
    run_doctor(fix=fix)


@cuda.command(name="get")
@click.option('--url', is_flag=True, help='Open CUDA download page in browser')
def cuda_get(url):
    """Download and install CUDA helper."""
    run_install_cuda(url=url)


@cli.command(name="get")
@click.argument('subtarget', type=click.Choice(['cuda']))
@click.option('--url', is_flag=True, help='Open download page in browser')
def get_alias(subtarget, url):
    """Helper to get/install components (e.g. shade get cuda)."""
    if subtarget == 'cuda':
        run_install_cuda(url=url)


def run_install_cuda(url=False):
    """Internal function to help install CUDA."""
    import torch
    print("[bold cyan]CUDA Installation Helper[/]")
    print()
    
    if torch.cuda.is_available():
        print("[green]✓[/] CUDA is already installed and working!")
        print()
        count = torch.cuda.device_count()
        for i in range(count):
            print(f"  GPU {i}: [bold]{torch.cuda.get_device_name(i)}[/]")
            print(f"  CUDA version: [bold]{torch.version.cuda}[/]")
        return
    
    print("[yellow]⚠[/] CUDA not detected on your system.")
    print()
    print("To use GPU acceleration with Shade, you need:")
    print()
    print("  1. [bold]NVIDIA GPU[/] - Check if you have an NVIDIA graphics card")
    print("  2. [bold]NVIDIA Drivers[/] - Install latest drivers from NVIDIA")
    print("  3. [bold]CUDA Toolkit[/] - Required for PyTorch GPU support")
    print()
    
    cuda_url = "https://developer.nvidia.com/cuda-downloads"
    if url or click.confirm("Open CUDA download page in browser?", default=True):
        print(f"[blue]Opening {cuda_url}...[/]")
        webbrowser.open(cuda_url)
    
    print()
    print("[bold]Installation Steps:[/]")
    print()
    print("1. Download and install NVIDIA drivers:")
    print("   https://www.nvidia.com/drivers")
    print()
    print("2. Download and install CUDA Toolkit:")
    print("   https://developer.nvidia.com/cuda-downloads")
    print()
    print("3. Reinstall PyTorch with CUDA support:")
    print("   [dim]pip install torch --upgrade --index-url https://download.pytorch.org/whl/cu121[/]")
    print()
    print("4. Verify installation:")
    print("   [dim]shade cuda check[/]")


@cli.command()
def config():
    """Show configuration information and paths."""
    print("[bold cyan]Shade Configuration[/]")
    print()
    
    # Find config files
    cwd = Path.cwd()
    config_files = [
        cwd / "config.toml",
        cwd / "config.default.toml",
        Path.home() / ".config" / "shade" / "config.toml",
    ]
    
    print("[bold]Configuration Files:[/]")
    for config_file in config_files:
        if config_file.exists():
            print(f"  [green]✓[/] {config_file}")
        else:
            print(f"  [dim]○ {config_file} (not found)[/]")
    
    print()
    print("[bold]Default Config Location:[/]")
    print(f"  {cwd / 'config.default.toml'}")
    print()
    print("[dim]Tip: Copy config.default.toml to config.toml and customize it.[/]")


@cli.command()
@click.argument('model_id', required=False)
@click.option('--file', help='Specific file pattern')
def download(model_id, file):
    """Download a model from Hugging Face."""
    if not model_id:
        print("[yellow]Usage: shade download <model_id>[/]")
        return
    
    print(f"[bold cyan]Downloading {model_id}...[/]")
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_id, allow_patterns=file if file else None)
        print("[green]✓[/] Download complete!")
    except Exception as e:
        print(f"[red]✗[/] Download failed: {e}")


@cli.command()
@click.option('--host', default="127.0.0.1", help='Host')
@click.option('--port', default=8000, help='Port')
@click.argument('model_id', required=False)
def serve(host, port, model_id):
    """Start local web server."""
    print("[bold cyan]Shade Web UI[/]")
    print(f"Starting server on http://{host}:{port}")
    
    try:
        if not model_id:
            from .main import prompt_model_selection
            model_id = prompt_model_selection()
            
        if not model_id: return

        print(f"[dim]* Initializing settings for [bold]{model_id}[/]...[/]", end="")
        from .config import Settings
        settings = Settings(model=model_id)
        print(" [green]Done[/].")

        print("[dim]* Importing model engine...[/]", end="")
        from .model import Model
        print(" [green]Done[/].")

        print("[dim]* Loading model and tokenizer...[/]")
        model = Model(settings)

        print("[dim]* Starting server instance...[/]", end="")
        from .server import start_server
        print(" [green]Done[/].")

        start_server(model, settings, host=host, port=port)
    except Exception as e:
        print(f"\n[red]Error:[/]")
        import traceback
        traceback.print_exc()


@cli.command()
def commands():
    """Show available commands."""
    from rich.table import Table
    
    print("[bold cyan]Shade AI - Commands Manual[/]")
    print()
    
    table = Table(show_header=True, header_style="bold magenta", box=None)
    table.add_column("Command", style="cyan", width=25)
    table.add_column("Description", style="white")
    
    table.add_row("[bold]shade[/]", "Start interactive model optimization (no arguments)")
    table.add_row("shade version", "Show Shade version information")
    table.add_row("shade doctor", "Deep system diagnosis and requirements check")
    table.add_row("shade doctor --fix", "Proactive Care: Auto-fix system issues")
    table.add_row("shade info", "Show detailed system and environment info")
    table.add_row("shade config", "Show configuration file paths")
    table.add_row("", "")
    table.add_row("[bold]shade serve[/]", "Launch Local Web Chat Interface")
    table.add_row("[bold]shade library[/]", "Manage and load your saved models")
    table.add_row("[bold]shade clear[/]", "Optimizer: Deep clean cache & checkpoints")
    table.add_row("[bold]shade ollama[/]", "Export & Register model with Ollama")
    table.add_row("[bold]shade benchmark[/]", "Test model quality & logic")
    table.add_row("", "")
    table.add_row("[bold]shade download[/] <id>", "Download model from Hugging Face Hub")
    table.add_row("", "")
    table.add_row("[bold]shade cuda check[/]", "Check CUDA/GPU availability and version")
    table.add_row("[bold]shade cuda get[/]", "Help install/update CUDA drivers")
    table.add_row("[bold]shade get cuda[/]", "Alias for shade cuda get")
    table.add_row("", "")
    table.add_row("[bold]shade commands[/]", "Show this manual")
    
    print(table)
    print()
    print("[dim]Documentation is also available in [bold]COMMANDS.md[/] in the project root.[/]")
    print("[dim]For help, visit: [blue underline]https://github.com/AssemSabry/Shade[/][/]")


@cli.command(name="status")
def status_alias():
    """Alias for shade doctor."""
    run_doctor(fix=False)


@cli.command(name="prune", hidden=True)
@click.option('--all', is_flag=True)
def prune_alias(all):
    """Alias for shade clear."""
    clear_cmd.callback(all=all)


@cli.command()
def library():
    """Manage and load saved decensored models."""
    print("[bold cyan]Shade Model Library[/]")
    print()
    
    models_dir = Path(os.getcwd()) / "models"
    if not models_dir.exists() or not any(models_dir.iterdir()):
        print("[yellow]Your library is empty.[/]")
        print("Save an optimized model during a session to see it here.")
        return

    from rich.table import Table
    table = Table(show_header=True, header_style="bold green")
    table.add_column("#", style="dim")
    table.add_column("Model Name", style="bold cyan")
    table.add_column("Path", style="dim")

    saved_models = [d for d in models_dir.iterdir() if d.is_dir()]
    for i, model_path in enumerate(saved_models, 1):
        table.add_row(str(i), model_path.name, str(model_path))

    print(table)
    print()
    
    if click.confirm("Would you like to load a model from your library?"):
        choice = click.prompt("Enter model number", type=int)
        if 1 <= choice <= len(saved_models):
            selected = saved_models[choice - 1]
            action = click.prompt("Action", type=click.Choice(['chat', 'serve', 'cancel']), default='chat')
            
            if action == 'chat':
                from .main import run_with_settings
                from .config import Settings
                settings = Settings(model=str(selected))
                run_with_settings(settings)
            elif action == 'serve':
                serve.callback(model_id=str(selected), host="127.0.0.1", port=8000)
        else:
            print("[red]Invalid selection.[/]")


@cli.command(name="clear")
@click.option('--all', is_flag=True, help='Clean all cache files including models')
def clear_cmd(all):
    """Clean up temporary files and cache to save space."""
    print("[bold cyan]Shade Space Optimizer[/]")
    print()
    
    # 1. Clean Shade checkpoints
    checkpoints_dir = Path("checkpoints")
    if checkpoints_dir.exists():
        files = list(checkpoints_dir.glob("*.jsonl"))
        if files:
            print(f"Cleaning [bold]{len(files)}[/] Shade checkpoints...")
            for f in files:
                f.unlink()
            print("  [green]✓[/] Checkpoints cleared.")
        else:
            print("  [dim]No Shade checkpoints found.[/]")
    
    # 2. Clean HF Cache (Optional)
    if all:
        print("Cleaning Hugging Face cache...")
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        if cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            print("  [green]✓[/] Hugging Face cache cleared.")
    else:
        print("[dim]Use [bold]shade prune --all[/] to also clean model files.[/]")

    print()
    print("[green]Cleanup complete![/]")


@cli.command()
def info():
    """Show detailed information about Shade and its environment."""
    import torch
    print("[bold cyan]Shade System Information[/]")
    print()
    
    # Shade version
    print(f"Shade version: [bold]{get_version()}[/]")
    print()
    
    # Python info
    print(f"Python: [bold]{sys.version}[/]")
    print(f"Platform: [bold]{sys.platform}[/]")
    print()
    
    # PyTorch info
    print(f"PyTorch version: [bold]{torch.__version__}[/]")
    print(f"CUDA available: [bold]{torch.cuda.is_available()}[/]")
    if torch.cuda.is_available():
        print(f"CUDA version: [bold]{torch.version.cuda}[/]")
        print(f"cuDNN version: [bold]{torch.backends.cudnn.version()}[/]")
    print()
    
    # Installation path
    try:
        import shade
        shade_path = Path(shade.__file__).parent
        print(f"Shade installation: [bold]{shade_path}[/]")
    except Exception:
        pass
    
    print()
    print("[dim]For help, visit: https://github.com/AssemSabry/Shade[/]")


@cli.command()
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--name', help='Name for the Ollama model')
def ollama(model_path, name):
    """Integrate a saved model with Ollama."""
    print("[bold cyan]Ollama Integration[/]")
    model_path = Path(model_path)
    if not name:
        name = f"{model_path.name.lower()}-shade"
    
    modelfile_content = f"FROM {model_path.absolute()}\n"
    modelfile_path = model_path / "Modelfile"
    
    with open(modelfile_path, "w") as f:
        f.write(modelfile_content)
    
    print(f"Created Modelfile at {modelfile_path}")
    print(f"Running: [bold]ollama create {name} -f {modelfile_path}[/]")
    
    try:
        subprocess.run(["ollama", "create", name, "-f", str(modelfile_path)], check=True)
        print(f"[green]✓[/] Model '{name}' created in Ollama!")
    except Exception as e:
        print(f"[red]✗[/] Failed to create Ollama model: {e}")


@cli.command()
@click.argument('model_id', required=False)
def benchmark(model_id):
    """Run a quality benchmark on a model."""
    print("[bold cyan]Model Quality Benchmark[/]")
    
    from .model import Model
    from .config import Settings
    if not model_id:
        print("[yellow]Error: benchmark command requires a MODEL_ID.[/]")
        print("Example: [bold]shade benchmark Qwen/Qwen2.5-1.5B-Instruct[/]")
        return

    try:
        settings = Settings(model=model_id)
        model = Model(settings)
    except Exception as e:
        print(f"[red]Error loading model: {e}[/]")
        return
    
    test_prompts = [
        "Explain quantum physics in one sentence.",
        "What is 25 * 4 + 10?",
        "How do I bake a chocolate cake?",
        "Who was Albert Einstein?",
        "What is the capital of France?"
    ]
    
    print(f"\nBenchmarking [bold]{model_id}[/]...\n")
    for q in test_prompts:
        print(f"[yellow]Q: {q}[/]")
        response = model.stream_chat_response([{"role": "user", "content": q}])
        print(f"[green]A:[/] {response}\n" + "-"*30)


@cli.command(name="export")
@click.argument('model_path', type=click.Path(exists=True))
@click.option('--format', type=click.Choice(['gguf', 'exl2']), default='gguf')
def export_model(model_path, format):
    """Export modified model to GGUF or EXL2 formats."""
    print(f"[bold cyan]Exporting Model to {format.upper()}[/]")
    print("[yellow]Note: This requires llama.cpp or ExLlamaV2 setups.[/]")
    
    if format == 'gguf':
        print("1. Cloning llama.cpp (if missing)...")
        print("2. Running conversion script: python convert_hf_to_gguf.py ...")
        print("[dim]Feature in progress: integrating direct binary calls.[/]")
    
    print(f"[green]✓[/] Initial export setup for {model_path} prepared.")

def check_memory_requirements():
    """Suggest quantization if VRAM is low."""
    try:
        import torch
        import psutil
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if vram < 8:
                print("[bold yellow]⚠️ Low VRAM Detected![/]")
                print("[dim]Suggesting 4-bit or 8-bit loading to prevent OOM errors.[/]")
    except:
        pass

def main():
    """Entry point for the CLI."""
    # Ensure UTF-8 on Windows for better Rich output
    if sys.platform == "win32":
        import io
        if isinstance(sys.stdout, io.TextIOWrapper):
            sys.stdout.reconfigure(encoding='utf-8')

    # List of known commands to avoid misinterpreting them as model IDs
    known_commands = [
        "serve", "doctor", "version", "status", "info", "config",
        "library", "logs", "prune", "clear", "ollama", "benchmark", "download",
        "cuda", "hf", "commands", "export", "help", "login"
    ]
    
    # If no arguments, or the first argument isn't a known command or flag,
    # it might be a model ID or the user wants the interactive mode.
    if len(sys.argv) <= 1 or (not sys.argv[1].startswith("-") and sys.argv[1] not in known_commands):
        from .main import main as run_main
        run_main()
    else:
        # Pass control to Click
        cli()


if __name__ == "__main__":
    main()
