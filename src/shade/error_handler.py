# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 Assem Sabry

"""User-friendly error handling for Shade."""

import sys
from typing import NoReturn

from huggingface_hub.errors import GatedRepoError
from requests.exceptions import ConnectionError, HTTPError, Timeout
from transformers import AutoTokenizer

from .utils import print


def format_error_message(error: Exception) -> str:
    """
    Formats an exception into a user-friendly error message.
    
    Args:
        error: The exception that was raised.
        
    Returns:
        A user-friendly error message string.
    """
    error_type = type(error).__name__
    error_msg = str(error)
    
    # Handle gated repository errors (401 Unauthorized)
    if isinstance(error, GatedRepoError) or (
        isinstance(error, OSError) and 
        ("gated repo" in error_msg.lower() or "401" in error_msg)
    ):
        return format_gated_repo_error(error_msg)
    
    # Handle HTTP 401 errors (authentication)
    if isinstance(error, HTTPError) and "401" in error_msg:
        return format_auth_error(error_msg)
    
    # Handle Disk Space errors
    if any(keyword in error_msg.lower() for keyword in ["no space left", "disk full", "storage full", "errno 28"]):
        return format_disk_error(error_msg)
    
    # Handle connection errors
    if isinstance(error, (ConnectionError, Timeout)) or (
        isinstance(error, OSError) and 
        any(keyword in error_msg.lower() for keyword in ["connection", "network", "timeout", "broken pipe", "errno 10054", "errno 10060"])
    ):
        return format_connection_error(error_msg)
    
    # Handle model loading failures
    if "failed to load model" in error_msg.lower():
        return format_model_load_error(error_msg)
    
    # Handle CUDA/GPU errors
    if any(keyword in error_msg.lower() for keyword in ["cuda", "gpu", "out of memory", "cudnn"]):
        return format_gpu_error(error_msg)
    
    # Handle tokenizer errors
    if "tokenizer" in error_msg.lower():
        return format_tokenizer_error(error_msg)
    
    # Default error message
    return format_generic_error(error_type, error_msg)


def format_gated_repo_error(error_msg: str) -> str:
    """Formats a gated repository error message."""
    # Extract model name from error message if possible
    model_name = None
    if "huggingface.co/" in error_msg:
        try:
            parts = error_msg.split("huggingface.co/")[1].split("/")
            model_name = f"{parts[0]}/{parts[1].split('/')[0]}"
        except (IndexError, AttributeError):
            pass
    
    lines = [
        "",
        "[bold red]Error: Gated Model Access Required[/]",
        "",
        "The model you're trying to use is [bold]gated/restricted[/] on Hugging Face.",
        "",
    ]
    
    if model_name:
        lines.append(f"Model: [bold]{model_name}[/]")
        lines.append("")
    
    lines.extend([
        "[bold]To fix this:[/]",
        "",
        "1. Visit the model page on Hugging Face:",
        f"   [blue]https://huggingface.co/{model_name or 'the-model'}[/]" if model_name else "   https://huggingface.co/the-model",
        "",
        "2. Click the 'Access repository' or 'Request access' button",
        "",
        "3. Accept the license agreement (if required)",
        "",
        "4. Login with your Hugging Face token:",
        "   [bold]shade login[/]",
        "",
        "   Or set the token manually:",
        "   [bold]huggingface-cli login[/]",
        "",
        "5. Try running the command again",
        "",
    ])
    
    return "\n".join(lines)


def format_auth_error(error_msg: str) -> str:
    """Formats an authentication error message."""
    lines = [
        "",
        "[bold red]Error: Authentication Failed[/]",
        "",
        "Failed to authenticate with Hugging Face.",
        "",
        "[bold]To fix this:[/]",
        "",
        "1. Login with your Hugging Face token:",
        "   [bold]huggingface-cli login[/]",
        "",
        "2. Or set the environment variable:",
        "   [bold]export HF_TOKEN=your_token_here[/]",
        "",
    ]
    
    return "\n".join(lines)


def format_connection_error(error_msg: str) -> str:
    """Formats a connection error message."""
    lines = [
        "",
        "[bold red]Error: Network / Connection Problem[/]",
        "",
        "Unable to connect to Hugging Face or download the model.",
        "",
        "[bold]To fix this:[/]",
        "",
        "1. Check your internet connection (Wifi/Ethernet)",
        "",
        "2. If you are in a restricted region, try using a VPN",
        "",
        "3. If you have slow internet, increasing [bold]timeout[/] in config might help",
        "",
        "4. Try again later (Hugging Face might be down)",
        "",
    ]
    
    return "\n".join(lines)


def format_disk_error(error_msg: str) -> str:
    """Formats a disk space error message."""
    lines = [
        "",
        "[bold red]Error: Out of Disk Space[/]",
        "",
        "Your hard drive is full. Hugging Face models can take [bold]10GB - 50GB[/] or more.",
        "",
        "[bold]To fix this:[/]",
        "",
        "1. Free up space on your disk",
        "",
        "2. Change the download folder to another drive:",
        "   Set [bold]HF_HOME[/] environment variable to a folder on a larger drive",
        "   Example: [bold]set HF_HOME=D:\\huggingface_cache[/] (on Windows)",
        "",
        "3. Clear Hugging Face cache to remove unused models:",
        "   [bold]huggingface-cli delete-cache[/]",
        "",
    ]
    
    return "\n".join(lines)


def format_model_load_error(error_msg: str) -> str:
    """Formats a model loading error message."""
    lines = [
        "",
        "[bold red]Error: Failed to Load Model[/]",
        "",
        "The model could not be loaded with any of the configured data types.",
        "",
        "[bold]Possible solutions:[/]",
        "",
        "1. Check that the model name is correct",
        "",
        "2. Make sure you have enough system RAM/VRAM",
        "",
        "3. Try using 4-bit quantization:",
        "   Set [bold]quantization = 'bnb_4bit'[/] in config.toml",
        "",
        "4. Try a smaller model",
        "",
    ]
    
    return "\n".join(lines)


def format_gpu_error(error_msg: str) -> str:
    """Formats a GPU/CUDA error message."""
    lines = [
        "",
        "[bold red]Error: GPU/CUDA Problem[/]",
        "",
        "An error occurred while trying to use the GPU.",
        "",
        "[bold]To fix this:[/]",
        "",
        "1. Check that your GPU drivers are up to date",
        "",
        "2. Make sure you have enough VRAM:",
        "   [bold]shade doctor[/]  # Check system info",
        "",
        "3. Try running on CPU instead:",
        "   Set [bold]device_map = 'cpu'[/] in config.toml",
        "",
        "4. Try using 4-bit quantization to reduce VRAM usage",
        "",
    ]
    
    if "out of memory" in error_msg.lower():
        lines.insert(4, "[bold red]GPU ran out of memory![/]")
        lines.insert(5, "")
    
    return "\n".join(lines)


def format_tokenizer_error(error_msg: str) -> str:
    """Formats a tokenizer error message."""
    lines = [
        "",
        "[bold red]Error: Tokenizer Problem[/]",
        "",
        "Failed to load the tokenizer for this model.",
        "",
        "[bold]To fix this:[/]",
        "",
        "1. The model might be incomplete or corrupted",
        "",
        "2. Try clearing the cache:",
        "   [bold]huggingface-cli delete-cache[/]",
        "",
        "3. Try downloading the model again",
        "",
    ]
    
    return "\n".join(lines)


def format_generic_error(error_type: str, error_msg: str) -> str:
    """Formats a generic error message."""
    lines = [
        "",
        f"[bold red]Error: {error_type}[/]",
        "",
        f"{error_msg}",
        "",
        "[bold]If this error persists:[/]",
        "",
        "1. Check the documentation at: [blue]https://github.com/AssemSabry/Shade[/]",
        "",
        "2. Run [bold]shade doctor[/] to check your system",
        "",
        "3. Report the issue with full error details",
        "",
    ]
    
    return "\n".join(lines)


def handle_error(error: Exception) -> NoReturn:
    """
    Handles an exception by printing a user-friendly error message and exiting.
    
    Args:
        error: The exception that was raised.
    """
    message = format_error_message(error)
    print(message)
    sys.exit(1)
