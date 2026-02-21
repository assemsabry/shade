import os
import json
import datetime
import uvicorn
import requests
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from .model import Model
from .config import Settings

app = FastAPI(title="Shade AI Server")

# Enable CORS for external frontends (React, Flutter, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for the model
active_model: Optional[Model] = None
active_settings: Optional[Settings] = None

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

class ChatHistoryManager:
    @staticmethod
    def save_chat(model_name: str, messages: list):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = LOGS_DIR / f"chat_{timestamp}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump({
                "model": model_name,
                "timestamp": timestamp,
                "messages": messages
            }, f, ensure_ascii=False, indent=2)
        return log_file

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

# Resolve paths relative to this file
BASE_DIR = Path(__file__).parent.parent.parent
WEB_DIR = BASE_DIR / "webchat"

# Ensure the web directory exists
if not WEB_DIR.exists():
    # Fallback to 'web chat' if the rename didn't happen for some reason
    WEB_DIR = BASE_DIR / "web chat"

@app.get("/api/info")
async def get_info():
    if not active_settings:
        return {"model": "No model loaded"}
    return {"model": active_settings.model}

@app.post("/api/chat")
async def chat(request: ChatRequest):
    if active_model is None or active_model.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded or initialized")
    
    chat_history = [{"role": "user", "content": request.message}]
    
    async def generate_chunks():
        from transformers import TextIteratorStreamer
        from threading import Thread
        
        # Prepare tokens
        chat_prompt = active_model.tokenizer.apply_chat_template(
            chat_history,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = active_model.tokenizer(
            chat_prompt,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(active_model.model.device)
        
        streamer = TextIteratorStreamer(active_model.tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.7,
        )
        
        thread = Thread(target=active_model.model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for new_text in streamer:
            if new_text:
                yield new_text

    return StreamingResponse(generate_chunks(), media_type="text/plain")

@app.post("/compare")
async def compare(request: ChatRequest):
    if not active_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        chat_history = [{"role": "user", "content": request.message}]
        
        # Non-streaming for compare is easier to sync, but we could stream both as well.
        # For simplicity and robustness in comparison, we'll get full responses.
        
        # 1. Modified
        modified_response = active_model.stream_chat_response(chat_history)
        
        # 2. Original (Base)
        lora_weights = {}
        for name, param in active_model.model.named_parameters():
            if "lora_B" in name:
                lora_weights[name] = param.data.clone()
        
        active_model.reset_model()
        original_response = active_model.stream_chat_response(chat_history) # Corrected from active_model.str
        
        # 3. Restore
        for name, param in active_model.model.named_parameters():
                if name in lora_weights:
                    param.data.copy_(lora_weights[name])
        
        return {
            "original": original_response,
            "modified": modified_response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/library")
async def get_library():
    models_dir = Path("models")
    if not models_dir.exists():
        return {"models": []}
    
    saved_models = []
    for d in models_dir.iterdir():
        if d.is_dir():
            saved_models.append({
                "name": d.name,
                "path": str(d.absolute()),
                "created": datetime.datetime.fromtimestamp(d.stat().st_ctime).isoformat()
            })
    return {"models": saved_models}

@app.get("/api/logs")
async def get_logs():
    logs = []
    # Use BASE_DIR/logs instead of LOGS_DIR if not defined
    logs_path = BASE_DIR / "logs"
    if logs_path.exists():
        for f in logs_path.glob("*.json"):
            with open(f, "r", encoding="utf-8") as file:
                logs.append(json.load(file))
    return {"history": logs}

# Mount static files (JS, CSS, Images, etc.) from the build folder
if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="static")
else:
    # Log a warning if the web directory is missing
    print(f"[yellow]Warning: Web interface directory not found at {WEB_DIR}[/]")

def start_server(model_instance: Model, settings_instance: Settings, host="127.0.0.1", port=8000):
    global active_model, active_settings
    active_model = model_instance
    active_settings = settings_instance
    
    url = f"http://{host}:{port}"
    print()
    print(f"[bold green]🚀 Shade Web Server started![/]")
    print(f"[bold cyan]Click to open:[/] [blue underline]{url}[/] ")
    print()
    
    import webbrowser
    webbrowser.open(url)
    
    uvicorn.run(app, host=host, port=port, log_level="error")
