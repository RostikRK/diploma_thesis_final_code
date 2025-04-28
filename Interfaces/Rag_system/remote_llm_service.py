import logging
import os
import re
import json
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import Optional, List, Dict, Any, Union
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("llm_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# Set up start
MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit")
HOST = os.environ.get("LLM_SERVICE_HOST", "0.0.0.0")
PORT = int(os.environ.get("LLM_SERVICE_PORT", "5000"))
# Set up end

class GenerateRequest(BaseModel):
    """Request model for text generation."""
    prompt: Optional[str] = None
    messages: Optional[List[Dict[str, str]]] = None
    system_prompt: Optional[str] = None
    temperature: float = 0.4
    max_tokens: int = 3024
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    remove_thinking: bool = True


class GenerateResponse(BaseModel):
    """Response model for text generation."""
    text: str
    usage: Dict[str, int] = {}


# Create FastAPI app
app = FastAPI(
    title="LLM Inference Service",
    description="Remote service for LLM inference",
    version="1.0.0"
)


@app.on_event("startup")
async def load_model():
    global model, tokenizer

    try:
        logger.info(f"Loading LLM model: {MODEL_NAME}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=MODEL_NAME,
            max_seq_length=10096,
            load_in_4bit=True
        )

        FastLanguageModel.for_inference(model)

        # Apply chat template
        if "phi" in MODEL_NAME.lower():
            template = "phi-3"
        elif "qwen" in MODEL_NAME.lower():
            template = "qwen-2.5"
        elif "mistral" in MODEL_NAME.lower():
            template = "mistral"
        else:
            template = "llama-3"

        tokenizer = get_chat_template(
            tokenizer,
            chat_template=template,
        )

        logger.info(f"Model loaded successfully with {template} template")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint returning service information."""
    return {
        "name": "LLM Inference Service",
        "model": MODEL_NAME,
        "status": "online"
    }



@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate text using the LLM model.
    """
    try:
        # Prepare messages
        if request.prompt is not None:
            messages = []
            if request.system_prompt:
                messages.append({"role": "system", "content": request.system_prompt})
            messages.append({"role": "user", "content": request.prompt})
        elif request.messages is not None:
            messages = request.messages
            if request.system_prompt and (not messages or messages[0]["role"] != "system"):
                messages = [{"role": "system", "content": request.system_prompt}] + messages
        else:
            raise HTTPException(status_code=400, detail="Either prompt or messages must be provided")

        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to("cuda")

        input_tokens = inputs.shape[1]
        output_ids = model.generate(
            input_ids=inputs,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty
        )

        completion_tokens = output_ids.shape[1] - input_tokens

        generated_text = tokenizer.decode(
            output_ids[0][input_tokens:],
            skip_special_tokens=True
        ).strip()


        if request.remove_thinking:
            generated_text = re.sub(r'<think>.*?</think>', '', generated_text, flags=re.DOTALL)

        response = GenerateResponse(
            text=generated_text,
            usage={
                "prompt_tokens": input_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": input_tokens + completion_tokens
            }
        )

        return response

    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")


def start_server():
    """Start the FastAPI server."""
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    start_server()