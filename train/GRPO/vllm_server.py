# 集成vllm到训练流程有两种实现方式，一种是把vllm作为独立的服务然后通过请求在训练过程中更新lora适配器，另一种是将vllm直接写到训练代码中
# 这里选择将vllm作为独立的服务，这样既可以满足训练的需求，又可以方便后续应用进行推理
# vllm_server.py
import argparse
import asyncio
import os
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import threading
import time
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
# 定义请求体模型


class InferenceRequest(BaseModel):
    prompts: List[str]
    num_generations: int = Field(
        1, description="Number of sequences to generate per prompt.")
    max_new_tokens: int = Field(
        256, description="Maximum number of tokens to generate per sequence.")
    temperature: float = Field(0.7, description="Temperature for sampling.")
    top_p: float = Field(1.0, description="Top-p sampling parameter.")
    top_k: int = Field(-1,
                       description="Top-k sampling parameter (-1 for no top-k).")
    lora_adapter_name: Optional[str] = Field(
        None, description="Name of the LoRA adapter to use for inference.")


class LoRAUpdateRequest(BaseModel):
    lora_name: str
    # Path to the LoRA adapter directory (e.g., /path/to/my_lora_dir)
    lora_path: str


class LoRARemoveRequest(BaseModel):
    lora_name: str


app = FastAPI()
llm_engine: Optional[LLM] = None
lora_manager_lock = threading.Lock()  # 保护 LoRA manager 的并发访问
active_lora_adapters: Dict[str, str] = {}  # {lora_name: lora_path}


def initialize_llm_engine(model_path: str, gpu_memory_utilization: float = 0.9):
    global llm_engine
    print(f"Initializing vLLM engine with model: {model_path}...")
    import torch
    print(f"Available GPUs:{torch.cuda.device_count()}")
    try:
        llm_engine = LLM(
            model=model_path,
            trust_remote_code=True,
            dtype="auto",
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,  # Set to True for debugging, False for performance
            tensor_parallel_size=torch.cuda.device_count(),
            quantization="bitsandbytes",
            enable_lora=True  # Important: Enable LoRA for the engine
        )
        print("vLLM engine initialized successfully.")
    except Exception as e:
        print(f"Failed to initialize vLLM engine: {e}")
        llm_engine = None
        raise


@app.on_event("startup")
async def startup_event():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the base model for vLLM.")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for the FastAPI server.")
    parser.add_argument("--gpu_memory_utilization", type=float,
                        default=0.9, help="GPU memory utilization for vLLM.")
    # Use parse_known_args to ignore uvicorn args
    args, unknown = parser.parse_known_args()
    try:
        initialize_llm_engine(args.model, args.gpu_memory_utilization)
    except Exception as e:
        print(
            f"Server failed to start due to vLLM engine initialization error: {e}")
        # Optionally, shut down uvicorn if engine fails to start
        os._exit(1)  # Force exit if engine cannot initialize


@app.post("/generate")
async def generate_text(request: InferenceRequest):
    if llm_engine is None:
        raise HTTPException(
            status_code=503, detail="vLLM engine not initialized.")
    sampling_params = SamplingParams(
        n=request.num_generations,
        temperature=request.temperature,
        max_tokens=request.max_new_tokens,
        top_p=request.top_p,
        top_k=request.top_k,
    )

    # If lora_adapter_name is provided, construct LoRARequest
    lora_request = LoRARequest(
        request.lora_adapter_name, 1) if request.lora_adapter_name else None
    try:
        # vLLM generate is async by default if using async with it.
        # But for FastAPI, running in a threadpool is common if LLM is blocking.
        # However, vLLM's LLM.generate is already designed to be cooperative if used with an async loop.
        outputs = await asyncio.to_thread(llm_engine.generate, request.prompts, sampling_params, lora_request)

        results = []
        for output in outputs:
            generated_texts = [gen.text for gen in output.outputs]
            results.append({
                "prompt": output.prompt,
                "generated_texts": generated_texts,
                "prompt_token_ids": output.prompt_token_ids,
                "completion_token_ids": [gen.token_ids for gen in output.outputs]
            })
        return {"status": "success", "results": results}
    except Exception as e:
        print(f"Error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/lora/add")
async def add_lora_adapter(request: LoRAUpdateRequest):
    if llm_engine is None:
        raise HTTPException(
            status_code=503, detail="vLLM engine not initialized.")

    with lora_manager_lock:  # Protect concurrent access to LoRA manager
        try:
            if not os.path.exists(request.lora_path):
                raise HTTPException(
                    status_code=400, detail=f"LoRA path does not exist: {request.lora_path}")

            # Check if this LoRA name is already added (vLLM will raise error if so)
            if request.lora_name in active_lora_adapters:
                print(
                    f"LoRA adapter '{request.lora_name}' already registered, attempting to replace.")
                # You might want to remove it first before adding again,
                # or just let vLLM handle the error if it's already there
                # llm_engine.remove_lora(request.lora_name) # This might cause issues if not graceful
                raise HTTPException(
                    status_code=409, detail=f"LoRA adapter '{request.lora_name}' already exists. Use /lora/remove first if you want to replace.")
            llm_engine.add_lora(request.lora_name, request.lora_path)
            active_lora_adapters[request.lora_name] = request.lora_path
            print(
                f"Added LoRA adapter: {request.lora_name} from {request.lora_path}")
            return {"status": "success", "message": f"LoRA adapter '{request.lora_name}' added."}
        except Exception as e:
            print(f"Error adding LoRA adapter: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/lora/remove")
async def remove_lora_adapter(request: LoRARemoveRequest):
    if llm_engine is None:
        raise HTTPException(
            status_code=503, detail="vLLM engine not initialized.")

    with lora_manager_lock:
        try:
            if request.lora_name not in active_lora_adapters:
                raise HTTPException(
                    status_code=404, detail=f"LoRA adapter '{request.lora_name}' not found.")

            llm_engine.remove_lora(request.lora_name)
            del active_lora_adapters[request.lora_name]
            print(f"Removed LoRA adapter: {request.lora_name}")
            return {"status": "success", "message": f"LoRA adapter '{request.lora_name}' removed."}
        except Exception as e:
            print(f"Error removing LoRA adapter: {e}")
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/lora/list")
async def list_lora_adapters():
    if llm_engine is None:
        raise HTTPException(
            status_code=503, detail="vLLM engine not initialized.")

    with lora_manager_lock:
        return {"status": "success", "adapters": list(active_lora_adapters.keys())}
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the base model for vLLM.")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host for the FastAPI server.")
    parser.add_argument("--port", type=int, default=8000,
                        help="Port for the FastAPI server.")
    parser.add_argument("--gpu_memory_utilization", type=float,
                        default=0.9, help="GPU memory utilization for vLLM.")
    # Add a dummy argument for `uvicorn.run` to consume `args.model` internally, so it doesn't complain
    # that it's not a known argument for uvicorn.
    parser.add_argument("--dummy_uvicorn_arg",
                        action="store_true", help=argparse.SUPPRESS)

    args = parser.parse_args()
    # Store args for startup_event
    import sys
    sys.argv = [sys.argv[0]] + [f"--model={args.model}", f"--port={args.port}",
                                f"--gpu_memory_utilization={args.gpu_memory_utilization}"]
    print(
        f"Starting vLLM server on {args.host}:{args.port} with model: {args.model}")
    uvicorn.run(app, host=args.host, port=args.port)
