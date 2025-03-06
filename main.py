from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from fastapi.responses import JSONResponse
from app.core.inference import FluxInference
from contextlib import asynccontextmanager

# Global inference instance
inference_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global inference_engine
    try:
        inference_engine = FluxInference()
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        raise
    yield
    # Shutdown logic (if needed)
    # For example: inference_engine.cleanup()

app = FastAPI(title="Stable Diffusion Inference API", lifespan=lifespan)

# Model configuration
class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    height: Optional[int] = 512
    width: Optional[int] = 512
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 7.5

class ImageGenerationResponse(BaseModel):
    image: str
    format: str

@app.post("/generate", response_model=ImageGenerationResponse)
async def generate_image(request: ImageGenerationRequest):
    global inference_engine
    
    if not inference_engine:
        raise HTTPException(status_code=500, detail="Inference engine not initialized")
    
    try:
        result = inference_engine.generate(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            output_format="b64"  # Always return base64 for API responses
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    global inference_engine
    
    if not inference_engine or not inference_engine.is_healthy():
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "message": "Model not loaded"}
        )
    return {
        "status": "healthy",
        "gpu_memory": inference_engine.get_gpu_info()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)