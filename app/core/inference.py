import gc
import torch
from diffusers import StableDiffusionPipeline
from typing import Optional, Dict, Any, Union
import base64
from io import BytesIO
from PIL import Image

class FluxInference:
    def __init__(self):
        self.model = None
        self.setup_gpu()
        self.initialize_model()

    def setup_gpu(self):
        # GPU memory optimizations
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.set_per_process_memory_fraction(0.95)

    def initialize_model(self):
        try:
            self.model = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float16
            )
            if torch.cuda.is_available():
                self.model = self.model.to("cuda")
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise

    def generate(self, 
                prompt: str,
                negative_prompt: Optional[str] = None,
                height: Optional[int] = 512,
                width: Optional[int] = 512,
                num_inference_steps: Optional[int] = 50,
                guidance_scale: Optional[float] = 7.5,
                output_format: Optional[str] = "b64") -> Dict[str, Any]:
        
        if self.model is None:
            raise RuntimeError("Model not initialized")

        image = self.model(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
        
        # Convert to base64 if requested
        if output_format == "b64":
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return {
                "image": img_str,
                "format": "base64"
            }
        else:
            # Return PIL image
            return {
                "image": image,
                "format": "pil"
            }

    def get_gpu_info(self) -> Optional[Dict[str, int]]:
        if torch.cuda.is_available():
            return {
                "total": torch.cuda.get_device_properties(0).total_memory,
                "allocated": torch.cuda.memory_allocated(0),
                "cached": torch.cuda.memory_reserved(0)
            }
        return None

    def is_healthy(self) -> bool:
        return self.model is not None 