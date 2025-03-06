import gc
import torch
from diffusers import StableDiffusionPipeline
from typing import Optional, Dict, Any
import base64
from io import BytesIO
import os

class FluxInference:
    def __init__(self):
        self.model = None
        self.setup_gpu()
        self.initialize_model()

    def setup_gpu(self):
        # GPU memory optimizations
        torch.cuda.empty_cache()
        gc.collect()
        
        # Enable performance optimizations
        torch.backends.cudnn.benchmark = True
        
        if torch.cuda.is_available():
            # Use maximum available VRAM (95% to leave some headroom)
            torch.cuda.set_per_process_memory_fraction(0.95)
            
            # Enable pinned memory for faster CPU->GPU transfers
            # (This uses CPU RAM, not GPU VRAM)
            torch.set_float32_matmul_precision('high')

    def initialize_model(self):
        try:
            # Get model path from environment variable or use default
            model_path = os.environ.get("MODEL_PATH", "runwayml/stable-diffusion-v1-5")
            
            self.model = StableDiffusionPipeline.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
                use_flash_attention_2=True,
            )
            
            if torch.cuda.is_available():
                # Enable sequential CPU<->GPU offloading
                self.model.enable_model_cpu_offload()

                # Enable attention slicing for lower memory usage
                self.model.enable_attention_slicing(slice_size="auto")

                # Try to import xformers first
                try:
                    import xformers
                    self.model.enable_xformers_memory_efficient_attention()
                    print("Using xformers for memory-efficient attention")
                except ImportError:
                    print("Xformers not installed, using standard attention mechanisms")
                except Exception as e:
                    print(f"Error enabling xformers: {e}")
            else:
                print("CUDA not available, running on CPU only")
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

        # Clear cache before generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        with torch.inference_mode():
            image = self.model(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).images[0]
        
        # Clear cache after generation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
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