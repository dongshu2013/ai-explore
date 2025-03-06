import gc
import torch
from typing import Optional, Dict, Any
import base64
from io import BytesIO
from PIL import Image
import numpy as np
import os
from llama_cpp import Llama

class GGUFInference:
    def __init__(self):
        self.model = None
        self.setup()
        self.initialize_model()

    def setup(self):
        # Basic setup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def initialize_model(self):
        try:
            # Get model path from environment variable
            model_path = os.environ.get("GGUF_MODEL_PATH")
            if not model_path:
                raise ValueError("GGUF_MODEL_PATH environment variable not set")
                
            # Initialize llama-cpp model with GGUF file
            self.model = Llama(
                model_path=model_path,
                n_gpu_layers=-1,  # Use all GPU layers
                n_ctx=2048,       # Context size
                verbose=False
            )
            print(f"Loaded GGUF model from {model_path}")
            
        except Exception as e:
            print(f"Error initializing GGUF model: {str(e)}")
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

        # Format the prompt for the GGUF model
        formatted_prompt = f"""
        Generate a detailed image based on this description:
        Prompt: {prompt}
        Negative prompt: {negative_prompt if negative_prompt else 'None'}
        Settings: {width}x{height}, Steps: {num_inference_steps}, CFG: {guidance_scale}
        
        Return the image data as a base64 encoded string.
        """
        
        # Generate with the GGUF model
        response = self.model(
            formatted_prompt,
            max_tokens=1024,
            temperature=0.1,
            stop=["</image>"]
        )
        
        # Extract base64 image data from response
        # This is hypothetical - actual implementation depends on model output format
        try:
            # Parse the response to extract base64 image data
            # This is a placeholder - you'll need to adapt based on your model's output
            image_data = self._extract_image_data(response["choices"][0]["text"])
            
            if output_format == "b64":
                return {
                    "image": image_data,
                    "format": "base64"
                }
            else:
                # Convert base64 to PIL Image
                import base64
                from io import BytesIO
                image = Image.open(BytesIO(base64.b64decode(image_data)))
                return {
                    "image": image,
                    "format": "pil"
                }
        except Exception as e:
            raise RuntimeError(f"Failed to process model output: {str(e)}")

    def _extract_image_data(self, text):
        # Placeholder function to extract base64 image data from model output
        # You'll need to implement this based on your model's output format
        import re
        match = re.search(r'<image>(.*?)</image>', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        raise ValueError("Could not extract image data from model output")

    def is_healthy(self) -> bool:
        return self.model is not None 