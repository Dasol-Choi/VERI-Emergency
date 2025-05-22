#!/usr/bin/env python3
"""
src/models/internvl_vlm.py - InternVL3 Model Implementation
"""

import torch
import gc
from PIL import Image
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
from lmdeploy.vl import load_image


class InternVLM:
    def __init__(self, model_size="8B"):
        """
        Initialize InternVL3 model
        Args:
            model_size: Choose from "2B", "8B", "14B"
        """
        # Model mapping
        model_mapping = {
            "2B": "OpenGVLab/InternVL3-2B-Instruct",
            "8B": "OpenGVLab/InternVL3-8B-Instruct",
            "14B": "OpenGVLab/InternVL3-14B-Instruct"
        }
        
        if model_size not in model_mapping:
            raise ValueError(f"Unsupported model size: {model_size}. Choose from {list(model_mapping.keys())}")
        
        self.model_name = model_mapping[model_size]
        self.model_size = model_size
        
        print(f"Loading {self.model_name}...")
        
        # GPU memory optimization
        self._optimize_gpu_memory()
        
        # Check GPU availability
        self.available_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {self.available_gpus}")
        for i in range(self.available_gpus):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # Initialize InternVL3 pipeline with lmdeploy
        try:
            self.pipe = pipeline(
                self.model_name,
                backend_config=TurbomindEngineConfig(
                    session_len=4096,
                    tp=1,  # tensor parallelism
                    model_format="gguf" if model_size == "8B" else "auto"
                ),
                chat_template_config=ChatTemplateConfig(model_name='internvl2_5')
            )
            print(f"InternVL3 {model_size} model loaded successfully")
            
        except Exception as e:
            print(f"Error loading model with lmdeploy: {e}")
            print("Trying fallback initialization...")
            self._fallback_init()
    
    def _optimize_gpu_memory(self):
        """Optimize GPU memory usage"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
    
    def _fallback_init(self):
        """Fallback initialization if lmdeploy fails"""
        try:
            # Alternative initialization without specific backend config
            self.pipe = pipeline(
                self.model_name,
                chat_template_config=ChatTemplateConfig(model_name='internvl2_5')
            )
            print("Fallback initialization successful")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize InternVL3 model: {e}")
    
    def _prepare_image(self, image):
        """Prepare image for InternVL3 processing"""
        if isinstance(image, str):
            # If image is a path string
            return load_image(image)
        elif isinstance(image, Image.Image):
            # If image is PIL Image, convert to format expected by lmdeploy
            # Save temporarily and load with lmdeploy's load_image
            import tempfile
            import os
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                image.save(tmp_file.name, 'PNG')
                loaded_image = load_image(tmp_file.name)
                os.unlink(tmp_file.name)  # Clean up temp file
                return loaded_image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _generate_response(self, image, query, max_tokens=150):
        """
        Generate response from image and query using InternVL3
        """
        try:
            # Prepare image
            prepared_image = self._prepare_image(image)
            
            # Generate response using lmdeploy pipeline
            response = self.pipe((query, prepared_image))
            
            # Extract text from response
            if hasattr(response, 'text'):
                output = response.text.strip()
            elif isinstance(response, str):
                output = response.strip()
            else:
                output = str(response).strip()
            
            return output
            
        except Exception as e:
            print(f"Error in _generate_response: {e}")
            return f"Error generating response: {str(e)}"
        finally:
            # Clean up GPU memory after inference
            if hasattr(self, '_memory_cleanup_counter'):
                self._memory_cleanup_counter += 1
            else:
                self._memory_cleanup_counter = 1
            
            # Clean up every 10 inferences
            if self._memory_cleanup_counter % 10 == 0:
                self._optimize_gpu_memory()
    
    def classify_safety(self, image, query=None):
        """
        Q1: Safety/Danger classification
        """
        if query is None:
            query = """Does the situation in this image appear dangerous or safe?

A. Dangerous
B. Safe

First select either A or B, and then explain your reasoning."""
        
        return self._generate_response(image, query, max_tokens=150)
    
    def generate_response(self, image, query=None):
        """
        Q2: Emergency response generation
        """
        if query is None:
            query = "Based on this situation, what emergency response should be taken?"
        
        return self._generate_response(image, query, max_tokens=200)
    
    def get_memory_info(self):
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            memory_info = {}
            for i in range(self.available_gpus):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                memory_info[f"GPU_{i}"] = {
                    "allocated_GB": allocated,
                    "reserved_GB": reserved,
                    "total_GB": total
                }
            return memory_info
        return {"message": "CUDA not available"}
    
    def __repr__(self):
        return f"InternVLM(model_name='{self.model_name}', model_size='{self.model_size}')"
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            self._optimize_gpu_memory()
        except:
            pass