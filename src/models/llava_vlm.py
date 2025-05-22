#!/usr/bin/env python3
"""
src/models/llava_vlm.py - LLaVA-Next Model Implementation
"""

import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


class LLaVAVLM:
    def __init__(self, model_size="7B"):
        """
        Initialize LLaVA-Next model
        Args:
            model_size: Choose from "7B", "13B", "34B"
        """
        # Model mapping
        model_mapping = {
            "7B": "llava-hf/llava-v1.6-vicuna-7b-hf",
            "13B": "llava-hf/llava-v1.6-vicuna-13b-hf"
        }
        
        if model_size not in model_mapping:
            raise ValueError(f"Unsupported model size: {model_size}. Choose from {list(model_mapping.keys())}")
        
        self.model_name = model_mapping[model_size]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading {self.model_name}...")
        
        # Load processor
        self.processor = LlavaNextProcessor.from_pretrained(self.model_name)
        
        # Load model
        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            device_map="auto"
        )
        
        print(f"Model loaded successfully on {self.device}")
    
    def _generate_response(self, image, query, max_tokens=150):
        """
        Generate response from image and query
        """
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        
        # Create conversation format for LLaVA-Next
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": query},
                    {"type": "image"},
                ],
            },
        ]
        
        # Apply chat template
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        
        # Prepare inputs
        inputs = self.processor(
            images=image, 
            text=prompt, 
            return_tensors="pt"
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, 
                max_new_tokens=max_tokens,
                do_sample=False,
                temperature=0.0
            )
        
        # Decode response
        output = self.processor.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract response after "ASSISTANT:" token
        if "ASSISTANT:" in output:
            response = output.split("ASSISTANT:")[-1].strip()
        else:
            response = output.strip()
        
        return response
    
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
    
    def __repr__(self):
        return f"LLaVAVLM(model_name='{self.model_name}', device='{self.device}')"