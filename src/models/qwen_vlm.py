#!/usr/bin/env python3
"""
src/models/qwen_vlm.py - Qwen2.5-VL Model Implementation
"""

import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor


class QwenVLM:
    def __init__(self, model_size="7B"):
        """
        Initialize Qwen2.5-VL model
        Args:
            model_size: Choose from "3B", "7B", "32B", "72B"
        """
        self.model_name = f"Qwen/Qwen2.5-VL-{model_size}-Instruct"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"Loading {self.model_name}...")
        
        # Load model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        print(f"Model loaded successfully on {self.device}")
    
    def _generate_response(self, image, query, max_tokens=150):
        """
        Generate response from image and query
        """
        # Convert to PIL Image if needed
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        
        # Create message format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": query},
                ],
            }
        ]
        
        # Prepare input
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], 
            images=[image], 
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
        output = self.processor.batch_decode(
            output_ids[:, inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        )[0].strip()
        
        return output
    
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
        return f"QwenVLM(model_name='{self.model_name}', device='{self.device}')"