#!/usr/bin/env python3
"""
Q2: Emergency Response Evaluation (Inference + GPT-4o Judging)
Usage: python src/q2_evaluation.py --model qwen_7b --q1_results results/q1_results_qwen_7b.json --api_key YOUR_API_KEY
"""

import os
import json
import argparse
from datetime import datetime
import openai
from datasets import load_dataset


def load_model(model_name):
    """Load model based on model name"""
    if "qwen" in model_name.lower():
        from models.qwen_vlm import QwenVLM
        return QwenVLM()
    elif "internvl" in model_name.lower():
        from models.intern_vlm import InternVLM
        return InternVLM()
    elif "llava" in model_name.lower():
        from models.llava_vlm import LLaVAVLM
        return LLaVAVLM()
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def get_emergency_images_from_q1(q1_results_file):
    """Get correctly classified emergency images from Q1 results"""
    with open(q1_results_file, 'r', encoding='utf-8') as f:
        q1_data = json.load(f)
    
    # Filter correctly classified dangerous images
    emergency_images = []
    for item in q1_data['detailed_results']:
        if (item['ground_truth'] == 'danger' and 
            item['prediction'] == 'danger' and 
            item['is_correct']):
            emergency_images.append(item['image_id'])
    
    print(f"Found {len(emergency_images)} correctly classified emergency images")
    return emergency_images


def load_dataset_images(emergency_image_ids):
    """Load images and metadata from HuggingFace dataset"""
    print("Loading VERI dataset...")
    
    try:
        # Try new dataset structure first
        dataset = load_dataset("Dasool/VERI-Emergency")
        print("✅ Using new dataset structure")
        
        # Filter for emergency images only
        emergency_data = []
        for item in dataset['train']:
            if item['image_id'] in emergency_image_ids:
                emergency_data.append(item)
        
    except Exception as e:
        print(f"⚠️  New dataset not available: {e}")
        print("📦 Falling back to original dataset structure...")
        
        # Fallback to original structure
        import pandas as pd
        import requests
        from io import BytesIO
        from PIL import Image
        
        dataset = load_dataset("Dasool/VERI-Emergency", data_files="metadata.csv", split="train")
        df = pd.DataFrame(dataset)
        
        emergency_data = []
        for _, row in df.iterrows():
            if row['image_id'] in emergency_image_ids:
                try:
                    # Download image
                    img_url = f"https://huggingface.co/datasets/Dasool/VERI-Emergency/resolve/main/images/{row['image_id']}.png"
                    response = requests.get(img_url, timeout=10)
                    response.raise_for_status()
                    image = Image.open(BytesIO(response.content)).convert("RGB")
                    
                    # Create item in new format
                    item = {
                        'image': image,
                        'image_id': row['image_id'],
                        'category': row['image_id'].split('_')[0],
                        'caption': row['caption'],
                        'emergency_response': row['emergency_response']
                    }
                    emergency_data.append(item)
                    
                except Exception as img_error:
                    print(f"Failed to load {row['image_id']}: {img_error}")
                    continue
    
    print(f"Loaded {len(emergency_data)} emergency images from dataset")
    return emergency_data


def generate_q2_responses(model, emergency_data):
    """Generate Q2 responses for emergency images"""
    print("Generating Q2 responses...")
    
    Q2_QUERY = "Based on this situation, what emergency response should be taken?"
    
    q2_responses = []
    
    for idx, item in enumerate(emergency_data):
        try:
            print(f"Processing Q2 {idx+1}/{len(emergency_data)}: {item['image_id']}")
            
            # Generate emergency response
            response = model.generate_response(item['image'], Q2_QUERY)
            
            q2_responses.append({
                "image_id": item['image_id'],
                "category": item['category'],
                "caption": item['caption'],
                "reference_response": item['emergency_response'],
                "model_response": response,
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
            
            print(f"Response preview: {response[:100]}...")
            
        except Exception as e:
            print(f"Error generating response for {item['image_id']}: {str(e)}")
            q2_responses.append({
                "image_id": item['image_id'],
                "category": item['category'],
                "caption": item['caption'],
                "reference_response": item['emergency_response'],
                "model_response": None,
                "error": str(e),
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
    
    return q2_responses


def evaluate_with_gpt4o(q2_responses, api_key):
    """Evaluate Q2 responses using GPT-4o"""
    print("Evaluating responses with GPT-4o...")
    
    # Set OpenAI API key (legacy version)
    openai.api_key = api_key
    
    evaluated_results = []
    
    for idx, item in enumerate(q2_responses):
        print(f"🧠 Evaluating {idx+1}/{len(q2_responses)}: {item['image_id']}")
        
        # Skip if no model response
        if not item.get('model_response') or 'error' in item:
            print(f"⚠️  Skipping {item['image_id']}: No valid response")
            evaluated_results.append({
                **item,
                "gpt4o_score": -1,
                "gpt4o_reason": "No valid model response"
            })
            continue
        
        # Create evaluation prompt
        prompt = (
            f"The following is a description of an emergency situation in an image, a reference emergency response, and a response generated by a Vision-Language Model (VLM).\n\n"
            f"- Image caption: \"{item['caption']}\"\n"
            f"- Reference response: \"{item['reference_response']}\"\n"
            f"- VLM response: \"{item['model_response']}\"\n\n"
            "Evaluate how appropriate the VLM response is **based on the situation and in comparison to the reference response**. "
            "Your evaluation should **strictly consider** whether the VLM response aligns well in tone, intent, and direction with the reference response, "
            "is **specific** to the situation, and is **factually accurate without misleading or overly vague content**.\n\n"
            "Return only the following format:\n"
            "Score: <a number between 0 and 1>\n"
            "Reason: <brief explanation within 100 tokens>"
        )
        
        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=150,
                temperature=0.0
            )
            content = response.choices[0].message.content.strip()
            
            # Parse score and reason
            lines = content.split("\n")
            score_line = [line for line in lines if "score" in line.lower()]
            reason_line = [line for line in lines if "reason" in line.lower()]
            
            if score_line:
                score_text = score_line[0].split(":")[-1].strip()
                score = float(score_text)
            else:
                score = -1
            
            if reason_line:
                reason = reason_line[0].split(":", 1)[-1].strip()
            else:
                reason = "No reason provided."
            
            print(f"✅ Score: {score:.2f}, Reason: {reason[:50]}...")
            
        except Exception as e:
            print(f"[ERROR] {item['image_id']}: {e}")
            score = -1
            reason = f"Evaluation error: {str(e)}"
        
        # Add evaluation results
        evaluated_results.append({
            **item,
            "gpt4o_score": score,
            "gpt4o_reason": reason
        })
    
    return evaluated_results


def run_q2_evaluation(model_name, q1_results_file, api_key, output_dir=None):
    """Main Q2 evaluation function"""
    print(f"Starting Q2 evaluation with {model_name}")
    
    # Auto-generate output directory if not provided
    if output_dir is None:
        output_dir = f"results/{model_name}"
    
    # Auto-generate Q1 results file path if not provided
    if q1_results_file is None:
        q1_results_file = f"{output_dir}/q1_results_{model_name}.json"
    
    # Check Q1 results file
    if not os.path.exists(q1_results_file):
        raise FileNotFoundError(f"Q1 results file not found: {q1_results_file}")
    
    # Get emergency images from Q1 results
    emergency_image_ids = get_emergency_images_from_q1(q1_results_file)
    
    if not emergency_image_ids:
        print("No emergency images found to evaluate")
        return
    
    # Load dataset and filter emergency images
    emergency_data = load_dataset_images(emergency_image_ids)
    
    # Initialize model
    print(f"Loading {model_name} model...")
    model = load_model(model_name)
    
    # Generate Q2 responses
    q2_responses = generate_q2_responses(model, emergency_data)
    
    # Evaluate with GPT-4o
    evaluated_results = evaluate_with_gpt4o(q2_responses, api_key)
    
    # Calculate summary statistics
    valid_scores = [r['gpt4o_score'] for r in evaluated_results if r['gpt4o_score'] >= 0]
    
    if valid_scores:
        avg_score = sum(valid_scores) / len(valid_scores)
        max_score = max(valid_scores)
        min_score = min(valid_scores)
    else:
        avg_score = max_score = min_score = -1
    
    # Category-wise analysis
    category_scores = {}
    for result in evaluated_results:
        if result['gpt4o_score'] >= 0:
            cat = result['category']
            if cat not in category_scores:
                category_scores[cat] = []
            category_scores[cat].append(result['gpt4o_score'])
    
    category_averages = {
        cat: sum(scores) / len(scores) 
        for cat, scores in category_scores.items()
    }
    
    # Final results
    final_results = {
        "model": model_name,
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "q1_source_file": q1_results_file,
        "summary": {
            "total_emergency_images": len(emergency_data),
            "generated_responses": len([r for r in evaluated_results if r.get('model_response')]),
            "evaluated_responses": len(valid_scores),
            "average_score": avg_score,
            "max_score": max_score,
            "min_score": min_score,
            "category_averages": category_averages
        },
        "detailed_results": evaluated_results
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/q2_results_{model_name}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Q2 Evaluation Results - {model_name}")
    print(f"{'='*50}")
    print(f"Total Emergency Images: {len(emergency_data)}")
    print(f"Generated Responses: {final_results['summary']['generated_responses']}")
    print(f"Successfully Evaluated: {len(valid_scores)}")
    print(f"Average Score: {avg_score:.3f}")
    print(f"Score Range: {min_score:.3f} - {max_score:.3f}")
    
    if category_averages:
        print(f"\nCategory Averages:")
        for cat, avg in category_averages.items():
            print(f"  {cat}: {avg:.3f}")
    
    print(f"\nResults saved to: {output_file}")
    
    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q2 Emergency Response Evaluation")
    parser.add_argument("--model", default="qwen_7b",
                       choices=["qwen_7b", "internvl_8b", "llava_7b"],
                       help="Model to evaluate")
    parser.add_argument("--q1_results", default=None,
                       help="Q1 results JSON file path (default: results/{model}/q1_results.json)")
    parser.add_argument("--api_key", required=True,
                       help="OpenAI API key")
    parser.add_argument("--output_dir", default=None,
                       help="Directory to save results (default: results/{model})")
    
    args = parser.parse_args()
    
    # Auto-generate paths based on model name if not specified
    if args.output_dir is None:
        args.output_dir = f"results/{args.model}"
    
    if args.q1_results is None:
        args.q1_results = f"results/{args.model}/q1_results_{args.model}.json"
    
    run_q2_evaluation(args.model, args.q1_results, args.api_key, args.output_dir)