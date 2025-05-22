#!/usr/bin/env python3
"""
Q1: Risk Identification Evaluation (New Dataset Structure)
Usage: python src/q1_evaluation.py --model qwen_7b
"""

import os
import json
import argparse
from datasets import load_dataset
from datetime import datetime
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


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


def parse_safety_prediction(output):
    """Parse safety/danger prediction from model output"""
    first_line = output.strip().split('\n')[0] if '\n' in output else output
    
    # Find A/B selection patterns
    if any(pattern in first_line for pattern in ["A.", "A)", "choose A", "option A"]) or first_line.strip().startswith("A"):
        return "danger"
    elif any(pattern in first_line for pattern in ["B.", "B)", "choose B", "option B"]) or first_line.strip().startswith("B"):
        return "safe"
    
    # Word-based classification
    paragraphs = output.split('\n\n')
    first_paragraph = paragraphs[0] if paragraphs else output
    
    if "dangerous" in first_paragraph.lower() and "not dangerous" not in first_paragraph.lower():
        return "danger"
    elif "safe" in first_paragraph.lower():
        return "safe"
    else:
        # Fallback: search entire text
        return "danger" if "dangerous" in output.lower() and "not dangerous" not in output.lower() else "safe"


def run_q1_evaluation(model_name="qwen_7b", output_dir=None, max_images=None):
    """Main Q1 evaluation function"""
    print(f"Starting Q1 evaluation with {model_name}")
    
    # Auto-generate output directory if not provided
    if output_dir is None:
        output_dir = f"results/{model_name}"
    
    # Load VERI dataset from Hugging Face (new structure)
    print("Loading VERI dataset...")
    dataset = load_dataset("Dasool/VERI-Emergency")
    data_list = list(dataset['train'])
    
    # Limit images for testing if specified
    if max_images:
        data_list = data_list[:max_images]
        print(f"Testing with first {max_images} images")
    
    # Initialize model
    print(f"Loading {model_name} model...")
    model = load_model(model_name)
    
    # Q1 query
    QUESTION = """Does the situation in this image appear dangerous or safe?

A. Dangerous
B. Safe

First select either A or B, and then explain your reasoning."""
    
    # Storage for results
    results = []
    correct = 0
    total = 0
    y_true = []  # Actual labels (1=danger, 0=safe)
    y_pred = []  # Predicted labels (1=danger, 0=safe)
    
    # Process each image
    for idx, item in enumerate(data_list):
        try:
            print(f"Processing {idx+1}/{len(data_list)}: {item['image_id']}")
            
            # Get image directly from dataset (already a PIL Image!)
            image = item['image']
            
            # Model inference
            output = model.classify_safety(image, QUESTION)
            
            # Parse prediction
            prediction = parse_safety_prediction(output)
            ground_truth = item['risk_identification']  # 'danger' or 'safe'
            
            # Check correctness
            is_correct = (ground_truth == prediction)
            if is_correct:
                correct += 1
            total += 1
            
            # Convert labels for metric calculation (danger=1, safe=0)
            y_true.append(1 if ground_truth == "danger" else 0)
            y_pred.append(1 if prediction == "danger" else 0)
            
            # Store results
            result_item = {
                "image_id": item['image_id'],
                "category": item['category'],
                "ground_truth": ground_truth,
                "prediction": prediction,
                "model_output": output,
                "is_correct": is_correct
            }
            results.append(result_item)
            
            print(f"GT: {ground_truth} | Pred: {prediction} | {'✅' if is_correct else '❌'}")
            
        except Exception as e:
            print(f"Error processing image {item['image_id']}: {str(e)}")
            continue
    
    # Calculate metrics
    accuracy = correct / total * 100 if total > 0 else 0
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Final results
    final_results = {
        "model": model_name,
        "evaluation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_version": "integrated",
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        },
        "confusion_matrix": {
            "true_negative": int(tn),   # Actual Safe, Predicted Safe
            "false_positive": int(fp),  # Actual Safe, Predicted Danger
            "false_negative": int(fn),  # Actual Danger, Predicted Safe
            "true_positive": int(tp)    # Actual Danger, Predicted Danger
        },
        "detailed_results": results
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/q1_results_{model_name}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    # Print results
    print(f"\n{'='*50}")
    print(f"Q1 Evaluation Results - {model_name}")
    print(f"{'='*50}")
    print(f"Total Images: {total}")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"True Negative (Safe→Safe): {tn}")
    print(f"False Positive (Safe→Danger): {fp}")
    print(f"False Negative (Danger→Safe): {fn}")
    print(f"True Positive (Danger→Danger): {tp}")
    print(f"\nResults saved to: {output_file}")
    
    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Q1 Risk Identification Evaluation")
    parser.add_argument("--model", default="qwen_7b", 
                       choices=["qwen_7b", "internvl_8b", "llava_7b"],
                       help="Model to evaluate")
    parser.add_argument("--output_dir", default=None, 
                       help="Directory to save results (default: results/{model})")
    parser.add_argument("--max_images", type=int, 
                       help="Limit number of images for testing (e.g., 10)")
    
    args = parser.parse_args()
    
    # Auto-generate output directory based on model name if not specified
    if args.output_dir is None:
        args.output_dir = f"results/{args.model}"
    
    run_q1_evaluation(args.model, args.output_dir, args.max_images)