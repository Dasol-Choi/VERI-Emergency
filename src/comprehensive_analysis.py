#!/usr/bin/env python3
"""
Comprehensive Analysis: Q1 + Q2 Results Summary
Usage: python src/comprehensive_analysis.py --model qwen_7b --results_dir results
"""

import os
import json
import argparse
import pandas as pd
from datetime import datetime


def load_results(model_name, results_dir):
    """Load Q1 and Q2 results for a model"""
    q1_file = f"{results_dir}/q1_results_{model_name}.json"
    q2_file = f"{results_dir}/q2_results_{model_name}.json"
    
    results = {"model": model_name}
    
    # Load Q1 results
    if os.path.exists(q1_file):
        with open(q1_file, 'r') as f:
            results["q1"] = json.load(f)
        print(f"✅ Loaded Q1 results: {q1_file}")
    else:
        print(f"❌ Q1 results not found: {q1_file}")
        results["q1"] = None
    
    # Load Q2 results
    if os.path.exists(q2_file):
        with open(q2_file, 'r') as f:
            results["q2"] = json.load(f)
        print(f"✅ Loaded Q2 results: {q2_file}")
    else:
        print(f"❌ Q2 results not found: {q2_file}")
        results["q2"] = None
    
    return results


def analyze_q1_performance(q1_data):
    """Analyze Q1 performance by category"""
    if not q1_data:
        return None
    
    detailed_results = q1_data.get("detailed_results", [])
    
    # Overall metrics
    metrics = q1_data.get("metrics", {})
    
    # Category-wise analysis
    categories = {}
    for item in detailed_results:
        cat = item.get("category", "unknown")
        if cat not in categories:
            categories[cat] = {"total": 0, "correct": 0, "tp": 0, "fp": 0, "tn": 0, "fn": 0}
        
        categories[cat]["total"] += 1
        if item.get("is_correct"):
            categories[cat]["correct"] += 1
        
        # Confusion matrix
        gt = item.get("ground_truth")
        pred = item.get("prediction")
        
        if gt == "danger" and pred == "danger":
            categories[cat]["tp"] += 1
        elif gt == "safe" and pred == "danger":
            categories[cat]["fp"] += 1
        elif gt == "safe" and pred == "safe":
            categories[cat]["tn"] += 1
        elif gt == "danger" and pred == "safe":
            categories[cat]["fn"] += 1
    
    # Calculate category metrics
    category_metrics = {}
    for cat, stats in categories.items():
        if stats["total"] > 0:
            accuracy = stats["correct"] / stats["total"]
            
            # Precision and Recall
            precision = stats["tp"] / (stats["tp"] + stats["fp"]) if (stats["tp"] + stats["fp"]) > 0 else 0
            recall = stats["tp"] / (stats["tp"] + stats["fn"]) if (stats["tp"] + stats["fn"]) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            category_metrics[cat] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "total_images": stats["total"],
                "false_positive_rate": stats["fp"] / (stats["fp"] + stats["tn"]) if (stats["fp"] + stats["tn"]) > 0 else 0
            }
    
    return {
        "overall_metrics": metrics,
        "category_metrics": category_metrics,
        "confusion_matrix": q1_data.get("confusion_matrix", {})
    }


def analyze_q2_performance(q2_data):
    """Analyze Q2 performance by category"""
    if not q2_data:
        return None
    
    detailed_results = q2_data.get("detailed_results", [])
    summary = q2_data.get("summary", {})
    
    # Category-wise Q2 analysis
    categories = {}
    valid_scores = []
    
    for item in detailed_results:
        cat = item.get("category", "unknown")
        score = item.get("gpt4o_score", -1)
        
        if cat not in categories:
            categories[cat] = {"scores": [], "total": 0, "valid": 0}
        
        categories[cat]["total"] += 1
        
        if score >= 0:
            categories[cat]["scores"].append(score)
            categories[cat]["valid"] += 1
            valid_scores.append(score)
    
    # Calculate category averages
    category_averages = {}
    for cat, data in categories.items():
        if data["scores"]:
            category_averages[cat] = {
                "average_score": sum(data["scores"]) / len(data["scores"]),
                "max_score": max(data["scores"]),
                "min_score": min(data["scores"]),
                "total_responses": data["total"],
                "valid_responses": data["valid"],
                "success_rate": data["valid"] / data["total"] if data["total"] > 0 else 0
            }
        else:
            category_averages[cat] = {
                "average_score": 0,
                "max_score": 0,
                "min_score": 0,
                "total_responses": data["total"],
                "valid_responses": 0,
                "success_rate": 0
            }
    
    return {
        "overall_summary": summary,
        "category_performance": category_averages,
        "overall_average": sum(valid_scores) / len(valid_scores) if valid_scores else 0,
        "total_valid_scores": len(valid_scores)
    }


def generate_comprehensive_report(results):
    """Generate comprehensive analysis report"""
    model_name = results["model"]
    q1_analysis = analyze_q1_performance(results["q1"])
    q2_analysis = analyze_q2_performance(results["q2"])
    
    report = {
        "model": model_name,
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "q1_analysis": q1_analysis,
        "q2_analysis": q2_analysis
    }
    
    # Calculate pipeline success rate (Q1 correct → Q2 generated)
    if q1_analysis and q2_analysis:
        # How many dangerous images were correctly identified in Q1
        q1_emergency_correct = 0
        if q1_analysis["confusion_matrix"]:
            q1_emergency_correct = q1_analysis["confusion_matrix"].get("true_positive", 0)
        
        # How many Q2 responses were generated
        q2_generated = q2_analysis["overall_summary"].get("generated_responses", 0)
        
        # Pipeline efficiency
        pipeline_efficiency = q2_generated / q1_emergency_correct if q1_emergency_correct > 0 else 0
        
        report["pipeline_analysis"] = {
            "q1_emergency_correct": q1_emergency_correct,
            "q2_responses_generated": q2_generated,
            "pipeline_efficiency": pipeline_efficiency,
            "end_to_end_performance": q2_analysis["overall_average"] if q2_analysis else 0
        }
    
    return report


def print_summary(report):
    """Print comprehensive summary"""
    model_name = report["model"]
    
    print(f"\n{'='*60}")
    print(f"📊 COMPREHENSIVE ANALYSIS - {model_name.upper()}")
    print(f"{'='*60}")
    
    # Q1 Summary
    if report["q1_analysis"]:
        q1 = report["q1_analysis"]
        print(f"\n🎯 Q1: Risk Identification Performance")
        print(f"   Overall Accuracy: {q1['overall_metrics'].get('accuracy', 0):.1f}%")
        print(f"   Precision: {q1['overall_metrics'].get('precision', 0):.3f}")
        print(f"   Recall: {q1['overall_metrics'].get('recall', 0):.3f}")
        print(f"   F1 Score: {q1['overall_metrics'].get('f1', 0):.3f}")
        
        # False positive analysis
        cm = q1["confusion_matrix"]
        if cm:
            fp_rate = cm.get("false_positive", 0) / (cm.get("false_positive", 0) + cm.get("true_negative", 1))
            print(f"   False Positive Rate: {fp_rate:.1%} (Overreaction Problem)")
        
        print(f"\n   📂 Category Breakdown:")
        for cat, metrics in q1["category_metrics"].items():
            cat_name = {"AB": "Accidents", "PME": "Medical", "ND": "Disasters"}.get(cat, cat)
            print(f"      {cat_name}: F1={metrics['f1']:.3f}, FP Rate={metrics['false_positive_rate']:.1%}")
    
    # Q2 Summary
    if report["q2_analysis"]:
        q2 = report["q2_analysis"]
        print(f"\n🧠 Q2: Emergency Response Performance")
        print(f"   Overall Average Score: {q2['overall_average']:.3f}")
        print(f"   Total Responses Evaluated: {q2['total_valid_scores']}")
        
        print(f"\n   📂 Category Performance:")
        for cat, perf in q2["category_performance"].items():
            cat_name = {"AB": "Accidents", "PME": "Medical", "ND": "Disasters"}.get(cat, cat)
            print(f"      {cat_name}: Avg={perf['average_score']:.3f}, Success={perf['success_rate']:.1%}")
    
    # Pipeline Analysis
    if "pipeline_analysis" in report:
        pipeline = report["pipeline_analysis"]
        print(f"\n🔄 End-to-End Pipeline Analysis")
        print(f"   Q1 Emergencies Correctly Identified: {pipeline['q1_emergency_correct']}")
        print(f"   Q2 Responses Generated: {pipeline['q2_responses_generated']}")
        print(f"   Pipeline Efficiency: {pipeline['pipeline_efficiency']:.1%}")
        print(f"   End-to-End Quality Score: {pipeline['end_to_end_performance']:.3f}")
    
    print(f"\n{'='*60}")


def save_csv_summary(report, output_dir, model_name):
    """Save summary as CSV for easy comparison"""
    
    # Flatten key metrics for CSV
    csv_data = {
        "model": model_name,
        "analysis_date": report["analysis_date"]
    }
    
    # Q1 metrics
    if report["q1_analysis"]:
        q1_metrics = report["q1_analysis"]["overall_metrics"]
        csv_data.update({
            "q1_accuracy": q1_metrics.get("accuracy", 0),
            "q1_precision": q1_metrics.get("precision", 0),
            "q1_recall": q1_metrics.get("recall", 0),
            "q1_f1": q1_metrics.get("f1", 0)
        })
        
        # Category F1 scores
        for cat, metrics in report["q1_analysis"]["category_metrics"].items():
            csv_data[f"q1_{cat}_f1"] = metrics["f1"]
            csv_data[f"q1_{cat}_fp_rate"] = metrics["false_positive_rate"]
    
    # Q2 metrics
    if report["q2_analysis"]:
        csv_data["q2_overall_score"] = report["q2_analysis"]["overall_average"]
        
        # Category Q2 scores
        for cat, perf in report["q2_analysis"]["category_performance"].items():
            csv_data[f"q2_{cat}_score"] = perf["average_score"]
    
    # Pipeline metrics
    if "pipeline_analysis" in report:
        pipeline = report["pipeline_analysis"]
        csv_data.update({
            "pipeline_efficiency": pipeline["pipeline_efficiency"],
            "end_to_end_score": pipeline["end_to_end_performance"]
        })
    
    # Save to CSV
    df = pd.DataFrame([csv_data])
    csv_file = f"{output_dir}/comprehensive_summary_{model_name}.csv"
    df.to_csv(csv_file, index=False)
    print(f"📁 CSV summary saved: {csv_file}")


def run_comprehensive_analysis(model_name, results_dir, output_dir):
    """Main comprehensive analysis function"""
    print(f"🔍 Starting comprehensive analysis for {model_name}")
    
    # Load results
    results = load_results(model_name, results_dir)
    
    if not results["q1"] and not results["q2"]:
        print("❌ No results found to analyze")
        return
    
    # Generate comprehensive report
    report = generate_comprehensive_report(results)
    
    # Print summary
    print_summary(report)
    
    # Save detailed report
    os.makedirs(output_dir, exist_ok=True)
    report_file = f"{output_dir}/comprehensive_analysis_{model_name}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Save CSV summary
    save_csv_summary(report, output_dir, model_name)
    
    print(f"\n📄 Detailed report saved: {report_file}")
    
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comprehensive Q1+Q2 Analysis")
    parser.add_argument("--model", required=True,
                       help="Model name (e.g., qwen_7b)")
    parser.add_argument("--results_dir", default=None,
                       help="Directory containing Q1 and Q2 results (default: results/{model})")
    parser.add_argument("--output_dir", default=None,
                       help="Directory to save analysis output (default: results/{model})")
    
    args = parser.parse_args()
    
    # Auto-generate paths based on model name if not specified
    if args.results_dir is None:
        args.results_dir = f"results/{args.model}"
    
    if args.output_dir is None:
        args.output_dir = f"results/{args.model}"
    
    run_comprehensive_analysis(args.model, args.results_dir, args.output_dir)