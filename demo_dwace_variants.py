#!/usr/bin/env python3
"""
DWACE Variants Comparison Demo
Shows the differences between Original, Paper, and Simplified DWACE implementations
"""

import numpy as np
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Add the current directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from datasets.loaders import load_karate
from main import run_method, run_clustering_evaluation

def demo_dwace_variants():
    """Compare all DWACE variants on the same dataset"""
    
    print("🔬 DWACE Variants Comparison Demo")
    print("=" * 60)
    print("Comparing different DWACE implementations:")
    print("1. dwace (Original): DE + AutoEncoder + Contrastive")  
    print("2. dwace_paper: Paper Implementation with Modularity Loss")
    print("3. dwace_simplified: MSE Loss Only (No Graph Losses)")
    print("=" * 60)
    
    # Load dataset
    print("\n📊 Loading Karate Club dataset...")
    G, ground_truth = load_karate()
    print(f"   Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"   Communities: {len(np.unique(ground_truth))}")
    
    # DWACE variants to compare
    variants = [
        ("dwace", "Original DWACE (DE + AE + Contrastive)"),
        ("dwace_paper", "Paper DWACE (Modularity Loss + InfoNCE)"),
        ("dwace_simplified", "Simplified DWACE (MSE Loss Only)")
    ]
    
    results = []
    
    print(f"\n🔬 Running DWACE Variants...")
    print("-" * 60)
    
    for method_name, description in variants:
        print(f"\n🔄 Testing {method_name}...")
        print(f"   Description: {description}")
        
        try:
            # Run the method
            embedding, duration = run_method(method_name, G, ground_truth, max_epochs=10)
            
            if embedding is not None:
                # Run clustering evaluation
                clustering_results = run_clustering_evaluation(embedding, G, ground_truth, method_name)
                
                if clustering_results:
                    # Get best ARI result
                    best_result = max(clustering_results, key=lambda x: x['ARI'])
                    
                    result_info = {
                        'method': method_name,
                        'description': description,
                        'embedding_shape': embedding.shape,
                        'duration': duration,
                        'best_ari': best_result['ARI'],
                        'best_nmi': best_result['NMI'],
                        'best_modularity': best_result['Modularity'],
                        'clustering_method': best_result['Clustering']
                    }
                    
                    results.append(result_info)
                    
                    print(f"   ✅ Completed: Shape={embedding.shape}, Time={duration:.2f}s")
                    print(f"   📊 Best Result ({best_result['Clustering']}): "
                          f"ARI={best_result['ARI']:.4f}, "
                          f"NMI={best_result['NMI']:.4f}, "
                          f"Modularity={best_result['Modularity']:.4f}")
                else:
                    print(f"   ❌ Clustering evaluation failed")
            else:
                print(f"   ❌ Method failed to return embedding")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Summary comparison
    if results:
        print(f"\n📊 DWACE VARIANTS COMPARISON SUMMARY")
        print("=" * 80)
        print(f"{'Method':<20} {'Description':<35} {'Shape':<12} {'Time':<8} {'ARI':<8} {'Modularity':<10}")
        print("-" * 80)
        
        for result in results:
            print(f"{result['method']:<20} "
                  f"{result['description'][:34]:<35} "
                  f"{str(result['embedding_shape']):<12} "
                  f"{result['duration']:<8.1f} "
                  f"{result['best_ari']:<8.4f} "
                  f"{result['best_modularity']:<10.4f}")
        
        # Find best performers
        best_ari = max(results, key=lambda x: x['best_ari'])
        best_modularity = max(results, key=lambda x: x['best_modularity'])
        fastest = min(results, key=lambda x: x['duration'])
        
        print(f"\n🏆 PERFORMANCE INSIGHTS:")
        print(f"   🥇 Best ARI: {best_ari['method']} ({best_ari['best_ari']:.4f})")
        print(f"   🏗️  Best Modularity: {best_modularity['method']} ({best_modularity['best_modularity']:.4f})")
        print(f"   ⚡ Fastest: {fastest['method']} ({fastest['duration']:.2f}s)")
        
        print(f"\n💡 KEY OBSERVATIONS:")
        print(f"   • DWACE Simplified uses pure MSE loss (no graph structure)")
        print(f"   • DWACE Paper includes modularity loss for graph awareness")
        print(f"   • DWACE Original combines DE optimization with contrastive learning")
        print(f"   • Different architectures lead to different modularity vs ARI trade-offs")
        
    else:
        print(f"\n❌ No results generated!")

if __name__ == "__main__":
    demo_dwace_variants()
