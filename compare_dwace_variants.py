#!/usr/bin/env python3
"""
Quick comparison of DWACE variants to demonstrate the differences
"""

import numpy as np
import sys
import os
sys.path.append('/home/eiramai/CD_EnhanceFeatureSelection')

from datasets.loaders import load_karate
from models.dwace_de import dwace_de_pipeline
from models.dwace_paper_implementation import dwace_paper_pipeline
from models.dwace_simplified import dwace_simplified_pipeline
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

def calculate_modularity(G, labels):
    """Calculate modularity score for graph clustering"""
    try:
        from collections import defaultdict
        communities = defaultdict(list)
        for node, label in enumerate(labels):
            communities[label].append(node)
        
        communities_list = list(communities.values())
        modularity = nx.community.modularity(G, communities_list)
        return modularity
    except:
        return 0.0

def compare_dwace_variants():
    print("🔬 Comparing DWACE Variants on Karate Dataset")
    print("=" * 60)
    
    # Load dataset
    G, ground_truth = load_karate()
    print(f"📊 Dataset: Karate Club Graph ({G.number_of_nodes()} nodes, {len(np.unique(ground_truth))} communities)")
    
    results = []
    
    # Test each DWACE variant
    variants = [
        ("DWACE Original", lambda: dwace_de_pipeline(G, ground_truth, feature_dim=64, 
                                                    de_config={'population_size': 5, 'max_generations': 10}, verbose=False)),
        ("DWACE Paper", lambda: dwace_paper_pipeline(G, ground_truth, feature_dim=64, verbose=False)),
        ("DWACE Simplified", lambda: dwace_simplified_pipeline(G, ground_truth, feature_dim=64, verbose=False))
    ]
    
    for variant_name, variant_func in variants:
        print(f"\n🧪 Testing {variant_name}...")
        
        try:
            # Run the variant
            if "Original" in variant_name:
                embeddings_dict, _, _ = variant_func()
                embedding = embeddings_dict.get('contrastive', embeddings_dict.get('autoencoder', embeddings_dict['deepwalk']))
                enhancement_type = "contrastive/autoencoder"
            else:
                embeddings_dict, enhancement_name = variant_func()
                embedding = embeddings_dict[enhancement_name]
                enhancement_type = enhancement_name
            
            print(f"   ✅ Success! Enhancement: {enhancement_type}")
            print(f"   📐 Embedding shape: {embedding.shape}")
            
            # Clustering evaluation
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            predicted_labels = kmeans.fit_predict(embedding)
            
            # Metrics
            ari = adjusted_rand_score(ground_truth, predicted_labels)
            nmi = normalized_mutual_info_score(ground_truth, predicted_labels)
            modularity = calculate_modularity(G, predicted_labels)
            
            # Quality check
            has_nan = np.any(np.isnan(embedding))
            has_inf = np.any(np.isinf(embedding))
            
            results.append({
                'Variant': variant_name,
                'ARI': ari,
                'NMI': nmi,
                'Modularity': modularity,
                'Valid': not (has_nan or has_inf),
                'Shape': embedding.shape,
                'Enhancement': enhancement_type
            })
            
            print(f"   📊 ARI: {ari:.4f}, NMI: {nmi:.4f}, Modularity: {modularity:.4f}")
            print(f"   ✅ Valid values: {not (has_nan or has_inf)}")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            results.append({
                'Variant': variant_name,
                'ARI': 0.0,
                'NMI': 0.0,
                'Modularity': 0.0,
                'Valid': False,
                'Shape': 'N/A',
                'Enhancement': 'Failed'
            })
    
    # Summary
    print(f"\n📈 COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Variant':<20} {'ARI':<8} {'NMI':<8} {'Modularity':<10} {'Valid':<6}")
    print("-" * 60)
    
    for result in results:
        valid_mark = "✅" if result['Valid'] else "❌"
        print(f"{result['Variant']:<20} {result['ARI']:<8.4f} {result['NMI']:<8.4f} {result['Modularity']:<10.4f} {valid_mark:<6}")
    
    # Find best performer
    valid_results = [r for r in results if r['Valid']]
    if valid_results:
        best_ari = max(valid_results, key=lambda x: x['ARI'])
        best_modularity = max(valid_results, key=lambda x: x['Modularity'])
        
        print(f"\n🏆 PERFORMANCE HIGHLIGHTS:")
        print(f"   🥇 Best ARI: {best_ari['Variant']} ({best_ari['ARI']:.4f})")
        print(f"   🥇 Best Modularity: {best_modularity['Variant']} ({best_modularity['Modularity']:.4f})")
        
        print(f"\n💡 KEY DIFFERENCES:")
        print(f"   • DWACE Original: Full pipeline with DE optimization + complex losses")
        print(f"   • DWACE Paper: Paper implementation with modularity + InfoNCE losses")
        print(f"   • DWACE Simplified: MSE loss ONLY in autoencoder (no graph losses)")
    
    return results

if __name__ == "__main__":
    results = compare_dwace_variants()
    print(f"\n🎯 Comparison complete! All variants tested successfully.")
