#!/usr/bin/env python3
"""
Enhanced Feature Selection for Community Detection in Graphs
============================================================

This is the main benchmarking script that compares different embedding methods
for community detection, with focus on DWACE (DeepWalk with Autoencoder and 
Contrastive Embedding) and optimized Differential Evolution approaches.

Methods compared:
- DeepWalk (baseline)
- Node2Vec
- DWACE (DeepWalk + Autoencoder + Contrastive)
- Optimized DE (DeepWalk + DE feature selection)
- Optimized DE Full (DeepWalk + DE + Autoencoder + Contrastive)

Author: Enhanced Community Detection Pipeline
"""

import numpy as np
import networkx as nx
import pandas as pd
import warnings
from collections import defaultdict
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
import time

warnings.filterwarnings('ignore')

# Import modules
from datasets.loaders import load_karate, load_football, load_dolphins, load_email, load_facebook
from models.feature_utils import deepwalk_embedding, node2vec_embedding
from models.dwace_de import dwace_de_pipeline
from models.optimized_de import OptimizedDifferentialEvolution, simple_autoencoder, simple_contrastive_learning
from evaluate import find_best_k
from clustering import cluster_all


def calculate_modularity(G, labels):
    """Calculate modularity score for graph clustering"""
    try:
        # Convert labels to communities format for NetworkX
        communities = defaultdict(list)
        for node, label in enumerate(labels):
            communities[label].append(node)
        
        communities_list = list(communities.values())
        modularity = nx.community.modularity(G, communities_list)
        return modularity
    except:
        return 0.0


def calculate_conductance(G, labels):
    """Calculate average conductance for all communities"""
    try:
        communities = defaultdict(list)
        for node, label in enumerate(labels):
            communities[label].append(node)
        
        conductances = []
        for community in communities.values():
            if len(community) > 0:
                subgraph = G.subgraph(community)
                internal_edges = subgraph.number_of_edges()
                
                # Count edges from community to outside
                external_edges = 0
                for node in community:
                    for neighbor in G.neighbors(node):
                        if neighbor not in community:
                            external_edges += 1
                
                total_edges = internal_edges + external_edges
                if total_edges > 0:
                    conductance = external_edges / total_edges
                    conductances.append(conductance)
        
        return np.mean(conductances) if conductances else 1.0
    except:
        return 1.0


def calculate_coverage(G, labels):
    """Calculate coverage score for graph clustering"""
    try:
        communities = defaultdict(list)
        for node, label in enumerate(labels):
            communities[label].append(node)
        
        total_internal_edges = 0
        for community in communities.values():
            if len(community) > 1:
                subgraph = G.subgraph(community)
                total_internal_edges += subgraph.number_of_edges()
        
        total_edges = G.number_of_edges()
        coverage = total_internal_edges / total_edges if total_edges > 0 else 0.0
        return coverage
    except:
        return 0.0


def evaluate_communities(G, true_labels, predicted_labels):
    """Comprehensive evaluation of community detection"""
    metrics = {}
    
    # Basic metrics
    metrics['ari'] = adjusted_rand_score(true_labels, predicted_labels)
    metrics['nmi'] = normalized_mutual_info_score(true_labels, predicted_labels)
    
    # Silhouette score (requires embeddings, we'll calculate it separately)
    metrics['silhouette'] = 0.0  # Placeholder
    
    # Graph-based metrics
    metrics['modularity'] = calculate_modularity(G, predicted_labels)
    metrics['conductance'] = calculate_conductance(G, predicted_labels)
    metrics['coverage'] = calculate_coverage(G, predicted_labels)
    
    return metrics


def load_dataset(choice):
    """Load dataset based on user choice"""
    datasets = {
        1: ("Karate Club Graph", load_karate),
        2: ("Football", load_football), 
        3: ("Dolphins", load_dolphins),
        4: ("Email", load_email),
        5: ("Facebook", load_facebook)
    }
    
    if choice not in datasets:
        raise ValueError(f"Invalid dataset choice: {choice}")
    
    name, loader = datasets[choice]
    G, ground_truth = loader()
    return G, ground_truth, name


def run_method(method_name, G, ground_truth, max_epochs=20):
    """Run a specific embedding method"""
    print(f"  Running {method_name}...")
    start_time = time.time()
    
    try:
        if method_name == "deepwalk":
            embedding = deepwalk_embedding(G, dim=64, walk_length=30, num_walks=200)
            
        elif method_name == "node2vec":
            embedding = node2vec_embedding(G, dim=64, walk_length=30, num_walks=200)
            
        elif method_name == "dwace":
            # DWACE: DeepWalk + Autoencoder + Contrastive Embedding
            embeddings_dict, _, _ = dwace_de_pipeline(
                G, ground_truth, 
                feature_dim=64,
                de_config={'population_size': 10, 'max_generations': max_epochs}
            )
            # Use the final enhanced embedding
            embedding = embeddings_dict.get('contrastive', embeddings_dict.get('autoencoder', embeddings_dict['deepwalk']))
            
        elif method_name == "deepwalk_optimized_de":
            # DeepWalk + Optimized DE feature selection
            deepwalk_emb = deepwalk_embedding(G, dim=64, walk_length=30, num_walks=200)
            
            de_optimizer = OptimizedDifferentialEvolution(
                population_size=8,
                max_generations=max_epochs,
                F=0.5, CR=0.7,
                min_features_ratio=0.6,
                max_features_ratio=0.8
            )
            
            feature_mask, _, _ = de_optimizer.optimize(deepwalk_emb, G, ground_truth)
            selected_features = np.where(feature_mask)[0]
            embedding = deepwalk_emb[:, selected_features]
            
        elif method_name == "deepwalk_optimized_de_full":
            # DeepWalk + DE + Autoencoder + Contrastive (Full Pipeline)
            deepwalk_emb = deepwalk_embedding(G, dim=64, walk_length=30, num_walks=200)
            
            # DE feature selection
            de_optimizer = OptimizedDifferentialEvolution(
                population_size=8,
                max_generations=max_epochs,
                F=0.5, CR=0.7,
                min_features_ratio=0.6,
                max_features_ratio=0.8
            )
            
            feature_mask, _, _ = de_optimizer.optimize(deepwalk_emb, G, ground_truth)
            selected_features = np.where(feature_mask)[0]
            de_embedding = deepwalk_emb[:, selected_features]
            
            # Autoencoder
            ae_embedding = simple_autoencoder(
                de_embedding,
                output_dim=min(32, len(selected_features)),
                epochs=max_epochs
            )
            
            # Contrastive learning
            if ground_truth is not None and len(np.unique(ground_truth)) > 1:
                embedding = simple_contrastive_learning(
                    ae_embedding,
                    ground_truth,
                    epochs=max_epochs,
                    temperature=0.5  # Increased temperature to prevent overflow
                )
            else:
                embedding = ae_embedding
                
        else:
            raise ValueError(f"Unknown method: {method_name}")
        
        duration = time.time() - start_time
        return embedding, duration
        
    except Exception as e:
        print(f"    Error in {method_name}: {e}")
        return None, 0


def run_clustering_evaluation(embedding, G, ground_truth, method_name):
    """Run clustering and evaluation on embedding"""
    if embedding is None:
        return []
    
    n_clusters = len(np.unique(ground_truth)) if ground_truth is not None else find_best_k(embedding)
    clustering_methods = {
        'KMeans': KMeans(n_clusters=n_clusters, random_state=42, n_init=10),
        'Agglomerative': AgglomerativeClustering(n_clusters=n_clusters),
        'Spectral': SpectralClustering(n_clusters=n_clusters, random_state=42)
    }
    
    results = []
    
    for cluster_name, clusterer in clustering_methods.items():
        try:
            predicted_labels = clusterer.fit_predict(embedding)
            
            if ground_truth is not None:
                metrics = evaluate_communities(G, ground_truth, predicted_labels)
                
                # Calculate silhouette score
                if embedding.shape[0] > n_clusters:
                    silhouette = silhouette_score(embedding, predicted_labels)
                else:
                    silhouette = 0.0
                
                results.append({
                    'Method': method_name,
                    'Clustering': cluster_name,
                    'ARI': metrics['ari'],
                    'NMI': metrics['nmi'], 
                    'Modularity': metrics['modularity'],
                    'Silhouette': silhouette,
                    'Conductance': metrics['conductance'],
                    'Coverage': metrics['coverage'],
                    'Run_Time': 0  # Will be added later
                })
            else:
                # No ground truth - only calculate graph-based metrics
                metrics = calculate_modularity(G, predicted_labels)
                silhouette = silhouette_score(embedding, predicted_labels) if embedding.shape[0] > n_clusters else 0.0
                
                results.append({
                    'Method': method_name,
                    'Clustering': cluster_name,
                    'ARI': 0.0,  # Cannot calculate without ground truth
                    'NMI': 0.0,  # Cannot calculate without ground truth
                    'Modularity': metrics,
                    'Silhouette': silhouette,
                    'Conductance': calculate_conductance(G, predicted_labels),
                    'Coverage': calculate_coverage(G, predicted_labels),
                    'Run_Time': 0  # Will be added later
                })
                
        except Exception as e:
            print(f"    Clustering error with {cluster_name}: {e}")
            continue
    
    return results


def run_benchmark():
    """Main benchmarking function"""
    print("🚀 Enhanced Feature Selection for Community Detection")
    print("=" * 60)
    
    # Dataset selection
    print("\nAvailable datasets:")
    print("1. Karate Club Graph (34 nodes, 2 communities)")
    print("2. Football (115 nodes, 12 communities)")  
    print("3. Dolphins (62 nodes, 2 communities)")
    print("4. Email (1133 nodes)")
    print("5. Facebook (4039 nodes)")
    
    while True:
        try:
            dataset_choice = int(input("\nSelect dataset (1-5): "))
            if 1 <= dataset_choice <= 5:
                break
            print("Please enter a number between 1 and 5")
        except ValueError:
            print("Please enter a valid number")
    
    # Number of runs
    while True:
        try:
            n_runs = int(input("Number of runs (minimum 1): "))
            if n_runs >= 1:
                break
            print("Please enter a number >= 1")
        except ValueError:
            print("Please enter a valid number")
    
    # Max epochs
    while True:
        try:
            max_epochs = int(input("Max epochs (default 20, minimum 1): ") or "20")
            if max_epochs >= 1:
                break
            print("Please enter a number >= 1")
        except ValueError:
            print("Please enter a valid number")
    
    # Load dataset
    G, ground_truth, dataset_name = load_dataset(dataset_choice)
    print(f"\n📊 Dataset: {dataset_name}")
    print(f"   Nodes: {G.number_of_nodes()}")
    print(f"   Edges: {G.number_of_edges()}")
    if ground_truth is not None:
        print(f"   Communities: {len(np.unique(ground_truth))}")
    else:
        print("   Communities: Unknown (no ground truth)")
    
    # Methods to compare
    methods = [
        "deepwalk",
        "node2vec", 
        "dwace",
        "deepwalk_optimized_de",
        "deepwalk_optimized_de_full"
    ]
    
    print(f"\n🔬 Methods to compare:")
    for i, method in enumerate(methods, 1):
        print(f"   {i}. {method}")
    
    print(f"\n🏃 Running benchmark with {n_runs} runs, max {max_epochs} epochs...")
    print("=" * 60)
    
    all_results = []
    
    for run in range(n_runs):
        print(f"\n🔄 Run {run + 1}/{n_runs}")
        print("-" * 40)
        
        for method_name in methods:
            # Run embedding method
            embedding, duration = run_method(method_name, G, ground_truth, max_epochs)
            
            if embedding is not None:
                print(f"    ✅ {method_name} completed in {duration:.2f}s")
                
                # Run clustering evaluation
                method_results = run_clustering_evaluation(embedding, G, ground_truth, method_name)
                
                # Add run number and duration to results
                for result in method_results:
                    result['Run'] = run + 1
                    result['Dataset'] = dataset_name
                    result['Duration'] = duration
                    result['Run_Time'] = duration  # For table display
                
                all_results.extend(method_results)
            else:
                print(f"    ❌ {method_name} failed")
        
        # Display results table after each run
        if run == 0 or (run + 1) % 1 == 0:  # Show after each run
            print(f"\n📊 RESULTS AFTER RUN {run + 1}/{n_runs}")
            print("=" * 90)
            display_current_run_results(all_results, dataset_name, run + 1)
    
    # Create results DataFrame
    if all_results:
        df = pd.DataFrame(all_results)
        
        # Save detailed results
        output_file = f"detailed_results_{dataset_name.replace(' ', '_')}_{n_runs}runs.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Detailed results saved to: {output_file}")
        
        # Generate summary statistics
        print(f"\n📊 RESULTS SUMMARY - {dataset_name}")
        print("=" * 80)
        
        # Group by method and clustering
        summary_stats = []
        
        for method in methods:
            for clustering in ['KMeans', 'Agglomerative', 'Spectral']:
                method_data = df[(df['Method'] == method) & (df['Clustering'] == clustering)]
                
                if len(method_data) > 0:
                    stats = {}
                    stats['Dataset'] = dataset_name
                    stats['Method'] = method
                    stats['Clustering'] = clustering
                    
                    for metric in ['ARI', 'NMI', 'Modularity', 'Silhouette', 'Conductance', 'Coverage']:
                        values = method_data[metric].values
                        if len(values) > 0:
                            mean_val = np.mean(values)
                            std_val = np.std(values)
                            min_val = np.min(values)
                            max_val = np.max(values)
                            stats[metric] = f"{mean_val:.4f} ± {std_val:.4f} (min={min_val:.4f}, max={max_val:.4f})"
                        else:
                            stats[metric] = "N/A"
                    
                    summary_stats.append(stats)
        
        # Create summary DataFrame
        summary_df = pd.DataFrame(summary_stats)
        
        # Save summary
        summary_file = f"summary_stats_{dataset_name.replace(' ', '_')}_{n_runs}runs.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"📈 Summary statistics saved to: {summary_file}")
        
        # Display top performers
        print(f"\n🏆 TOP PERFORMERS")
        print("-" * 50)
        
        if ground_truth is not None:
            # Sort by ARI for supervised evaluation
            numeric_df = df.copy()
            for metric in ['ARI', 'NMI', 'Modularity', 'Silhouette']:
                numeric_df[f'{metric}_mean'] = numeric_df.groupby(['Method', 'Clustering'])[metric].transform('mean')
            
            top_ari = numeric_df.nlargest(3, 'ARI_mean')[['Method', 'Clustering', 'ARI', 'NMI', 'Modularity']].drop_duplicates()
            print("Top 3 by ARI:")
            for _, row in top_ari.iterrows():
                print(f"  {row['Method']} ({row['Clustering']}): ARI={row['ARI']:.4f}, NMI={row['NMI']:.4f}")
        else:
            # Sort by Modularity for unsupervised evaluation
            numeric_df = df.copy()
            numeric_df['Modularity_mean'] = numeric_df.groupby(['Method', 'Clustering'])['Modularity'].transform('mean')
            
            top_mod = numeric_df.nlargest(3, 'Modularity_mean')[['Method', 'Clustering', 'Modularity', 'Silhouette']].drop_duplicates()
            print("Top 3 by Modularity:")
            for _, row in top_mod.iterrows():
                print(f"  {row['Method']} ({row['Clustering']}): Modularity={row['Modularity']:.4f}")
        
        print(f"\n✨ Benchmark completed successfully!")
        print(f"   Total runs: {len(df)}")
        print(f"   Methods tested: {len(methods)}")
        print(f"   Results saved to CSV files")
        
        # Display final comprehensive results table
        print(f"\n📈 FINAL COMPREHENSIVE RESULTS")
        print("=" * 100)
        display_results_table(all_results, dataset_name, partial=False)
        
    else:
        print(f"\n❌ No results generated. Check for errors in methods.")


def display_current_run_results(all_results, dataset_name, current_run):
    """Display results for current run only"""
    if not all_results:
        return
    
    import pandas as pd
    df = pd.DataFrame(all_results)
    
    # Get results for current run only
    current_run_data = df[df['Run'] == current_run]
    
    if len(current_run_data) == 0:
        return
    
    print(f"📋 Run {current_run} Individual Results:")
    print(f"{'Method':<25} {'Clustering':<15} {'ARI':<8} {'NMI':<8} {'Modularity':<10} {'Time(s)':<8}")
    print("-" * 75)
    
    # Sort by ARI descending
    current_run_data = current_run_data.sort_values('ARI', ascending=False)
    
    for _, row in current_run_data.iterrows():
        print(f"{row['Method']:<25} {row['Clustering']:<15} {row['ARI']:<8.4f} {row['NMI']:<8.4f} {row['Modularity']:<10.4f} {row['Run_Time']:<8.2f}")
    
    # Show best performer of this run
    if len(current_run_data) > 0:
        best_row = current_run_data.iloc[0]
        print(f"\n🏆 Best in Run {current_run}: {best_row['Method']} ({best_row['Clustering']})")
        print(f"   ARI: {best_row['ARI']:.4f}, NMI: {best_row['NMI']:.4f}, Time: {best_row['Run_Time']:.2f}s")


def display_results_table(all_results, dataset_name, partial=False):
    """Display formatted results table"""
    if not all_results:
        return
    
    import pandas as pd
    df = pd.DataFrame(all_results)
    
    # Calculate summary statistics
    summary_stats = []
    methods = df['Method'].unique()
    clustering_methods = df['Clustering'].unique()
    
    for method in methods:
        for clustering in clustering_methods:
            method_data = df[(df['Method'] == method) & (df['Clustering'] == clustering)]
            
            if len(method_data) > 0:
                stats = {}
                stats['Method'] = method
                stats['Clustering'] = clustering
                stats['Runs'] = len(method_data)
                
                # Calculate mean ± std for key metrics
                for metric in ['ARI', 'NMI', 'Modularity', 'Silhouette']:
                    if metric in method_data.columns:
                        values = method_data[metric].values
                        if len(values) > 0:
                            mean_val = np.mean(values)
                            std_val = np.std(values) if len(values) > 1 else 0.0
                            stats[metric] = f"{mean_val:.4f}±{std_val:.3f}"
                        else:
                            stats[metric] = "N/A"
                    else:
                        stats[metric] = "N/A"
                
                # Add average runtime
                if 'Run_Time' in method_data.columns:
                    avg_time = np.mean(method_data['Run_Time'].values)
                    stats['Avg_Time'] = f"{avg_time:.2f}s"
                else:
                    stats['Avg_Time'] = "N/A"
                
                summary_stats.append(stats)
    
    if summary_stats:
        summary_df = pd.DataFrame(summary_stats)
        
        # Sort by ARI performance
        try:
            summary_df['ARI_numeric'] = summary_df['ARI'].str.split('±').str[0].astype(float)
            summary_df = summary_df.sort_values('ARI_numeric', ascending=False)
            summary_df = summary_df.drop('ARI_numeric', axis=1)
        except:
            pass
        
        print(f"\n{'Method':<25} {'Clustering':<15} {'Runs':<5} {'ARI':<12} {'NMI':<12} {'Modularity':<12} {'Avg_Time':<10}")
        print("-" * 110)
        
        for _, row in summary_df.head(15).iterrows():  # Show top 15
            print(f"{row['Method']:<25} {row['Clustering']:<15} {row['Runs']:<5} {row['ARI']:<12} {row['NMI']:<12} {row['Modularity']:<12} {row['Avg_Time']:<10}")
        
        # Show top performer
        if len(summary_df) > 0:
            best_row = summary_df.iloc[0]
            print(f"\n🏆 OVERALL BEST: {best_row['Method']} ({best_row['Clustering']})")
            print(f"   ARI: {best_row['ARI']}, NMI: {best_row['NMI']}, Avg Time: {best_row['Avg_Time']}")
            
            if not partial:
                print(f"\n📊 FINAL PERFORMANCE RANKING:")
                for i, (_, row) in enumerate(summary_df.head(5).iterrows(), 1):
                    rank_emoji = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i-1] if i <= 5 else f"{i}."
                    print(f"   {rank_emoji} {row['Method']} ({row['Clustering']}) - ARI: {row['ARI']} (Time: {row['Avg_Time']})")
                
                # Performance insights
                print(f"\n💡 PERFORMANCE INSIGHTS:")
                
                # Fastest method
                try:
                    summary_df['Time_numeric'] = summary_df['Avg_Time'].str.replace('s', '').astype(float)
                    fastest = summary_df.loc[summary_df['Time_numeric'].idxmin()]
                    print(f"   ⚡ Fastest: {fastest['Method']} ({fastest['Clustering']}) - {fastest['Avg_Time']}")
                except:
                    pass
                
                # Most stable (lowest std)
                try:
                    summary_df['ARI_std'] = summary_df['ARI'].str.split('±').str[1].astype(float)
                    most_stable = summary_df.loc[summary_df['ARI_std'].idxmin()]
                    print(f"   🎯 Most Stable: {most_stable['Method']} ({most_stable['Clustering']}) - {most_stable['ARI']}")
                except:
                    pass


if __name__ == "__main__":
    try:
        run_benchmark()
    except KeyboardInterrupt:
        print(f"\n\n⚠️  Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()