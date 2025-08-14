import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import networkx as nx
from collections import defaultdict

class DifferentialEvolution:
    """
    Differential Evolution for feature selection in graph embeddings
    Optimizes feature subset to maximize clustering quality (modularity + silhouette)
    """
    
    def __init__(self, population_size=20, max_generations=50, F=0.5, CR=0.9, 
                 elite_ratio=0.1, mutation_strategy='best/1/bin'):
        """
        Args:
            population_size: Size of population
            max_generations: Maximum number of generations
            F: Differential weight [0,2]
            CR: Crossover probability [0,1]
            elite_ratio: Ratio of elite individuals to preserve
            mutation_strategy: DE mutation strategy
        """
        self.population_size = population_size
        self.max_generations = max_generations
        self.F = F
        self.CR = CR
        self.elite_ratio = elite_ratio
        self.mutation_strategy = mutation_strategy
        
    def _initialize_population(self, n_features, min_features=0.3, max_features=0.8):
        """Initialize population with random binary vectors"""
        population = []
        for _ in range(self.population_size):
            # Random selection ratio between min_features and max_features
            selection_ratio = np.random.uniform(min_features, max_features)
            n_selected = max(1, int(selection_ratio * n_features))
            
            individual = np.zeros(n_features, dtype=bool)
            selected_indices = np.random.choice(n_features, n_selected, replace=False)
            individual[selected_indices] = True
            population.append(individual)
            
        return np.array(population)
    
    def _fitness_function(self, individual, embeddings, graph, ground_truth=None, n_clusters=None):
        """
        Fitness function combining multiple clustering quality metrics
        """
        selected_features = individual.astype(bool)
        if np.sum(selected_features) < 2:  # Need at least 2 features
            return -np.inf
            
        X_selected = embeddings[:, selected_features]
        
        try:
            # Auto-determine number of clusters if not provided
            if n_clusters is None:
                if ground_truth is not None:
                    n_clusters = len(np.unique(ground_truth))
                else:
                    # Use elbow method or default
                    n_clusters = min(8, max(2, int(np.sqrt(len(embeddings) / 2))))
            
            # Clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_selected)
            
            # Compute fitness components
            fitness_components = []
            
            # 1. Silhouette score (internal clustering quality)
            if len(np.unique(labels)) > 1:
                sil_score = silhouette_score(X_selected, labels)
                fitness_components.append(sil_score)
            else:
                fitness_components.append(-1.0)
            
            # 2. Modularity (graph structure preservation)
            if graph is not None:
                modularity = self._compute_modularity(graph, labels)
                fitness_components.append(modularity)
            else:
                fitness_components.append(0.0)
                
            # 3. Feature selection ratio penalty (encourage parsimony)
            selection_ratio = np.sum(selected_features) / len(selected_features)
            parsimony_bonus = 1.0 - abs(selection_ratio - 0.5)  # Bonus for ~50% selection
            fitness_components.append(parsimony_bonus * 0.1)  # Small weight
            
            # 4. Supervised metric (if ground truth available)
            if ground_truth is not None:
                from sklearn.metrics import adjusted_rand_score
                ari = adjusted_rand_score(ground_truth, labels)
                fitness_components.append(ari)
            
            # Weighted combination
            weights = [0.4, 0.4, 0.05, 0.15] if ground_truth is not None else [0.5, 0.4, 0.1]
            fitness = np.sum([w * c for w, c in zip(weights[:len(fitness_components)], fitness_components)])
            
            return fitness
            
        except Exception as e:
            print(f"[WARNING] Fitness computation error: {e}")
            return -np.inf
    
    def _compute_modularity(self, graph, labels):
        """Compute modularity score"""
        try:
            communities = defaultdict(list)
            for idx, node in enumerate(graph.nodes()):
                communities[labels[idx]].append(node)
            
            modularity = nx.algorithms.community.quality.modularity(graph, communities.values())
            return modularity
        except:
            return 0.0
    
    def _mutation(self, population, fitness_scores):
        """DE mutation with multiple strategies"""
        mutated_population = []
        
        # Sort population by fitness (descending)
        sorted_indices = np.argsort(fitness_scores)[::-1]
        best_individual = population[sorted_indices[0]]
        
        for i in range(len(population)):
            if self.mutation_strategy == 'best/1/bin':
                # Select random individuals (different from current)
                candidates = [j for j in range(len(population)) if j != i]
                r1, r2 = np.random.choice(candidates, 2, replace=False)
                
                # DE mutation: best + F * (r1 - r2)
                # For binary vectors, we use probabilistic approach
                diff_prob = np.abs(population[r1].astype(float) - population[r2].astype(float))
                mutation_prob = best_individual.astype(float) + self.F * diff_prob
                mutation_prob = np.clip(mutation_prob, 0, 1)
                
                # Generate mutant
                mutant = np.random.random(len(population[i])) < mutation_prob
                
            elif self.mutation_strategy == 'rand/1/bin':
                candidates = [j for j in range(len(population)) if j != i]
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                
                diff_prob = np.abs(population[r2].astype(float) - population[r3].astype(float))
                mutation_prob = population[r1].astype(float) + self.F * diff_prob
                mutation_prob = np.clip(mutation_prob, 0, 1)
                
                mutant = np.random.random(len(population[i])) < mutation_prob
            
            # Ensure at least some features are selected
            if np.sum(mutant) < 2:
                n_select = max(2, int(0.3 * len(mutant)))
                selected_indices = np.random.choice(len(mutant), n_select, replace=False)
                mutant = np.zeros_like(mutant, dtype=bool)
                mutant[selected_indices] = True
                
            mutated_population.append(mutant)
            
        return np.array(mutated_population)
    
    def _crossover(self, population, mutated_population):
        """Binomial crossover"""
        offspring = []
        
        for i in range(len(population)):
            parent = population[i]
            mutant = mutated_population[i]
            
            # Binomial crossover
            child = parent.copy()
            crossover_mask = np.random.random(len(parent)) < self.CR
            
            # Ensure at least one gene is inherited from mutant
            if not np.any(crossover_mask):
                crossover_mask[np.random.randint(len(parent))] = True
                
            child[crossover_mask] = mutant[crossover_mask]
            
            # Ensure minimum feature selection
            if np.sum(child) < 2:
                n_select = max(2, int(0.3 * len(child)))
                selected_indices = np.random.choice(len(child), n_select, replace=False)
                child = np.zeros_like(child, dtype=bool)
                child[selected_indices] = True
            
            offspring.append(child)
            
        return np.array(offspring)
    
    def _selection(self, population, offspring, fitness_scores, offspring_fitness):
        """Selection: keep better individuals"""
        new_population = []
        new_fitness = []
        
        for i in range(len(population)):
            if offspring_fitness[i] > fitness_scores[i]:
                new_population.append(offspring[i])
                new_fitness.append(offspring_fitness[i])
            else:
                new_population.append(population[i])
                new_fitness.append(fitness_scores[i])
                
        return np.array(new_population), np.array(new_fitness)
    
    def optimize(self, embeddings, graph=None, ground_truth=None, n_clusters=None, verbose=True):
        """
        Main DE optimization loop
        
        Args:
            embeddings: Node embeddings (N x D)
            graph: NetworkX graph object
            ground_truth: True labels (optional)
            n_clusters: Number of clusters (auto-determined if None)
            verbose: Print progress
            
        Returns:
            best_features: Binary mask of selected features
            best_fitness: Best fitness score achieved
            convergence_history: Fitness evolution over generations
        """
        n_features = embeddings.shape[1]
        
        if verbose:
            print(f"[DE] Starting optimization: {n_features} features, pop_size={self.population_size}, generations={self.max_generations}")
        
        # Initialize population
        population = self._initialize_population(n_features)
        
        # Evaluate initial population
        fitness_scores = np.array([
            self._fitness_function(ind, embeddings, graph, ground_truth, n_clusters) 
            for ind in population
        ])
        
        convergence_history = []
        best_fitness_ever = -np.inf
        best_individual_ever = None
        stagnation_count = 0
        
        for generation in range(self.max_generations):
            # Mutation
            mutated_population = self._mutation(population, fitness_scores)
            
            # Crossover
            offspring = self._crossover(population, mutated_population)
            
            # Evaluate offspring
            offspring_fitness = np.array([
                self._fitness_function(ind, embeddings, graph, ground_truth, n_clusters)
                for ind in offspring
            ])
            
            # Selection
            population, fitness_scores = self._selection(
                population, offspring, fitness_scores, offspring_fitness
            )
            
            # Track best solution
            current_best_fitness = np.max(fitness_scores)
            if current_best_fitness > best_fitness_ever:
                best_fitness_ever = current_best_fitness
                best_individual_ever = population[np.argmax(fitness_scores)].copy()
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            convergence_history.append({
                'generation': generation,
                'best_fitness': current_best_fitness,
                'avg_fitness': np.mean(fitness_scores),
                'n_features_best': np.sum(population[np.argmax(fitness_scores)])
            })
            
            if verbose and generation % 10 == 0:
                best_idx = np.argmax(fitness_scores)
                n_selected = np.sum(population[best_idx])
                print(f"[DE] Gen {generation:3d}: Best={current_best_fitness:.4f}, "
                      f"Avg={np.mean(fitness_scores):.4f}, Features={n_selected}/{n_features}")
            
            # Early stopping if stagnation
            if stagnation_count > 15:
                if verbose:
                    print(f"[DE] Early stopping at generation {generation} due to stagnation")
                break
                
        if verbose:
            final_n_features = np.sum(best_individual_ever)
            print(f"[DE] Optimization complete: {final_n_features}/{n_features} features selected, "
                  f"fitness={best_fitness_ever:.4f}")
        
        return best_individual_ever, best_fitness_ever, convergence_history
    
    def select_features(self, embeddings, feature_mask):
        """Apply feature selection mask to embeddings"""
        return embeddings[:, feature_mask.astype(bool)]
