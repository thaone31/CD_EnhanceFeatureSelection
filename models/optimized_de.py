import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.cluster import KMeans
import networkx as nx


class OptimizedDifferentialEvolution:
    """
    Optimized Differential Evolution for feature selection in graph embeddings
    
    This is a simplified, robust version of DE that:
    - Uses conservative mutation/crossover rates
    - Has adaptive parameters
    - Focuses on stability over aggressive optimization
    - Includes early stopping to prevent overfitting
    """
    
    def __init__(self, population_size=8, max_generations=20, F=0.5, CR=0.7,
                 min_features_ratio=0.6, max_features_ratio=0.8, early_stop_patience=5):
        self.population_size = population_size
        self.max_generations = max_generations
        self.F = F  # Mutation factor
        self.CR = CR  # Crossover rate
        self.min_features_ratio = min_features_ratio
        self.max_features_ratio = max_features_ratio
        self.early_stop_patience = early_stop_patience
        
    def _fitness_function(self, features_subset, embeddings, G, ground_truth):
        """
        Simplified fitness function focusing on clustering quality
        """
        try:
            if np.sum(features_subset) < 2:  # Need at least 2 features
                return 0.0
                
            # Apply feature selection
            selected_embeddings = embeddings[:, features_subset.astype(bool)]
            
            if ground_truth is not None:
                n_clusters = len(np.unique(ground_truth))
                
                # KMeans clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                predicted_labels = kmeans.fit_predict(selected_embeddings)
                
                # Primary metric: ARI
                ari_score = adjusted_rand_score(ground_truth, predicted_labels)
                
                # Secondary metric: Silhouette (for cluster quality)
                if len(np.unique(predicted_labels)) > 1 and selected_embeddings.shape[0] > n_clusters:
                    silhouette = silhouette_score(selected_embeddings, predicted_labels)
                else:
                    silhouette = 0.0
                
                # Combined fitness with efficiency bonus
                n_selected = np.sum(features_subset)
                efficiency_bonus = 1.0 - (n_selected / embeddings.shape[1]) * 0.1  # Small bonus for fewer features
                
                fitness = ari_score * 0.7 + silhouette * 0.2 + efficiency_bonus * 0.1
                return max(0.0, fitness)
            else:
                # Unsupervised: use modularity + silhouette
                kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)  # Default clusters
                predicted_labels = kmeans.fit_predict(selected_embeddings)
                
                # Calculate modularity
                communities = {i: [] for i in range(8)}
                for node, label in enumerate(predicted_labels):
                    communities[label].append(node)
                
                communities_list = [comm for comm in communities.values() if len(comm) > 0]
                modularity = nx.community.modularity(G, communities_list) if len(communities_list) > 1 else 0.0
                
                # Silhouette score
                if len(np.unique(predicted_labels)) > 1:
                    silhouette = silhouette_score(selected_embeddings, predicted_labels)
                else:
                    silhouette = 0.0
                    
                fitness = modularity * 0.6 + silhouette * 0.4
                return max(0.0, fitness)
                
        except Exception as e:
            return 0.0
    
    def _initialize_population(self, n_features):
        """Initialize population with diverse feature selections"""
        population = []
        
        for _ in range(self.population_size):
            # Random selection ratio between min and max
            selection_ratio = np.random.uniform(self.min_features_ratio, self.max_features_ratio)
            n_select = int(selection_ratio * n_features)
            
            # Create binary mask
            mask = np.zeros(n_features)
            selected_indices = np.random.choice(n_features, n_select, replace=False)
            mask[selected_indices] = 1
            
            population.append(mask)
            
        return np.array(population)
    
    def optimize(self, embeddings, G, ground_truth):
        """
        Run optimized DE algorithm
        
        Returns:
            best_features: Binary mask of selected features
            best_fitness: Fitness score of best solution
            convergence_history: List of best fitness per generation
        """
        n_features = embeddings.shape[1]
        
        # Initialize population
        population = self._initialize_population(n_features)
        
        # Evaluate initial population
        fitness_scores = []
        for individual in population:
            fitness = self._fitness_function(individual, embeddings, G, ground_truth)
            fitness_scores.append(fitness)
        
        fitness_scores = np.array(fitness_scores)
        convergence_history = []
        best_fitness = np.max(fitness_scores)
        convergence_history.append(best_fitness)
        
        # Early stopping tracking
        no_improvement_count = 0
        
        # Evolution loop
        for generation in range(self.max_generations):
            new_population = []
            new_fitness_scores = []
            
            for i in range(self.population_size):
                # Select three random individuals (different from current)
                candidates = [j for j in range(self.population_size) if j != i]
                a, b, c = np.random.choice(candidates, 3, replace=False)
                
                # Mutation: V = Xa + F * (Xb - Xc)
                mutant = population[a] + self.F * (population[b] - population[c])
                mutant = np.clip(mutant, 0, 1)  # Keep in valid range
                
                # Crossover
                trial = population[i].copy()
                for j in range(n_features):
                    if np.random.random() < self.CR:
                        trial[j] = mutant[j]
                
                # Convert to binary (threshold at 0.5)
                trial_binary = (trial > 0.5).astype(float)
                
                # Ensure we have minimum number of features
                if np.sum(trial_binary) < self.min_features_ratio * n_features:
                    # Add random features to meet minimum
                    n_add = int(self.min_features_ratio * n_features) - int(np.sum(trial_binary))
                    available_features = np.where(trial_binary == 0)[0]
                    if len(available_features) >= n_add:
                        add_features = np.random.choice(available_features, n_add, replace=False)
                        trial_binary[add_features] = 1
                
                # Selection: keep better individual
                trial_fitness = self._fitness_function(trial_binary, embeddings, G, ground_truth)
                
                if trial_fitness > fitness_scores[i]:
                    new_population.append(trial_binary)
                    new_fitness_scores.append(trial_fitness)
                else:
                    new_population.append(population[i])
                    new_fitness_scores.append(fitness_scores[i])
            
            population = np.array(new_population)
            fitness_scores = np.array(new_fitness_scores)
            
            # Track convergence
            current_best = np.max(fitness_scores)
            convergence_history.append(current_best)
            
            # Early stopping check
            if current_best > best_fitness + 1e-6:  # Improvement threshold
                best_fitness = current_best
                no_improvement_count = 0
            else:
                no_improvement_count += 1
                
            if no_improvement_count >= self.early_stop_patience:
                print(f"   Early stopping at generation {generation+1} (no improvement for {self.early_stop_patience} generations)")
                break
        
        # Return best solution
        best_idx = np.argmax(fitness_scores)
        best_features = population[best_idx].astype(bool)
        
        return best_features, best_fitness, convergence_history


def simple_autoencoder(X, output_dim=None, epochs=50):
    """
    Simple autoencoder for dimensionality reduction
    
    Args:
        X: Input features
        output_dim: Output dimension (default: half of input)
        epochs: Training epochs
        
    Returns:
        Encoded features
    """
    if output_dim is None:
        output_dim = max(8, X.shape[1] // 2)
    
    # Build autoencoder
    input_dim = X.shape[1]
    
    # Encoder
    input_layer = tf.keras.Input(shape=(input_dim,))
    encoded = layers.Dense(output_dim, activation='relu')(input_layer)
    
    # Decoder
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
    
    # Models
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)
    
    # Compile and train
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X, X, epochs=epochs, batch_size=32, verbose=0)
    
    # Return encoded features
    encoded_features = encoder.predict(X, verbose=0)
    
    # Clear session to prevent memory leaks
    tf.keras.backend.clear_session()
    
    return encoded_features


def simple_contrastive_learning(X, labels, epochs=50, temperature=0.1):
    """
    Simple contrastive learning enhancement
    
    Args:
        X: Input features
        labels: True labels for contrastive learning
        epochs: Training epochs
        temperature: Temperature parameter
        
    Returns:
        Enhanced features
    """
    n_features = X.shape[1]
    n_classes = len(np.unique(labels))
    
    # Build projection network
    input_layer = tf.keras.Input(shape=(n_features,))
    projected = layers.Dense(n_features, activation='relu')(input_layer)
    projected = layers.Dense(n_features, activation=None)(projected)  # No activation for projection
    
    model = Model(input_layer, projected)
    
    # Custom contrastive loss
    def contrastive_loss(y_true, y_pred):
        # Normalize features
        y_pred = tf.nn.l2_normalize(y_pred, axis=1)
        
        # Compute similarity matrix
        similarity_matrix = tf.matmul(y_pred, y_pred, transpose_b=True) / temperature
        
        # Create mask for positive pairs
        labels_equal = tf.equal(tf.expand_dims(y_true, 0), tf.expand_dims(y_true, 1))
        labels_equal = tf.cast(labels_equal, tf.float32)
        
        # Mask out diagonal
        batch_size = tf.shape(y_true)[0]
        mask = tf.ones_like(similarity_matrix) - tf.eye(batch_size)
        labels_equal = labels_equal * mask
        
        # Compute loss
        exp_sim = tf.exp(similarity_matrix) * mask
        log_prob = similarity_matrix - tf.math.log(tf.reduce_sum(exp_sim, axis=1, keepdims=True))
        
        # Mean over positive pairs
        positive_pairs = tf.reduce_sum(labels_equal)
        loss = -tf.reduce_sum(labels_equal * log_prob) / (positive_pairs + 1e-8)
        
        return loss
    
    # Compile and train
    model.compile(optimizer='adam', loss=contrastive_loss)
    
    # Convert labels to categorical for training
    labels_array = np.array(labels)
    model.fit(X, labels_array, epochs=epochs, batch_size=32, verbose=0)
    
    # Get enhanced features
    enhanced_features = model.predict(X, verbose=0)
    
    # Clear session
    tf.keras.backend.clear_session()
    
    return enhanced_features
