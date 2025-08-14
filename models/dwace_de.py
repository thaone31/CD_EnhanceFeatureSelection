import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, backend as K
import tensorflow.keras.models as keras_models
from models.differential_evolution import DifferentialEvolution
from models.feature_utils import deepwalk_embedding, node2vec_embedding
from collections import defaultdict

def dwace_de_pipeline(G, ground_truth=None, feature_dim=128, walk_params=None, 
                      de_config=None, ae_config=None, contrastive_config=None, verbose=True):
    """
    DWACE-DE Pipeline: DeepWalk → Differential Evolution → AutoEncoder → Contrastive Enhancement
    
    Args:
        G: NetworkX graph
        ground_truth: True labels (optional)
        feature_dim: Target feature dimension
        walk_params: DeepWalk parameters
        de_config: Differential Evolution configuration
        ae_config: AutoEncoder configuration  
        contrastive_config: Contrastive learning configuration
        verbose: Print progress
        
    Returns:
        embeddings_dict: Dictionary of embeddings at each stage
        enhancement_name: Final enhancement method used
        de_info: Information about DE optimization
    """
    
    # Default configurations
    if walk_params is None:
        walk_params = {'num_walks': 20, 'walk_length': 40}
    
    if de_config is None:
        de_config = {
            'population_size': 20,
            'max_generations': 50, 
            'F': 0.5,
            'CR': 0.9,
            'elite_ratio': 0.1
        }
    
    if ae_config is None:
        ae_config = {
            'epochs': 100,
            'batch_size': 64,
            'lambda_recon': 1.0,
            'lambda_mod': 0.1, 
            'lambda_neigh': 0.1,
            'lambda_sup': 1.0
        }
        
    if contrastive_config is None:
        contrastive_config = {
            'epochs': 150,
            'temperature': 0.05,
            'lambda_contrastive': 1.0,
            'lambda_sup': 0.3
        }
    
    embeddings_dict = {}
    
    if verbose:
        print("🔄 Starting DWACE-DE Pipeline...")
        print("=" * 50)
    
    # ================== STEP 1: Base Embedding Generation ==================
    if verbose:
        print("📊 Step 1: Generating base embeddings...")
    
    # DeepWalk embedding
    embedding_deepwalk = deepwalk_embedding(
        G, dim=feature_dim, 
        walk_length=walk_params.get('walk_length', 40),
        num_walks=walk_params.get('num_walks', 20)
    )
    embeddings_dict['deepwalk'] = embedding_deepwalk
    
    # Node2Vec embedding (for comparison)
    embedding_node2vec = node2vec_embedding(
        G, dim=feature_dim,
        walk_length=walk_params.get('walk_length', 40), 
        num_walks=walk_params.get('num_walks', 20)
    )
    embeddings_dict['node2vec'] = embedding_node2vec
    
    if verbose:
        print(f"   ✓ DeepWalk embedding: {embedding_deepwalk.shape}")
        print(f"   ✓ Node2Vec embedding: {embedding_node2vec.shape}")
    
    # ================== STEP 2: Differential Evolution Feature Selection ==================
    if verbose:
        print("🧬 Step 2: Differential Evolution feature selection...")
    
    # Initialize DE optimizer
    de_optimizer = DifferentialEvolution(**de_config)
    
    # Optimize feature selection on DeepWalk embedding
    best_features, best_fitness, convergence_history = de_optimizer.optimize(
        embeddings=embedding_deepwalk,
        graph=G,
        ground_truth=ground_truth,
        verbose=verbose
    )
    
    # Apply feature selection
    embedding_deepwalk_de = de_optimizer.select_features(embedding_deepwalk, best_features)
    embeddings_dict['deepwalk_de'] = embedding_deepwalk_de
    
    de_info = {
        'selected_features': best_features,
        'n_selected': np.sum(best_features),
        'selection_ratio': np.sum(best_features) / len(best_features),
        'best_fitness': best_fitness,
        'convergence_history': convergence_history
    }
    
    if verbose:
        print(f"   ✓ DE selected {de_info['n_selected']}/{len(best_features)} features "
              f"(ratio: {de_info['selection_ratio']:.3f})")
        print(f"   ✓ Best fitness: {best_fitness:.4f}")
    
    # ================== STEP 3: AutoEncoder Processing ==================
    if verbose:
        print("🔧 Step 3: AutoEncoder processing...")
    
    # Apply AutoEncoder on DE-selected features
    embedding_deepwalk_de_ae = autoencoder_with_graph_loss(
        embedding_deepwalk_de, G, ground_truth, 
        out_dim=feature_dim, **ae_config
    )
    embeddings_dict['deepwalk_de_ae'] = embedding_deepwalk_de_ae
    
    if verbose:
        print(f"   ✓ AutoEncoder output: {embedding_deepwalk_de_ae.shape}")
    
    # ================== STEP 4: Adaptive Enhancement ==================
    if verbose:
        print("🔀 Step 4: Adaptive enhancement selection...")
    
    # Check if ground truth is available for contrastive learning
    if ground_truth is not None and len(np.unique(ground_truth)) > 1:
        if verbose:
            print(f"   ✅ Ground truth available ({len(np.unique(ground_truth))} classes)")
            print("   📊 Applying contrastive learning...")
        
        # Contrastive enhancement
        embedding_final = contrastive_projection_enhanced(
            embedding_deepwalk_de_ae,
            comm_labels=ground_truth,
            out_dim=feature_dim,
            **contrastive_config
        )
        enhancement_name = "deepwalk_de_ae_contrast"
        
    else:
        if verbose:
            print("   ❌ No ground truth available")
            print("   ⏹️ Using AutoEncoder output directly")
        
        # Use AutoEncoder output directly
        embedding_final = embedding_deepwalk_de_ae
        enhancement_name = "deepwalk_de_ae"
    
    embeddings_dict[enhancement_name] = embedding_final
    
    if verbose:
        print("=" * 50)
        print("🎯 DWACE-DE Pipeline Complete!")
        print(f"   Final embedding: {enhancement_name} - {embedding_final.shape}")
        print(f"   Feature reduction: {feature_dim} → {de_info['n_selected']} → {embedding_final.shape[1]}")
    
    return embeddings_dict, enhancement_name, de_info


def autoencoder_with_graph_loss(X, G, ground_truth=None, out_dim=64, epochs=100, 
                                batch_size=64, lambda_recon=1.0, lambda_mod=0.1, 
                                lambda_neigh=0.1, lambda_sup=1.0, verbose=True):
    """
    AutoEncoder with graph-aware loss (modularity + neighborhood preservation)
    """
    # Prepare adjacency matrix if graph is not too large
    if G.number_of_nodes() < 10000:
        import networkx as nx
        A = nx.to_numpy_array(G)
    else:
        A = None
        if verbose:
            print(f"   [WARNING] Graph too large ({G.number_of_nodes()} nodes), skipping graph loss")
    
    # Prepare community labels
    if ground_truth is not None:
        comm_labels = np.array(ground_truth)
    else:
        comm_labels = np.zeros(X.shape[0], dtype=int)
    
    # Build AutoEncoder architecture
    input_layer = layers.Input(shape=(X.shape[1],))
    
    # Encoder
    hidden_dim = max(out_dim * 2, 32)
    encoded = layers.Dense(hidden_dim, activation='relu')(input_layer)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dropout(0.2)(encoded)
    
    encoded = layers.Dense(out_dim, activation='relu')(encoded)
    encoded = layers.BatchNormalization()(encoded)
    
    # Decoder
    decoded = layers.Dense(hidden_dim, activation='relu')(encoded)
    decoded = layers.BatchNormalization()(decoded)
    decoded = layers.Dropout(0.2)(decoded)
    
    output = layers.Dense(X.shape[1], activation='linear')(decoded)
    
    # Create models
    autoencoder = keras_models.Model(input_layer, output)
    encoder = keras_models.Model(input_layer, encoded)
    
    # Custom loss with graph-aware components
    def graph_aware_loss(y_true, y_pred):
        # Reconstruction loss
        mse_loss = losses.MeanSquaredError()(y_true, y_pred)
        
        total_loss = lambda_recon * mse_loss
        
        if A is not None:
            # Get encoder output
            Z = encoder(y_true)
            n = tf.shape(Z)[0]
            A_tf = tf.convert_to_tensor(A, dtype=tf.float32)
            
            # Modularity loss
            deg = tf.reduce_sum(A_tf, axis=1, keepdims=True)
            m = tf.reduce_sum(deg) / 2.0 + 1e-8
            
            Z_norm = tf.math.l2_normalize(Z, axis=1) 
            dot = tf.matmul(Z_norm, Z_norm, transpose_b=True)
            
            deg_prod = tf.matmul(deg, tf.transpose(deg)) / (2.0 * m)
            modularity_matrix = A_tf - deg_prod
            modularity_score = tf.reduce_sum(modularity_matrix * dot) / (2.0 * m)
            modularity_loss = -modularity_score
            
            # Neighborhood preservation loss
            comm_labels_tf = tf.convert_to_tensor(comm_labels, dtype=tf.int32)
            mask_same = tf.cast(tf.equal(tf.expand_dims(comm_labels_tf, 1), 
                                        tf.expand_dims(comm_labels_tf, 0)), tf.float32)
            mask_diff = 1.0 - mask_same
            
            dists = tf.norm(tf.expand_dims(Z, 1) - tf.expand_dims(Z, 0), axis=2)
            same_mean = tf.reduce_sum(dists * mask_same) / (tf.reduce_sum(mask_same) + 1e-8)
            diff_mean = tf.reduce_sum(dists * mask_diff) / (tf.reduce_sum(mask_diff) + 1e-8)
            neigh_loss = same_mean / (diff_mean + 1e-8)
            
            total_loss += lambda_mod * modularity_loss + lambda_neigh * neigh_loss
        
        return total_loss
    
    # Compile and train
    optimizer = optimizers.Adam(learning_rate=0.001)
    autoencoder.compile(optimizer=optimizer, loss=graph_aware_loss)
    
    autoencoder.fit(
        X, X,
        epochs=epochs, 
        batch_size=min(batch_size, X.shape[0]),
        verbose=1 if verbose else 0
    )
    
    # Get enhanced features
    enhanced = encoder.predict(X)
    K.clear_session()
    
    return enhanced


def contrastive_projection_enhanced(X, comm_labels, out_dim=64, epochs=150, 
                                  temperature=0.05, lambda_contrastive=1.0, 
                                  lambda_sup=0.3, verbose=True):
    """
    Enhanced contrastive projection with InfoNCE + supervised learning
    """
    batch_size = min(128, X.shape[0])
    
    # Architecture
    input_layer = layers.Input(shape=(X.shape[1],))
    
    # Improved architecture with gradual dimension reduction
    x = layers.Dense(256, activation='relu')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    # Projection head
    proj = layers.Dense(out_dim, activation=None)(x)
    proj = layers.LayerNormalization()(proj)
    
    # Classifier head
    n_classes = int(np.max(comm_labels)) + 1 if comm_labels is not None else 1
    if n_classes > 1:
        classifier_out = layers.Dense(n_classes, activation='softmax', name='classifier')(proj)
        model = keras_models.Model(input_layer, [proj, classifier_out])
        y_class = tf.keras.utils.to_categorical(comm_labels, num_classes=n_classes)
    else:
        model = keras_models.Model(input_layer, proj)
        y_class = None
    
    # Optimizer with learning rate scheduling
    initial_lr = 0.001
    optimizer = optimizers.Adam(learning_rate=initial_lr, beta_1=0.9, beta_2=0.999)
    
    if verbose:
        print(f"   Starting contrastive training: {epochs} epochs, temp={temperature}")
    
    # Training loop
    for epoch in range(epochs):
        # Learning rate decay
        current_lr = initial_lr * (0.95 ** (epoch // 10))
        optimizer.learning_rate = current_lr
        
        # Data augmentation
        idx = np.random.permutation(X.shape[0])
        X1 = graph_augment_advanced(X[idx], drop_prob=0.1, noise_std=0.05)
        X2 = graph_augment_advanced(X[idx], drop_prob=0.15, noise_std=0.08)
        
        with tf.GradientTape() as tape:
            if n_classes > 1:
                z1, class_pred1 = model(X1, training=True)
                z2, class_pred2 = model(X2, training=True)
            else:
                z1 = model(X1, training=True)
                z2 = model(X2, training=True)
            
            # InfoNCE loss
            loss_contrastive = info_nce_loss_enhanced(z1, z2, temperature=temperature)
            
            # Supervised loss
            if n_classes > 1 and y_class is not None:
                sup_loss1 = losses.CategoricalCrossentropy()(y_class[idx], class_pred1)
                sup_loss2 = losses.CategoricalCrossentropy()(y_class[idx], class_pred2)
                sup_loss = (sup_loss1 + sup_loss2) / 2.0
            else:
                sup_loss = 0.0
            
            total_loss = lambda_contrastive * loss_contrastive + lambda_sup * sup_loss
        
        # Gradient clipping
        grads = tape.gradient(total_loss, model.trainable_weights)
        clipped_grads = [tf.clip_by_norm(grad, 1.0) if grad is not None else grad for grad in grads]
        optimizer.apply_gradients(zip(clipped_grads, model.trainable_weights))
        
        if verbose and epoch % 20 == 0:
            print(f"   Epoch {epoch:3d}: loss={total_loss.numpy():.4f}, "
                  f"contrast={loss_contrastive.numpy():.4f}, "
                  f"sup={sup_loss.numpy() if n_classes > 1 else 0.0:.4f}")
    
    # Get final embeddings
    if n_classes > 1:
        out, _ = model.predict(X, verbose=0)
    else:
        out = model.predict(X, verbose=0)
    
    K.clear_session()
    return out


def graph_augment_advanced(X, drop_prob=0.2, noise_std=0.1, shuffle_prob=0.3):
    """Advanced graph augmentation with multiple strategies"""
    X_aug = X.copy()
    
    # 1. Feature dropout
    if drop_prob > 0:
        mask = np.random.binomial(1, 1-drop_prob, X.shape)
        X_aug = X_aug * mask
    
    # 2. Gaussian noise
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, X.shape)
        X_aug = X_aug + noise
    
    # 3. Feature shuffling
    if np.random.random() < shuffle_prob:
        for i in range(X_aug.shape[0]):
            n_shuffle = max(1, int(0.05 * X_aug.shape[1]))
            shuffle_idx = np.random.choice(X_aug.shape[1], n_shuffle, replace=False)
            X_aug[i, shuffle_idx] = np.random.permutation(X_aug[i, shuffle_idx])
    
    return X_aug.astype(np.float32)


def info_nce_loss_enhanced(z1, z2, temperature=0.1):
    """Enhanced InfoNCE loss with better numerical stability"""
    z1 = tf.math.l2_normalize(z1, axis=1)
    z2 = tf.math.l2_normalize(z2, axis=1)
    batch_size = tf.shape(z1)[0]
    
    # Compute similarity matrix
    representations = tf.concat([z1, z2], axis=0)
    similarity_matrix = tf.matmul(representations, representations, transpose_b=True)
    
    # Remove self-similarity
    mask = tf.eye(2 * batch_size, dtype=tf.bool)
    similarity_matrix = tf.where(mask, -tf.float32.max, similarity_matrix)
    
    # Temperature scaling
    similarity_matrix = similarity_matrix / temperature
    
    # Positive pair labels
    labels_a = tf.range(batch_size, dtype=tf.int32) + batch_size
    labels_b = tf.range(batch_size, dtype=tf.int32)
    
    # Cross-entropy loss
    loss_a = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels_a, logits=similarity_matrix[:batch_size]
    )
    loss_b = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels_b, logits=similarity_matrix[batch_size:]
    )
    
    return tf.reduce_mean(loss_a + loss_b)
