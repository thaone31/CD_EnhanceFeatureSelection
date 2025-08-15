import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, backend as K
import tensorflow.keras.models as keras_models
import networkx as nx
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings('ignore')

# Configure TensorFlow
tf.config.run_functions_eagerly(True)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


def dwace_enhanced_pipeline(G, ground_truth=None, feature_dim=128, initial_embedding=None, verbose=True):
    """
    DWACE Enhanced Pipeline with Neighborhood-preserving and Supervised Cross-entropy losses:
    
    1. DeepWalk embedding generation (or use provided initial_embedding)
    2. Enhanced AutoEncoder with multiple losses:
       - L_recon: MSE reconstruction loss
       - L_modularity: Modularity preservation loss
       - L_neighborhood: Neighborhood-preserving loss (preserves local graph structure)
       - L_supervised: Supervised cross-entropy loss (if labels available)
    3. Advanced contrastive learning with InfoNCE
    
    Combined Loss: L = L_recon + λ1*L_modularity + λ2*L_neighborhood + λ3*L_supervised
    """
    from models.feature_utils import deepwalk_embedding
    
    if verbose:
        print("🚀 Starting DWACE Enhanced Implementation...")
        print("=" * 70)
    
    # ========== STEP 1: Embedding Input ==========
    if initial_embedding is not None:
        if verbose:
            print("📊 Step 1: Using provided initial embedding...")
        embedding_dw = initial_embedding
        actual_feature_dim = embedding_dw.shape[1]
        if verbose:
            print(f"   ✓ Initial embedding: {embedding_dw.shape}")
    else:
        if verbose:
            print("📊 Step 1: DeepWalk embedding generation...")
        
        embedding_dw = deepwalk_embedding(G, dim=feature_dim, walk_length=40, num_walks=20)
        actual_feature_dim = feature_dim
        
        if verbose:
            print(f"   ✓ DeepWalk embedding: {embedding_dw.shape}")
    
    # ========== STEP 2: Enhanced AutoEncoder ==========
    if verbose:
        print("🔧 Step 2: Enhanced AutoEncoder with multiple losses...")
    
    embedding_ae = enhanced_autoencoder_multi_loss(
        embedding_dw, G, ground_truth,
        out_dim=actual_feature_dim//2,
        epochs=150,
        verbose=verbose
    )
    
    if verbose:
        print(f"   ✓ Enhanced AutoEncoder output: {embedding_ae.shape}")
    
    # ========== STEP 3: Advanced Contrastive Learning ==========
    if ground_truth is not None:
        if verbose:
            print("🎯 Step 3: Advanced contrastive learning...")
        
        embedding_final = advanced_contrastive_learning(
            embedding_ae, ground_truth,
            out_dim=feature_dim//2,
            epochs=100,
            verbose=verbose
        )
        
        enhancement_name = "dwace_enhanced_full"
    else:
        if verbose:
            print("⚠️ Step 3: No ground truth - using Enhanced AutoEncoder output")
        
        embedding_final = embedding_ae
        enhancement_name = "dwace_enhanced_ae_only"
    
    if verbose:
        print("=" * 70)
        print("🎯 DWACE Enhanced Pipeline Complete!")
        print(f"   Final embedding: {embedding_final.shape}")
    
    return {
        'deepwalk': embedding_dw,
        'dwace_enhanced_ae': embedding_ae,
        enhancement_name: embedding_final
    }, enhancement_name


def compute_neighborhood_graph(X, k=5):
    """
    Compute k-nearest neighbor graph from embedding space.
    Used for neighborhood-preserving loss.
    """
    try:
        nbrs = NearestNeighbors(n_neighbors=k+1, metric='cosine').fit(X)
        distances, indices = nbrs.kneighbors(X)
        
        # Create adjacency matrix (exclude self-connections)
        n_samples = X.shape[0]
        adj_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(1, k+1):  # Skip self (index 0)
                neighbor_idx = indices[i, j]
                distance = distances[i, j]
                # Convert distance to similarity (closer = higher similarity)
                similarity = np.exp(-distance)
                adj_matrix[i, neighbor_idx] = similarity
                adj_matrix[neighbor_idx, i] = similarity  # Make symmetric
        
        return adj_matrix
    except:
        # Fallback: identity matrix
        return np.eye(X.shape[0])


def neighborhood_preserving_loss(embeddings_original, embeddings_encoded, k=5):
    """
    Neighborhood-preserving loss to maintain local structure.
    
    Loss = ||D_original - D_encoded||_F^2
    where D_* are the distance matrices in original and encoded spaces.
    """
    try:
        # Compute pairwise distances in both spaces
        embeddings_original = tf.nn.l2_normalize(embeddings_original, axis=1, epsilon=1e-8)
        embeddings_encoded = tf.nn.l2_normalize(embeddings_encoded, axis=1, epsilon=1e-8)
        
        # Cosine distance matrices
        dist_original = 1.0 - tf.matmul(embeddings_original, embeddings_original, transpose_b=True)
        dist_encoded = 1.0 - tf.matmul(embeddings_encoded, embeddings_encoded, transpose_b=True)
        
        # Focus on k-nearest neighbors to reduce computational cost
        batch_size = tf.shape(embeddings_original)[0]
        k_neighbors = tf.minimum(k, batch_size - 1)
        
        # Get top-k smallest distances (nearest neighbors)
        _, indices_original = tf.nn.top_k(-dist_original, k=k_neighbors)
        _, indices_encoded = tf.nn.top_k(-dist_encoded, k=k_neighbors)
        
        # Create masks for k-nearest neighbors
        batch_indices = tf.expand_dims(tf.range(batch_size), 1)
        batch_indices = tf.tile(batch_indices, [1, k_neighbors])
        
        # Gather distances for k-nearest neighbors
        indices_original_flat = tf.reshape(tf.stack([batch_indices, indices_original], axis=-1), [-1, 2])
        indices_encoded_flat = tf.reshape(tf.stack([batch_indices, indices_encoded], axis=-1), [-1, 2])
        
        dist_original_knn = tf.gather_nd(dist_original, indices_original_flat)
        dist_encoded_knn = tf.gather_nd(dist_encoded, indices_encoded_flat)
        
        # Mean squared error between distance matrices
        neighborhood_loss = tf.reduce_mean(tf.square(dist_original_knn - dist_encoded_knn))
        
        return tf.clip_by_value(neighborhood_loss, 0.0, 10.0)
        
    except Exception as e:
        # Fallback: simple MSE between normalized embeddings
        return tf.reduce_mean(tf.square(tf.nn.l2_normalize(embeddings_original, axis=1) - 
                                        tf.nn.l2_normalize(embeddings_encoded, axis=1)))


def supervised_cross_entropy_loss(embeddings, labels, n_classes):
    """
    Supervised cross-entropy loss for labeled nodes.
    
    Creates a simple classifier head and computes cross-entropy loss
    between predicted and true labels.
    """
    try:
        # Simple linear classifier
        W = tf.Variable(tf.random.normal([tf.shape(embeddings)[1], n_classes], stddev=0.1))
        b = tf.Variable(tf.zeros([n_classes]))
        
        logits = tf.matmul(embeddings, W) + b
        probabilities = tf.nn.softmax(logits)
        
        # One-hot encode labels
        labels_onehot = tf.one_hot(labels, depth=n_classes)
        
        # Cross-entropy loss
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=labels_onehot, logits=logits))
        
        return tf.clip_by_value(cross_entropy, 0.0, 10.0)
        
    except Exception as e:
        return tf.constant(0.0)


def enhanced_autoencoder_multi_loss(X, G, ground_truth=None, out_dim=64, epochs=150, verbose=True):
    """
    Enhanced AutoEncoder with multiple losses:
    - L_recon: MSE reconstruction loss
    - L_modularity: Modularity preservation loss  
    - L_neighborhood: Neighborhood-preserving loss
    - L_supervised: Supervised cross-entropy loss (if labels available)
    """
    
    # Input normalization
    X = np.array(X, dtype=np.float32)
    X_mean = np.mean(X, axis=0, keepdims=True)
    X_std = np.std(X, axis=0, keepdims=True) + 1e-8
    X_normalized = (X - X_mean) / X_std
    X_normalized = np.clip(X_normalized, -3, 3)
    
    # Prepare graph adjacency matrix
    try:
        A = nx.to_numpy_array(G, dtype=np.float32)
    except:
        A = None
        if verbose:
            print("   [WARNING] Cannot create adjacency matrix")
    
    # Prepare ground truth labels
    labels = None
    n_classes = 0
    if ground_truth is not None:
        labels = np.array(ground_truth)
        n_classes = len(np.unique(labels))
        if verbose:
            print(f"   ✓ Using supervised loss with {n_classes} classes")
    
    input_dim = X_normalized.shape[1]
    hidden_dim = (input_dim + out_dim) // 2
    
    if verbose:
        print(f"   Architecture: {input_dim} → {hidden_dim} → {out_dim} → {hidden_dim} → {input_dim}")
    
    # ========== ENHANCED ARCHITECTURE ==========
    input_layer = layers.Input(shape=(input_dim,))
    
    # Encoder with advanced architecture
    x = layers.Dense(hidden_dim, kernel_initializer='he_normal')(input_layer)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    # Bottleneck layer
    encoded = layers.Dense(out_dim, kernel_initializer='he_normal')(x)
    encoded = layers.ELU()(encoded)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dropout(0.1)(encoded)
    
    # Decoder (mirror encoder)
    y = layers.Dense(hidden_dim, kernel_initializer='he_normal')(encoded)
    y = layers.ELU()(y)
    y = layers.BatchNormalization()(y)
    y = layers.Dropout(0.1)(y)
    
    decoded = layers.Dense(input_dim, kernel_initializer='he_normal', name='autoencoder_output')(y)
    
    # Classifier head (if supervised)
    classifier_out = None
    if labels is not None:
        classifier_out = layers.Dense(n_classes, activation='softmax', name='classifier')(encoded)
    
    # ========== MODELS ==========
    if classifier_out is not None:
        autoencoder = keras_models.Model(input_layer, {'autoencoder_output': decoded, 'classifier': classifier_out})
    else:
        autoencoder = keras_models.Model(input_layer, decoded)
    
    encoder = keras_models.Model(input_layer, encoded)
    
    # ========== ENHANCED COMBINED LOSS ==========
    def enhanced_combined_loss(y_true, y_pred):
        """
        Enhanced combined loss with all four components:
        L = L_recon + λ1*L_modularity + λ2*L_neighborhood + λ3*L_supervised
        """
        if isinstance(y_pred, list):
            y_pred_recon, y_pred_class = y_pred
        else:
            y_pred_recon = y_pred
            y_pred_class = None
        
        # L_recon: MSE reconstruction loss
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred_recon))
        total_loss = mse_loss
        
        # Get encoded representations
        Z = encoder(y_true)
        
        # L_neighborhood: Neighborhood-preserving loss (λ2 = 0.5)
        try:
            neighborhood_loss = neighborhood_preserving_loss(y_true, Z, k=5)
            total_loss += 0.5 * neighborhood_loss
        except Exception as e:
            pass
        
        # L_modularity: Modularity preservation loss (λ1 = 0.3)
        if A is not None:
            try:
                batch_size = tf.shape(y_true)[0]
                
                if tf.equal(batch_size, tf.constant(A.shape[0])):
                    Z_norm = tf.nn.l2_normalize(Z, axis=1, epsilon=1e-8)
                    sim_matrix = tf.matmul(Z_norm, Z_norm, transpose_b=True)
                    sim_matrix = tf.clip_by_value(sim_matrix, -1.0, 1.0)
                    
                    A_tf = tf.convert_to_tensor(A, dtype=tf.float32)
                    deg = tf.reduce_sum(A_tf, axis=1, keepdims=True)
                    m = tf.maximum(tf.reduce_sum(deg) / 2.0, 1e-8)
                    
                    expected_edges = tf.matmul(deg, tf.transpose(deg)) / (2.0 * m)
                    modularity_matrix = A_tf - expected_edges
                    
                    modularity_score = tf.reduce_sum(modularity_matrix * sim_matrix) / (2.0 * m)
                    modularity_loss = -modularity_score
                    
                    modularity_loss = tf.clip_by_value(modularity_loss, -10.0, 10.0)
                    total_loss += 0.3 * modularity_loss
                    
            except Exception as e:
                pass
        
        # L_supervised: Cross-entropy loss for labeled nodes (λ3 = 1.0)
        if y_pred_class is not None and labels is not None:
            try:
                labels_batch = tf.convert_to_tensor(labels[:tf.shape(y_true)[0]], dtype=tf.int32)
                supervised_loss = supervised_cross_entropy_loss(Z, labels_batch, n_classes)
                total_loss += 1.0 * supervised_loss
            except Exception as e:
                pass
        
        return tf.clip_by_value(total_loss, 0.0, 100.0)
    
    # ========== TRAINING ==========
    optimizer = optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    
    if classifier_out is not None:
        autoencoder.compile(
            optimizer=optimizer,
            loss={'autoencoder_output': enhanced_combined_loss, 'classifier': 'sparse_categorical_crossentropy'},
            loss_weights={'autoencoder_output': 1.0, 'classifier': 0.5}
        )
        
        y_class = labels[:X_normalized.shape[0]]  # Match batch size
        training_data = (X_normalized, {'autoencoder_output': X_normalized, 'classifier': y_class})
    else:
        autoencoder.compile(optimizer=optimizer, loss=enhanced_combined_loss)
        training_data = (X_normalized, X_normalized)
    
    # Training callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=30, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.7, patience=15, min_lr=1e-6)
    ]
    
    # Training loop with progress tracking
    if verbose:
        print("   🔄 Training Enhanced AutoEncoder...")
        history = autoencoder.fit(
            training_data[0], training_data[1],
            epochs=epochs,
            batch_size=min(32, X_normalized.shape[0]),
            callbacks=callbacks,
            verbose=0
        )
        
        final_loss = history.history['loss'][-1]
        print(f"   ✓ Training complete - Final loss: {final_loss:.6f}")
    else:
        autoencoder.fit(
            training_data[0], training_data[1],
            epochs=epochs,
            batch_size=min(32, X_normalized.shape[0]),
            callbacks=callbacks,
            verbose=0
        )
    
    # Generate enhanced embeddings
    enhanced_embeddings = encoder.predict(X_normalized, verbose=0)
    
    # Post-processing
    enhanced_embeddings = np.clip(enhanced_embeddings, -5, 5)
    
    if verbose:
        print(f"   ✓ Enhanced embeddings shape: {enhanced_embeddings.shape}")
        nan_count = np.isnan(enhanced_embeddings).sum()
        if nan_count > 0:
            print(f"   ⚠️ Warning: {nan_count} NaN values detected and will be fixed")
            enhanced_embeddings = np.nan_to_num(enhanced_embeddings, nan=0.0)
    
    return enhanced_embeddings


def advanced_contrastive_learning(X, labels, out_dim=32, epochs=100, verbose=True):
    """
    Advanced contrastive learning with InfoNCE and supervised signals.
    """
    X = np.array(X, dtype=np.float32)
    X_normalized = (X - np.mean(X, axis=0, keepdims=True)) / (np.std(X, axis=0, keepdims=True) + 1e-8)
    X_normalized = np.clip(X_normalized, -3, 3)
    
    labels_array = np.array(labels)
    n_classes = len(np.unique(labels_array))
    
    input_dim = X_normalized.shape[1]
    
    if verbose:
        print(f"   Contrastive learning: {input_dim} → {out_dim}, Classes: {n_classes}")
    
    # ========== ADVANCED CONTRASTIVE MODEL ==========
    input_layer = layers.Input(shape=(input_dim,))
    
    # Projection head with residual connections
    x = layers.Dense(input_dim, kernel_initializer='he_normal')(input_layer)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.BatchNormalization()(x)
    
    # Residual connection
    residual = layers.Dense(input_dim, kernel_initializer='he_normal')(input_layer)
    x = layers.Add()([x, residual])
    
    x = layers.Dense(out_dim * 2, kernel_initializer='he_normal')(x)
    x = layers.LeakyReLU(alpha=0.01)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    # Final projection
    projection = layers.Dense(out_dim, kernel_initializer='he_normal')(x)
    projection = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(projection)
    
    # Classification head (only if multiple classes)
    if n_classes > 1:
        classification = layers.Dense(n_classes, activation='softmax')(projection)
        model = keras_models.Model(input_layer, [projection, classification])
    else:
        # No classification head for single class
        model = keras_models.Model(input_layer, projection)
        classification = None
    projector = keras_models.Model(input_layer, projection)
    
    # ========== ADVANCED CONTRASTIVE LOSS ==========
    def advanced_contrastive_loss(y_true, y_pred):
        # Handle different output formats
        if isinstance(y_pred, list) and len(y_pred) == 2:
            projection_out, classification_out = y_pred
        else:
            projection_out = y_pred
            classification_out = None
        
        # InfoNCE loss for contrastive learning
        try:
            temperature = 0.1
            batch_size = tf.shape(projection_out)[0]
            
            # Similarity matrix
            similarity_matrix = tf.matmul(projection_out, projection_out, transpose_b=True) / temperature
            
            # Create positive/negative masks based on labels
            labels_batch = tf.convert_to_tensor(labels_array[:batch_size], dtype=tf.int32)
            labels_eq = tf.equal(tf.expand_dims(labels_batch, 0), tf.expand_dims(labels_batch, 1))
            positives_mask = tf.cast(labels_eq, tf.float32)
            
            # Remove diagonal (self-similarity)
            positives_mask = positives_mask * (1.0 - tf.eye(batch_size))
            
            # InfoNCE loss
            logits = similarity_matrix - tf.reduce_max(similarity_matrix, axis=1, keepdims=True)
            exp_logits = tf.exp(logits)
            
            log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-8)
            
            # Mean log-likelihood of positive pairs
            mean_log_prob_pos = tf.reduce_sum(positives_mask * log_prob, axis=1) / (tf.reduce_sum(positives_mask, axis=1) + 1e-8)
            contrastive_loss = -tf.reduce_mean(mean_log_prob_pos)
            
            contrastive_loss = tf.clip_by_value(contrastive_loss, 0.0, 10.0)
            
        except:
            contrastive_loss = 0.0
        
        # Classification loss (only if we have classification head)
        classification_loss = 0.0
        if classification_out is not None and n_classes > 1:
            try:
                labels_onehot = tf.one_hot(labels_batch, depth=n_classes)
                classification_loss = tf.reduce_mean(
                    tf.keras.losses.categorical_crossentropy(labels_onehot, classification_out)
                )
            except:
                classification_loss = 0.0
        
        # Combined loss
        total_loss = contrastive_loss + 0.5 * classification_loss
        return tf.clip_by_value(total_loss, 0.0, 100.0)
    
    # ========== TRAINING ==========
    optimizer = optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=advanced_contrastive_loss)
    
    # Prepare training targets
    if n_classes > 1:
        # Dummy targets for multi-output model
        dummy_targets = [X_normalized, X_normalized]
    else:
        # Single target for single-output model  
        dummy_targets = X_normalized
    
    # Training
    if verbose:
        print("   🔄 Advanced contrastive training...")
        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=10)
        ]
        
        model.fit(
            X_normalized, dummy_targets,
            epochs=epochs,
            batch_size=min(64, X_normalized.shape[0]),
            callbacks=callbacks,
            verbose=0
        )
        print("   ✓ Contrastive learning complete")
    else:
        model.fit(
            X_normalized, [X_normalized, X_normalized],
            epochs=epochs,
            batch_size=min(64, X_normalized.shape[0]),
            verbose=0
        )
    
    # Generate final embeddings
    final_embeddings = projector.predict(X_normalized, verbose=0)
    final_embeddings = np.clip(final_embeddings, -5, 5)
    final_embeddings = np.nan_to_num(final_embeddings, nan=0.0)
    
    if verbose:
        print(f"   ✓ Final embeddings shape: {final_embeddings.shape}")
    
    return final_embeddings
