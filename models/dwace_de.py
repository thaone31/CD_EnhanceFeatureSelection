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
    # Input normalization to prevent NaN
    X = np.array(X, dtype=np.float32)
    X_normalized = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    X_normalized = np.clip(X_normalized, -5, 5)  # Clip extreme values
    
    # Check for NaN in input
    if np.isnan(X_normalized).any():
        print("   [WARNING] NaN detected in autoencoder input, replacing with zeros")
        X_normalized = np.nan_to_num(X_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Prepare adjacency matrix if graph is not too large
    if G.number_of_nodes() < 5000:  # Reduce threshold for stability
        import networkx as nx
        A = nx.to_numpy_array(G)
    else:
        A = None
        if verbose:
            print(f"   [WARNING] Graph too large ({G.number_of_nodes()} nodes), skipping graph loss")
    
    # Prepare community labels
    if ground_truth is not None:
        comm_labels = np.array(ground_truth, dtype=np.int32)
    else:
        comm_labels = np.zeros(X_normalized.shape[0], dtype=np.int32)
    
    # Build AutoEncoder architecture
    input_layer = layers.Input(shape=(X_normalized.shape[1],))
    
    # Encoder with regularization
    hidden_dim = max(out_dim * 2, 32)
    encoded = layers.Dense(hidden_dim, activation='relu', 
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4))(input_layer)
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dropout(0.3)(encoded)
    
    encoded = layers.Dense(out_dim, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4))(encoded)
    encoded = layers.BatchNormalization()(encoded)
    
    # Decoder with regularization
    decoded = layers.Dense(hidden_dim, activation='relu',
                          kernel_regularizer=tf.keras.regularizers.l2(1e-4))(encoded)
    decoded = layers.BatchNormalization()(decoded)
    decoded = layers.Dropout(0.3)(decoded)
    
    output = layers.Dense(X_normalized.shape[1], activation='linear',
                         kernel_regularizer=tf.keras.regularizers.l2(1e-4))(decoded)
    
    # Create models
    autoencoder = keras_models.Model(input_layer, output)
    encoder = keras_models.Model(input_layer, encoded)
    
    # Custom loss with graph-aware components
    def graph_aware_loss(y_true, y_pred):
        # Check for NaN in predictions
        if tf.reduce_any(tf.math.is_nan(y_pred)):
            return tf.constant(10.0)  # Return high loss if NaN detected
        
        # Reconstruction loss with clipping
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
        mse_loss = tf.clip_by_value(mse_loss, 0.0, 100.0)
        
        total_loss = lambda_recon * mse_loss
        
        if A is not None and lambda_mod > 0:
            try:
                # Get encoder output
                Z = encoder(y_true)
                
                # Check for NaN in encoder output
                if tf.reduce_any(tf.math.is_nan(Z)):
                    return total_loss  # Skip graph loss if NaN in Z
                
                A_tf = tf.convert_to_tensor(A, dtype=tf.float32)
                
                # Modularity loss with better numerical stability
                deg = tf.reduce_sum(A_tf, axis=1, keepdims=True)
                m = tf.maximum(tf.reduce_sum(deg) / 2.0, 1e-8)
                
                Z_norm = tf.nn.l2_normalize(Z, axis=1, epsilon=1e-8)
                dot = tf.matmul(Z_norm, Z_norm, transpose_b=True)
                dot = tf.clip_by_value(dot, -1.0, 1.0)
                
                deg_prod = tf.matmul(deg, tf.transpose(deg)) / (2.0 * m)
                modularity_matrix = A_tf - deg_prod
                modularity_score = tf.reduce_sum(modularity_matrix * dot) / (2.0 * m)
                modularity_loss = -tf.clip_by_value(modularity_score, -10.0, 10.0)
                
                # Neighborhood preservation loss with stability checks
                if lambda_neigh > 0 and ground_truth is not None:
                    comm_labels_tf = tf.convert_to_tensor(comm_labels, dtype=tf.int32)
                    mask_same = tf.cast(tf.equal(tf.expand_dims(comm_labels_tf, 1), 
                                               tf.expand_dims(comm_labels_tf, 0)), tf.float32)
                    mask_diff = 1.0 - mask_same
                    
                    dists = tf.norm(tf.expand_dims(Z, 1) - tf.expand_dims(Z, 0), axis=2)
                    dists = tf.clip_by_value(dists, 0.0, 10.0)
                    
                    same_sum = tf.reduce_sum(mask_same)
                    diff_sum = tf.reduce_sum(mask_diff)
                    
                    if same_sum > 0 and diff_sum > 0:
                        same_mean = tf.reduce_sum(dists * mask_same) / same_sum
                        diff_mean = tf.reduce_sum(dists * mask_diff) / diff_sum
                        neigh_loss = same_mean / tf.maximum(diff_mean, 1e-8)
                        neigh_loss = tf.clip_by_value(neigh_loss, 0.0, 10.0)
                        
                        total_loss += lambda_neigh * neigh_loss
                
                # Add modularity loss with clipping
                modularity_loss = tf.where(tf.math.is_nan(modularity_loss), 0.0, modularity_loss)
                total_loss += lambda_mod * modularity_loss
                
            except Exception as e:
                # If any error occurs in graph loss computation, just use reconstruction loss
                pass
        
        # Final NaN check and clipping
        total_loss = tf.where(tf.math.is_nan(total_loss), mse_loss, total_loss)
        return tf.clip_by_value(total_loss, 0.0, 100.0)
    
    # Compile and train with gradient clipping
    optimizer = optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    autoencoder.compile(optimizer=optimizer, loss=graph_aware_loss)
    
    # Train with callbacks for stability
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10, min_lr=1e-6)
    ]
    
    try:
        autoencoder.fit(
            X_normalized, X_normalized,
            epochs=epochs, 
            batch_size=min(batch_size, X_normalized.shape[0]),
            verbose=1 if verbose else 0,
            callbacks=callbacks
        )
    except Exception as e:
        print(f"   [WARNING] Training error: {e}, using input as output")
        K.clear_session()
        return X_normalized[:, :out_dim]
    
    # Get enhanced features
    try:
        enhanced = encoder.predict(X_normalized, verbose=0)
        
        # Check for NaN in output
        if np.isnan(enhanced).any():
            print("   [WARNING] NaN in autoencoder output, using normalized input")
            enhanced = X_normalized[:, :out_dim]
        
    except Exception as e:
        print(f"   [WARNING] Prediction error: {e}, using normalized input")
        enhanced = X_normalized[:, :out_dim]
    
    K.clear_session()
    return enhanced


def contrastive_projection_enhanced(X, comm_labels, out_dim=64, epochs=150, 
                                  temperature=0.5, lambda_contrastive=1.0, 
                                  lambda_sup=0.3, verbose=True):
    """
    Enhanced contrastive projection with InfoNCE + supervised learning
    """
    # Input normalization to prevent NaN
    X = np.array(X, dtype=np.float32)
    X_normalized = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    X_normalized = np.clip(X_normalized, -3, 3)  # More conservative clipping
    
    # Check for NaN in input
    if np.isnan(X_normalized).any():
        print("   [WARNING] NaN detected in contrastive input, replacing with zeros")
        X_normalized = np.nan_to_num(X_normalized, nan=0.0, posinf=1.0, neginf=-1.0)
    
    batch_size = min(64, X_normalized.shape[0])  # Smaller batch size for stability
    
    # Ensure temperature is reasonable
    temperature = max(temperature, 0.1)
    
    # Architecture with better regularization
    input_layer = layers.Input(shape=(X_normalized.shape[1],))
    
    # Simpler, more stable architecture
    x = layers.Dense(128, activation='relu', 
                    kernel_regularizer=tf.keras.regularizers.l2(1e-3))(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(out_dim * 2, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    # Projection head with normalization
    proj = layers.Dense(out_dim, activation=None,
                       kernel_regularizer=tf.keras.regularizers.l2(1e-3))(x)
    proj = layers.BatchNormalization()(proj)
    
    # Classifier head
    n_classes = int(np.max(comm_labels)) + 1 if comm_labels is not None else 1
    if n_classes > 1 and n_classes < 100:  # Reasonable number of classes
        classifier_out = layers.Dense(n_classes, activation='softmax', 
                                    name='classifier',
                                    kernel_regularizer=tf.keras.regularizers.l2(1e-3))(proj)
        model = keras_models.Model(input_layer, [proj, classifier_out])
        y_class = tf.keras.utils.to_categorical(comm_labels, num_classes=n_classes)
    else:
        model = keras_models.Model(input_layer, proj)
        y_class = None
        n_classes = 1
    
    # Optimizer with conservative learning rate
    initial_lr = 0.0005
    optimizer = optimizers.Adam(learning_rate=initial_lr, clipnorm=0.5)
    
    if verbose:
        print(f"   Starting contrastive training: {epochs} epochs, temp={temperature:.3f}")
        print(f"   Model: {n_classes} classes, batch_size={batch_size}")
    
    # Training loop with better error handling
    successful_epochs = 0
    for epoch in range(epochs):
        try:
            # Learning rate decay
            current_lr = initial_lr * (0.98 ** (epoch // 5))
            optimizer.learning_rate = max(current_lr, 1e-6)
            
            # Simple data augmentation
            idx = np.random.permutation(X_normalized.shape[0])
            X1 = add_noise(X_normalized[idx], noise_level=0.02)
            X2 = add_noise(X_normalized[idx], noise_level=0.03)
            
            with tf.GradientTape() as tape:
                if n_classes > 1:
                    z1, class_pred1 = model(X1, training=True)
                    z2, class_pred2 = model(X2, training=True)
                else:
                    z1 = model(X1, training=True)
                    z2 = model(X2, training=True)
                
                # Check for NaN in outputs
                if tf.reduce_any(tf.math.is_nan(z1)) or tf.reduce_any(tf.math.is_nan(z2)):
                    if verbose and epoch % 20 == 0:
                        print(f"   Epoch {epoch:3d}: NaN in outputs, skipping...")
                    continue
                
                # Contrastive loss with better stability
                loss_contrastive = info_nce_loss_stable(z1, z2, temperature=temperature)
                
                # Supervised loss
                if n_classes > 1 and y_class is not None:
                    try:
                        sup_loss1 = tf.keras.losses.categorical_crossentropy(y_class[idx], class_pred1)
                        sup_loss2 = tf.keras.losses.categorical_crossentropy(y_class[idx], class_pred2)
                        sup_loss = (tf.reduce_mean(sup_loss1) + tf.reduce_mean(sup_loss2)) / 2.0
                        sup_loss = tf.clip_by_value(sup_loss, 0.0, 10.0)
                    except:
                        sup_loss = 0.0
                else:
                    sup_loss = 0.0
                
                total_loss = lambda_contrastive * loss_contrastive + lambda_sup * sup_loss
                
                # Check for NaN in loss
                if tf.math.is_nan(total_loss) or total_loss > 100:
                    if verbose and epoch % 20 == 0:
                        print(f"   Epoch {epoch:3d}: Invalid loss, skipping...")
                    continue
            
            # Compute gradients with error checking
            grads = tape.gradient(total_loss, model.trainable_weights)
            if grads is None or any(g is None for g in grads):
                continue
            
            # Check for NaN gradients
            if any(tf.reduce_any(tf.math.is_nan(g)) for g in grads if g is not None):
                if verbose and epoch % 20 == 0:
                    print(f"   Epoch {epoch:3d}: NaN gradients, skipping...")
                continue
            
            # Apply gradients
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            successful_epochs += 1
            
            if verbose and epoch % 30 == 0:
                print(f"   Epoch {epoch:3d}: loss={total_loss.numpy():.4f}, "
                      f"contrast={loss_contrastive.numpy():.4f}, "
                      f"sup={sup_loss.numpy() if n_classes > 1 else 0.0:.4f}")
        
        except Exception as e:
            if verbose and epoch % 50 == 0:
                print(f"   Epoch {epoch}: Training error: {e}")
            continue
    
    if verbose:
        print(f"   Completed {successful_epochs}/{epochs} successful training epochs")
    
    # Get final embeddings with error handling
    try:
        if n_classes > 1:
            out, _ = model.predict(X_normalized, verbose=0, batch_size=batch_size)
        else:
            out = model.predict(X_normalized, verbose=0, batch_size=batch_size)
        
        # Check for NaN in output
        if np.isnan(out).any():
            print("   [WARNING] NaN in contrastive output, using normalized input")
            out = X_normalized[:, :out_dim]
            
    except Exception as e:
        print(f"   [WARNING] Prediction error: {e}, using normalized input")
        out = X_normalized[:, :out_dim]
    
    K.clear_session()
    return out


def add_noise(X, noise_level=0.02):
    """Add small amount of gaussian noise for data augmentation"""
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise.astype(np.float32)


def info_nce_loss_stable(z1, z2, temperature=0.5):
    """More stable InfoNCE loss implementation"""
    # L2 normalize with small epsilon
    z1 = tf.nn.l2_normalize(z1, axis=1, epsilon=1e-8)
    z2 = tf.nn.l2_normalize(z2, axis=1, epsilon=1e-8)
    
    batch_size = tf.shape(z1)[0]
    temperature = tf.maximum(temperature, 0.1)
    
    try:
        # Compute cosine similarity
        logits_11 = tf.matmul(z1, z1, transpose_b=True) / temperature
        logits_22 = tf.matmul(z2, z2, transpose_b=True) / temperature
        logits_12 = tf.matmul(z1, z2, transpose_b=True) / temperature
        logits_21 = tf.matmul(z2, z1, transpose_b=True) / temperature
        
        # Clip logits to prevent overflow
        logits_11 = tf.clip_by_value(logits_11, -10.0, 10.0)
        logits_22 = tf.clip_by_value(logits_22, -10.0, 10.0)
        logits_12 = tf.clip_by_value(logits_12, -10.0, 10.0)
        logits_21 = tf.clip_by_value(logits_21, -10.0, 10.0)
        
        # Mask out self-similarity
        mask = tf.eye(batch_size, dtype=tf.bool)
        logits_11 = tf.where(mask, -1e9, logits_11)
        logits_22 = tf.where(mask, -1e9, logits_22)
        
        # Positive logits are on the diagonal of logits_12 and logits_21
        pos_logits_1 = tf.linalg.diag_part(logits_12)
        pos_logits_2 = tf.linalg.diag_part(logits_21)
        
        # Negative logits
        neg_logits_1 = tf.concat([
            tf.boolean_mask(logits_11, ~mask),
            tf.boolean_mask(logits_12, ~tf.eye(batch_size, dtype=tf.bool))
        ], axis=0)
        neg_logits_1 = tf.reshape(neg_logits_1, [batch_size, -1])
        
        neg_logits_2 = tf.concat([
            tf.boolean_mask(logits_22, ~mask),
            tf.boolean_mask(logits_21, ~tf.eye(batch_size, dtype=tf.bool))
        ], axis=0)
        neg_logits_2 = tf.reshape(neg_logits_2, [batch_size, -1])
        
        # Compute loss
        logits_1 = tf.concat([tf.expand_dims(pos_logits_1, 1), neg_logits_1], axis=1)
        logits_2 = tf.concat([tf.expand_dims(pos_logits_2, 1), neg_logits_2], axis=1)
        
        labels = tf.zeros(batch_size, dtype=tf.int32)  # Positive is always at index 0
        
        loss_1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits_1)
        loss_2 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits_2)
        
        total_loss = tf.reduce_mean(loss_1 + loss_2)
        
        # Final stability check
        total_loss = tf.where(tf.math.is_nan(total_loss), tf.constant(1.0), total_loss)
        total_loss = tf.clip_by_value(total_loss, 0.0, 10.0)
        
        return total_loss
        
    except Exception:
        # Fallback to simple contrastive loss
        sim_matrix = tf.matmul(z1, z2, transpose_b=True)
        pos_sim = tf.linalg.diag_part(sim_matrix)
        loss = -tf.reduce_mean(pos_sim)
        return tf.clip_by_value(loss, 0.0, 10.0)


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
