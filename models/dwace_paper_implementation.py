import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses, backend as K
import tensorflow.keras.models as keras_models
import networkx as nx
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import warnings
warnings.filterwarnings('ignore')

# Configure TensorFlow
tf.config.run_functions_eagerly(True)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


def dwace_paper_pipeline(G, ground_truth=None, feature_dim=128, initial_embedding=None, verbose=True):
    """
    DWACE Pipeline according to the paper specification:
    1. DeepWalk embedding generation (or use provided initial_embedding)
    2. AutoEncoder with proper architecture (Dense→LeakyReLU→Dense→ELU+BatchNorm+Dropout)
    3. Contrastive projection learning with InfoNCE loss
    4. Combined loss: L = L_recon + λ1*L_modularity + λ2*L_classifier
    """
    from models.feature_utils import deepwalk_embedding
    
    if verbose:
        print("🔄 Starting DWACE Paper Implementation...")
        print("=" * 60)
    
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
    
    # ========== STEP 2: AutoEncoder with Paper Architecture ==========
    if verbose:
        print("🔧 Step 2: AutoEncoder with graph-aware loss...")
    
    embedding_ae = autoencoder_paper_architecture(
        embedding_dw, G, ground_truth, 
        out_dim=actual_feature_dim//2,  # Dimensionality reduction based on actual input
        epochs=100,
        verbose=verbose
    )
    
    if verbose:
        print(f"   ✓ AutoEncoder output: {embedding_ae.shape}")
    
    # ========== STEP 3: Contrastive Projection Learning ==========
    if ground_truth is not None:
        if verbose:
            print("🎯 Step 3: Contrastive projection learning...")
        
        embedding_final = contrastive_infoNCE_learning(
            embedding_ae, ground_truth,
            out_dim=feature_dim//2,
            epochs=150,
            verbose=verbose
        )
        
        enhancement_name = "dwace_paper_full"
    else:
        if verbose:
            print("⚠️ Step 3: No ground truth - using AutoEncoder output")
        
        embedding_final = embedding_ae
        enhancement_name = "dwace_paper_ae_only"
    
    if verbose:
        print("=" * 60)
        print("🎯 DWACE Paper Pipeline Complete!")
        print(f"   Final embedding: {embedding_final.shape}")
    
    return {
        'deepwalk': embedding_dw,
        'dwace_ae': embedding_ae,
        enhancement_name: embedding_final
    }, enhancement_name


def autoencoder_paper_architecture(X, G, ground_truth=None, out_dim=64, epochs=100, verbose=True):
    """
    AutoEncoder with exact paper specification:
    - Encoder: Dense → LeakyReLU → Dense → ELU → BatchNorm → Dropout
    - Decoder: mirrors encoder structure  
    - Loss: MSE + Modularity loss + (Classifier loss if labels exist)
    """
    
    # Input normalization
    X = np.array(X, dtype=np.float32)
    X_mean = np.mean(X, axis=0, keepdims=True)
    X_std = np.std(X, axis=0, keepdims=True) + 1e-8
    X_normalized = (X - X_mean) / X_std
    X_normalized = np.clip(X_normalized, -3, 3)
    
    # Prepare adjacency matrix
    try:
        A = nx.to_numpy_array(G)
    except:
        A = None
        print("   [WARNING] Cannot create adjacency matrix")
    
    input_dim = X_normalized.shape[1]
    hidden_dim = (input_dim + out_dim) // 2
    
    # ========== ENCODER (Paper Specification) ==========
    input_layer = layers.Input(shape=(input_dim,))
    
    # Dense → LeakyReLU
    x = layers.Dense(hidden_dim, kernel_initializer='he_normal')(input_layer)
    x = layers.LeakyReLU(alpha=0.01)(x)  # Paper uses LeakyReLU
    
    # Dense → ELU  
    encoded = layers.Dense(out_dim, kernel_initializer='he_normal')(x)
    encoded = layers.ELU()(encoded)  # Paper uses ELU
    
    # BatchNormalization and Dropout
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dropout(0.2)(encoded)  # Paper mentions dropout
    
    # ========== DECODER (Mirror Structure) ==========
    # Dense → ELU
    y = layers.Dense(hidden_dim, kernel_initializer='he_normal')(encoded)
    y = layers.ELU()(y)
    
    # Dense → LeakyReLU
    y = layers.Dense(input_dim, kernel_initializer='he_normal')(y)
    decoded = layers.LeakyReLU(alpha=0.01)(y)
    
    # ========== CLASSIFIER HEAD (if labels exist) ==========
    classifier_out = None
    if ground_truth is not None:
        n_classes = len(np.unique(ground_truth))
        classifier_out = layers.Dense(n_classes, activation='softmax', name='classifier')(encoded)
    
    # ========== MODELS ==========
    if classifier_out is not None:
        autoencoder = keras_models.Model(input_layer, [decoded, classifier_out])
    else:
        autoencoder = keras_models.Model(input_layer, decoded)
    
    encoder = keras_models.Model(input_layer, encoded)
    
    # ========== COMBINED LOSS FUNCTION ==========
    def combined_loss(y_true, y_pred):
        """
        Combined loss as per paper: L = L_recon + λ1*L_modularity + λ2*L_classifier
        """
        if isinstance(y_pred, list):
            y_pred_recon, y_pred_class = y_pred
        else:
            y_pred_recon = y_pred
            y_pred_class = None
        
        # L_recon: MSE reconstruction loss
        mse_loss = tf.reduce_mean(tf.square(y_true - y_pred_recon))
        total_loss = mse_loss
        
        # L_modularity: Modularity preservation loss  
        if A is not None:
            try:
                # Get embeddings from the input (not the reconstruction)
                batch_size = tf.shape(y_true)[0]
                
                # Only compute modularity loss if batch size matches adjacency matrix
                if tf.equal(batch_size, tf.constant(A.shape[0])):
                    Z = encoder(y_true)  # Get embeddings from input
                    
                    # Ensure Z has correct shape
                    Z = tf.ensure_shape(Z, [None, out_dim])
                    
                    # Modularity loss computation
                    A_tf = tf.convert_to_tensor(A, dtype=tf.float32)
                    deg = tf.reduce_sum(A_tf, axis=1, keepdims=True)
                    m = tf.maximum(tf.reduce_sum(deg) / 2.0, 1e-8)
                    
                    # Normalized embeddings for similarity
                    Z_norm = tf.nn.l2_normalize(Z, axis=1, epsilon=1e-8)
                    sim_matrix = tf.matmul(Z_norm, Z_norm, transpose_b=True)
                    
                    # Clip similarity to prevent extreme values
                    sim_matrix = tf.clip_by_value(sim_matrix, -1.0, 1.0)
                    
                    # Expected edges (null model)
                    expected_edges = tf.matmul(deg, tf.transpose(deg)) / (2.0 * m)
                    modularity_matrix = A_tf - expected_edges
                    
                    # Modularity score - ensure shapes match
                    modularity_score = tf.reduce_sum(modularity_matrix * sim_matrix) / (2.0 * m)
                    modularity_loss = -modularity_score  # Negative because we want to maximize modularity
                    
                    # Add to total loss with clipping
                    modularity_loss = tf.clip_by_value(modularity_loss, -10.0, 10.0)
                    total_loss += 0.1 * modularity_loss
                    
            except Exception as e:
                # If modularity computation fails, just use reconstruction loss
                pass
        
        # L_classifier: Classification loss (if labels exist)
        if y_pred_class is not None and ground_truth is not None:
            try:
                y_true_class = tf.keras.utils.to_categorical(ground_truth, num_classes=len(np.unique(ground_truth)))
                classification_loss = tf.keras.losses.categorical_crossentropy(y_true_class, y_pred_class)
                
                # λ2 = 1.0 (paper typical value) 
                total_loss += 1.0 * tf.reduce_mean(classification_loss)
            except:
                pass
        
        return tf.clip_by_value(total_loss, 0.0, 100.0)
    
    # ========== TRAINING ==========
    optimizer = optimizers.Adam(learning_rate=0.001, clipnorm=1.0)
    
    if classifier_out is not None:
        autoencoder.compile(
            optimizer=optimizer,
            loss={'model': combined_loss, 'classifier': 'sparse_categorical_crossentropy'},
            loss_weights={'model': 1.0, 'classifier': 1.0}
        )
        
        # Prepare training data
        y_class = np.array(ground_truth)
        training_data = (X_normalized, [X_normalized, y_class])
    else:
        autoencoder.compile(optimizer=optimizer, loss=combined_loss)
        training_data = (X_normalized, X_normalized)
    
    # Training with callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='loss', patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=10)
    ]
    
    try:
        # Use full batch size to ensure adjacency matrix compatibility
        effective_batch_size = X_normalized.shape[0] if A is not None else min(32, X_normalized.shape[0])
        
        if classifier_out is not None:
            autoencoder.fit(
                training_data[0], training_data[1],
                epochs=epochs,
                batch_size=effective_batch_size,
                callbacks=callbacks,
                verbose=1 if verbose else 0
            )
        else:
            autoencoder.fit(
                training_data[0], training_data[1],
                epochs=epochs,
                batch_size=effective_batch_size,
                callbacks=callbacks,
                verbose=1 if verbose else 0
            )
    except Exception as e:
        if verbose:
            print(f"   [WARNING] Training failed: {e}")
        K.clear_session()
        return X_normalized[:, :out_dim]
    
    # Get enhanced features
    try:
        enhanced = encoder.predict(X_normalized, verbose=0)
        
        if np.isnan(enhanced).any():
            enhanced = np.nan_to_num(enhanced)
            
    except Exception as e:
        if verbose:
            print(f"   [WARNING] Prediction failed: {e}")
        enhanced = X_normalized[:, :out_dim]
    
    K.clear_session()
    return enhanced


def contrastive_infoNCE_learning(X, labels, out_dim=64, epochs=150, temperature=0.5, verbose=True):
    """
    Contrastive projection learning with InfoNCE loss as per paper:
    - Creates two stochastic views via dropout/masking
    - Uses InfoNCE to pull positive pairs closer, push negative pairs apart
    - Combines with supervised loss when labels available
    """
    
    X = np.array(X, dtype=np.float32)
    X_normalized = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-8)
    X_normalized = np.clip(X_normalized, -3, 3)
    
    n_classes = len(np.unique(labels))
    
    # ========== PROJECTION NETWORK ==========
    input_layer = layers.Input(shape=(X_normalized.shape[1],))
    
    # Two fully connected layers with BatchNorm (as per paper)
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(1e-3))(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    
    # Projection head
    projection = layers.Dense(out_dim, activation=None, name='projection')(x)
    projection = layers.BatchNormalization()(projection)
    
    # Classifier head for supervised learning
    classifier = layers.Dense(n_classes, activation='softmax', name='classifier')(projection)
    
    model = keras_models.Model(input_layer, [projection, classifier])
    
    # ========== InfoNCE + Supervised Loss ==========
    def combined_contrastive_loss(y_true, y_pred):
        proj_out, class_out = y_pred
        
        # InfoNCE Loss
        # Normalize projections
        proj_norm = tf.nn.l2_normalize(proj_out, axis=1, epsilon=1e-8)
        
        # Compute similarity matrix
        sim_matrix = tf.matmul(proj_norm, proj_norm, transpose_b=True) / temperature
        sim_matrix = tf.clip_by_value(sim_matrix, -10, 10)
        
        # Create positive pair mask (same labels)
        labels_tensor = tf.cast(y_true[:, 0], tf.int32)  # Assuming y_true contains labels
        pos_mask = tf.cast(tf.equal(tf.expand_dims(labels_tensor, 1), 
                                  tf.expand_dims(labels_tensor, 0)), tf.float32)
        
        # Remove diagonal (self-pairs)
        pos_mask = pos_mask - tf.eye(tf.shape(pos_mask)[0])
        
        # InfoNCE computation
        exp_sim = tf.exp(sim_matrix) 
        log_prob = sim_matrix - tf.math.log(tf.reduce_sum(exp_sim, axis=1, keepdims=True) + 1e-8)
        
        # Average over positive pairs
        pos_log_prob = tf.reduce_sum(pos_mask * log_prob, axis=1)
        pos_count = tf.reduce_sum(pos_mask, axis=1)
        pos_count = tf.maximum(pos_count, 1.0)  # Avoid division by zero
        
        infoNCE_loss = -tf.reduce_mean(pos_log_prob / pos_count)
        
        # Supervised classification loss  
        y_class_true = tf.keras.utils.to_categorical(labels_tensor, num_classes=n_classes)
        classification_loss = tf.keras.losses.categorical_crossentropy(y_class_true, class_out)
        classification_loss = tf.reduce_mean(classification_loss)
        
        # Combined loss: λ1=1.0 for contrastive, λ2=0.3 for supervised (paper values)
        total_loss = 1.0 * infoNCE_loss + 0.3 * classification_loss
        
        return tf.clip_by_value(total_loss, 0.0, 50.0)
    
    # ========== TRAINING ==========
    optimizer = optimizers.Adam(learning_rate=0.0005, clipnorm=1.0)
    model.compile(optimizer=optimizer, loss=combined_contrastive_loss)
    
    # Prepare labels for training
    y_labels = np.expand_dims(labels, 1)
    
    try:
        # Use full batch for contrastive learning to avoid shape mismatches
        model.fit(
            X_normalized, [X_normalized, labels],  # Dummy targets
            epochs=epochs,
            batch_size=X_normalized.shape[0],  # Full batch size
            verbose=1 if verbose else 0
        )
    except Exception as e:
        if verbose:
            print(f"   [WARNING] Contrastive training failed: {e}")
        K.clear_session()
        return X_normalized
    
    # Get final projections
    try:
        projections, _ = model.predict(X_normalized, verbose=0)
        
        if np.isnan(projections).any():
            projections = np.nan_to_num(projections)
            
    except Exception as e:
        if verbose:
            print(f"   [WARNING] Projection failed: {e}")
        projections = X_normalized
    
    K.clear_session()
    return projections
