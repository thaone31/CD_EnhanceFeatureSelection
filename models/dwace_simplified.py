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


def dwace_simplified_pipeline(G, ground_truth=None, feature_dim=128, initial_embedding=None, verbose=True):
    """
    DWACE Simplified Pipeline:
    1. DeepWalk embedding generation (or use provided initial_embedding)
    2. Simple AutoEncoder with MSE loss ONLY (no modularity or graph losses)
    3. Simple contrastive learning (if ground truth available)
    
    This version focuses purely on dimensionality reduction and feature enhancement
    without complex graph-aware losses.
    """
    from models.feature_utils import deepwalk_embedding
    
    if verbose:
        print("🔄 Starting DWACE Simplified Implementation...")
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
    
    # ========== STEP 2: Simple AutoEncoder with MSE Loss Only ==========
    if verbose:
        print("🔧 Step 2: Simple AutoEncoder with MSE loss only...")
    
    embedding_ae = simple_autoencoder_mse_only(
        embedding_dw, 
        out_dim=actual_feature_dim//2,  # Dimensionality reduction
        epochs=100,
        verbose=verbose
    )
    
    if verbose:
        print(f"   ✓ AutoEncoder output: {embedding_ae.shape}")
    
    # ========== STEP 3: Simple Contrastive Learning ==========
    if ground_truth is not None:
        if verbose:
            print("🎯 Step 3: Simple contrastive learning...")
        
        embedding_final = simple_contrastive_learning(
            embedding_ae, ground_truth,
            out_dim=feature_dim//2,
            epochs=100,
            verbose=verbose
        )
        
        enhancement_name = "dwace_simplified_full"
    else:
        if verbose:
            print("⚠️ Step 3: No ground truth - using AutoEncoder output")
        
        embedding_final = embedding_ae
        enhancement_name = "dwace_simplified_ae_only"
    
    if verbose:
        print("=" * 60)
        print("🎯 DWACE Simplified Pipeline Complete!")
        print(f"   Final embedding: {embedding_final.shape}")
    
    return {
        'deepwalk': embedding_dw,
        'dwace_ae': embedding_ae,
        enhancement_name: embedding_final
    }, enhancement_name


def simple_autoencoder_mse_only(X, out_dim=64, epochs=100, verbose=True):
    """
    Simple AutoEncoder with MSE loss ONLY.
    No modularity loss, no graph awareness - just pure dimensionality reduction.
    
    Architecture:
    - Encoder: Dense → ReLU → Dense → ReLU → BatchNorm → Dropout
    - Decoder: Dense → ReLU → Dense → ReLU
    - Loss: MSE reconstruction loss only
    """
    
    # Input normalization
    X = np.array(X, dtype=np.float32)
    X_mean = np.mean(X, axis=0, keepdims=True)
    X_std = np.std(X, axis=0, keepdims=True) + 1e-8
    X_normalized = (X - X_mean) / X_std
    X_normalized = np.clip(X_normalized, -3, 3)
    
    input_dim = X_normalized.shape[1]
    hidden_dim = (input_dim + out_dim) // 2
    
    if verbose:
        print(f"   Input dim: {input_dim}, Hidden dim: {hidden_dim}, Output dim: {out_dim}")
    
    # ========== ENCODER (Simple Architecture) ==========
    input_layer = layers.Input(shape=(input_dim,))
    
    # First hidden layer
    x = layers.Dense(hidden_dim, kernel_initializer='he_normal')(input_layer)
    x = layers.ReLU()(x)
    
    # Encoded representation
    encoded = layers.Dense(out_dim, kernel_initializer='he_normal')(x)
    encoded = layers.ReLU()(encoded)
    
    # Regularization
    encoded = layers.BatchNormalization()(encoded)
    encoded = layers.Dropout(0.1)(encoded)
    
    # ========== DECODER (Mirror Structure) ==========
    # First decoder layer
    y = layers.Dense(hidden_dim, kernel_initializer='he_normal')(encoded)
    y = layers.ReLU()(y)
    
    # Output reconstruction
    decoded = layers.Dense(input_dim, kernel_initializer='he_normal')(y)
    decoded = layers.ReLU()(decoded)
    
    # ========== MODELS ==========
    autoencoder = keras_models.Model(input_layer, decoded)
    encoder = keras_models.Model(input_layer, encoded)
    
    # ========== SIMPLE MSE LOSS ==========
    autoencoder.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='mse'  # Simple MSE loss only
    )
    
    if verbose:
        print(f"   Model compiled with MSE loss")
    
    # ========== TRAINING ==========
    try:
        # Use full batch training for stability
        batch_size = len(X_normalized)
        
        history = autoencoder.fit(
            X_normalized, X_normalized,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            shuffle=False  # No shuffling for full batch
        )
        
        # Extract features
        embedding_ae = encoder.predict(X_normalized, verbose=0)
        
        # Normalize output embeddings
        embedding_ae = embedding_ae / (np.linalg.norm(embedding_ae, axis=1, keepdims=True) + 1e-8)
        
        if verbose:
            final_loss = history.history['loss'][-1]
            print(f"   ✓ Training complete. Final MSE loss: {final_loss:.6f}")
            print(f"   ✓ Output embedding shape: {embedding_ae.shape}")
        
        return embedding_ae
        
    except Exception as e:
        if verbose:
            print(f"   ❌ AutoEncoder training failed: {e}")
        
        # Fallback: simple PCA-like dimensionality reduction
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(out_dim, X_normalized.shape[1]))
        embedding_fallback = pca.fit_transform(X_normalized)
        
        if verbose:
            print(f"   🔄 Using PCA fallback: {embedding_fallback.shape}")
        
        return embedding_fallback


def simple_contrastive_learning(X, labels, out_dim=64, epochs=100, temperature=0.1, verbose=True):
    """
    Simple contrastive learning with basic positive/negative sampling.
    Much simpler than InfoNCE - just basic contrastive head training.
    """
    
    # Input normalization
    X = np.array(X, dtype=np.float32)
    X_mean = np.mean(X, axis=0, keepdims=True)
    X_std = np.std(X, axis=0, keepdims=True) + 1e-8
    X_normalized = (X - X_mean) / X_std
    X_normalized = np.clip(X_normalized, -2, 2)
    
    # Prepare labels
    labels = np.array(labels)
    n_classes = len(np.unique(labels))
    
    input_dim = X_normalized.shape[1]
    
    if verbose:
        print(f"   Input: {X_normalized.shape}, Classes: {n_classes}, Output: {out_dim}")
    
    # ========== SIMPLE CONTRASTIVE HEAD ==========
    input_layer = layers.Input(shape=(input_dim,))
    
    # Projection head
    x = layers.Dense(input_dim//2, kernel_initializer='he_normal')(input_layer)
    x = layers.ReLU()(x)
    x = layers.BatchNormalization()(x)
    
    # Final projection
    projection = layers.Dense(out_dim, kernel_initializer='he_normal')(x)
    projection = layers.ReLU()(projection)
    
    # L2 normalization for contrastive learning
    projection_normalized = layers.Lambda(
        lambda x: tf.nn.l2_normalize(x, axis=1, epsilon=1e-8)
    )(projection)
    
    # ========== MODEL ==========
    model = keras_models.Model(input_layer, projection_normalized)
    
    # ========== SIMPLE TRIPLET-LIKE LOSS ==========
    def simple_contrastive_loss(y_true, y_pred):
        """
        Simple contrastive loss based on same/different class similarity
        """
        # y_pred is normalized embeddings
        # Compute pairwise cosine similarities
        similarities = tf.matmul(y_pred, tf.transpose(y_pred))
        
        # Create mask for same class pairs
        y_true_int = tf.cast(y_true, tf.int32)
        same_class_mask = tf.equal(
            tf.expand_dims(y_true_int, 1), 
            tf.expand_dims(y_true_int, 0)
        )
        same_class_mask = tf.cast(same_class_mask, tf.float32)
        
        # Positive pairs (same class) should have high similarity
        positive_loss = tf.reduce_mean(
            same_class_mask * (1.0 - similarities)
        )
        
        # Negative pairs (different class) should have low similarity  
        negative_loss = tf.reduce_mean(
            (1.0 - same_class_mask) * tf.maximum(0.0, similarities + 0.2)
        )
        
        total_loss = positive_loss + negative_loss
        return total_loss
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=simple_contrastive_loss
    )
    
    if verbose:
        print(f"   Model compiled with simple contrastive loss")
    
    # ========== TRAINING ==========
    try:
        # Use full batch training
        batch_size = len(X_normalized)
        
        history = model.fit(
            X_normalized, labels,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
            shuffle=False
        )
        
        # Extract final embeddings
        embedding_contrastive = model.predict(X_normalized, verbose=0)
        
        if verbose:
            final_loss = history.history['loss'][-1]
            print(f"   ✓ Contrastive training complete. Final loss: {final_loss:.6f}")
            print(f"   ✓ Output embedding shape: {embedding_contrastive.shape}")
        
        return embedding_contrastive
        
    except Exception as e:
        if verbose:
            print(f"   ❌ Contrastive learning failed: {e}")
        
        # Fallback: return input embeddings
        if verbose:
            print(f"   🔄 Using input embeddings as fallback")
        
        return X_normalized
