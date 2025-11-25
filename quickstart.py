"""
Quick Start Example - Train Your First Model
=============================================
Simple script to get started with wildfire burn severity prediction.

Usage:
    python quick_start.py
"""

import sys
sys.path.append('/home/claude')
from models import *

# ============================================================================
# SIMPLE CONFIGURATION
# ============================================================================

# Which features to use?
FEATURE_GROUPS = [
    'spectral',      # 6 features: Blue, Green, Red, NIR, SWIR1, SWIR2
    'indices',       # 5 features: NDVI, NDBI, MNDWI, NBR, BSI
    'diff_indices',  # 5 features: dNDVI, dNDBI, dMNDWI, dNBR, dBSI
    'terrain',       # ~60 features: Elevation, slope, aspect, etc.
    'climate',       # 19 features: WorldClim BIO variables
]

# Model choice: 'mlp', 'resnet', or 'attention'
MODEL_TYPE = 'mlp'  # Start with MLP - it's fast and works well

# Training settings
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001

# ============================================================================
# RUN TRAINING
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("QUICK START: Wildfire Burn Severity Prediction")
    print("="*70 + "\n")
    
    # Load data
    data_loader = WildfireDataLoader('/mnt/user-data/uploads/all_fires_complete_covariates.csv')
    data_loader.load_and_clean(drop_missing=True)
    
    # Prepare features
    X, y, feature_names = data_loader.prepare_features(FEATURE_GROUPS)
    splits = data_loader.split_data(X, y)
    
    # Create PyTorch datasets
    train_dataset = WildfireDataset(splits['X_train'], splits['y_train'])
    val_dataset = WildfireDataset(splits['X_val'], splits['y_val'])
    test_dataset = WildfireDataset(splits['X_test'], splits['y_test'])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Initialize model
    input_dim = X.shape[1]
    num_classes = len(np.unique(y))
    
    if MODEL_TYPE == 'mlp':
        model = MLPClassifier(input_dim, num_classes)
    elif MODEL_TYPE == 'resnet':
        model = ResNetClassifier(input_dim, num_classes)
    elif MODEL_TYPE == 'attention':
        model = AttentionClassifier(input_dim, num_classes)
    
    print(f"Model: {MODEL_TYPE.upper()}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Handle class imbalance
    class_weights = compute_class_weights(splits['y_train'])
    
    # Train!
    trainer = Trainer(model)
    trainer.fit(
        train_loader, val_loader,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        class_weights=class_weights
    )
    
    # Evaluate on test set
    results = trainer.evaluate(test_loader, data_loader.label_encoder)
    
    print("\n" + "="*70)
    print("TEST RESULTS")
    print("="*70)
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Macro F1: {results['macro_f1']:.3f}")
    print(f"Weighted F1: {results['weighted_f1']:.3f}")
    
    # Save outputs
    plot_training_history(trainer.history)
    plot_confusion_matrix(results['confusion_matrix'], data_loader.label_encoder.classes_)
    
    print("\n✓ Training complete! Check training_history.png and confusion_matrix.png")