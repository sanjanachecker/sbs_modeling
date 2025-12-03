"""
Wildfire Burn Severity Prediction Pipeline
==========================================
Complete PyTorch training pipeline for predicting burn severity from satellite and terrain data.

Features:
- Multi-model support (MLP, ResNet-style, Attention-based)
- Automatic feature engineering
- Class imbalance handling
- Stratified cross-validation
- Comprehensive metrics and logging
- Model checkpointing
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

class WildfireDataset(Dataset):
    """PyTorch Dataset for wildfire burn severity data"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class WildfireDataLoader:
    """Handle data loading, cleaning, and preprocessing"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self.df = None
        self.feature_names = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_and_clean(self, drop_missing: bool = True) -> pd.DataFrame:
        """Load CSV and handle missing data"""
        print("Loading data...")
        self.df = pd.read_csv(self.csv_path)
        print(f"Initial shape: {self.df.shape}")
        
        if drop_missing:
            # Drop rows with missing spectral data
            initial_len = len(self.df)
            self.df = self.df.dropna(subset=['blueBand'])
            dropped = initial_len - len(self.df)
            print(f"Dropped {dropped} rows with missing spectral data ({dropped/initial_len*100:.1f}%)")
            print(f"Final shape: {self.df.shape}")
        
        # Handle 'mod' vs 'moderate' inconsistency
        self.df['SBS'] = self.df['SBS'].replace({'mod': 'moderate'})
        
        return self.df
    
    def get_feature_groups(self) -> Dict[str, List[str]]:
        """Categorize features into logical groups"""
        all_cols = self.df.columns.tolist()
        
        groups = {
            'spectral': ['blueBand', 'greenBand', 'redBand', 'nirBand', 'swir1Band', 'swir2Band'],
            'indices': ['ndvi', 'ndbi', 'mndwi', 'nbr', 'bsi'],
            'diff_indices': ['dndvi', 'dndbi', 'dmndwi', 'dnbr', 'dbsi'],
            'terrain': [col for col in all_cols if any(x in col for x in 
                       ['aspct_', 'crosc_', 'elev', 'slope', 'tpi', 'tri', 'twi', 'spi'])],
            'climate': [col for col in all_cols if col.startswith('wc_')],
            'metadata': ['Fire_year', 'PointX', 'PointY', 'latitude', 'longitude', 'SBS', 'Source', 'system:index']
        }
        
        return groups
    
    def prepare_features(self, feature_groups: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare feature matrix and labels
        
        Args:
            feature_groups: List of group names to include. If None, use all except metadata.
                          Options: ['spectral', 'indices', 'diff_indices', 'terrain', 'climate']
        """
        if feature_groups is None:
            feature_groups = ['spectral', 'indices', 'diff_indices', 'terrain', 'climate']
        
        groups = self.get_feature_groups()
        
        # Collect feature columns
        feature_cols = []
        for group in feature_groups:
            if group in groups:
                feature_cols.extend(groups[group])
        
        # Ensure all columns exist
        feature_cols = [col for col in feature_cols if col in self.df.columns]
        self.feature_names = feature_cols
        
        print(f"\n{'='*60}")
        print("FEATURE SELECTION")
        print(f"{'='*60}")
        for group in feature_groups:
            if group in groups:
                cols = [col for col in groups[group] if col in self.df.columns]
                print(f"{group:15s}: {len(cols):3d} features")
        print(f"{'='*60}")
        print(f"Total features: {len(feature_cols)}")
        
        # Extract features and labels
        X = self.df[feature_cols].values
        y = self.label_encoder.fit_transform(self.df['SBS'].values)
        
        # Handle any remaining NaNs
        nan_mask = np.isnan(X).any(axis=1)
        if nan_mask.sum() > 0:
            print(f"Warning: Dropping {nan_mask.sum()} rows with NaN values in selected features")
            X = X[~nan_mask]
            y = y[~nan_mask]
        
        print(f"Final dataset shape: {X.shape}")
        print(f"\nClass distribution:")
        for idx, label in enumerate(self.label_encoder.classes_):
            count = (y == idx).sum()
            print(f"  {label:10s}: {count:4d} ({count/len(y)*100:5.1f}%)")
        
        return X, y, feature_cols
    
    def split_data(self, X: np.ndarray, y: np.ndarray, 
                   test_size: float = 0.2, val_size: float = 0.1,
                   random_state: int = 42) -> Dict:
        """Split data into train/val/test sets with stratification"""
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=random_state
        )
        
        # Second split: train vs val
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
        )
        
        # Fit scaler on training data only
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)
        
        print(f"\n{'='*60}")
        print("DATA SPLIT")
        print(f"{'='*60}")
        print(f"Train set: {X_train.shape[0]:4d} samples ({X_train.shape[0]/len(X)*100:5.1f}%)")
        print(f"Val set:   {X_val.shape[0]:4d} samples ({X_val.shape[0]/len(X)*100:5.1f}%)")
        print(f"Test set:  {X_test.shape[0]:4d} samples ({X_test.shape[0]/len(X)*100:5.1f}%)")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }


# ============================================================================
# 2. MODEL ARCHITECTURES
# ============================================================================

class MLPClassifier(nn.Module):
    """Multi-Layer Perceptron with dropout and batch norm"""
    
    def __init__(self, input_dim: int, num_classes: int, 
                 hidden_dims: List[int] = [256, 128, 64],
                 dropout: float = 0.3):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class ResidualBlock(nn.Module):
    """Residual block for tabular data"""
    
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.dropout(out)
        out = self.bn2(self.fc2(out))
        out += identity  # Residual connection
        out = F.relu(out)
        return out


class ResNetClassifier(nn.Module):
    """ResNet-style architecture for tabular data"""
    
    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dim: int = 256, num_blocks: int = 3,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU()
        )
        
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)
        ])
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        return self.classifier(x)


class AttentionClassifier(nn.Module):
    """Feature attention-based classifier"""
    
    def __init__(self, input_dim: int, num_classes: int,
                 hidden_dim: int = 256, num_heads: int = 4,
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Self-attention on features
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        self.classifier = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        # Project input
        x = self.input_proj(x)
        x = x.unsqueeze(1)  # Add sequence dimension for attention
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        
        x = x.squeeze(1)  # Remove sequence dimension
        return self.classifier(x)


# ============================================================================
# 3. TRAINING FRAMEWORK
# ============================================================================

class Trainer:
    """Handle model training, validation, and evaluation"""
    
    def __init__(self, model: nn.Module, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
    def train_epoch(self, dataloader: DataLoader, optimizer, criterion) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for features, labels in dataloader:
            features, labels = features.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        
        return total_loss / len(dataloader), 100. * correct / total
    
    def validate(self, dataloader: DataLoader, criterion) -> Tuple[float, float]:
        """Validate on validation set"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for features, labels in dataloader:
                features, labels = features.to(self.device), labels.to(self.device)
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        return total_loss / len(dataloader), 100. * correct / total
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader,
            epochs: int = 50, lr: float = 0.001, weight_decay: float = 1e-4,
            patience: int = 10, class_weights: Optional[torch.Tensor] = None):
        """Full training loop with early stopping"""
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        else:
            criterion = nn.CrossEntropyLoss()
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        best_val_acc = 0
        patience_counter = 0
        
        print(f"\n{'='*60}")
        print("TRAINING")
        print(f"{'='*60}")
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            scheduler.step(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'models/best_model.pth')
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% | "
                      f"Best: {best_val_acc:.2f}%")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('models/best_model.pth'))
        print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")
    
    def predict(self, dataloader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions"""
        self.model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in dataloader:
                features = features.to(self.device)
                outputs = self.model(features)
                _, predicted = outputs.max(1)
                all_preds.append(predicted.cpu().numpy())
                all_labels.append(labels.numpy())
        
        return np.concatenate(all_preds), np.concatenate(all_labels)
    
    def evaluate(self, dataloader: DataLoader, label_encoder: LabelEncoder) -> Dict:
        """Comprehensive evaluation"""
        y_pred, y_true = self.predict(dataloader)
        
        # Classification report
        report = classification_report(
            y_true, y_pred,
            target_names=label_encoder.classes_,
            output_dict=True
        )
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        return {
            'predictions': y_pred,
            'true_labels': y_true,
            'report': report,
            'confusion_matrix': cm,
            'accuracy': report['accuracy'],
            'macro_f1': report['macro avg']['f1-score'],
            'weighted_f1': report['weighted avg']['f1-score']
        }


# ============================================================================
# 4. UTILITY FUNCTIONS
# ============================================================================

def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    """Compute class weights for imbalanced data"""
    unique, counts = np.unique(y, return_counts=True)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(unique)  # Normalize
    return torch.FloatTensor(weights)


def plot_training_history(history: Dict, save_path: str = 'eval/training_history.png'):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Accuracy
    ax2.plot(history['train_acc'], label='Train Acc', linewidth=2)
    ax2.plot(history['val_acc'], label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history saved to {save_path}")


def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         save_path: str = 'eval/confusion_matrix.png'):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")


def save_results(results: Dict, label_encoder: LabelEncoder, 
                save_path: str = 'results.json'):
    """Save evaluation results to JSON"""
    # Convert numpy arrays to lists for JSON serialization
    results_copy = results.copy()
    results_copy['predictions'] = results['predictions'].tolist()
    results_copy['true_labels'] = results['true_labels'].tolist()
    results_copy['confusion_matrix'] = results['confusion_matrix'].tolist()
    results_copy['class_names'] = label_encoder.classes_.tolist()
    
    with open(save_path, 'w') as f:
        json.dump(results_copy, f, indent=2)
    print(f"Results saved to {save_path}")


# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================

def main():
    """Main training pipeline"""
    
    # Configuration
    CONFIG = {
        'csv_path': '../data/all_fires_complete_covariates.csv
',
        'model_type': 'mlp',  # Options: 'mlp', 'resnet', 'attention'
        'feature_groups': ['spectral', 'indices', 'diff_indices', 'terrain', 'climate'],
        'batch_size': 64,
        'epochs': 100,
        'lr': 0.001,
        'patience': 15,
        'random_state': 42
    }
    
    print("="*70)
    print("WILDFIRE BURN SEVERITY PREDICTION - PYTORCH PIPELINE")
    print("="*70)
    
    # Set random seeds
    torch.manual_seed(CONFIG['random_state'])
    np.random.seed(CONFIG['random_state'])
    
    # Load and prepare data
    data_loader = WildfireDataLoader(CONFIG['csv_path'])
    data_loader.load_and_clean(drop_missing=True)
    
    X, y, feature_names = data_loader.prepare_features(CONFIG['feature_groups'])
    splits = data_loader.split_data(X, y)
    
    # Create datasets
    train_dataset = WildfireDataset(splits['X_train'], splits['y_train'])
    val_dataset = WildfireDataset(splits['X_val'], splits['y_val'])
    test_dataset = WildfireDataset(splits['X_test'], splits['y_test'])
    
    # Compute class weights for imbalanced data
    class_weights = compute_class_weights(splits['y_train'])
    print(f"\nClass weights: {class_weights.numpy()}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'])
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'])
    
    # Initialize model
    input_dim = X.shape[1]
    num_classes = len(np.unique(y))
    
    print(f"\n{'='*60}")
    print("MODEL ARCHITECTURE")
    print(f"{'='*60}")
    
    if CONFIG['model_type'] == 'mlp':
        model = MLPClassifier(input_dim, num_classes, hidden_dims=[256, 128, 64])
        print("Using: Multi-Layer Perceptron (MLP)")
    elif CONFIG['model_type'] == 'resnet':
        model = ResNetClassifier(input_dim, num_classes, hidden_dim=256, num_blocks=3)
        print("Using: ResNet-style architecture")
    elif CONFIG['model_type'] == 'attention':
        model = AttentionClassifier(input_dim, num_classes, hidden_dim=256, num_heads=4)
        print("Using: Attention-based architecture")
    else:
        raise ValueError(f"Unknown model type: {CONFIG['model_type']}")
    
    print(f"Input dimension: {input_dim}")
    print(f"Number of classes: {num_classes}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    trainer = Trainer(model)
    trainer.fit(
        train_loader, val_loader,
        epochs=CONFIG['epochs'],
        lr=CONFIG['lr'],
        patience=CONFIG['patience'],
        class_weights=class_weights
    )
    
    # Evaluate on test set
    print(f"\n{'='*60}")
    print("TEST SET EVALUATION")
    print(f"{'='*60}")
    
    results = trainer.evaluate(test_loader, data_loader.label_encoder)
    
    print("\nClassification Report:")
    print("-" * 60)
    for class_name in data_loader.label_encoder.classes_:
        metrics = results['report'][class_name]
        print(f"{class_name:10s} | Precision: {metrics['precision']:.3f} | "
              f"Recall: {metrics['recall']:.3f} | F1: {metrics['f1-score']:.3f}")
    
    print("-" * 60)
    print(f"{'Overall':10s} | Accuracy: {results['accuracy']:.3f} | "
          f"Macro F1: {results['macro_f1']:.3f} | Weighted F1: {results['weighted_f1']:.3f}")
    
    # Save outputs
    plot_training_history(trainer.history, 'eval/training_history.png')
    plot_confusion_matrix(results['confusion_matrix'], 
                         data_loader.label_encoder.classes_,
                         'eval/confusion_matrix.png')
    save_results(results, data_loader.label_encoder, 'results.json')
    
    print(f"\n{'='*60}")
    print("PIPELINE COMPLETE")
    print(f"{'='*60}")
    print("Generated files:")
    print("  - best_model.pth (trained model weights)")
    print("  - training_history.png (training curves)")
    print("  - confusion_matrix.png (confusion matrix)")
    print("  - results.json (detailed metrics)")


if __name__ == '__main__':
    main()