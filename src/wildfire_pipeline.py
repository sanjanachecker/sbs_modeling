"""
High-Severity Focused Wildfire Pipeline
========================================
Diagnostic approach: Train ONLY on fires with abundant 'high' severity samples
to determine if the model CAN learn to predict 'high' class when given enough data.

Strategy:
- Train on: Fires with most 'high' samples (top 10 fires)
- Test on: Other fires
- No class weight boosting (to see raw performance)

This will answer: "Is the problem insufficient data or feature separability?"

Author: RajC
Date: December 2025
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Central configuration"""
    
    # File paths
    COMPLETE_DATA_PATH = 'data/all_fires_complete_covariates_fixed_129.csv'
    UPSAMPLED_DATA_PATH = 'data/all_fires_upsampled_points_with_covariates_fixed_v2.csv'
    OUTPUT_DIR = Path('results_high_severity_focused')
    
    # Top features
    TOP_FEATURES = [
        'dnbr', 'wc_bio06', 'vd_6', 'wc_bio18', 'wc_bio10',
        'greenBand', 'planc_32', 'crosc_16', 'swir2Band', 'tpi_2',
        'planc_8', 'minc_4', 'mbi_1', 'nbr', 'devmeanelev_4',
        'redBand', 'aspct_2', 'vrm_2', 'planc_4', 'relmeanelev_32',
        'pisrdir_2021-06-22', 'planc_2', 'nirBand', 'pisrdir_2021-04-22',
        'diffmeanelev_2', 'relmeanelev_4', 'relelev_16', 'profc_16',
        'wc_bio07', 'minc_2', 'dah', 'msp', 'diffmeanelev_4',
        'aspct_32', 'devmeanelev_32', 'wc_bio02', 'profc_4', 'hn',
        'po_2', 'rdgh_6', 'tsc_16', 'devmeanelev_8', 'relmeanelev_16',
        'pisrdir_2021-03-22', 'devmeanelev_16', 'mbi_001', 'longc_8',
        'relmeanelev_8', 'pisrdir_2021-12-22', 'wc_bio08'
    ]
    
    # Fires with most 'high' severity samples (for training)
    TRAIN_FIRES = [
        'dixie_2021',      # 67 high samples
        'caldor_2021',     # 57 high samples
        'creek_2020',      # 51 high samples
        'line_2024',       # 40 high samples
        'rim_2013',        # 31 high samples
        'oak_2022',        # 25 high samples
        'sugar_2021',      # 19 high samples
        'apple_2020',      # 18 high samples
        'eldorado_2020',   # 18 high samples
        'donnell_2018'     # 17 high samples
    ]
    
    # Training parameters (NO CLASS WEIGHT BOOSTING)
    BATCH_SIZE = 64
    LEARNING_RATE = 0.0005
    EPOCHS = 150
    PATIENCE = 30
    WEIGHT_DECAY = 1e-4
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __init__(self):
        self.OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_and_prepare_data(config: Config):
    """Load data and split by high-severity fire presence"""
    
    print("="*80)
    print("HIGH-SEVERITY FOCUSED PIPELINE")
    print("="*80)
    print("\nStrategy: Train on fires with abundant 'high' samples")
    print("          Test on other fires to see if model generalizes")
    print("          Add upsampling for UNBURNED class only")
    
    # Load complete data
    df_complete = pd.read_csv(config.COMPLETE_DATA_PATH)
    df_complete['SBS'] = df_complete['SBS'].replace({'mod': 'moderate'})
    
    # Load upsampled data
    df_upsampled = pd.read_csv(config.UPSAMPLED_DATA_PATH)
    df_upsampled['SBS'] = df_upsampled['SBS'].replace({'mod': 'moderate'})
    
    print(f"\nComplete dataset: {df_complete.shape}")
    print(f"Upsampled dataset: {df_upsampled.shape}")
    
    # Calculate upsampling quota for UNBURNED class only
    print("\n" + "="*80)
    print("CALCULATING UPSAMPLING QUOTAS (UNBURNED CLASS ONLY)")
    print("="*80)
    
    quotas = {}
    for fire in df_complete['Fire_year'].unique():
        fire_data = df_complete[df_complete['Fire_year'] == fire]
        moderate_count = (fire_data['SBS'] == 'moderate').sum()
        
        if moderate_count > 0:
            quota = int(np.sqrt(moderate_count))  # Quota = sqrt(moderate_count)
            quotas[fire] = quota
            print(f"{fire:30s}: {moderate_count:4d} moderate → sample {quota:3d} unburned upsampled points")
    
    # Select upsampled points
    print("\n" + "="*80)
    print("SELECTING UPSAMPLED POINTS (UNBURNED ONLY)")
    print("="*80)
    
    selected_dfs = []
    total_upsampled = 0
    
    for fire, quota in quotas.items():
        # Get upsampled UNBURNED points for this fire only
        fire_upsampled = df_upsampled[
            (df_upsampled['Fire_year'] == fire) & 
            (df_upsampled['SBS'] == 'unburned')
        ]
        
        if len(fire_upsampled) == 0:
            print(f"{fire:30s}: No upsampled unburned points available")
            continue
        
        # Sample the quota
        if len(fire_upsampled) < quota:
            sampled = fire_upsampled
            print(f"{fire:30s}: Only {len(fire_upsampled)} available (wanted {quota})")
        else:
            sampled = fire_upsampled.sample(n=quota, random_state=RANDOM_STATE)
            print(f"{fire:30s}: Sampled {quota} unburned points")
        
        selected_dfs.append(sampled)
        total_upsampled += len(sampled)
    
    # Combine complete and upsampled data
    df_complete['data_source'] = 'complete'
    
    if selected_dfs:
        df_upsampled_selected = pd.concat(selected_dfs, ignore_index=True)
        df_upsampled_selected['data_source'] = 'upsampled'
        df = pd.concat([df_complete, df_upsampled_selected], ignore_index=True)
        print(f"\nTotal upsampled unburned points: {total_upsampled}")
    else:
        df = df_complete
        print(f"\nNo upsampled points selected")
    
    print(f"Combined dataset: {len(df)} samples")
    
    # Analyze high severity distribution
    print("\n" + "="*80)
    print("HIGH SEVERITY DISTRIBUTION BY FIRE (AFTER UPSAMPLING)")
    print("="*80)
    
    high_by_fire = df[df['SBS'] == 'high'].groupby('Fire_year').size().sort_values(ascending=False)
    print("\nTop 10 fires by 'high' severity count:")
    for fire, count in high_by_fire.head(10).items():
        total = (df['Fire_year'] == fire).sum()
        pct = count / total * 100
        marker = " ← TRAINING" if fire in config.TRAIN_FIRES else ""
        print(f"  {fire:30s}: {count:3d} high / {total:4d} total ({pct:5.1f}%){marker}")
    
    # Split into train and test fires
    train_fires = config.TRAIN_FIRES
    test_fires = [f for f in df['Fire_year'].unique() if f not in train_fires]
    
    print(f"\n" + "="*80)
    print("TRAIN/TEST SPLIT")
    print("="*80)
    print(f"Training fires: {len(train_fires)}")
    print(f"Testing fires: {len(test_fires)}")
    
    # Create masks
    train_mask = df['Fire_year'].isin(train_fires)
    test_mask = df['Fire_year'].isin(test_fires)
    
    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()
    
    print(f"\nTraining set: {len(df_train)} samples")
    print("  Class distribution:")
    for cls, count in df_train['SBS'].value_counts().items():
        print(f"    {cls:12s}: {count:4d} ({count/len(df_train)*100:5.1f}%)")
    
    print(f"\nTest set: {len(df_test)} samples")
    print("  Class distribution:")
    for cls, count in df_test['SBS'].value_counts().items():
        print(f"    {cls:12s}: {count:4d} ({count/len(df_test)*100:5.1f}%)")
    
    return df_train, df_test


# ============================================================================
# IMPUTATION
# ============================================================================

def impute_missing_values(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """Simple imputation using IterativeImputer"""
    
    print("\n" + "="*80)
    print("IMPUTING MISSING VALUES")
    print("="*80)
    
    df_imputed = df.copy()
    
    # Check for missing values
    missing_cols = [col for col in feature_cols if col in df.columns and df[col].isnull().sum() > 0]
    
    if not missing_cols:
        print("No missing values to impute")
        return df_imputed
    
    print(f"Features with missing values: {len(missing_cols)}")
    
    # Use IterativeImputer
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=50, max_depth=10, random_state=RANDOM_STATE),
        max_iter=10,
        random_state=RANDOM_STATE,
        verbose=0
    )
    
    cols_to_impute = [col for col in feature_cols if col in df.columns]
    X = df[cols_to_impute].values
    X_imputed = imputer.fit_transform(X)
    
    for i, col in enumerate(cols_to_impute):
        df_imputed[col] = X_imputed[:, i]
    
    print(f"Imputation complete. Remaining missing: {df_imputed[feature_cols].isnull().sum().sum()}")
    
    return df_imputed


# ============================================================================
# MODEL
# ============================================================================

class ImprovedMLPClassifier(nn.Module):
    """Enhanced MLP classifier"""
    
    def __init__(self, input_dim: int, num_classes: int, 
                 hidden_dims: List[int] = [512, 256, 128, 64],
                 dropout: float = 0.3):
        super().__init__()
        
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU()
        )
        
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Sequential(
                nn.Linear(hidden_dims[i], hidden_dims[i+1]),
                nn.BatchNorm1d(hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ))
        
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)
    
    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x)


class WildfireDataset(Dataset):
    """PyTorch Dataset"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# ============================================================================
# TRAINING
# ============================================================================

def train_model(X_train, y_train, X_test, y_test, label_encoder, config):
    """Train and evaluate model"""
    
    print("\n" + "="*80)
    print("MODEL TRAINING")
    print("="*80)
    
    # Create datasets
    train_dataset = WildfireDataset(X_train, y_train)
    test_dataset = WildfireDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    
    # Create model
    model = ImprovedMLPClassifier(
        input_dim=X_train.shape[1],
        num_classes=len(label_encoder.classes_),
        hidden_dims=[512, 256, 128, 64],
        dropout=0.3
    ).to(config.DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # NO CLASS WEIGHT BOOSTING - using standard weights only
    class_counts = np.bincount(y_train)
    class_weights = torch.FloatTensor(len(class_counts) / (len(class_counts) * class_counts))
    
    print(f"\nStandard class weights (NO BOOSTING):")
    for idx, label in enumerate(label_encoder.classes_):
        print(f"  {label:12s}: {class_weights[idx].item():.4f}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(config.DEVICE))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, 
                                   weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                            factor=0.5, patience=7)
    
    # Training loop
    best_kappa = 0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'test_kappa': []}
    
    print(f"\nTraining for up to {config.EPOCHS} epochs...")
    
    for epoch in range(config.EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for features, labels in train_loader:
            features, labels = features.to(config.DEVICE), labels.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_correct += predicted.eq(labels).sum().item()
            train_total += labels.size(0)
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Evaluate on test
        model.eval()
        test_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = model(features)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_loss /= len(test_loader)
        test_acc = 100. * np.mean(np.array(all_preds) == np.array(all_labels))
        test_kappa = cohen_kappa_score(all_labels, all_preds)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['test_kappa'].append(test_kappa)
        
        scheduler.step(test_kappa)
        
        if test_kappa > best_kappa:
            best_kappa = test_kappa
            patience_counter = 0
            torch.save(model.state_dict(), config.OUTPUT_DIR / 'best_model.pth')
        else:
            patience_counter += 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{config.EPOCHS} | "
                  f"Train: L={train_loss:.4f} A={train_acc:.2f}% | "
                  f"Test: L={test_loss:.4f} A={test_acc:.2f}% κ={test_kappa:.4f} | "
                  f"Best κ={best_kappa:.4f}")
        
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load(config.OUTPUT_DIR / 'best_model.pth'))
    print(f"\nTraining complete. Best Kappa: {best_kappa:.4f}")
    
    return model, history


def evaluate_model(model, X_test, y_test, label_encoder, config):
    """Comprehensive evaluation"""
    
    print("\n" + "="*80)
    print("TEST SET EVALUATION")
    print("="*80)
    
    model.eval()
    test_dataset = WildfireDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(config.DEVICE)
            outputs = model(features)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Reports
    report = classification_report(all_labels, all_preds, 
                                   target_names=label_encoder.classes_,
                                   output_dict=True, zero_division=0)
    
    cm = confusion_matrix(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    
    print("\nClassification Report:")
    print("-" * 80)
    for class_name in label_encoder.classes_:
        if class_name in report:
            metrics = report[class_name]
            print(f"{class_name:12s} | Precision: {metrics['precision']:.3f} | "
                  f"Recall: {metrics['recall']:.3f} | F1: {metrics['f1-score']:.3f} | "
                  f"Support: {int(metrics['support'])}")
    
    print("-" * 80)
    print(f"{'Overall':12s} | Accuracy: {report['accuracy']:.3f} | "
          f"Macro F1: {report['macro avg']['f1-score']:.3f} | "
          f"Kappa: {kappa:.3f}")
    
    # Analyze 'high' class specifically
    print("\n" + "="*80)
    print("HIGH CLASS ANALYSIS")
    print("="*80)
    high_idx = label_encoder.transform(['high'])[0]
    high_metrics = report['high']
    
    print(f"High class performance:")
    print(f"  Precision: {high_metrics['precision']:.3f}")
    print(f"  Recall: {high_metrics['recall']:.3f}")
    print(f"  F1-Score: {high_metrics['f1-score']:.3f}")
    print(f"  Support: {int(high_metrics['support'])}")
    
    # How many times was 'high' predicted?
    high_predictions = (all_preds == high_idx).sum()
    print(f"\nTimes 'high' was predicted: {high_predictions} / {len(all_preds)} ({high_predictions/len(all_preds)*100:.1f}%)")
    
    return {
        'report': report,
        'confusion_matrix': cm,
        'kappa': kappa,
        'predictions': all_preds,
        'true_labels': all_labels
    }


def plot_results(history, cm, class_names, config):
    """Plot training history and confusion matrix"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Training curves
    ax = axes[0]
    ax.plot(history['train_acc'], label='Train Acc', linewidth=2)
    ax.plot(history['test_acc'], label='Test Acc', linewidth=2)
    ax.plot([k*100 for k in history['test_kappa']], label='Test Kappa×100', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Metric Value')
    ax.set_title('Training Progress')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Confusion matrix
    ax = axes[1]
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    im = ax.imshow(cm_norm, cmap='Blues')
    
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, f'{cm[i, j]}\n({cm_norm[i, j]:.1%})',
                          ha="center", va="center", color="white" if cm_norm[i, j] > 0.5 else "black",
                          fontsize=9)
    
    ax.set_title('Confusion Matrix')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(config.OUTPUT_DIR / 'results.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plots to {config.OUTPUT_DIR / 'results.png'}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Execute pipeline"""
    
    config = Config()
    
    print(f"Device: {config.DEVICE}")
    print(f"Output directory: {config.OUTPUT_DIR}\n")
    
    # Load and split data
    df_train, df_test = load_and_prepare_data(config)
    
    # Get features
    available_features = [f for f in config.TOP_FEATURES if f in df_train.columns]
    print(f"\nUsing {len(available_features)} features")
    
    # Impute
    df_train = impute_missing_values(df_train, available_features)
    df_test = impute_missing_values(df_test, available_features)
    
    # Prepare X, y
    X_train = df_train[available_features].values
    y_train_raw = df_train['SBS'].values
    
    X_test = df_test[available_features].values
    y_test_raw = df_test['SBS'].values
    
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(np.concatenate([y_train_raw, y_test_raw]))
    
    y_train = label_encoder.transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)
    
    print(f"\nClasses: {label_encoder.classes_}")
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Train
    model, history = train_model(X_train, y_train, X_test, y_test, label_encoder, config)
    
    # Evaluate
    results = evaluate_model(model, X_test, y_test, label_encoder, config)
    
    # Plot
    plot_results(history, results['confusion_matrix'], label_encoder.classes_, config)
    
    # Save results
    results_dict = {
        'accuracy': float(results['report']['accuracy']),
        'kappa': float(results['kappa']),
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'class_names': label_encoder.classes_.tolist(),
        'classification_report': results['report'],
        'train_fires': config.TRAIN_FIRES
    }
    
    with open(config.OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(results_dict, f, indent=2)
    
    print(f"\n✓ Saved results to {config.OUTPUT_DIR / 'results.json'}")
    
    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    print("\nKEY QUESTION: Did the model learn to predict 'high' class?")
    print(f"Answer: {'YES - Model CAN learn high class with enough data!' if results['report']['high']['recall'] > 0.15 else 'NO - Deeper feature separability issue exists'}")
    print("\nIf YES: Problem is insufficient 'high' samples in training")
    print("        → Solution: Aggressive upsampling or SMOTE")
    print("\nIf NO: Problem is feature separability")
    print("       → Solution: Better features or different architecture")


if __name__ == '__main__':
    main()