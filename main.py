#!/usr/bin/env python3
"""
History-Based Data Preprocessing for Breast Cancer Risk Prediction

This system selects data based on patient history length:
- HISTORY = 0: Only latest year for each patient
- HISTORY = 1: Latest 2 years for patients with ≥2 years data
- HISTORY = 2: Latest 3 years for patients with ≥3 years data
- etc.

Automatically runs experiments for HISTORY 0,1,2,3,4 and compares results.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class HistoryConfig:
    """Configuration for history-based experiments"""
    CSV_PATH = '/kaggle/input/updatedcsv-having-pngs/CSAW-CC_breast_cancer_screening_data_backup.csv'
    IMAGE_DIR = '/kaggle/input/batch-3-pngs/PNGs'
    
    # Model settings
    IMAGE_SIZE = (512, 384)
    FEATURE_DIM = 256
    BATCH_SIZE = 8
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10  # Reduced for multiple experiments
    
    # History experiments to run
    HISTORY_VALUES = [0, 1, 2, 3, 4]
    
    # Splits
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def analyze_patient_history(df):
    """Analyze patient history patterns in the dataset"""
    
    print("ANALYZING PATIENT HISTORY PATTERNS")
    print("=" * 60)
    
    # Get patient history lengths
    patient_history = df.groupby('anon_patientid')['exam_year'].agg(['count', 'nunique', 'min', 'max']).reset_index()
    patient_history['years_span'] = patient_history['max'] - patient_history['min'] + 1
    patient_history['years_with_data'] = patient_history['nunique']
    
    print("1. OVERALL STATISTICS:")
    print(f"   Total patients: {len(patient_history):,}")
    print(f"   Total records: {len(df):,}")
    print(f"   Records per patient (avg): {len(df) / len(patient_history):.1f}")
    
    print("\n2. HISTORY LENGTH DISTRIBUTION:")
    history_dist = patient_history['years_with_data'].value_counts().sort_index()
    for years, count in history_dist.items():
        pct = count / len(patient_history) * 100
        print(f"   {years} year(s): {count:,} patients ({pct:.1f}%)")
    
    print("\n3. HISTORY AVAILABILITY FOR EXPERIMENTS:")
    for history in HistoryConfig.HISTORY_VALUES:
        required_years = history + 1
        eligible_patients = (patient_history['years_with_data'] >= required_years).sum()
        pct = eligible_patients / len(patient_history) * 100
        print(f"   HISTORY={history} (≥{required_years} years): {eligible_patients:,} patients ({pct:.1f}%)")
    
    return patient_history

def select_data_by_history(df, history_value):
    """
    Select data based on history value
    
    Args:
        df: Full dataset
        history_value: Number of additional years (0 = latest only, 1 = latest 2 years, etc.)
    
    Returns:
        Filtered dataframe with selected records
    """
    
    required_years = history_value + 1
    print(f"\nSELECTING DATA FOR HISTORY={history_value} (requires ≥{required_years} years)")
    print("-" * 60)
    
    selected_records = []
    
    # Process each patient
    for patient_id, patient_data in df.groupby('anon_patientid'):
        # Get unique years for this patient
        unique_years = sorted(patient_data['exam_year'].unique())
        
        # Check if patient has enough history
        if len(unique_years) >= required_years:
            # Select the latest N years
            selected_years = unique_years[-required_years:]
            
            # Get all records for selected years
            patient_selected = patient_data[patient_data['exam_year'].isin(selected_years)]
            selected_records.append(patient_selected)
    
    if selected_records:
        selected_df = pd.concat(selected_records, ignore_index=True)
    else:
        selected_df = pd.DataFrame()
    
    print(f"Selected {len(selected_df):,} records from {selected_df['anon_patientid'].nunique():,} patients")
    
    # Analyze cancer distribution
    cancer_records = selected_df[selected_df['x_case'] == 1]
    cancer_patients = cancer_records['anon_patientid'].nunique()
    
    print(f"Cancer records: {len(cancer_records):,}")
    print(f"Cancer patients: {cancer_patients:,}")
    print(f"Cancer rate (records): {len(cancer_records)/len(selected_df):.3f}")
    print(f"Cancer rate (patients): {cancer_patients/selected_df['anon_patientid'].nunique():.3f}")
    
    return selected_df

def create_temporal_pairs(df, history_value):
    """
    Create prior-current pairs based on history value
    
    For HISTORY=0: No pairs (single time point)
    For HISTORY≥1: Create pairs using temporal relationship
    """
    
    if history_value == 0:
        print("HISTORY=0: Creating single-timepoint data (no temporal pairs)")
        
        # For single timepoint, we use the current exam as both "prior" and "current"
        # This allows the same model architecture to work
        single_point_data = []
        
        for _, exam in df.iterrows():
            record = {
                'patient_id': exam['anon_patientid'],
                'current_filename': exam['anon_filename'],
                'current_year': exam['exam_year'],
                'current_age': exam['x_age'],
                'current_percent_density': exam['libra_percentdensity'],
                'current_breast_area': exam['libra_breastarea'],
                'prior_filename': exam['anon_filename'],  # Same as current
                'prior_year': exam['exam_year'],
                'prior_percent_density': exam['libra_percentdensity'],
                'prior_breast_area': exam['libra_breastarea'],
                'time_between_exams': 0,  # No time difference
                'density_change': 0,      # No change
                'area_change': 0,         # No change
                'event': exam['x_case'],  # Use current cancer status
                'time_to_event': 0 if exam['x_case'] == 1 else 5,  # Immediate or max follow-up
                'is_cancer_exam': exam['x_case'] == 1
            }
            single_point_data.append(record)
        
        pairs_df = pd.DataFrame(single_point_data)
        
    else:
        print(f"HISTORY={history_value}: Creating temporal pairs")
        
        pairs = []
        
        for patient_id, patient_data in df.groupby('anon_patientid'):
            patient_data = patient_data.sort_values(['exam_year', 'anon_filename'])
            
            # Get unique exam years
            exam_years = sorted(patient_data['exam_year'].unique())
            
            # Create pairs for consecutive years
            for i in range(len(exam_years) - 1):
                prior_year = exam_years[i]
                current_year = exam_years[i + 1]
                
                # Get exams for each year (take first view for simplicity)
                prior_exams = patient_data[patient_data['exam_year'] == prior_year]
                current_exams = patient_data[patient_data['exam_year'] == current_year]
                
                if len(prior_exams) > 0 and len(current_exams) > 0:
                    prior_exam = prior_exams.iloc[0]
                    current_exam = current_exams.iloc[0]
                    
                    # Calculate changes
                    density_change = current_exam['libra_percentdensity'] - prior_exam['libra_percentdensity']
                    area_change = current_exam['libra_breastarea'] - prior_exam['libra_breastarea']
                    time_between = current_exam['exam_year'] - prior_exam['exam_year']
                    
                    pair_record = {
                        'patient_id': patient_id,
                        'current_filename': current_exam['anon_filename'],
                        'current_year': current_exam['exam_year'],
                        'current_age': current_exam['x_age'],
                        'current_percent_density': current_exam['libra_percentdensity'],
                        'current_breast_area': current_exam['libra_breastarea'],
                        'prior_filename': prior_exam['anon_filename'],
                        'prior_year': prior_exam['exam_year'],
                        'prior_percent_density': prior_exam['libra_percentdensity'],
                        'prior_breast_area': prior_exam['libra_breastarea'],
                        'time_between_exams': time_between,
                        'density_change': density_change,
                        'area_change': area_change,
                        'event': current_exam['x_case'],
                        'time_to_event': 0 if current_exam['x_case'] == 1 else 5,
                        'is_cancer_exam': current_exam['x_case'] == 1
                    }
                    pairs.append(pair_record)
        
        pairs_df = pd.DataFrame(pairs)
    
    print(f"Created {len(pairs_df):,} pairs with {pairs_df['event'].sum():,} cancer events")
    return pairs_df

def create_risk_labels(pairs_df):
    """Create binary risk labels for different time horizons"""
    
    # For this simplified version, we'll use binary classification
    # 1-year risk is the most important
    pairs_df['cancer_within_1y'] = pairs_df['event'].astype(int)
    pairs_df['valid_for_1y'] = 1  # All samples are valid for 1-year prediction
    
    return pairs_df

# Simplified model for multiple experiments
class SimpleBreastCancerModel(nn.Module):
    """Simplified model for quick experiments across different history values"""
    
    def __init__(self, use_prior=True):
        super().__init__()
        self.use_prior = use_prior
        
        # Simple CNN backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Feature dimensions
        image_features = 128 * (2 if use_prior else 1)  # Prior + current or just current
        clinical_features = 6
        total_features = image_features + clinical_features
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, prior_images, current_images, clinical_features):
        # Extract features from current images
        current_features = self.backbone(current_images)
        current_features = current_features.view(current_features.size(0), -1)
        
        if self.use_prior:
            # Extract features from prior images
            prior_features = self.backbone(prior_images)
            prior_features = prior_features.view(prior_features.size(0), -1)
            
            # Combine features
            image_features = torch.cat([prior_features, current_features], dim=1)
        else:
            image_features = current_features
        
        # Combine with clinical features
        all_features = torch.cat([image_features, clinical_features], dim=1)
        
        # Predict
        output = self.classifier(all_features)
        return output.squeeze(1)

# Simplified dataset
class SimpleBreastCancerDataset(Dataset):
    def __init__(self, pairs_df, image_dir, mode='train'):
        self.pairs_df = pairs_df
        self.image_dir = Path(image_dir)
        self.mode = mode
    
    def __len__(self):
        return len(self.pairs_df)
    
    def load_mammogram(self, filename):
        # Simplified: return dummy image tensor for this example
        # In practice, you'd load and preprocess the actual images
        return torch.randn(1, 384, 512)  # Dummy mammogram
    
    def __getitem__(self, idx):
        row = self.pairs_df.iloc[idx]
        
        # Load images (dummy for this example)
        prior_image = self.load_mammogram(row['prior_filename'])
        current_image = self.load_mammogram(row['current_filename'])
        
        # Clinical features
        features = torch.tensor([
            (row['current_age'] - 50) / 15,
            row['density_change'] / 10,
            row['area_change'] / 1000,
            row['time_between_exams'],
            row['current_percent_density'] / 100,
            row['prior_percent_density'] / 100
        ], dtype=torch.float32)
        
        # Target
        target = torch.tensor(row['cancer_within_1y'], dtype=torch.float32)
        
        return {
            'prior_image': prior_image,
            'current_image': current_image,
            'features': features,
            'target': target,
            'patient_id': row['patient_id']
        }

def train_simple_model(model, train_loader, val_loader, history_value):
    """Train model for one history value"""
    
    print(f"\nTraining model for HISTORY={history_value}")
    print("-" * 40)
    
    optimizer = optim.Adam(model.parameters(), lr=HistoryConfig.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    
    best_auc = 0.0
    results = {'train_losses': [], 'val_aucs': []}
    
    for epoch in range(HistoryConfig.NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            prior_images = batch['prior_image'].to(HistoryConfig.DEVICE)
            current_images = batch['current_image'].to(HistoryConfig.DEVICE)
            features = batch['features'].to(HistoryConfig.DEVICE)
            targets = batch['target'].to(HistoryConfig.DEVICE)
            
            optimizer.zero_grad()
            
            predictions = model(prior_images, current_images, features)
            loss = criterion(predictions, targets)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        results['train_losses'].append(train_loss)
        
        # Validation
        model.eval()
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                prior_images = batch['prior_image'].to(HistoryConfig.DEVICE)
                current_images = batch['current_image'].to(HistoryConfig.DEVICE)
                features = batch['features'].to(HistoryConfig.DEVICE)
                targets = batch['target'].to(HistoryConfig.DEVICE)
                
                predictions = model(prior_images, current_images, features)
                
                val_predictions.extend(torch.sigmoid(predictions).cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
        
        # Calculate AUC
        if len(set(val_targets)) > 1:  # Check if we have both classes
            val_auc = roc_auc_score(val_targets, val_predictions)
        else:
            val_auc = 0.5
        
        results['val_aucs'].append(val_auc)
        
        if val_auc > best_auc:
            best_auc = val_auc
        
        print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, AUC={val_auc:.4f}")
    
    print(f"Best AUC for HISTORY={history_value}: {best_auc:.4f}")
    return best_auc, results

def run_history_experiments(df):
    """Run experiments for all history values and compare results"""
    
    print("\n" + "="*80)
    print("RUNNING HISTORY-BASED EXPERIMENTS")
    print("="*80)
    
    all_results = {}
    
    for history_value in HistoryConfig.HISTORY_VALUES:
        print(f"\n{'='*20} HISTORY = {history_value} {'='*20}")
        
        # Select data for this history value
        selected_df = select_data_by_history(df, history_value)
        
        if len(selected_df) < 100:  # Skip if too few samples
            print(f"Skipping HISTORY={history_value}: Too few samples ({len(selected_df)})")
            continue
        
        # Create pairs
        pairs_df = create_temporal_pairs(selected_df, history_value)
        pairs_df = create_risk_labels(pairs_df)
        
        if pairs_df['event'].sum() < 10:  # Skip if too few cancer cases
            print(f"Skipping HISTORY={history_value}: Too few cancer cases ({pairs_df['event'].sum()})")
            continue
        
        # Split data by patient
        unique_patients = pairs_df['patient_id'].unique()
        np.random.seed(42)
        np.random.shuffle(unique_patients)
        
        n_patients = len(unique_patients)
        train_end = int(n_patients * HistoryConfig.TRAIN_RATIO)
        val_end = int(n_patients * (HistoryConfig.TRAIN_RATIO + HistoryConfig.VAL_RATIO))
        
        train_patients = unique_patients[:train_end]
        val_patients = unique_patients[train_end:val_end]
        test_patients = unique_patients[val_end:]
        
        train_df = pairs_df[pairs_df['patient_id'].isin(train_patients)]
        val_df = pairs_df[pairs_df['patient_id'].isin(val_patients)]
        test_df = pairs_df[pairs_df['patient_id'].isin(test_patients)]
        
        # Create datasets and loaders
        train_dataset = SimpleBreastCancerDataset(train_df, HistoryConfig.IMAGE_DIR, 'train')
        val_dataset = SimpleBreastCancerDataset(val_df, HistoryConfig.IMAGE_DIR, 'val')
        test_dataset = SimpleBreastCancerDataset(test_df, HistoryConfig.IMAGE_DIR, 'test')
        
        train_loader = DataLoader(train_dataset, batch_size=HistoryConfig.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=HistoryConfig.BATCH_SIZE, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=HistoryConfig.BATCH_SIZE, shuffle=False)
        
        # Create model (use prior images for HISTORY > 0)
        use_prior = history_value > 0
        model = SimpleBreastCancerModel(use_prior=use_prior).to(HistoryConfig.DEVICE)
        
        # Train model
        best_auc, training_results = train_simple_model(model, train_loader, val_loader, history_value)
        
        # Test evaluation
        model.eval()
        test_predictions = []
        test_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                prior_images = batch['prior_image'].to(HistoryConfig.DEVICE)
                current_images = batch['current_image'].to(HistoryConfig.DEVICE)
                features = batch['features'].to(HistoryConfig.DEVICE)
                targets = batch['target'].to(HistoryConfig.DEVICE)
                
                predictions = model(prior_images, current_images, features)
                
                test_predictions.extend(torch.sigmoid(predictions).cpu().numpy())
                test_targets.extend(targets.cpu().numpy())
        
        # Calculate test AUC
        if len(set(test_targets)) > 1:
            test_auc = roc_auc_score(test_targets, test_predictions)
        else:
            test_auc = 0.5
        
        # Store results
        all_results[history_value] = {
            'n_patients': len(unique_patients),
            'n_pairs': len(pairs_df),
            'n_cancer': pairs_df['event'].sum(),
            'cancer_rate': pairs_df['event'].mean(),
            'best_val_auc': best_auc,
            'test_auc': test_auc,
            'training_results': training_results
        }
        
        print(f"Final test AUC for HISTORY={history_value}: {test_auc:.4f}")
    
    return all_results

def create_comparison_plots(all_results):
    """Create comparison plots for different history values"""
    
    if not all_results:
        print("No results to plot")
        return
    
    history_values = list(all_results.keys())
    test_aucs = [all_results[h]['test_auc'] for h in history_values]
    val_aucs = [all_results[h]['best_val_auc'] for h in history_values]
    n_patients = [all_results[h]['n_patients'] for h in history_values]
    cancer_rates = [all_results[h]['cancer_rate'] for h in history_values]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # AUC comparison
    axes[0, 0].bar(history_values, test_aucs, alpha=0.7, color='skyblue', label='Test AUC')
    axes[0, 0].bar(history_values, val_aucs, alpha=0.7, color='lightcoral', label='Best Val AUC')
    axes[0, 0].set_xlabel('History Value')
    axes[0, 0].set_ylabel('AUC')
    axes[0, 0].set_title('Model Performance by History Value')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Sample size
    axes[0, 1].bar(history_values, n_patients, alpha=0.7, color='lightgreen')
    axes[0, 1].set_xlabel('History Value')
    axes[0, 1].set_ylabel('Number of Patients')
    axes[0, 1].set_title('Sample Size by History Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Cancer rate
    axes[1, 0].bar(history_values, cancer_rates, alpha=0.7, color='orange')
    axes[1, 0].set_xlabel('History Value')
    axes[1, 0].set_ylabel('Cancer Rate')
    axes[1, 0].set_title('Cancer Rate by History Value')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning curves for first few history values
    for i, h in enumerate(history_values[:3]):  # Show first 3
        training_results = all_results[h]['training_results']
        axes[1, 1].plot(training_results['val_aucs'], label=f'HISTORY={h}', linewidth=2)
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation AUC')
    axes[1, 1].set_title('Learning Curves')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_results_table(all_results):
    """Create a comprehensive results table"""
    
    if not all_results:
        print("No results to display")
        return
    
    print("\n" + "="*100)
    print("HISTORY-BASED EXPERIMENT RESULTS SUMMARY")
    print("="*100)
    
    # Create results table
    table_data = []
    for history_value, results in all_results.items():
        row = {
            'History': history_value,
            'Required Years': history_value + 1,
            'Patients': f"{results['n_patients']:,}",
            'Pairs': f"{results['n_pairs']:,}",
            'Cancer Cases': f"{results['n_cancer']:,}",
            'Cancer Rate': f"{results['cancer_rate']:.3f}",
            'Best Val AUC': f"{results['best_val_auc']:.4f}",
            'Test AUC': f"{results['test_auc']:.4f}"
        }
        table_data.append(row)
    
    results_df = pd.DataFrame(table_data)
    print(results_df.to_string(index=False))
    
    # Find best performing history value
    best_history = max(all_results.keys(), key=lambda h: all_results[h]['test_auc'])
    best_auc = all_results[best_history]['test_auc']
    
    print(f"\n🏆 BEST PERFORMING CONFIGURATION:")
    print(f"   HISTORY = {best_history} (requires ≥{best_history + 1} years)")
    print(f"   Test AUC = {best_auc:.4f}")
    print(f"   Patients = {all_results[best_history]['n_patients']:,}")
    print(f"   Cancer Rate = {all_results[best_history]['cancer_rate']:.3f}")

def main_history_experiments():
    """Main function to run all history experiments"""
    
    print("HISTORY-BASED BREAST CANCER PREDICTION EXPERIMENTS")
    print("="*80)
    
    # Load data
    print("Loading dataset...")
    df = pd.read_csv(HistoryConfig.CSV_PATH)
    
    # Analyze patient history patterns
    patient_history = analyze_patient_history(df)
    
    # Run experiments for all history values
    all_results = run_history_experiments(df)
    
    # Create visualizations and summary
    create_comparison_plots(all_results)
    create_results_table(all_results)
    
    return all_results

if __name__ == "__main__":
    """Run the complete history-based experiment suite"""
    
    try:
        results = main_history_experiments()
        print("\n All history experiments completed successfully!")
        
    except Exception as e:
        print(f" Error in history experiments: {e}")
        import traceback
        traceback.print_exc()
