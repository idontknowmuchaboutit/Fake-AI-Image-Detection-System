"""
Model Training Script for Fake AI Image Detection
Author: Siddhant Mishra
University: Sage University Indore
Course: MCA Final Year Project

This script trains the CNN model to detect AI-generated images.
Target: 90-95% accuracy on test set
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_curve, auc, precision_recall_curve
)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
    TensorBoard, CSVLogger
)

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from models.architectures import FakeImageDetectorModels
from utils.data_loader import DataLoader


class ModelTrainer:
    """Handles model training and evaluation"""
    
    def __init__(self, config):
        """
        Initialize trainer
        
        Args:
            config: Dictionary containing training configuration
        """
        self.config = config
        self.model = None
        self.history = None
        self.results_dir = Path(config['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        
    def build_model(self):
        """Build and compile the model"""
        print("\n🏗️  Building model...")
        
        builder = FakeImageDetectorModels(
            input_shape=(self.config['img_size'], self.config['img_size'], 3),
            num_classes=2
        )
        
        self.model = builder.get_model(self.config['architecture'])
        
        # Compile model
        optimizer = keras.optimizers.Adam(learning_rate=self.config['learning_rate'])
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        print(f"✓ Model built: {self.config['architecture']}")
        print(f"✓ Total parameters: {self.model.count_params():,}")
        
        return self.model
    
    def prepare_data(self):
        """Prepare data loaders"""
        print("\n📊 Preparing data...")
        
        loader = DataLoader(
            data_dir=self.config['data_dir'],
            img_size=self.config['img_size'],
            batch_size=self.config['batch_size']
        )
        
        # Get dataset info
        info = loader.get_dataset_info()
        print("\nDataset split:")
        for split, counts in info.items():
            print(f"  {split}: {counts['total']} images "
                  f"(Real: {counts['real']}, Fake: {counts['fake']})")
        
        # Create generators
        train_gen, val_gen, test_gen = loader.create_data_generators(
            augment_train=self.config['augment']
        )
        
        return train_gen, val_gen, test_gen
    
    def get_callbacks(self):
        """Create training callbacks"""
        callbacks = []
        
        # Model checkpoint - save best model
        checkpoint_path = self.results_dir / 'best_model.keras'
        callbacks.append(ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ))
        
        # Early stopping
        callbacks.append(EarlyStopping(
            monitor='val_loss',
            patience=self.config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ))
        
        # Learning rate reduction
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ))
        
        # TensorBoard logging
        log_dir = self.results_dir / 'logs' / datetime.now().strftime("%Y%m%d-%H%M%S")
        callbacks.append(TensorBoard(log_dir=str(log_dir), histogram_freq=1))
        
        # CSV logger
        csv_path = self.results_dir / 'training_history.csv'
        callbacks.append(CSVLogger(str(csv_path)))
        
        return callbacks
    
    def train(self, train_gen, val_gen):
        """Train the model"""
        print("\n🚀 Starting training...")
        print("="*70)
        
        callbacks = self.get_callbacks()
        
        # Calculate steps
        steps_per_epoch = train_gen.n // train_gen.batch_size
        validation_steps = val_gen.n // val_gen.batch_size
        
        # Train model
        self.history = self.model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            epochs=self.config['epochs'],
            validation_data=val_gen,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n✅ Training completed!")
        
        return self.history
    
    def evaluate(self, test_gen):
        """Evaluate model on test set"""
        print("\n📈 Evaluating on test set...")
        
        # Load best model
        best_model_path = self.results_dir / 'best_model.keras'
        if best_model_path.exists():
            self.model = keras.models.load_model(str(best_model_path))
            print(f"✓ Loaded best model from {best_model_path}")
        
        # Evaluate
        test_steps = test_gen.n // test_gen.batch_size
        test_loss, test_acc, test_precision, test_recall, test_auc = self.model.evaluate(
            test_gen, steps=test_steps, verbose=1
        )
        
        # Calculate F1 score
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        
        # Get predictions for detailed metrics
        test_gen.reset()
        predictions = self.model.predict(test_gen, steps=test_steps, verbose=1)
        y_pred = np.argmax(predictions, axis=1)
        y_true = test_gen.classes[:len(y_pred)]
        
        results = {
            'test_loss': float(test_loss),
            'test_accuracy': float(test_acc),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'test_auc': float(test_auc),
            'test_f1_score': float(f1_score)
        }
        
        # Print results
        print("\n" + "="*70)
        print("TEST SET RESULTS:")
        print("="*70)
        print(f"Accuracy:  {test_acc*100:.2f}%")
        print(f"Precision: {test_precision*100:.2f}%")
        print(f"Recall:    {test_recall*100:.2f}%")
        print(f"F1-Score:  {f1_score*100:.2f}%")
        print(f"AUC:       {test_auc:.4f}")
        print(f"Loss:      {test_loss:.4f}")
        print("="*70)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, 
                                   target_names=['Real', 'Fake'],
                                   digits=4))
        
        # Save results
        results_file = self.results_dir / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n✓ Results saved to {results_file}")
        
        return results, y_true, y_pred, predictions
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Train')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Train')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        fig_path = self.results_dir / 'training_history.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training history plot saved to {fig_path}")
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'],
                   yticklabels=['Real', 'Fake'])
        plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save figure
        fig_path = self.results_dir / 'confusion_matrix.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to {fig_path}")
        plt.close()
    
    def plot_roc_curve(self, y_true, predictions):
        """Plot ROC curve"""
        # Get probabilities for positive class (fake)
        y_scores = predictions[:, 1]
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve',
                 fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # Save figure
        fig_path = self.results_dir / 'roc_curve.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to {fig_path}")
        plt.close()
    
    def save_model_summary(self):
        """Save model architecture summary"""
        summary_path = self.results_dir / 'model_summary.txt'
        with open(summary_path, 'w') as f:
            self.model.summary(print_fn=lambda x: f.write(x + '\n'))
        print(f"✓ Model summary saved to {summary_path}")
    
    def run_complete_training(self):
        """Run complete training pipeline"""
        print("\n" + "="*70)
        print("FAKE AI IMAGE DETECTION - MODEL TRAINING")
        print("University: Sage University Indore")
        print("="*70)
        
        # Build model
        self.build_model()
        
        # Prepare data
        train_gen, val_gen, test_gen = self.prepare_data()
        
        # Save model summary
        self.save_model_summary()
        
        # Train
        self.train(train_gen, val_gen)
        
        # Plot training history
        self.plot_training_history()
        
        # Evaluate
        results, y_true, y_pred, predictions = self.evaluate(test_gen)
        
        # Plot evaluation metrics
        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_roc_curve(y_true, predictions)
        
        # Save configuration
        config_path = self.results_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        print(f"✓ Configuration saved to {config_path}")
        
        print("\n" + "="*70)
        print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"\nAll results saved to: {self.results_dir}")
        print("\nIMPORTANT NOTE:")
        print("This model was trained on a specific dataset (real photos vs")
        print("Stable Diffusion/Midjourney images). Performance on images from")
        print("other AI generators or different domains may vary.")
        print("="*70)
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Train Fake AI Image Detector')
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Path to processed data directory')
    parser.add_argument('--architecture', type=str, default='custom',
                       choices=['custom', 'efficientnet', 'resnet', 'mobilenet', 'hybrid'],
                       help='Model architecture to use')
    parser.add_argument('--img_size', type=int, default=224,
                       help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Initial learning rate')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Training configuration
    config = {
        'data_dir': args.data_dir,
        'architecture': args.architecture,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'learning_rate': args.learning_rate,
        'augment': True,
        'early_stopping_patience': 10,
        'results_dir': args.results_dir
    }
    
    # Create trainer and run
    trainer = ModelTrainer(config)
    results = trainer.run_complete_training()


if __name__ == "__main__":
    main()
