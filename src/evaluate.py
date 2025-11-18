import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report,
                            roc_curve, auc, precision_recall_curve)
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    def __init__(self, model, test_generator):
        self.model = model
        self.test_generator = test_generator
        self.y_true = None
        self.y_pred = None
        self.y_pred_proba = None

    def evaluate(self):
        """Evaluate model on test data"""
        print(f"\n{'='*80}")
        print("🔍 Evaluating Model on Test Data...")
        print(f"{'='*80}\n")

        # Reset generator
        self.test_generator.reset()

        # Get predictions
        print("Generating predictions...")
        self.y_pred_proba = self.model.predict(self.test_generator, verbose=0)
        self.y_pred = (self.y_pred_proba > 0.5).astype(int).flatten()

        # Get true labels
        self.y_true = self.test_generator.classes

        # Calculate metrics
        print("Calculating metrics...")
        test_loss, test_acc, test_precision, test_recall, test_auc = \
            self.model.evaluate(self.test_generator, verbose=0)

        # Calculate F1 Score
        f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall)

        print(f"\n{'='*80}")
        print("📊 Test Results:")
        print(f"{'='*80}")
        print(f"  Loss:       {test_loss:.4f}")
        print(f"  Accuracy:   {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"  Precision:  {test_precision:.4f} ({test_precision*100:.2f}%)")
        print(f"  Recall:     {test_recall:.4f} ({test_recall*100:.2f}%)")
        print(f"  F1-Score:   {f1_score:.4f} ({f1_score*100:.2f}%)")
        print(f"  AUC:        {test_auc:.4f}")
        print(f"{'='*80}\n")

        return {
            'loss': test_loss,
            'accuracy': test_acc,
            'precision': test_precision,
            'recall': test_recall,
            'f1_score': f1_score,
            'auc': test_auc
        }

    def plot_confusion_matrix(self, save_path='results/plots/confusion_matrix.png'):
        """Plot confusion matrix with improved styling"""
        cm = confusion_matrix(self.y_true, self.y_pred)

        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

        plt.figure(figsize=(10, 8))

        # Create annotations with count and percentage
        annotations = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

        sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                   xticklabels=['Normal', 'Pneumonia'],
                   yticklabels=['Normal', 'Pneumonia'],
                   cbar_kws={'label': 'Count'},
                   linewidths=2, linecolor='white')

        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=13, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Confusion matrix saved to {save_path}")

        # Print confusion matrix details
        tn, fp, fn, tp = cm.ravel()
        print(f"\nConfusion Matrix Breakdown:")
        print(f"  True Negatives (TN):  {tn:4d} | Correctly predicted Normal")
        print(f"  False Positives (FP): {fp:4d} | Normal predicted as Pneumonia")
        print(f"  False Negatives (FN): {fn:4d} | Pneumonia predicted as Normal (⚠ Critical!)")
        print(f"  True Positives (TP):  {tp:4d} | Correctly predicted Pneumonia")

        return cm

    def plot_roc_curve(self, save_path='results/plots/roc_curve.png'):
        """Plot ROC curve with improved styling"""
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='#e74c3c', lw=3,
                label=f'ROC Curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='#95a5a6', lw=2, linestyle='--',
                label='Random Classifier (AUC = 0.5000)')
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.05])
        plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=13, fontweight='bold')
        plt.title('Receiver Operating Characteristic (ROC) Curve',
                 fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ ROC curve saved to {save_path}")
        print(f"  AUC Score: {roc_auc:.4f}")

        return roc_auc

    def plot_precision_recall_curve(self, save_path='results/plots/pr_curve.png'):
        """Plot Precision-Recall curve with improved styling"""
        precision, recall, thresholds = precision_recall_curve(
            self.y_true, self.y_pred_proba)

        pr_auc = auc(recall, precision)

        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='#3498db', lw=3,
                label=f'PR Curve (AUC = {pr_auc:.4f})')
        plt.xlim([-0.02, 1.02])
        plt.ylim([-0.02, 1.05])
        plt.xlabel('Recall', fontsize=13, fontweight='bold')
        plt.ylabel('Precision', fontsize=13, fontweight='bold')
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold', pad=20)
        plt.legend(loc="lower left", fontsize=11)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Precision-Recall curve saved to {save_path}")

    def generate_classification_report(self, save_path='results/metrics/classification_report.txt'):
        """Generate and save classification report"""
        report = classification_report(self.y_true, self.y_pred,
                                      target_names=['Normal', 'Pneumonia'],
                                      digits=4)

        print(f"\n{'='*80}")
        print("Classification Report:")
        print(f"{'='*80}")
        print(report)

        # Save to file
        os.makedirs('results/metrics', exist_ok=True)
        with open(save_path, 'w') as f:
            f.write("PNEUMONIA DETECTION - CLASSIFICATION REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(report)

        print(f"✓ Classification report saved to {save_path}")

        return report

    def visualize_predictions(self, num_samples=16,
                            save_path='results/plots/predictions.png'):
        """Visualize sample predictions with improved styling"""
        # Get a batch of images
        self.test_generator.reset()
        images, labels = next(self.test_generator)
        predictions = self.model.predict(images, verbose=0)

        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.ravel()

        correct_count = 0

        for i in range(min(num_samples, len(images))):
            axes[i].imshow(images[i])

            true_label = 'Pneumonia' if labels[i] == 1 else 'Normal'
            pred_label = 'Pneumonia' if predictions[i] > 0.5 else 'Normal'
            confidence = predictions[i][0] if predictions[i] > 0.5 else 1 - predictions[i][0]

            is_correct = true_label == pred_label
            if is_correct:
                correct_count += 1

            color = '#2ecc71' if is_correct else '#e74c3c'
            symbol = '✓' if is_correct else '✗'

            axes[i].set_title(f'{symbol} True: {true_label}\nPred: {pred_label} ({confidence:.1%})',
                            color=color, fontweight='bold', fontsize=11)
            axes[i].axis('off')

        # Add overall accuracy to the plot
        sample_accuracy = correct_count / min(num_samples, len(images))
        fig.suptitle(f'Sample Predictions (Accuracy: {sample_accuracy:.1%})',
                    fontsize=18, fontweight='bold', y=0.995)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"✓ Prediction visualizations saved to {save_path}")

    def generate_all_visualizations(self):
        """Generate all evaluation visualizations"""
        print(f"\n{'='*80}")
        print("📈 Generating Evaluation Visualizations...")
        print(f"{'='*80}\n")

        self.plot_confusion_matrix()
        self.plot_roc_curve()
        self.plot_precision_recall_curve()
        self.generate_classification_report()
        self.visualize_predictions()

        print(f"\n{'='*80}")
        print("✅ All visualizations generated successfully!")
        print(f"{'='*80}\n")
