import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    def __init__(self, model, model_name='pneumonia_model'):
        self.model = model
        self.model_name = model_name
        self.history = None


    def compute_class_weights(self, train_generator):
        """Compute class weights to fix class imbalance"""
        labels = train_generator.classes  # 0 = Normal, 1 = Pneumonia

        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )

        return {0: class_weights[0], 1: class_weights[1]}


    def get_callbacks(self):
        """Callbacks optimized for stability"""
        os.makedirs('models', exist_ok=True)

        return [
            ModelCheckpoint(
                filepath=f'models/{self.model_name}_best.keras',
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            ),

            EarlyStopping(
                monitor='val_accuracy',
                patience=7,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),

            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.3,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]


    def train(self, train_generator, val_generator, epochs=20):
        """Main training loop"""
        print("\n" + "="*80)
        print(f"🚀 Starting Training: {self.model_name}")
        print("="*80)
        print(f"Training samples:    {train_generator.samples}")
        print(f"Validation samples:  {val_generator.samples}")
        print(f"Batch size:          {train_generator.batch_size}")
        print(f"Steps per epoch:     {len(train_generator)}")
        print("="*80 + "\n")

        # Safe mixed precision
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
            print("✓ Mixed precision enabled\n")
        except:
            print("⚠ Mixed precision unavailable\n")

        # Compute class weights
        class_weights = self.compute_class_weights(train_generator)
        print("Class Weights Applied:", class_weights, "\n")

        # Train model (NO workers / multiprocessing)
        self.history = self.model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=epochs,
            callbacks=self.get_callbacks(),
            class_weight=class_weights,   # ← balance dataset
            verbose=1
        )

        # Save final model
        self.model.save(f'models/{self.model_name}_final.keras')
        print("\n✓ Final model saved\n")

        return self.history


    def plot_training_history(self):
        """Plot accuracy, loss, precision, recall curves"""
        if not self.history:
            print("No history available.")
            return

        hist = self.history.history
        epochs = range(1, len(hist['accuracy']) + 1)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        titles = ['Accuracy', 'Loss', 'Precision', 'Recall']
        keys = [('accuracy', 'val_accuracy'),
                ('loss', 'val_loss'),
                ('precision', 'val_precision'),
                ('recall', 'val_recall')]

        for ax, (title, (train_key, val_key)) in zip(axes.flat, zip(titles, keys)):
            ax.plot(epochs, hist[train_key], label='Train', linewidth=2)
            ax.plot(epochs, hist[val_key], label='Val', linewidth=2)
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.3)
            ax.legend()

        plt.tight_layout()
        plt.savefig(f'results/plots/{self.model_name}_training_history.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        print("✓ Training history plot saved")
