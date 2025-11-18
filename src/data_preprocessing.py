import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')


class DataPreprocessor:
    def __init__(self, data_dir, img_size=(224, 224), batch_size=64):
        self.data_dir = data_dir
        self.img_size = img_size
        self.batch_size = batch_size

    def analyze_dataset(self):
        """Analyze and visualize dataset distribution"""

        train_normal = len(os.listdir(os.path.join(self.data_dir, 'train/NORMAL')))
        train_pneumonia = len(os.listdir(os.path.join(self.data_dir, 'train/PNEUMONIA')))
        test_normal = len(os.listdir(os.path.join(self.data_dir, 'test/NORMAL')))
        test_pneumonia = len(os.listdir(os.path.join(self.data_dir, 'test/PNEUMONIA')))

        # Print stats
        print("\nDataset Statistics:")
        print(f"  Training   - Normal: {train_normal}, Pneumonia: {train_pneumonia}")
        print(f"  Testing    - Normal: {test_normal}, Pneumonia: {test_pneumonia}")
        print(f"  Total Images: {train_normal + train_pneumonia + test_normal + test_pneumonia}")

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        axes[0].bar(['Normal', 'PNEUMONIA'], [train_normal, train_pneumonia],
                    color=['#2ecc71', '#e74c3c'])
        axes[0].set_title('Training Data Distribution')

        axes[1].bar(['Normal', 'PNEUMONIA'], [test_normal, test_pneumonia],
                    color=['#2ecc71', '#e74c3c'])
        axes[1].set_title('Test Data Distribution')

        plt.tight_layout()
        os.makedirs('results/plots', exist_ok=True)
        plt.savefig('results/plots/data_distribution.png', dpi=150)
        plt.close()

        print("✓ Data distribution plot saved\n")

    def create_data_generators(self, use_validation_split=True):
        """Create training, validation, and test generators"""

        if use_validation_split:
            # Train + Validation split from train folder
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                validation_split=0.2
            )

            train_gen = train_datagen.flow_from_directory(
                os.path.join(self.data_dir, 'train'),
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='binary',
                shuffle=True,
                subset='training'
            )

            val_gen = train_datagen.flow_from_directory(
                os.path.join(self.data_dir, 'train'),
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='binary',
                shuffle=False,
                subset='validation'
            )

        else:
            # Use small "val" folder (not recommended)
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True
            )

            train_gen = train_datagen.flow_from_directory(
                os.path.join(self.data_dir, 'train'),
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='binary',
                shuffle=True
            )

            val_datagen = ImageDataGenerator(rescale=1./255)
            val_gen = val_datagen.flow_from_directory(
                os.path.join(self.data_dir, 'val'),
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='binary',
                shuffle=False
            )

        # Test generator
        test_datagen = ImageDataGenerator(rescale=1./255)
        test_gen = test_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'test'),
            target_size=self.img_size,
            batch_size=self.batch_size,
            class_mode='binary',
            shuffle=False
        )

        return train_gen, val_gen, test_gen

    def visualize_samples(self, generator, num_samples=9):
        """Save sample images"""
        images, labels = next(generator)

        plt.figure(figsize=(12, 12))
        for i in range(min(num_samples, len(images))):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            plt.title("PNEUMONIA" if labels[i] == 1 else "NORMAL")
            plt.axis("off")

        plt.tight_layout()
        plt.savefig("results/plots/sample_images.png", dpi=150)
        plt.close()
        print("✓ Sample images saved\n")
