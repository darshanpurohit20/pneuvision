from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense,
                                      Dropout, BatchNormalization, GlobalAveragePooling2D, Input)
from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

class PneumoniaDetectionModel:
    def __init__(self, img_size=(128, 128), model_type='cnn'):
        self.img_size = img_size
        self.model_type = model_type
        self.model = None

    def build_simple_cnn(self):
        """Build an optimized lightweight CNN"""
        model = Sequential([
            # Input layer
            Input(shape=(*self.img_size, 3)),

            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling2D(2, 2),

            # Global Average Pooling (faster than Flatten)
            GlobalAveragePooling2D(),

            # Dense layers
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')  # Binary classification
        ])

        return model

    def build_vgg16_transfer(self):
        """Build model using VGG16 transfer learning"""
        # Load pre-trained VGG16
        base_model = VGG16(weights='imagenet',
                          include_top=False,
                          input_shape=(*self.img_size, 3))

        # Freeze most layers, train only last few
        for layer in base_model.layers[:-4]:
            layer.trainable = False

        # Add custom layers
        inputs = Input(shape=(*self.img_size, 3))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=output)

        return model

    def build_resnet_transfer(self):
        """Build model using ResNet50 transfer learning"""
        # Load pre-trained ResNet50
        base_model = ResNet50(weights='imagenet',
                             include_top=False,
                             input_shape=(*self.img_size, 3))

        # Freeze most layers
        for layer in base_model.layers[:-10]:
            layer.trainable = False

        # Add custom layers
        inputs = Input(shape=(*self.img_size, 3))
        x = base_model(inputs, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        output = Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=output)

        return model

    def compile_model(self, learning_rate=0.001):
        """Build and compile the selected model"""
        if self.model_type == 'cnn':
            self.model = self.build_simple_cnn()
        elif self.model_type == 'vgg16':
            self.model = self.build_vgg16_transfer()
        elif self.model_type == 'resnet':
            self.model = self.build_resnet_transfer()
        else:
            raise ValueError("Model type must be 'cnn', 'vgg16', or 'resnet'")

        # Compile model with mixed precision for faster training
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy',
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall'),
                    tf.keras.metrics.AUC(name='auc')]
        )

        return self.model

    def get_model_summary(self):
        """Print model architecture"""
        if self.model:
            print("\nModel Architecture:")
            print("=" * 80)
            self.model.summary()
            print("=" * 80)

            # Count parameters
            total_params = self.model.count_params()
            print(f"\nTotal Parameters: {total_params:,}")
            print(f"Model Size: ~{total_params * 4 / (1024**2):.2f} MB")
        else:
            print("Model not compiled yet!")
