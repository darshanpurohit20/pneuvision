import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Suppress TensorFlow excessive logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'     # Disable oneDNN to reduce warnings
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

import tensorflow as tf

# Safe GPU memory growth configuration
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass

# Create result directories
os.makedirs('results/plots', exist_ok=True)
os.makedirs('results/metrics', exist_ok=True)

# Import custom modules
from src.data_preprocessing import DataPreprocessor
from src.model import PneumoniaDetectionModel
from src.train import ModelTrainer
from src.evaluate import ModelEvaluator


def print_header(text):
    print("\n" + "=" * 80)
    print(f"  {text}")
    print("=" * 80 + "\n")


def main():
    # ================= CONFIG =================
    DATA_DIR = 'data/chest_xray'
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 64
    EPOCHS = 20
    MODEL_TYPE = 'resnet'          # cnn | vgg16 | resnet
    LEARNING_RATE = 1e-5
    USE_VALIDATION_SPLIT = True

    print_header("🏥 PNEUMONIA DETECTION FROM CHEST X-RAYS")

    print("Configuration:")
    print(f"  Image Size:       {IMG_SIZE}")
    print(f"  Batch Size:       {BATCH_SIZE}")
    print(f"  Epochs:           {EPOCHS}")
    print(f"  Model Type:       {MODEL_TYPE.upper()}")
    print(f"  Learning Rate:    {LEARNING_RATE}")
    print(f"  Validation Split: {USE_VALIDATION_SPLIT}")

    # ================= STEP 1: DATA =================
    print_header("STEP 1: Loading and Preprocessing Data")

    preprocessor = DataPreprocessor(DATA_DIR, IMG_SIZE, BATCH_SIZE)

    preprocessor.analyze_dataset()

    train_gen, val_gen, test_gen = preprocessor.create_data_generators(
        use_validation_split=USE_VALIDATION_SPLIT
    )

    print("\nGenerating sample visualizations...")
    preprocessor.visualize_samples(train_gen)

    print("\nData Split Summary:")
    print(f"  Training samples:   {train_gen.samples}")
    print(f"  Validation samples: {val_gen.samples}")
    print(f"  Test samples:       {test_gen.samples}")
    print(f"  Total: {train_gen.samples + val_gen.samples + test_gen.samples}")

    # ================= STEP 2: MODEL =================
    print_header(f"STEP 2: Building Model ({MODEL_TYPE.upper()})")

    model_builder = PneumoniaDetectionModel(IMG_SIZE, MODEL_TYPE)
    model = model_builder.compile_model(learning_rate=LEARNING_RATE)
    model_builder.get_model_summary()

    # ================= STEP 3: TRAIN =================
    print_header("STEP 3: Training Model")

    trainer = ModelTrainer(model, model_name=f'{MODEL_TYPE}_pneumonia')

    try:
        history = trainer.train(train_gen, val_gen, epochs=EPOCHS)

        print("\nGenerating training history plots...")
        trainer.plot_training_history()

    except KeyboardInterrupt:
        print("\nTraining interrupted! Saving current model...")
        model.save(f'models/{MODEL_TYPE}_pneumonia_interrupted.keras')
        print("Saved.")
        return

    # ================= STEP 4: EVAL =================
    print_header("STEP 4: Evaluating Model")

    evaluator = ModelEvaluator(model, test_gen)
    metrics = evaluator.evaluate()
    evaluator.generate_all_visualizations()

    # ================= FINAL =================
    print_header("✅ TRAINING AND EVALUATION COMPLETE!")

    print("Final Results:")
    print(f"  Test Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Test Precision: {metrics['precision']:.4f}")
    print(f"  Test Recall:    {metrics['recall']:.4f}")
    print(f"  Test F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  Test AUC:       {metrics['auc']:.4f}")

    print("\nGenerated Outputs:")
    print("  📊 Plots:     results/plots/")
    print("  📝 Metrics:   results/metrics/")
    print("  🤖 Model:     models/")
    print("  🔍 Predictions: use src/predict.py")

    print("\n" + "=" * 80)
    print("  Thank you for using the Pneumonia Detection System!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
