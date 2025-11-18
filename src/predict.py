import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
import os
import warnings
warnings.filterwarnings('ignore')

class PneumoniaPredictor:
    def __init__(self, model_path, img_size=(224, 224)):
        """Initialize predictor with trained model"""
        print(f"Loading model from: {model_path}")
        self.model = load_model(model_path)
        self.img_size = img_size
        print("✓ Model loaded successfully\n")

    def preprocess_image(self, img_path):
        """Load and preprocess a single image"""
        img = image.load_img(img_path, target_size=self.img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        return img_array

    def predict(self, img_path):
        """Predict pneumonia for a single image"""
        img_array = self.preprocess_image(img_path)
        prediction = self.model.predict(img_array, verbose=0)

        probability = float(prediction[0][0])
        result = 'Pneumonia' if probability > 0.5 else 'Normal'
        confidence = probability if probability > 0.5 else 1 - probability

        return result, confidence, probability

    def visualize_prediction(self, img_path, save_path=None):
        """Visualize prediction with original image"""
        result, confidence, probability = self.predict(img_path)

        # Load original image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(img)

        # Color based on prediction
        color = '#e74c3c' if result == 'Pneumonia' else '#2ecc71'

        # Title with prediction
        title = f'Prediction: {result}\nConfidence: {confidence:.2%}'
        if result == 'Pneumonia':
            title += f'\n⚠ Pneumonia Probability: {probability:.2%}'

        ax.set_title(title, fontsize=16, fontweight='bold', color=color, pad=20)
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Prediction visualization saved to {save_path}")

        plt.show()

        return result, confidence

    def batch_predict(self, image_folder, output_csv='results/predictions.csv'):
        """Predict on multiple images and save results"""
        import pandas as pd

        results = []
        image_files = [f for f in os.listdir(image_folder)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"Processing {len(image_files)} images...")

        for idx, img_file in enumerate(image_files, 1):
            img_path = os.path.join(image_folder, img_file)
            result, confidence, probability = self.predict(img_path)

            results.append({
                'filename': img_file,
                'prediction': result,
                'confidence': f'{confidence:.4f}',
                'pneumonia_probability': f'{probability:.4f}'
            })

            print(f"  [{idx}/{len(image_files)}] {img_file}: {result} ({confidence:.2%})")

        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        print(f"\n✓ Results saved to {output_csv}")

        # Summary statistics
        pneumonia_count = sum(1 for r in results if r['prediction'] == 'Pneumonia')
        normal_count = len(results) - pneumonia_count

        print(f"\nSummary:")
        print(f"  Normal:    {normal_count:3d} ({normal_count/len(results)*100:.1f}%)")
        print(f"  Pneumonia: {pneumonia_count:3d} ({pneumonia_count/len(results)*100:.1f}%)")

        return df

# Example usage
if __name__ == "__main__":
    print("="*80)
    print("  PNEUMONIA PREDICTION TOOL")
    print("="*80 + "\n")

    # Initialize predictor
    model_path = 'models/cnn_pneumonia_best.keras'

    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        print("Please train the model first using main.py")
        exit(1)

    predictor = PneumoniaPredictor(model_path)

    # Example 1: Single image prediction
    print("Example 1: Single Image Prediction")
    print("-" * 80)
    test_image = 'data/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg'

    if os.path.exists(test_image):
        result, confidence = predictor.visualize_prediction(
            test_image,
            save_path='results/plots/single_prediction.png'
        )
        print(f"\nPrediction: {result}")
        print(f"Confidence: {confidence:.2%}\n")
    else:
        print(f"Test image not found: {test_image}\n")

    # Example 2: Batch prediction
    print("\nExample 2: Batch Prediction")
    print("-" * 80)
    test_folder = 'data/chest_xray/test/PNEUMONIA'

    if os.path.exists(test_folder):
        # Predict on first 10 images
        all_files = [f for f in os.listdir(test_folder)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:10]

        if all_files:
            # Create temporary folder
            temp_folder = 'temp_batch_test'
            os.makedirs(temp_folder, exist_ok=True)

            # Copy files
            import shutil
            for f in all_files:
                shutil.copy(os.path.join(test_folder, f),
                          os.path.join(temp_folder, f))

            # Run batch prediction
            df = predictor.batch_predict(temp_folder)

            # Cleanup
            shutil.rmtree(temp_folder)
        else:
            print("No images found in test folder")
    else:
        print(f"Test folder not found: {test_folder}")

    print("\n" + "="*80)
    print("  Prediction Complete!")
    print("="*80)
