#!/usr/bin/env python3
"""
Enhanced handwriting recognition with preprocessing for real-world images
"""

import torch
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import cv2
from transformers import VisionEncoderDecoderModel, TrOCRProcessor
import matplotlib.pyplot as plt

def preprocess_real_world_image(image_path, debug=True):
    """
    Preprocess real-world handwriting images to match IAM dataset format.

    Steps:
    1. Load image
    2. Convert to grayscale
    3. Enhance contrast
    4. Binarize (threshold)
    5. Crop to text region
    6. Resize appropriately
    """
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Binarize using adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=15,
        C=10
    )

    # Find contours to crop to text region
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get bounding box of all text
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter out very small contours (noise)
            if w > 10 and h > 10:
                x_min = min(x_min, x)
                y_min = min(y_min, y)
                x_max = max(x_max, x + w)
                y_max = max(y_max, y + h)

        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(binary.shape[1], x_max + padding)
        y_max = min(binary.shape[0], y_max + padding)

        # Crop to text region
        cropped = binary[y_min:y_max, x_min:x_max]
    else:
        cropped = binary

    # Invert back (black text on white background)
    inverted = cv2.bitwise_not(cropped)

    # Convert to PIL Image
    pil_image = Image.fromarray(inverted)

    # Convert to RGB (model expects RGB)
    pil_image = pil_image.convert('RGB')

    if debug:
        # Show preprocessing steps
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))

        axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('1. Original Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(gray, cmap='gray')
        axes[0, 1].set_title('2. Grayscale')
        axes[0, 1].axis('off')

        axes[0, 2].imshow(enhanced, cmap='gray')
        axes[0, 2].set_title('3. Contrast Enhanced')
        axes[0, 2].axis('off')

        axes[1, 0].imshow(binary, cmap='gray')
        axes[1, 0].set_title('4. Binarized')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(cropped, cmap='gray')
        axes[1, 1].set_title('5. Cropped')
        axes[1, 1].axis('off')

        axes[1, 2].imshow(pil_image)
        axes[1, 2].set_title('6. Final (Ready for Model)')
        axes[1, 2].axis('off')

        plt.tight_layout()
        plt.savefig('/workspace/preprocessing_steps.png', dpi=150, bbox_inches='tight')
        print("✓ Preprocessing steps saved to: preprocessing_steps.png")
        plt.show()

    return pil_image


def predict_with_preprocessing(image_path, model_path="/workspace/best_trocr_model"):
    """
    Predict handwriting with enhanced preprocessing.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    print(f"Loading model from {model_path}...")
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    processor = TrOCRProcessor.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print("✓ Model loaded!")

    # Preprocess image
    print(f"\nPreprocessing image: {image_path}...")
    processed_image = preprocess_real_world_image(image_path, debug=True)

    # Run inference
    print("\nRunning inference...")
    pixel_values = processor(processed_image, return_tensors="pt").pixel_values.to(device)

    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
        predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print(f"\n{'='*60}")
    print(f"PREDICTION RESULT")
    print(f"{'='*60}")
    print(f"📝 Predicted Text: '{predicted_text}'")
    print(f"{'='*60}\n")

    # Show final result
    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
    ax.imshow(processed_image)
    ax.set_title(f"Predicted: '{predicted_text}'", fontsize=16, weight='bold', color='green')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig('/workspace/final_prediction.png', dpi=150, bbox_inches='tight')
    print("✓ Final prediction saved to: final_prediction.png")
    plt.show()

    return predicted_text, processed_image


def test_multiple_approaches(image_path):
    """
    Try multiple preprocessing approaches and compare results.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    print("Loading model...")
    model = VisionEncoderDecoderModel.from_pretrained("/workspace/best_trocr_model")
    processor = TrOCRProcessor.from_pretrained("/workspace/best_trocr_model")
    model.to(device)
    model.eval()

    results = {}

    # Approach 1: Direct (no preprocessing)
    print("\n1. Testing WITHOUT preprocessing...")
    img_direct = Image.open(image_path).convert('RGB')
    pixel_values = processor(img_direct, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
        results['direct'] = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"   Result: '{results['direct']}'")

    # Approach 2: With preprocessing
    print("\n2. Testing WITH preprocessing...")
    img_processed = preprocess_real_world_image(image_path, debug=False)
    pixel_values = processor(img_processed, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
        results['preprocessed'] = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"   Result: '{results['preprocessed']}'")

    # Approach 3: Simple grayscale + resize
    print("\n3. Testing with simple grayscale...")
    img_simple = Image.open(image_path).convert('L').convert('RGB')
    pixel_values = processor(img_simple, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
        results['simple_gray'] = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"   Result: '{results['simple_gray']}'")

    # Approach 4: Inverted
    print("\n4. Testing with inverted colors...")
    img_inv = ImageOps.invert(Image.open(image_path).convert('L')).convert('RGB')
    pixel_values = processor(img_inv, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values)
        results['inverted'] = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"   Result: '{results['inverted']}'")

    print(f"\n{'='*60}")
    print("COMPARISON OF ALL APPROACHES")
    print(f"{'='*60}")
    for approach, prediction in results.items():
        print(f"{approach:20s}: '{prediction}'")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    image_path = "/workspace/WhatsApp Image 2025-10-19 at 07.17.58_d5d96fa4.jpg"

    print("="*60)
    print("ENHANCED HANDWRITING RECOGNITION TEST")
    print("="*60)

    # Test all approaches
    print("\n🔬 Testing multiple preprocessing approaches...\n")
    results = test_multiple_approaches(image_path)

    # Run detailed prediction with best approach
    print("\n\n🎯 Running detailed prediction with preprocessing...\n")
    predicted_text, processed_img = predict_with_preprocessing(image_path)

    print("\n✅ DONE!")
