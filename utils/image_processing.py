import cv2
import os
import pytesseract
import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt

def process_images(wireframe_path, live_site_path, output_folder):
    wireframe = cv2.imread(wireframe_path)
    live_site = cv2.imread(live_site_path)

    if wireframe is None or live_site is None:
        print("Error: Unable to load one or both of the images.")
        return None, None

    # Preprocess images (resize, grayscale, etc.)
    height, width = 1000, 1440
    wireframe_resized = cv2.resize(wireframe, (width, height))
    live_site_resized = cv2.resize(live_site, (width, height))

    # Calculate the SSIM score
    gray_wireframe = cv2.cvtColor(wireframe_resized, cv2.COLOR_BGR2GRAY)
    gray_live_site = cv2.cvtColor(live_site_resized, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray_wireframe, gray_live_site, full=True)

    # Find differences
    diff = cv2.absdiff(gray_wireframe, gray_live_site)
    blurred_diff = cv2.GaussianBlur(diff, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred_diff, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Annotate and classify discrepancies
    summary = {
        "Font-size/Style Issues": 0,
        "Spacing Issues": 0,
        "Padding Issues": 0,
        "Positioning Issues": 0,
    }
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter small irrelevant contours
            x, y, w, h = cv2.boundingRect(contour)
            summary["Font-size/Style Issues"] += 1  # Just as an example
            cv2.rectangle(live_site_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Overlay wireframe
    overlay = live_site_resized.copy()
    alpha = 0.4
    cv2.addWeighted(wireframe_resized, alpha, overlay, 1 - alpha, 0, overlay)

    # Save annotated image
    annotated_image_path = os.path.join(output_folder, "annotated_overlay_combined.png")
    cv2.imwrite(annotated_image_path, overlay)

    return annotated_image_path, summary


def extract_text_from_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load image at {image_path}")
        return ""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray_img)
    return text


def generate_heatmap(wireframe_path, live_site_path, annotated_image_path):
    try:
        wireframe = cv2.imread(wireframe_path)
        live_site = cv2.imread(live_site_path)

        if wireframe is None or live_site is None:
            print("Error: Unable to load one or both of the images for heatmap generation.")
            return None, None

        # Resize images to ensure the same dimensions
        height, width = 1000, 1440
        wireframe_resized = cv2.resize(wireframe, (width, height))
        live_site_resized = cv2.resize(live_site, (width, height))

        # Convert images to grayscale before calculating differences
        gray_wireframe = cv2.cvtColor(wireframe_resized, cv2.COLOR_BGR2GRAY)
        gray_live_site = cv2.cvtColor(live_site_resized, cv2.COLOR_BGR2GRAY)

        # Compute difference and generate heatmap
        diff = cv2.absdiff(gray_wireframe, gray_live_site)

        # Ensure the diff image is in a compatible format (grayscale)
        heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)

        # Overlay heatmap on annotated image
        annotated_image = cv2.imread(annotated_image_path)
        if annotated_image is None:
            print("Error: Unable to load annotated image for overlay.")
            return None, None
        heatmap_overlay = cv2.addWeighted(annotated_image, 0.7, heatmap, 0.3, 0)

        # Save heatmap image
        heatmap_image_path = annotated_image_path.replace("annotated_overlay_combined.png", "heatmap_overlay.png")
        cv2.imwrite(heatmap_image_path, heatmap_overlay)

        # Calculate SSIM score
        score, _ = ssim(gray_wireframe, gray_live_site, full=True)

        return score, heatmap_image_path
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return None, None
