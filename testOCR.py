import pytesseract
from PIL import Image
import easyocr

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Initialize OCR tools
easyocr_reader = easyocr.Reader(['en'])


# Function to preprocess the image (same as before)
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_bin = cv2.bitwise_not(img_bin)
    return img_bin

# Function to use Tesseract OCR
def ocr_tesseract(image_path):
    img = preprocess_image(image_path)
    pil_img = Image.fromarray(img)
    text = pytesseract.image_to_string(pil_img)
    return text

# Function to use EasyOCR
def ocr_easyocr(image_path):
    img = preprocess_image(image_path)
    result = easyocr_reader.readtext(img)
    text = " ".join([res[1] for res in result])
    return text


# Directory containing the subdirectories with images
data_directory = '/home/laptop/ECGR_5105/Project/random_tables'

# Initialize lists to hold OCR results
tesseract_results = []
easyocr_results = []


# Loop through each subdirectory and process files
count = 0
for subdir in os.listdir(data_directory):
    subdir_path = os.path.join(data_directory, subdir)
    if os.path.isdir(subdir_path):
        for filename in os.listdir(subdir_path):
            if filename.endswith(".png"):
                image_path = os.path.join(subdir_path, filename)
                
                # Extract text using Tesseract OCR
                tesseract_text = ocr_tesseract(image_path)
                tesseract_results.append(tesseract_text)
                
                # Extract text using EasyOCR
                easyocr_text = ocr_easyocr(image_path)
                easyocr_results.append(easyocr_text)
                

                
                count += 1
                if count % 50 == 0:
                    # Display the image using matplotlib
                    img = Image.open(image_path)
                    plt.imshow(img)
                    plt.title(f"Sample Image {count}")
                    plt.axis('off')  # Hide the axis
                    plt.show()
                    
                    # Print OCR results
                    print(f"Tesseract OCR Result for {image_path}:\n{tesseract_text}\n")
                    print(f"EasyOCR Result for {image_path}:\n{easyocr_text}\n")


# Example analysis of results (e.g., comparing results, calculating accuracy, etc.)
# This part can be customized based on specific requirements or metrics for comparison
for i in range(len(tesseract_results)):
    print(f"Image {i+1}:")
    print(f"Tesseract: {tesseract_results[i]}")
    print(f"EasyOCR: {easyocr_results[i]}")

    print("-" * 50)
