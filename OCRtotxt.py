import easyocr
from PIL import Image
import os
import cv2
import matplotlib.pyplot as plt

# Initialize EasyOCR
easyocr_reader = easyocr.Reader(['en'])

# Function to preprocess the image
def preprocess_image(image_path):
    # Read the image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Resize the image to enhance text recognition
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    # Apply Gaussian Blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    # Apply thresholding
    _, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_bin

# Function to use EasyOCR
def ocr_easyocr(image_path):
    img = preprocess_image(image_path)
    result = easyocr_reader.readtext(img)
    text = " ".join([res[1] for res in result])
    return text

# Directory containing the subdirectories
data_directory = '/home/laptop/ECGR_5105/Project/random_tables'

# Loop through each subdirectory and process files
count = 0
for subdir in os.listdir(data_directory):
    subdir_path = os.path.join(data_directory, subdir)
    if os.path.isdir(subdir_path):
        for filename in os.listdir(subdir_path):
            if filename.endswith(".png"):
                image_path = os.path.join(subdir_path, filename)
                txt_filename = filename.replace(".png", ".txt")
                txt_path = os.path.join(subdir_path, txt_filename)
                
                # Check if the txt file already exists
                if not os.path.exists(txt_path):
                    # Perform OCR on the image
                    ocr_text = ocr_easyocr(image_path)
                    if ocr_text.strip():  # Ensure OCR text is not empty
                        with open(txt_path, 'w') as txt_file:
                            txt_file.write(ocr_text)
                        count += 1
                        if count % 50 == 0:
                            # Display the image and extracted text
                            img = Image.open(image_path)
                            plt.imshow(img, cmap='gray')
                            plt.title(f"Sample Image {count}")
                            plt.axis('off')  # Hide the axis
                            plt.show()
                            print(f"EasyOCR Result for {image_path}:\n{ocr_text}\n")
                    else:
                        print(f"OCR text is empty for image: {image_path}")
                else:
                    print(f"Text file already exists for image: {image_path}")

print(f"Total processed images: {count}")
print("OCR extraction and saving completed successfully.")
