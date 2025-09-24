import cv2
import numpy as np
import os

def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply a blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    binary_img = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )
    
    return binary_img, img

def detect_dots(binary_img):
    dot_coordinates = []
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 5 < area < 500:  # Adjust these values based on your image
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                dot_coordinates.append((cX, cY))
                
    return dot_coordinates

def trace_lines(binary_img, dots):
    line_img = binary_img.copy()
    
    # Erase the dots
    for (x, y) in dots:
        cv2.circle(line_img, (x, y), 15, (0, 0, 0), -1)
        
    # Find line contours
    line_contours, _ = cv2.findContours(line_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return line_contours

def visualize_results(original_img, binary_img, dots, lines, output_dir="output"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Draw dots on original image
    dot_img = original_img.copy()
    for (x, y) in dots:
        cv2.circle(dot_img, (x, y), 5, (0, 0, 255), -1)  # Red dots
    
    # Draw lines on original image
    line_img = original_img.copy()
    cv2.drawContours(line_img, lines, -1, (0, 255, 0), 2)  # Green lines
    
    # Combine everything
    combined = original_img.copy()
    for (x, y) in dots:
        cv2.circle(combined, (x, y), 5, (0, 0, 255), -1)  # Red dots
    cv2.drawContours(combined, lines, -1, (0, 255, 0), 2)  # Green lines
    
    # Save results
    cv2.imwrite(os.path.join(output_dir, "1_original.jpg"), original_img)
    cv2.imwrite(os.path.join(output_dir, "2_binary.jpg"), binary_img)
    cv2.imwrite(os.path.join(output_dir, "3_dots.jpg"), dot_img)
    cv2.imwrite(os.path.join(output_dir, "4_lines.jpg"), line_img)
    cv2.imwrite(os.path.join(output_dir, "5_combined.jpg"), combined)
    
    print(f"Results saved to {output_dir} directory")

def main():
    # Input and output paths
    input_image = "image.png"  # Make sure this file exists
    output_dir = "output"
    
    try:
        # Process the image
        print("Processing image...")
        binary_img, original_img = preprocess_image(input_image)
        
        # Detect dots
        print("Detecting dots...")
        dots = detect_dots(binary_img)
        print(f"Found {len(dots)} dots")
        
        # Trace lines
        print("Tracing lines...")
        lines = trace_lines(binary_img, dots)
        print(f"Found {len(lines)} line segments")
        
        # Visualize and save results
        print("Saving results...")
        visualize_results(original_img, binary_img, dots, lines, output_dir)
        
        print("\nProcessing complete!")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()