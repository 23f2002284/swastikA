import os
import math
import svgwrite
import cv2
import numpy as np

def cluster_dots(coords, distance_threshold=40):
    """Groups nearby coordinates into clusters and finds the center of each cluster."""
    clusters = []
    for (x, y) in coords:
        found_cluster = False
        for cluster in clusters:
            if math.dist((x, y), cluster[0]) < distance_threshold:
                cluster.append((x, y))
                found_cluster = True
                break
        if not found_cluster:
            clusters.append([(x, y)])
    
    final_dots = []
    for cluster in clusters:
        x_coords = [p[0] for p in cluster]
        y_coords = [p[1] for p in cluster]
        centroid_x = int(np.mean(x_coords))
        centroid_y = int(np.mean(y_coords))
        final_dots.append((centroid_x, centroid_y))
        
    return final_dots

def create_recreation_svg(processed_image, dots, output_path, width, height):
    """Creates an animated SVG of the Kolam recreation."""
    contours, _ = cv2.findContours(processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    dwg = svgwrite.Drawing(output_path, profile='full', size=(width, height))

    for (x, y) in dots:
        dwg.add(dwg.circle(center=(x, y), r=5, fill='black'))

    path_group = dwg.g(stroke='black', stroke_width=4, fill='none')
    total_length = 0

    for contour in contours:
        if cv2.contourArea(contour) > 100:
            path_data = "M" + " L".join(f"{p[0][0]},{p[0][1]}" for p in contour) + " Z"
            path = dwg.path(d=path_data)
            path_group.add(path)
            total_length += cv2.arcLength(contour, closed=True)

    animation_style = f"""
        .path-animation {{
            stroke-dasharray: {total_length};
            stroke-dashoffset: {total_length};
            animation: draw 10s linear forwards;
        }}
        @keyframes draw {{ to {{ stroke-dashoffset: 0; }} }}
    """
    dwg.defs.add(dwg.style(animation_style))
    path_group['class'] = 'path-animation'
    dwg.add(path_group)
    dwg.save()
    return output_path

def analyze_kolam(image_path, output_dir='output'):
    """
    Analyzes a kolam image and creates an animated SVG recreation.
    
    Args:
        image_path: Path to the input kolam image
        output_dir: Directory to save output files
        
    Returns:
        dict: Contains paths to output files and analysis results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Read and process image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image.")
    
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations
    kernel = np.ones((7,7), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    erode_kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(closing, erode_kernel, iterations=2)
    
    # Save intermediate processing result
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    diag_path = os.path.join(output_dir, f'diag_{base_name}.png')
    cv2.imwrite(diag_path, erosion)

    # Find contours and process dots
    contours, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find potential dot fragments
    raw_dot_coords = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 10 < area < 10000:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                raw_dot_coords.append((cX, cY))
    
    # Cluster dots
    final_dots = cluster_dots(raw_dot_coords)

    # Generate debug image
    debug_image = image.copy()
    for (cX, cY) in final_dots:
        cv2.circle(debug_image, (cX, cY), 20, (0, 255, 0), 3)
    
    debug_path = os.path.join(output_dir, f'debug_{base_name}.png')
    cv2.imwrite(debug_path, debug_image)

    # Create SVG recreation
    svg_path = os.path.join(output_dir, f'recreation_{base_name}.svg')
    create_recreation_svg(closing, final_dots, svg_path, width, height)
    
    return {
        'dot_count': len(final_dots),
        'svg_path': svg_path,
        'debug_image': debug_path,
        'processed_image': diag_path
    }

# Example usage
if __name__ == '__main__':
    # Example usage
    result = analyze_kolam('image.png', 'output')
    print(f"Found {result['dot_count']} dots")
    print(f"SVG saved to: {result['svg_path']}")
    print(f"Debug image saved to: {result['debug_image']}")
