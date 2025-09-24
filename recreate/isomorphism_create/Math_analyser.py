import cv2
import numpy as np
import networkx as nx
from skimage import morphology, feature, measure
from sklearn.cluster import DBSCAN, KMeans
from scipy.spatial import KDTree, distance_matrix
from scipy.spatial.distance import pdist, squareform
from scipy import ndimage
from shapely.geometry import Point, Polygon, LineString
# from shapely.ops import cascaded_union
import json
import math
from collections import defaultdict, Counter
from itertools import combinations
import matplotlib.pyplot as plt

class KolamAnalyzer:
    """
    Advanced mathematical analyzer for kolam patterns.
    Extracts comprehensive geometric, topological, and structural features.
    """
    
    def __init__(self, config=None):
        # Default configuration
        self.config = {
            'DOT_THRESHOLD': 20,
            'DBSCAN_EPS': 12,
            'DBSCAN_MIN_SAMPLES': 2,
            'ANGLE_THRESHOLD': 15,
            'SYMMETRY_TOLERANCE': 4,
            'TARGET_SIZE': 512,
            'MIN_CYCLE_AREA': 50,
            'MAX_LATTICE_DEVIATION': 0.3,
            'BLOB_MIN_SIGMA': 1,
            'BLOB_MAX_SIGMA': 10,
            'BLOB_NUM_SIGMA': 10,
            'BLOB_THRESHOLD': 0.02
            # 'BLOB_THRESHOLD': 0.1
        }
        if config:
            self.config.update(config)
    
    # def preprocess(self, image):
    #     """Preprocess image for analysis"""
    #     # Resize maintaining aspect ratio
    #     h, w = image.shape[:2]
    #     scale = self.config['TARGET_SIZE'] / max(h, w)
    #     new_w, new_h = int(w * scale), int(h * scale)
    #     resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
    #     # Convert to grayscale
    #     if len(resized.shape) == 3:
    #         gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    #     else:
    #         gray = resized.copy()
        
    #     # Enhanced contrast and noise reduction
    #     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #     gray = clahe.apply(gray)
    #     gray = cv2.bilateralFilter(gray, 9, 75, 75)

    #     # --- EXTRA: smooth noise further ---
    #     gray = cv2.medianBlur(gray, 5)
        
    #     # Adaptive thresholding
    #     bin_img = cv2.adaptiveThreshold(
    #         gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #         cv2.THRESH_BINARY_INV, 21, 9
    #     )
        
    #     # Morphological cleaning
    #     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    #     bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
    #     bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)

    #     # --- EXTRA: kill tiny white noise specks ---
    #     bin_img = cv2.medianBlur(bin_img, 3)
        
    #     # Skeletonization
    #     skel = morphology.skeletonize(bin_img > 0)
        
    #     return {
    #         'original': resized,
    #         'gray': gray,
    #         'binary': bin_img,
    #         'skeleton': skel,
    #         'scale_factor': scale
    #     }
    def preprocess(self, image):
        """Preprocess image for analysis"""
        h, w = image.shape[:2]
        scale = self.config['TARGET_SIZE'] / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        if len(resized.shape) == 3:
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        else:
            gray = resized.copy()

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        gray = cv2.medianBlur(gray, 5)

        bin_img = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 21, 9
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, kernel)
        bin_img = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, kernel)
        bin_img = cv2.medianBlur(bin_img, 3)

        skel = morphology.skeletonize(bin_img > 0)

        return {
            'original': resized,
            'gray': gray,
            'binary': bin_img,
            'skeleton': skel,
            'scale_factor': scale
        }

    
    # def detect_dots(self, gray_img, binary_img):
    #     """Advanced dot detection with multiple methods"""
    #     dots = []
        
    #     # Method 1: Blob detection
    #     blobs = feature.blob_log(
    #         gray_img,
    #         min_sigma=self.config['BLOB_MIN_SIGMA'],
    #         max_sigma=self.config['BLOB_MAX_SIGMA'],
    #         num_sigma=self.config['BLOB_NUM_SIGMA'],
    #         threshold=self.config['BLOB_THRESHOLD']
    #     )
        
    #     for blob in blobs:
    #         y, x, r = blob
    #         dots.append({
    #             'x': float(x), 'y': float(y), 'r': float(r * np.sqrt(2)),
    #             'method': 'blob_log'
    #         })
        
    #     # Method 2: Contour-based dot detection
    #     contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     for contour in contours:
    #         area = cv2.contourArea(contour)
    #         if 20 < area < 500:  # Typical dot area range
    #             # Check circularity
    #             perimeter = cv2.arcLength(contour, True)
    #             if perimeter > 0:
    #                 circularity = 4 * np.pi * area / (perimeter * perimeter)
    #                 if circularity > 0.6:  # Reasonably circular
    #                     M = cv2.moments(contour)
    #                     if M["m00"] != 0:
    #                         cx = M["m10"] / M["m00"]
    #                         cy = M["m01"] / M["m00"]
    #                         equiv_diameter = np.sqrt(4 * area / np.pi)
    #                         dots.append({
    #                             'x': float(cx), 'y': float(cy), 
    #                             'r': float(equiv_diameter / 2),
    #                             'method': 'contour',
    #                             'area': float(area),
    #                             'circularity': float(circularity)
    #                         })
        
    #     # Remove duplicates (dots detected by multiple methods)
    #     if len(dots) > 1:
    #         coords = np.array([[d['x'], d['y']] for d in dots])
    #         distances = squareform(pdist(coords))
    #         to_remove = set()
            
    #         for i in range(len(dots)):
    #             if i in to_remove:
    #                 continue
    #             for j in range(i + 1, len(dots)):
    #                 if j in to_remove:
    #                     continue
    #                 if distances[i, j] < max(dots[i]['r'], dots[j]['r']) * 1.5:
    #                     # Keep the one with better properties
    #                     if dots[i].get('circularity', 0.5) >= dots[j].get('circularity', 0.5):
    #                         to_remove.add(j)
    #                     else:
    #                         to_remove.add(i)
            
    #         dots = [d for i, d in enumerate(dots) if i not in to_remove]
        
    #     return dots

    def detect_dots(self, gray_img, binary_img):
        """Advanced dot detection with filtering"""
        dots = []
        # Blob detection
        blobs = feature.blob_log(
            gray_img,
            min_sigma=self.config['BLOB_MIN_SIGMA'],
            max_sigma=self.config['BLOB_MAX_SIGMA'],
            num_sigma=self.config['BLOB_NUM_SIGMA'],
            threshold=self.config['BLOB_THRESHOLD']
        )
        for y, x, r in blobs:
            dots.append({'x': float(x), 'y': float(y), 'r': float(r * np.sqrt(2)), 'method': 'blob_log'})

        # Contour detection
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if 20 < area < 500:
            # if 50 < area < 200:
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity > 0.6:
                    # if circularity > 0.8:
                        M = cv2.moments(contour)
                        if M["m00"] != 0:
                            cx, cy = M["m10"]/M["m00"], M["m01"]/M["m00"]
                            equiv_diameter = np.sqrt(4 * area / np.pi)
                            dots.append({
                                'x': float(cx), 'y': float(cy),
                                'r': float(equiv_diameter / 2),
                                'method': 'contour',
                                'area': float(area),
                                'circularity': float(circularity)
                            })

        # Merge duplicates
        if len(dots) > 1:
            coords = np.array([[d['x'], d['y']] for d in dots])
            distances = squareform(pdist(coords))
            to_remove = set()
            for i in range(len(dots)):
                if i in to_remove:
                    continue
                for j in range(i+1, len(dots)):
                    if j in to_remove:
                        continue
                    if distances[i, j] < max(dots[i]['r'], dots[j]['r']) * 1.5:
                        if dots[i].get('circularity', 0.5) >= dots[j].get('circularity', 0.5):
                            to_remove.add(j)
                        else:
                            to_remove.add(i)
            dots = [d for i, d in enumerate(dots) if i not in to_remove]

        # Filter abnormal radii (tighter: ±1σ instead of ±2σ)
        if dots:
            radii = [d['r'] for d in dots]
            mean_r, std_r = np.mean(radii), np.std(radii)
            dots = [d for d in dots if (mean_r - 1*std_r) <= d['r'] <= (mean_r + 1*std_r)]

            # 🔹 Grid snapping to reduce noise
            if len(dots) > 5:
                coords = np.array([[d['x'], d['y']] for d in dots])
                lattice = self.estimate_lattice(dots)
                if lattice and 'basis' in lattice and len(lattice['basis']) >= 2:
                    v1, v2 = np.array(lattice['basis'][0]), np.array(lattice['basis'][1])
                    origin = np.mean(coords, axis=0)

                    snapped = []
                    seen = set()
                    for d in coords:
                        A = np.column_stack((v1, v2))
                        try:
                            coeffs = np.linalg.solve(A, d - origin)
                            i, j = np.round(coeffs).astype(int)
                            snapped_coord = origin + i*v1 + j*v2
                            key = (int(round(snapped_coord[0])), int(round(snapped_coord[1])))
                            if key not in seen:
                                seen.add(key)
                                snapped.append({
                                    'x': float(snapped_coord[0]),
                                    'y': float(snapped_coord[1]),
                                    'r': float(mean_r),
                                    'method': 'snapped'
                                })
                        except np.linalg.LinAlgError:
                            continue
                    dots = snapped
        return dots

    # def estimate_lattice(self, dots):
    #     """Enhanced lattice estimation with multiple lattice types"""
    #     if len(dots) < 4:
    #         return None
        
    #     coords = np.array([[d['x'], d['y']] for d in dots])
        
    #     # Compute all pairwise vectors
    #     vectors = []
    #     for i in range(len(coords)):
    #         for j in range(i + 1, len(coords)):
    #             vec = coords[j] - coords[i]
    #             length = np.linalg.norm(vec)
    #             if length > 0:
    #                 vectors.append({
    #                     'vector': vec,
    #                     'length': length,
    #                     'angle': np.arctan2(vec[1], vec[0]),
    #                     'indices': (i, j)
    #                 })
        
    #     if not vectors:
    #         return None
        
    #     # Cluster vectors by length and angle
    #     vector_data = np.array([[v['length'], v['angle']] for v in vectors])
        
    #     # Normalize for clustering
    #     vector_data_norm = vector_data.copy()
    #     vector_data_norm[:, 0] /= np.max(vector_data_norm[:, 0])  # Normalize length
    #     vector_data_norm[:, 1] = np.cos(vector_data_norm[:, 1])   # Convert angle for clustering
        
    #     # Find dominant vector directions
    #     clustering = DBSCAN(eps=0.2, min_samples=2).fit(vector_data_norm)
    #     labels = clustering.labels_
        
    #     # Analyze clusters to find basis vectors
    #     unique_labels = set(labels) - {-1}
    #     if len(unique_labels) < 1:
    #         return None
        
    #     cluster_info = []
    #     for label in unique_labels:
    #         cluster_vectors = [vectors[i] for i in range(len(vectors)) if labels[i] == label]
    #         if len(cluster_vectors) >= 2:
    #             # Compute representative vector for this cluster
    #             avg_vec = np.mean([v['vector'] for v in cluster_vectors], axis=0)
    #             avg_length = np.mean([v['length'] for v in cluster_vectors])
    #             cluster_info.append({
    #                 'vector': avg_vec,
    #                 'length': avg_length,
    #                 'count': len(cluster_vectors),
    #                 'std_length': np.std([v['length'] for v in cluster_vectors])
    #             })
        
    #     if not cluster_info:
    #         return None
        
    #     # Sort by frequency and select basis vectors
    #     cluster_info.sort(key=lambda x: x['count'], reverse=True)
        
    #     # Determine lattice type
    #     if len(cluster_info) >= 2:
    #         v1, v2 = cluster_info[0]['vector'], cluster_info[1]['vector']
            
    #         # Compute angle between basis vectors
    #         dot_product = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    #         angle_rad = np.arccos(np.clip(dot_product, -1, 1))
    #         angle_deg = np.degrees(angle_rad)
            
    #         # Length ratio
    #         length_ratio = cluster_info[1]['length'] / cluster_info[0]['length']
            
    #         # Classify lattice type
    #         lattice_type = self._classify_lattice_type(angle_deg, length_ratio)
            
    #         # Compute lattice properties
    #         spacing = cluster_info[0]['length']
    #         rotation = np.degrees(np.arctan2(v1[1], v1[0]))
            
    #         return {
    #             'type': lattice_type,
    #             'basis': [v1.tolist(), v2.tolist()],
    #             'spacing': float(spacing),
    #             'rotation_deg': float(rotation),
    #             'angle_between_basis': float(angle_deg),
    #             'length_ratio': float(length_ratio),
    #             'quality_score': float(min(cluster_info[0]['count'], cluster_info[1]['count'])),
    #             'regularity': 1.0 - cluster_info[0]['std_length'] / cluster_info[0]['length']
    #         }
    #     else:
    #         # Single dominant direction - might be a linear arrangement
    #         v1 = cluster_info[0]['vector']
    #         return {
    #             'type': 'linear',
    #             'basis': [v1.tolist()],
    #             'spacing': float(cluster_info[0]['length']),
    #             'rotation_deg': float(np.degrees(np.arctan2(v1[1], v1[0]))),
    #             'quality_score': float(cluster_info[0]['count']),
    #             'regularity': 1.0 - cluster_info[0]['std_length'] / cluster_info[0]['length']
    #         }
    
    def estimate_lattice(self, dots):
        """Improved lattice estimation with fallback methods"""
        if len(dots) < 4:
            return None

        coords = np.array([[d['x'], d['y']] for d in dots])
        vectors = []
        for i in range(len(coords)):
            for j in range(i+1, len(coords)):
                vec = coords[j] - coords[i]
                length = np.linalg.norm(vec)
                if length > 0:
                    vectors.append({'vector': vec, 'length': length, 'angle': np.arctan2(vec[1], vec[0])})
        if not vectors:
            return None

        vector_data = np.array([[v['length'], np.cos(v['angle'])] for v in vectors])
        try:
            clustering = DBSCAN(eps=0.2, min_samples=2).fit(vector_data)
            labels = clustering.labels_
        except Exception:
            clustering = KMeans(n_clusters=2).fit(vector_data)
            labels = clustering.labels_

        unique_labels = set(labels) - {-1}
        if not unique_labels:
            return None

        cluster_info = []
        for label in unique_labels:
            cluster_vectors = [vectors[i] for i in range(len(vectors)) if labels[i] == label]
            if len(cluster_vectors) >= 2:
                avg_vec = np.mean([v['vector'] for v in cluster_vectors], axis=0)
                avg_length = np.mean([v['length'] for v in cluster_vectors])
                cluster_info.append({
                    'vector': avg_vec,
                    'length': avg_length,
                    'count': len(cluster_vectors),
                    'std_length': np.std([v['length'] for v in cluster_vectors])
                })

        if not cluster_info:
            return None
        cluster_info.sort(key=lambda x: x['count'], reverse=True)

        if len(cluster_info) >= 2:
            v1, v2 = cluster_info[0]['vector'], cluster_info[1]['vector']
            dot_product = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            angle_deg = np.degrees(np.arccos(np.clip(dot_product, -1, 1)))
            length_ratio = cluster_info[1]['length'] / cluster_info[0]['length']
            lattice_type = self._classify_lattice_type(angle_deg, length_ratio)
            spacing = cluster_info[0]['length']
            rotation = np.degrees(np.arctan2(v1[1], v1[0]))
            return {
                'type': lattice_type,
                'basis': [v1.tolist(), v2.tolist()],
                'spacing': float(spacing),
                'rotation_deg': float(rotation),
                'angle_between_basis': float(angle_deg),
                'length_ratio': float(length_ratio),
                'quality_score': float(min(cluster_info[0]['count'], cluster_info[1]['count'])),
                'regularity': 1.0 - cluster_info[0]['std_length'] / cluster_info[0]['length']
            }
        else:
            v1 = cluster_info[0]['vector']
            return {
                'type': 'linear',
                'basis': [v1.tolist()],
                'spacing': float(cluster_info[0]['length']),
                'rotation_deg': float(np.degrees(np.arctan2(v1[1], v1[0]))),
                'quality_score': float(cluster_info[0]['count']),
                'regularity': 1.0 - cluster_info[0]['std_length'] / cluster_info[0]['length']
            }

    def _classify_lattice_type(self, angle_deg, length_ratio):
        """Classify lattice based on angle and length ratio"""
        angle_tol = 10
        ratio_tol = 0.2
        
        if abs(angle_deg - 90) < angle_tol and abs(length_ratio - 1.0) < ratio_tol:
            return 'square'
        elif abs(angle_deg - 60) < angle_tol or abs(angle_deg - 120) < angle_tol:
            if abs(length_ratio - 1.0) < ratio_tol:
                return 'triangular'
            else:
                return 'rhombic'
        elif abs(angle_deg - 90) < angle_tol:
            return 'rectangular'
        else:
            return 'oblique'
    
    def skeleton_to_graph(self, skeleton):
        """Convert skeleton to NetworkX graph with enhanced features"""
        # Find skeleton pixels
        y_coords, x_coords = np.where(skeleton)
        skeleton_pixels = set(zip(y_coords, x_coords))
        
        # Compute degree for each pixel
        pixel_degrees = {}
        for y, x in skeleton_pixels:
            degree = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    if (y + dy, x + dx) in skeleton_pixels:
                        degree += 1
            pixel_degrees[(y, x)] = degree
        
        # Find nodes (endpoints and junctions)
        nodes = [(y, x) for (y, x), deg in pixel_degrees.items() if deg != 2]
        
        # Build graph
        G = nx.Graph()
        
        # Add nodes with attributes
        for i, (y, x) in enumerate(nodes):
            G.add_node(i, pos=(x, y), pixel_coord=(y, x), degree=pixel_degrees[(y, x)])
        
        # Trace edges between nodes
        node_positions = {(y, x): i for i, (y, x) in enumerate(nodes)}
        
        for start_idx, (start_y, start_x) in enumerate(nodes):
            # For each neighbor of this node, trace path to next node
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    
                    ny, next_x = start_y + dy, start_x + dx
                    if (ny, next_x) not in skeleton_pixels:
                        continue
                    
                    # Trace path
                    path = self._trace_skeleton_path(skeleton_pixels, (start_y, start_x), (ny, next_x))
                    
                    if path and len(path) > 1:
                        end_y, end_x = path[-1]
                        if (end_y, end_x) in node_positions:
                            end_idx = node_positions[(end_y, end_x)]
                            
                            if start_idx != end_idx and not G.has_edge(start_idx, end_idx):
                                # Compute edge attributes
                                edge_attrs = self._compute_edge_attributes(path)
                                G.add_edge(start_idx, end_idx, **edge_attrs)
        
        return G, nodes
    
    def _trace_skeleton_path(self, skeleton_pixels, start, first_step):
        """Trace a path along skeleton from start through first_step until reaching a node"""
        path = [start, first_step]
        current = first_step
        prev = start
        
        max_iterations = 1000  # Prevent infinite loops
        iterations = 0
        
        while iterations < max_iterations:
            iterations += 1
            
            # Find next pixel
            neighbors = []
            cy, cx = current
            
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, next_x = cy + dy, cx + dx
                    if (ny, next_x) in skeleton_pixels and (ny, next_x) != prev:
                        neighbors.append((ny, next_x))
            
            if len(neighbors) == 0:
                # Dead end
                break
            elif len(neighbors) == 1:
                # Continue along path
                prev = current
                current = neighbors[0]
                path.append(current)
            else:
                # Junction reached
                break
        
        return path
    
    def _compute_edge_attributes(self, path):
        """Compute comprehensive edge attributes"""
        if len(path) < 2:
            return {}
        
        # Convert to numpy array
        path_array = np.array(path)
        
        # Basic measurements
        length = len(path)
        
        # Euclidean length
        euclidean_length = np.sum(np.sqrt(np.sum(np.diff(path_array, axis=0)**2, axis=1)))
        
        # Tortuosity (path length / euclidean distance)
        start, end = path_array[0], path_array[-1]
        straight_distance = np.linalg.norm(end - start)
        tortuosity = euclidean_length / max(straight_distance, 1e-6)
        
        # Average orientation
        if len(path) > 1:
            diffs = np.diff(path_array, axis=0)
            angles = np.arctan2(diffs[:, 0], diffs[:, 1])  # Note: y, x order
            avg_orientation = np.degrees(np.arctan2(np.mean(np.sin(angles)), np.mean(np.cos(angles))))
        else:
            avg_orientation = 0
        
        # Curvature estimation
        curvature = self._estimate_curvature(path_array)
        
        # Directional changes
        directional_changes = 0
        if len(path) > 2:
            for i in range(1, len(path) - 1):
                v1 = path_array[i] - path_array[i-1]
                v2 = path_array[i+1] - path_array[i]
                if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    angle_change = np.arccos(np.clip(cos_angle, -1, 1))
                    if angle_change > np.pi/4:  # 45 degrees
                        directional_changes += 1
        
        return {
            'length': int(length),
            'euclidean_length': float(euclidean_length),
            'tortuosity': float(tortuosity),
            'avg_orientation': float(avg_orientation),
            'curvature': float(curvature),
            'directional_changes': int(directional_changes),
            'pixels': path,
            'start': tuple(map(int, start)),
            'end': tuple(map(int, end))
        }
    
    def _estimate_curvature(self, path_array):
        """Estimate average curvature along path"""
        if len(path_array) < 3:
            return 0.0
        
        curvatures = []
        for i in range(1, len(path_array) - 1):
            # Three consecutive points
            p1, p2, p3 = path_array[i-1], path_array[i], path_array[i+1]
            
            # Vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Cross product magnitude (2D)
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            
            # Lengths
            len1, len2 = np.linalg.norm(v1), np.linalg.norm(v2)
            
            if len1 > 0 and len2 > 0:
                curvature = abs(cross) / (len1 * len2 * (len1 + len2) / 2)
                curvatures.append(curvature)
        
        return np.mean(curvatures) if curvatures else 0.0
    
    def detect_cycles(self, G, nodes):
        """Enhanced cycle detection with geometric properties"""
        try:
            cycle_basis = nx.cycle_basis(G)
        except:
            cycle_basis = []
        
        cycles = []
        for i, cycle in enumerate(cycle_basis):
            if len(cycle) >= 3:
                # Get node coordinates
                coords = []
                for node in cycle:
                    pos = G.nodes[node]['pos']
                    coords.append([pos[0], pos[1]])  # x, y
                
                coords = np.array(coords)
                
                # Compute cycle properties
                cycle_info = self._compute_cycle_properties(coords, cycle)
                cycle_info['id'] = i
                cycle_info['nodes'] = cycle
                
                cycles.append(cycle_info)
        
        # Sort by area (largest first)
        cycles.sort(key=lambda x: x['area'], reverse=True)
        
        # Detect nesting relationships
        cycles = self._detect_nested_cycles(cycles)
        
        return cycles
    
    def _compute_cycle_properties(self, coords, node_ids):
        """Compute comprehensive geometric properties of a cycle"""
        if len(coords) < 3:
            return {}
        
        # Close the polygon
        closed_coords = np.vstack([coords, coords[0]])
        
        # Area using shoelace formula
        area = 0.5 * abs(sum(closed_coords[i][0] * (closed_coords[i+1][1] - closed_coords[i-1][1]) 
                            for i in range(len(coords))))
        
        # Perimeter
        perimeter = np.sum([np.linalg.norm(closed_coords[i+1] - closed_coords[i]) 
                           for i in range(len(coords))])
        
        # Centroid
        cx = np.mean(coords[:, 0])
        cy = np.mean(coords[:, 1])
        centroid = [float(cx), float(cy)]
        
        # Bounding box
        min_x, max_x = np.min(coords[:, 0]), np.max(coords[:, 0])
        min_y, max_y = np.min(coords[:, 1]), np.max(coords[:, 1])
        bbox = [float(min_x), float(min_y), float(max_x - min_x), float(max_y - min_y)]
        
        # Convexity (area ratio to convex hull)
        try:
            from scipy.spatial import ConvexHull
            if len(coords) >= 3:
                hull = ConvexHull(coords)
                convex_area = hull.volume  # 2D volume is area
                convexity = area / max(convex_area, 1e-6)
            else:
                convexity = 1.0
        except:
            convexity = 1.0
        
        # Regularity (how close to regular polygon)
        regularity = self._compute_polygon_regularity(coords)
        
        # Aspect ratio
        aspect_ratio = (max_x - min_x) / max(max_y - min_y, 1e-6)
        
        return {
            'area': float(area),
            'perimeter': float(perimeter),
            'centroid': centroid,
            'bbox': bbox,
            'convexity': float(convexity),
            'regularity': float(regularity),
            'aspect_ratio': float(aspect_ratio),
            'num_vertices': len(coords)
        }
    
    def _compute_polygon_regularity(self, coords):
        """Compute how regular a polygon is (1.0 = perfect regular polygon)"""
        if len(coords) < 3:
            return 0.0
        
        # Compute side lengths
        n = len(coords)
        sides = []
        for i in range(n):
            side_length = np.linalg.norm(coords[(i+1) % n] - coords[i])
            sides.append(side_length)
        
        # Compute angles
        angles = []
        for i in range(n):
            v1 = coords[i] - coords[(i-1) % n]
            v2 = coords[(i+1) % n] - coords[i]
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                angles.append(angle)
        
        # Regularity based on coefficient of variation
        side_regularity = 1.0 - (np.std(sides) / max(np.mean(sides), 1e-6))
        if angles:
            angle_regularity = 1.0 - (np.std(angles) / max(np.mean(angles), 1e-6))
        else:
            angle_regularity = 0.0
        
        return max(0.0, min(1.0, (side_regularity + angle_regularity) / 2))
    
    def _detect_nested_cycles(self, cycles):
        """Detect which cycles are nested within others"""
        for i, cycle1 in enumerate(cycles):
            cycle1['nested_in'] = []
            cycle1['contains'] = []
            
            for j, cycle2 in enumerate(cycles):
                if i != j:
                    # Check if cycle1 is inside cycle2
                    if self._is_cycle_inside_cycle(cycle1, cycle2):
                        cycle1['nested_in'].append(j)
                        cycle2['contains'].append(i)
        
        return cycles
    
    def _is_cycle_inside_cycle(self, inner_cycle, outer_cycle):
        """Check if inner cycle is completely inside outer cycle"""
        inner_centroid = inner_cycle['centroid']
        
        # Simple approach: check if centroid is inside bounding box
        outer_bbox = outer_cycle['bbox']
        ox, oy, ow, oh = outer_bbox
        
        if (inner_centroid[0] >= ox and inner_centroid[0] <= ox + ow and
            inner_centroid[1] >= oy and inner_centroid[1] <= oy + oh):
            # More sophisticated check could use point-in-polygon
            return outer_cycle['area'] > inner_cycle['area'] * 1.2  # Some margin
        
        return False
    
    def detect_symmetries(self, G, nodes):
        """Advanced symmetry detection"""
        if len(nodes) < 2:
            return {'rotational_order': 1, 'mirror_axes': 0}
        
        # Get node coordinates
        coords = np.array([G.nodes[i]['pos'] for i in range(len(nodes))])
        
        # Compute centroid
        centroid = np.mean(coords, axis=0)
        
        # Center coordinates
        centered_coords = coords - centroid
        
        # Rotational symmetry detection
        rotational_order = self._detect_rotational_symmetry(centered_coords)
        
        # Mirror symmetry detection
        mirror_axes = self._detect_mirror_symmetry(centered_coords)
        
        return {
            'rotational_order': rotational_order,
            'mirror_axes': len(mirror_axes),
            'mirror_axis_angles': [float(np.degrees(angle)) for angle in mirror_axes],
            'centroid': centroid.tolist()
        }
    
    def _detect_rotational_symmetry(self, centered_coords):
        """Detect rotational symmetry order"""
        max_order = 12
        tolerance = self.config['SYMMETRY_TOLERANCE']
        
        for order in range(2, max_order + 1):
            angle = 2 * np.pi / order
            rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                      [np.sin(angle), np.cos(angle)]])
            
            rotated_coords = np.dot(centered_coords, rotation_matrix.T)
            
            # Find best matching using KDTree
            if len(centered_coords) > 0:
                tree = KDTree(centered_coords)
                distances, _ = tree.query(rotated_coords)
                median_distance = np.median(distances)
                
                if median_distance < tolerance:
                    return order
        
        return 1  # No rotational symmetry found
    
    def _detect_mirror_symmetry(self, centered_coords):
        """Detect mirror symmetry axes"""
        mirror_axes = []
        tolerance = self.config['SYMMETRY_TOLERANCE']
        
        # Test multiple potential mirror axes
        for angle in np.linspace(0, np.pi, 36):  # Test 36 directions
            # Mirror across line through origin at this angle
            cos_2a = np.cos(2 * angle)
            sin_2a = np.sin(2 * angle)
            
            reflection_matrix = np.array([[cos_2a, sin_2a],
                                        [sin_2a, -cos_2a]])
            
            reflected_coords = np.dot(centered_coords, reflection_matrix.T)
            
            # Check if reflected points match original points
            if len(centered_coords) > 0:
                tree = KDTree(centered_coords)
                distances, _ = tree.query(reflected_coords)
                median_distance = np.median(distances)
                
                if median_distance < tolerance:
                    mirror_axes.append(angle)
        
        # Remove similar axes (within 10 degrees)
        filtered_axes = []
        for axis in mirror_axes:
            is_new = True
            for existing in filtered_axes:
                if abs(axis - existing) < np.radians(10):
                    is_new = False
                    break
            if is_new:
                filtered_axes.append(axis)
        
        return filtered_axes
    
    def detect_foreign_lines(self, G, nodes, lattice_info, dots):
        """Enhanced foreign line detection"""
        foreign_lines = []
        
        if not lattice_info or not dots:
            return foreign_lines
        
        # Create lattice model
        lattice_model = self._create_lattice_model(lattice_info, dots)
        
        for edge in G.edges(data=True):
            u, v = edge[0], edge[1]
            edge_data = edge[2]
            
            # Get edge properties
            start_pos = G.nodes[u]['pos']
            end_pos = G.nodes[v]['pos']
            edge_angle = edge_data.get('avg_orientation', 0)
            
            # Check if edge aligns with lattice
            is_foreign, reason = self._is_edge_foreign(
                start_pos, end_pos, edge_angle, lattice_model, lattice_info
            )
            
            if is_foreign:
                foreign_lines.append({
                    'edge_id': f"{u}-{v}",
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'reason': reason,
                    'angle': float(edge_angle),
                    'length': edge_data.get('euclidean_length', 0)
                })
        
        return foreign_lines
    
    def _create_lattice_model(self, lattice_info, dots):
        """Create a model of the expected lattice"""
        if not lattice_info or 'basis' not in lattice_info:
            return None
        
        dot_coords = np.array([[d['x'], d['y']] for d in dots])
        basis_vectors = np.array(lattice_info['basis'])
        
        # Find lattice origin (closest dot to image center)
        image_center = np.array([256, 256])  # Assuming 512x512 after preprocessing
        distances_to_center = np.linalg.norm(dot_coords - image_center, axis=1)
        origin_idx = np.argmin(distances_to_center)
        origin = dot_coords[origin_idx]
        
        # Generate lattice points in a reasonable range
        max_range = 10  # Generate lattice points in a 21x21 grid around origin
        lattice_points = []
        
        if len(basis_vectors) >= 2:
            for i in range(-max_range, max_range + 1):
                for j in range(-max_range, max_range + 1):
                    point = origin + i * basis_vectors[0] + j * basis_vectors[1]
                    lattice_points.append(point)
        else:
            # Linear lattice
            for i in range(-max_range, max_range + 1):
                point = origin + i * basis_vectors[0]
                lattice_points.append(point)
        
        return {
            'origin': origin,
            'basis_vectors': basis_vectors,
            'lattice_points': np.array(lattice_points),
            'lattice_angles': [np.degrees(np.arctan2(v[1], v[0])) for v in basis_vectors]
        }
    
    def _is_edge_foreign(self, start_pos, end_pos, edge_angle, lattice_model, lattice_info):
        """Determine if an edge is foreign to the lattice structure"""
        if lattice_model is None:
            return False, "no_lattice"
        
        angle_threshold = self.config['ANGLE_THRESHOLD']
        
        # Check angle alignment with lattice directions
        lattice_angles = lattice_model['lattice_angles']
        min_angle_diff = float('inf')
        
        for lattice_angle in lattice_angles:
            # Consider both directions (angle and angle + 180)
            for test_angle in [lattice_angle, lattice_angle + 180]:
                angle_diff = abs(edge_angle - test_angle)
                angle_diff = min(angle_diff, 360 - angle_diff)  # Handle wraparound
                min_angle_diff = min(min_angle_diff, angle_diff)
        
        if min_angle_diff > angle_threshold:
            return True, f"angle_mismatch_{min_angle_diff:.1f}deg"
        
        # Check if endpoints are near lattice points
        lattice_points = lattice_model['lattice_points']
        tree = KDTree(lattice_points)
        
        start_dist, _ = tree.query(start_pos)
        end_dist, _ = tree.query(end_pos)
        
        lattice_spacing = lattice_info.get('spacing', 20)
        tolerance = lattice_spacing * 0.3  # 30% tolerance
        
        if start_dist > tolerance or end_dist > tolerance:
            return True, f"off_lattice_endpoints_{max(start_dist, end_dist):.1f}px"
        
        return False, "lattice_aligned"
    
    def analyze_complex_patterns(self, G, cycles, dots, lattice_info):
        """Analyze complex pattern features beyond basic topology"""
        patterns = {}
        
        # Pattern density analysis
        patterns['density'] = self._analyze_pattern_density(G, dots)
        
        # Connectivity analysis
        patterns['connectivity'] = self._analyze_connectivity(G)
        
        # Path complexity analysis
        patterns['path_complexity'] = self._analyze_path_complexity(G)
        
        # Cycle hierarchy analysis
        patterns['cycle_hierarchy'] = self._analyze_cycle_hierarchy(cycles)
        
        # Symmetry breaking analysis
        patterns['symmetry_breaking'] = self._analyze_symmetry_breaking(G, lattice_info)
        
        # Fractal dimension estimation
        patterns['fractal_dimension'] = self._estimate_fractal_dimension(G)
        
        # Pattern completeness (how much of expected pattern is present)
        patterns['completeness'] = self._analyze_pattern_completeness(dots, lattice_info)
        
        return patterns
    
    def _analyze_pattern_density(self, G, dots):
        """Analyze the density distribution of pattern elements"""
        if not dots:
            return {'overall': 0, 'local_variations': []}
        
        dot_coords = np.array([[d['x'], d['y']] for d in dots])
        
        # Overall density (dots per unit area)
        if len(dot_coords) > 0:
            min_x, max_x = np.min(dot_coords[:, 0]), np.max(dot_coords[:, 0])
            min_y, max_y = np.min(dot_coords[:, 1]), np.max(dot_coords[:, 1])
            area = (max_x - min_x) * (max_y - min_y)
            overall_density = len(dots) / max(area, 1)
        else:
            overall_density = 0
        
        # Local density variations using grid analysis
        grid_size = 5
        local_densities = []
        
        if len(dot_coords) > 0:
            x_bins = np.linspace(min_x, max_x, grid_size + 1)
            y_bins = np.linspace(min_y, max_y, grid_size + 1)
            
            for i in range(grid_size):
                for j in range(grid_size):
                    # Count dots in this grid cell
                    x_mask = (dot_coords[:, 0] >= x_bins[i]) & (dot_coords[:, 0] < x_bins[i + 1])
                    y_mask = (dot_coords[:, 1] >= y_bins[j]) & (dot_coords[:, 1] < y_bins[j + 1])
                    cell_count = np.sum(x_mask & y_mask)
                    cell_area = (x_bins[i + 1] - x_bins[i]) * (y_bins[j + 1] - y_bins[j])
                    cell_density = cell_count / max(cell_area, 1)
                    local_densities.append(cell_density)
        
        return {
            'overall': float(overall_density),
            'local_mean': float(np.mean(local_densities)) if local_densities else 0,
            'local_std': float(np.std(local_densities)) if local_densities else 0,
            'density_variation': float(np.std(local_densities) / max(np.mean(local_densities), 1e-6)) if local_densities else 0
        }
    
    def _analyze_connectivity(self, G):
        """Analyze graph connectivity properties"""
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()
        
        if n_nodes == 0:
            return {'components': 0, 'largest_component_ratio': 0, 'average_degree': 0}
        
        # Connected components
        components = list(nx.connected_components(G))
        n_components = len(components)
        largest_component_size = max(len(comp) for comp in components) if components else 0
        largest_component_ratio = largest_component_size / n_nodes
        
        # Average degree
        degrees = [d for n, d in G.degree()]
        average_degree = np.mean(degrees) if degrees else 0
        
        # Clustering coefficient
        try:
            clustering = nx.average_clustering(G)
        except:
            clustering = 0
        
        return {
            'components': n_components,
            'largest_component_ratio': float(largest_component_ratio),
            'average_degree': float(average_degree),
            'clustering_coefficient': float(clustering),
            'edge_density': float(2 * n_edges / max(n_nodes * (n_nodes - 1), 1))
        }
    
    def _analyze_path_complexity(self, G):
        """Analyze complexity of paths in the graph"""
        complexities = []
        
        for edge in G.edges(data=True):
            edge_data = edge[2]
            tortuosity = edge_data.get('tortuosity', 1.0)
            curvature = edge_data.get('curvature', 0.0)
            directional_changes = edge_data.get('directional_changes', 0)
            
            # Composite complexity measure
            complexity = tortuosity + curvature * 10 + directional_changes * 0.1
            complexities.append(complexity)
        
        if not complexities:
            return {'mean': 0, 'std': 0, 'max': 0}
        
        return {
            'mean': float(np.mean(complexities)),
            'std': float(np.std(complexities)),
            'max': float(np.max(complexities)),
            'high_complexity_ratio': float(np.sum(np.array(complexities) > 2.0) / len(complexities))
        }
    
    def _analyze_cycle_hierarchy(self, cycles):
        """Analyze the hierarchical structure of cycles"""
        if not cycles:
            return {'max_nesting_depth': 0, 'total_nested_cycles': 0}
        
        # Find maximum nesting depth
        max_depth = 0
        nested_count = 0
        
        for cycle in cycles:
            nested_in = cycle.get('nested_in', [])
            if nested_in:
                nested_count += 1
                # Simple depth calculation (could be more sophisticated)
                depth = len(nested_in)
                max_depth = max(max_depth, depth)
        
        # Area ratios between nested cycles
        area_ratios = []
        for cycle in cycles:
            if cycle.get('nested_in'):
                cycle_area = cycle['area']
                for parent_id in cycle['nested_in']:
                    if parent_id < len(cycles):
                        parent_area = cycles[parent_id]['area']
                        if parent_area > 0:
                            ratio = cycle_area / parent_area
                            area_ratios.append(ratio)
        
        return {
            'max_nesting_depth': max_depth,
            'total_nested_cycles': nested_count,
            'nesting_ratio': float(nested_count / len(cycles)),
            'average_area_ratio': float(np.mean(area_ratios)) if area_ratios else 0
        }
    
    def _analyze_symmetry_breaking(self, G, lattice_info):
        """Analyze deviations from perfect symmetry"""
        if not lattice_info:
            return {'score': 0, 'type': 'no_lattice'}
        
        # Get node positions
        if G.number_of_nodes() == 0:
            return {'score': 0, 'type': 'no_nodes'}
        
        positions = np.array([G.nodes[i]['pos'] for i in range(G.number_of_nodes())])
        
        # Expected vs actual symmetry
        expected_order = 4 if lattice_info.get('type') == 'square' else 3 if lattice_info.get('type') == 'triangular' else 2
        
        # Measure deviation from expected lattice positions
        if 'spacing' in lattice_info and positions.shape[0] > 0:
            spacing = lattice_info['spacing']
            
            # Simple symmetry breaking measure: standard deviation of nearest neighbor distances
            from scipy.spatial.distance import pdist
            if len(positions) > 1:
                distances = pdist(positions)
                expected_distance = spacing
                distance_deviation = np.std(distances) / max(expected_distance, 1)
            else:
                distance_deviation = 0
        else:
            distance_deviation = 0
        
        symmetry_breaking_score = min(1.0, distance_deviation)
        
        return {
            'score': float(symmetry_breaking_score),
            'distance_deviation': float(distance_deviation),
            'type': 'lattice_deviation'
        }
    
    def _estimate_fractal_dimension(self, G):
        """Estimate fractal dimension using box-counting method"""
        if G.number_of_nodes() == 0:
            return 1.0
        
        # Get all edge pixels
        all_pixels = []
        for edge in G.edges(data=True):
            edge_data = edge[2]
            pixels = edge_data.get('pixels', [])
            all_pixels.extend(pixels)
        
        if not all_pixels:
            return 1.0
        
        pixels_array = np.array(all_pixels)
        
        # Box counting at different scales
        scales = [2, 4, 8, 16, 32]
        counts = []
        
        for scale in scales:
            # Create grid at this scale
            min_x, max_x = np.min(pixels_array[:, 1]), np.max(pixels_array[:, 1])  # Note: pixels are (y, x)
            min_y, max_y = np.min(pixels_array[:, 0]), np.max(pixels_array[:, 0])
            
            x_bins = np.arange(min_x, max_x + scale, scale)
            y_bins = np.arange(min_y, max_y + scale, scale)
            
            # Count occupied boxes
            occupied_boxes = set()
            for pixel in pixels_array:
                y, x = pixel
                box_x = int((x - min_x) // scale)
                box_y = int((y - min_y) // scale)
                occupied_boxes.add((box_x, box_y))
            
            counts.append(len(occupied_boxes))
        
        # Fit line to log-log plot to estimate dimension
        if len(counts) > 1 and all(c > 0 for c in counts):
            log_scales = np.log(scales)
            log_counts = np.log(counts)
            
            # Linear regression
            A = np.vstack([log_scales, np.ones(len(log_scales))]).T
            slope, _ = np.linalg.lstsq(A, log_counts, rcond=None)[0]
            
            fractal_dim = -slope  # Box-counting dimension
            return max(1.0, min(2.0, fractal_dim))  # Clamp to reasonable range
        
        return 1.5  # Default estimate
    
    def _analyze_pattern_completeness(self, dots, lattice_info):
        """Analyze how complete the pattern is compared to expected lattice"""
        if not lattice_info or not dots:
            return {'score': 0, 'missing_dots': [], 'extra_dots': []}
        
        # Generate expected lattice points
        lattice_model = self._create_lattice_model(lattice_info, dots)
        if not lattice_model:
            return {'score': 0, 'missing_dots': [], 'extra_dots': []}
        
        expected_points = lattice_model['lattice_points']
        actual_points = np.array([[d['x'], d['y']] for d in dots])
        
        # Find missing and extra points
        tolerance = lattice_info.get('spacing', 20) * 0.3
        
        # Use KDTree for efficient nearest neighbor queries
        if len(actual_points) > 0:
            actual_tree = KDTree(actual_points)
            missing_dots = []
            
            for expected_point in expected_points:
                dist, _ = actual_tree.query(expected_point)
                if dist > tolerance:
                    missing_dots.append(expected_point.tolist())
        else:
            missing_dots = expected_points.tolist()
        
        if len(expected_points) > 0:
            expected_tree = KDTree(expected_points)
            extra_dots = []
            
            for actual_point in actual_points:
                dist, _ = expected_tree.query(actual_point)
                if dist > tolerance:
                    extra_dots.append(actual_point.tolist())
        else:
            extra_dots = actual_points.tolist()
        
        # Completeness score
        expected_count = len(expected_points)
        found_count = expected_count - len(missing_dots)
        completeness_score = found_count / max(expected_count, 1)
        
        return {
            'score': float(max(0, min(1, completeness_score))),
            'expected_dots': int(expected_count),
            'found_dots': int(found_count),
            'missing_dots': len(missing_dots),
            'extra_dots': len(extra_dots)
        }
    
    def generate_human_readable_rules(self, analysis_result):
        """Generate human-readable description of the pattern"""
        rules = []
        
        # Lattice description
        lattice = analysis_result.get('lattice')
        if lattice:
            lattice_type = lattice['type']
            spacing = lattice['spacing']
            rotation = lattice.get('rotation_deg', 0)
            
            rule = f"{lattice_type.title()} lattice with spacing ≈{spacing:.1f} px"
            if abs(rotation) > 5:
                rule += f" rotated {rotation:.1f}°"
            rules.append(rule)
        
        # Symmetry description
        symmetry = analysis_result.get('symmetry', {})
        rot_order = symmetry.get('rotational_order', 1)
        mirror_axes = symmetry.get('mirror_axes', 0)
        
        if rot_order > 1:
            rules.append(f"{rot_order}-fold rotational symmetry")
        if mirror_axes > 0:
            rules.append(f"{mirror_axes} mirror symmetr{'y' if mirror_axes == 1 else 'ies'}")
        
        # Cycle description
        cycles = analysis_result.get('cycles', [])
        if cycles:
            nested_cycles = [c for c in cycles if c.get('nested_in')]
            if nested_cycles:
                max_area_cycle = max(cycles, key=lambda x: x['area'])
                rules.append(f"{len(nested_cycles)} nested loops around central region")
            
            regular_cycles = [c for c in cycles if c.get('regularity', 0) > 0.8]
            if regular_cycles:
                rules.append(f"{len(regular_cycles)} highly regular cycle{'s' if len(regular_cycles) > 1 else ''}")
        
        # Foreign lines description
        foreign_lines = analysis_result.get('foreign_lines', [])
        if foreign_lines:
            angle_mismatches = [f for f in foreign_lines if 'angle_mismatch' in f.get('reason', '')]
            if angle_mismatches:
                avg_angle = np.mean([f['angle'] for f in angle_mismatches])
                rules.append(f"{len(angle_mismatches)} connecting lines at ≈{avg_angle:.0f}° to lattice")
        
        # Pattern complexity
        patterns = analysis_result.get('patterns', {})
        if patterns:
            path_complexity = patterns.get('path_complexity', {})
            if path_complexity.get('high_complexity_ratio', 0) > 0.3:
                rules.append("High path complexity with curved connections")
            
            connectivity = patterns.get('connectivity', {})
            if connectivity.get('components', 1) > 1:
                rules.append(f"Pattern consists of {connectivity['components']} separate components")
        
        # Completeness
        completeness = patterns.get('completeness', {}) if patterns else {}
        if completeness.get('score', 1) < 0.8:
            rules.append(f"Pattern appears {completeness['score']*100:.0f}% complete")
        
        return ". ".join(rules) + "." if rules else "Simple pattern with basic structure."
    
    # def _sanitize_for_json(self, obj):
    #     """Recursively convert NumPy types to Python native types for JSON serialization"""
    #     if isinstance(obj, dict):
    #         return {k: self._sanitize_for_json(v) for k, v in obj.items()}
    #     elif isinstance(obj, list):
    #         return [self._sanitize_for_json(v) for v in obj]
    #     elif isinstance(obj, (np.integer, np.int32, np.int64)):
    #         return int(obj)
    #     elif isinstance(obj, (np.floating, np.float32, np.float64)):
    #         return float(obj)
    #     elif isinstance(obj, np.ndarray):
    #         return obj.tolist()
    #     else:
    #         return obj


    def analyze(self, image):
        """Main analysis function that orchestrates all analysis steps"""
        try:
            # Step 1: Preprocessing
            preprocessed = self.preprocess(image)
            
            # Step 2: Dot detection
            dots = self.detect_dots(preprocessed['gray'], preprocessed['binary'])
            
            # Step 3: Lattice estimation
            lattice_info = None
            if len(dots) >= self.config['DOT_THRESHOLD']:
                lattice_info = self.estimate_lattice(dots)
            
            # Step 4: Skeleton to graph conversion
            G, nodes = self.skeleton_to_graph(preprocessed['skeleton'])
            
            # Step 5: Cycle detection
            cycles = self.detect_cycles(G, nodes)
            
            # Step 6: Symmetry detection
            symmetry = self.detect_symmetries(G, nodes)
            
            # Step 7: Foreign line detection
            foreign_lines = self.detect_foreign_lines(G, nodes, lattice_info, dots)
            
            # Step 8: Advanced pattern analysis
            patterns = self.analyze_complex_patterns(G, cycles, dots, lattice_info)
            
            # Step 9: Generate human-readable rules
            analysis_result = {
                'dots': dots,
                'lattice': lattice_info,
                'graph': {
                    'nodes': [[int(G.nodes[i]['pos'][1]), int(G.nodes[i]['pos'][0])] for i in range(len(nodes))],
                    'edges': [
                        {
                            'u': int(u), 'v': int(v),
                            'length': int(data.get('length', 0)),
                            'euclidean_length': float(data.get('euclidean_length', 0)),
                            'tortuosity': float(data.get('tortuosity', 1)),
                            'pixels': data.get('pixels', [])
                        }
                        for u, v, data in G.edges(data=True)
                    ]
                },
                'cycles': cycles,
                'symmetry': symmetry,
                'foreign_lines': foreign_lines,
                'patterns': patterns,
                'preprocessing_info': {
                    'original_size': preprocessed['original'].shape,
                    'scale_factor': preprocessed['scale_factor'],
                    'num_skeleton_pixels': int(np.sum(preprocessed['skeleton']))
                }
            }
            
            human_rules = self.generate_human_readable_rules(analysis_result)
            analysis_result['human_rules'] = human_rules
            
            # return self._sanitize_for_json(analysis_result)
            return analysis_result
            
        except Exception as e:
            return {
                'error': str(e),
                'dots': [],
                'lattice': None,
                'graph': {'nodes': [], 'edges': []},
                'cycles': [],
                'symmetry': {'rotational_order': 1, 'mirror_axes': 0},
                'foreign_lines': [],
                'patterns': {},
                'human_rules': f"Analysis failed: {str(e)}"
            }

# Example usage and testing function
def test_analyzer():
    """Test the analyzer with a synthetic kolam pattern"""
    # Create a simple test image (you would replace this with actual image loading)
    test_image = np.zeros((400, 400), dtype=np.uint8)
    
    # Add some dots in a grid pattern
    for i in range(5, 400, 40):
        for j in range(5, 400, 40):
            cv2.circle(test_image, (i, j), 3, 255, -1)
    
    # Add some connecting lines
    for i in range(5, 400, 40):
        for j in range(5, 360, 40):
            cv2.line(test_image, (i, j), (i, j+40), 255, 2)
            cv2.line(test_image, (i, j), (i+40, j), 255, 2)
    
    # Initialize analyzer
    analyzer = KolamAnalyzer()
    
    # Analyze the pattern
    result = analyzer.analyze(test_image)
    
    # Print results
    print("Analysis Results:")
    print(f"Dots found: {len(result['dots'])}")
    print(f"Lattice type: {result['lattice']['type'] if result['lattice'] else 'None'}")
    print(f"Graph nodes: {len(result['graph']['nodes'])}")
    print(f"Graph edges: {len(result['graph']['edges'])}")
    print(f"Cycles found: {len(result['cycles'])}")
    print(f"Human rules: {result['human_rules']}")
    
    return result

if __name__ == "__main__":
    # Run test
    test_result = test_analyzer()