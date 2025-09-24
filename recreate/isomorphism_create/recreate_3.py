# test_real_kolam.py
import cv2
import numpy as np
import json
import os
from Math_analyser import KolamAnalyzer
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def analyze_real_kolam(image_path, output_dir="public/results"):
    """Analyze a real kolam image"""
    print("=" * 60)
    print(f"ANALYZING REAL KOLAM IMAGE")
    print("=" * 60)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return None
    
    try:
        # Load the image
        print(f"📂 Loading image: {image_path}")
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"❌ Error: Could not load image. Check if file is a valid image format.")
            return None
        
        print(f"✅ Image loaded successfully: {img.shape}")
        print(f"   Width: {img.shape[1]}px, Height: {img.shape[0]}px")
        
        # Show image info
        if len(img.shape) == 3:
            print(f"   Color channels: {img.shape[2]}")
        else:
            print("   Grayscale image")
        
        # Initialize analyzer with custom config for better dot detection
        config = {
            'DOT_THRESHOLD': 15,  # Lower threshold for complex patterns
            # 'DOT_THRESHOLD': 10,  # Lower threshold for complex patterns
            'DBSCAN_EPS': 8,      # Smaller epsilon for tighter clustering
            # 'DBSCAN_EPS': 6,      # Smaller epsilon for tighter clustering
            'ANGLE_THRESHOLD': 20, # More tolerance for angle variations
            'SYMMETRY_TOLERANCE': 10, # More tolerance for symmetry detection
            'BLOB_THRESHOLD': 0.05,  # Lower threshold for blob detection
            # 'BLOB_THRESHOLD': 0.02,  # Lower threshold for blob detection
            'TARGET_SIZE': 512     # Good size for analysis
        }
        
        analyzer = KolamAnalyzer(config)
        
        # Run analysis
        print("\n🔍 Running comprehensive analysis...")
        print("   This may take a few moments for complex patterns...")
        
        result = analyzer.analyze(img)
        
        # Check for errors
        if 'error' in result:
            print(f"❌ Analysis error: {result['error']}")
            return result
        
        # Display detailed results
        print_detailed_results(result, output_dir)
        
        # Save results
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        json_output = f"{base_name}_analysis.json"
        
        # Save analysis results
        with open(os.path.join(output_dir, f"{base_name}_analysis.json"), 'w', encoding="utf-8") as f:
            json.dump(result, f, indent=2,default = str)
        # Create visualization
        create_analysis_visualization(img, result, base_name, output_dir)
        
        return result
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_detailed_results(result, output_dir):
    """Print comprehensive analysis results"""
    print("\n" + "=" * 60)
    print("🎯 KOLAM ANALYSIS RESULTS")
    print("=" * 60)
    with open(os.path.join(output_dir, "analysis.txt"), "w",encoding="utf-8") as f:
        f.write("KOLAM ANALYSIS RESULTS\n")
    # Preprocessing info
    prep_info = result.get('preprocessing_info', {})
    print(f"\n📊 IMAGE PROCESSING:")
    print(f"   Original size: {prep_info.get('original_size', 'Unknown')}")
    print(f"   Scale factor: {prep_info.get('scale_factor', 1.0):.2f}")
    print(f"   Skeleton pixels: {prep_info.get('num_skeleton_pixels', 0)}")
    with open(os.path.join(output_dir, "analysis.txt"), "a",encoding="utf-8") as f:
        f.write(f"   IMAGE PROCESSING:\n")
        f.write(f"   Original size: {prep_info.get('original_size', 'Unknown')}\n")
        f.write(f"   Scale factor: {prep_info.get('scale_factor', 1.0):.2f}\n")
        f.write(f"   Skeleton pixels: {prep_info.get('num_skeleton_pixels', 0)}\n")

    # Dot analysis
    dots = result.get('dots', [])
    print(f"\n 🔴 DOT ANALYSIS:")
    print(f"   Total dots detected: {len(dots)}")
    if dots:
        blob_dots = [d for d in dots if d.get('method') == 'blob_log']
        contour_dots = [d for d in dots if d.get('method') == 'contour']
        print(f"   - Blob detection: {len(blob_dots)}")
        print(f"   - Contour detection: {len(contour_dots)}")
        
        avg_radius = np.mean([d['r'] for d in dots])
        print(f"   Average dot radius: {avg_radius:.1f}px")
    with open(os.path.join(output_dir, "analysis.txt"), "a",encoding="utf-8") as f:
        f.write(f"   DOT ANALYSIS:\n")
        f.write(f"   Total dots detected: {len(dots)}\n")
        if dots:
            blob_dots = [d for d in dots if d.get('method') == 'blob_log']
            contour_dots = [d for d in dots if d.get('method') == 'contour']
            f.write(f"   - Blob detection: {len(blob_dots)}\n")
            f.write(f"   - Contour detection: {len(contour_dots)}\n")
            
            avg_radius = np.mean([d['r'] for d in dots])
            f.write(f"   Average dot radius: {avg_radius:.1f}px\n")
    
    # Lattice analysis
    lattice = result.get('lattice')
    print(f"\n🔲 LATTICE STRUCTURE:")
    if lattice:
        print(f"   Type: {lattice['type'].upper()}")
        print(f"   Spacing: {lattice['spacing']:.1f}px")
        print(f"   Rotation: {lattice['rotation_deg']:.1f}°")
        print(f"   Quality score: {lattice['quality_score']:.1f}")
        print(f"   Regularity: {lattice['regularity']:.2f}")
        if 'angle_between_basis' in lattice:
            print(f"   Basis angle: {lattice['angle_between_basis']:.1f}°")
        with open(os.path.join(output_dir, "analysis.txt"), "a",encoding="utf-8") as f:
            f.write(f"   LATTICE STRUCTURE:\n")
            f.write(f"   Type: {lattice['type'].upper()}\n")
            f.write(f"   Spacing: {lattice['spacing']:.1f}px\n")
            f.write(f"   Rotation: {lattice['rotation_deg']:.1f}°\n")
            f.write(f"   Quality score: {lattice['quality_score']:.1f}\n")
            f.write(f"   Regularity: {lattice['regularity']:.2f}\n")
            if 'angle_between_basis' in lattice:
                f.write(f"   Basis angle: {lattice['angle_between_basis']:.1f}°\n")
    else:
        print("   No regular lattice structure detected")
        with open(os.path.join(output_dir, "analysis.txt"), "a",encoding="utf-8") as f:
            f.write("   LATTICE STRUCTURE:\n")
            f.write("   No regular lattice structure detected\n")
    
    # Graph topology
    graph = result.get('graph', {})
    nodes = graph.get('nodes', [])
    edges = graph.get('edges', [])
    print(f"\n🌐 GRAPH TOPOLOGY:")
    print(f"   Nodes (junctions): {len(nodes)}")
    print(f"   Edges (connections): {len(edges)}")
    with open(os.path.join(output_dir, "analysis.txt"), "a",encoding="utf-8") as f:
        f.write(f"   GRAPH TOPOLOGY:\n")
        f.write(f"   Nodes (junctions): {len(nodes)}\n")
        f.write(f"   Edges (connections): {len(edges)}\n") 
    
    if edges:
        avg_length = np.mean([e.get('euclidean_length', 0) for e in edges])
        avg_tortuosity = np.mean([e.get('tortuosity', 1) for e in edges])
        print(f"   Average edge length: {avg_length:.1f}px")
        print(f"   Average tortuosity: {avg_tortuosity:.2f}")
    with open(os.path.join(output_dir, "analysis.txt"), "a",encoding="utf-8") as f:
        if edges:
            avg_length = np.mean([e.get('euclidean_length', 0) for e in edges])
            avg_tortuosity = np.mean([e.get('tortuosity', 1) for e in edges])
            f.write(f"   Average edge length: {avg_length:.1f}px\n")
            f.write(f"   Average tortuosity: {avg_tortuosity:.2f}\n")
    
    # Cycle analysis
    cycles = result.get('cycles', [])
    print(f"\n🔄 CYCLES & LOOPS:")
    print(f"   Total cycles found: {len(cycles)}")
    if cycles:
        nested_cycles = [c for c in cycles if c.get('nested_in')]
        print(f"   - Nested cycles: {len(nested_cycles)}")
        
        largest_cycle = max(cycles, key=lambda x: x.get('area', 0))
        print(f"   Largest cycle area: {largest_cycle.get('area', 0):.1f}px²")
        
        regular_cycles = [c for c in cycles if c.get('regularity', 0) > 0.7]
        print(f"   - Highly regular cycles: {len(regular_cycles)}")
    with open(os.path.join(output_dir, "analysis.txt"), "a",encoding="utf-8") as f:
        f.write(f"   CYCLES & LOOPS:\n")
        f.write(f"   Total cycles found: {len(cycles)}\n")
        if cycles:
            nested_cycles = [c for c in cycles if c.get('nested_in')]
            f.write(f"   - Nested cycles: {len(nested_cycles)}\n")
            
            largest_cycle = max(cycles, key=lambda x: x.get('area', 0))
            f.write(f"   Largest cycle area: {largest_cycle.get('area', 0):.1f}px²\n")
            
            regular_cycles = [c for c in cycles if c.get('regularity', 0) > 0.7]
            f.write(f"   - Highly regular cycles: {len(regular_cycles)}\n")
    
    # Symmetry analysis
    symmetry = result.get('symmetry', {})
    print(f"\n🔄 SYMMETRY PROPERTIES:")
    rot_order = symmetry.get('rotational_order', 1)
    mirror_axes = symmetry.get('mirror_axes', 0)
    
    if rot_order > 1:
        print(f"   Rotational symmetry: {rot_order}-fold")
    else:
        print("   No rotational symmetry detected")
    
    if mirror_axes > 0:
        print(f"   Mirror symmetries: {mirror_axes} axes")
        angles = symmetry.get('mirror_axis_angles', [])
        if angles:
            print(f"   Mirror axis angles: {[f'{a:.1f}°' for a in angles[:3]]}")
    else:
        print("   No mirror symmetry detected")
    with open(os.path.join(output_dir, "analysis.txt"), "a",encoding="utf-8") as f:
        f.write(f"   SYMMETRY PROPERTIES:\n")
        if rot_order > 1:
            f.write(f"   Rotational symmetry: {rot_order}-fold\n")
        else:
            f.write("   No rotational symmetry detected\n")
        
        if mirror_axes > 0:
            f.write(f"   Mirror symmetries: {mirror_axes} axes\n")
            angles = symmetry.get('mirror_axis_angles', [])
            if angles:
                f.write(f"   Mirror axis angles: {[f'{a:.1f}°' for a in angles[:3]]}\n")
        else:
            f.write("   No mirror symmetry detected\n")
    
    # Foreign lines
    foreign_lines = result.get('foreign_lines', [])
    print(f"\n🔀 FOREIGN ELEMENTS:")
    print(f"   Non-lattice lines: {len(foreign_lines)}")
    if foreign_lines:
        angle_mismatches = [f for f in foreign_lines if 'angle_mismatch' in f.get('reason', '')]
        off_lattice = [f for f in foreign_lines if 'off_lattice' in f.get('reason', '')]
        print(f"   - Angle mismatches: {len(angle_mismatches)}")
        print(f"   - Off-lattice endpoints: {len(off_lattice)}")
    with open(os.path.join(output_dir, "analysis.txt"), "a",encoding="utf-8") as f:
        f.write(f"   FOREIGN ELEMENTS:\n")
        f.write(f"   Non-lattice lines: {len(foreign_lines)}\n")
        if foreign_lines:
            angle_mismatches = [f for f in foreign_lines if 'angle_mismatch' in f.get('reason', '')]
            off_lattice = [f for f in foreign_lines if 'off_lattice' in f.get('reason', '')]
            f.write(f"   - Angle mismatches: {len(angle_mismatches)}\n")
            f.write(f"   - Off-lattice endpoints: {len(off_lattice)}\n")
    
    # Advanced pattern analysis
    patterns = result.get('patterns', {})
    if patterns:
        print(f"\n📈 PATTERN COMPLEXITY:")
        
        density = patterns.get('density', {})
        print(f"   Pattern density: {density.get('overall', 0):.6f}")
        print(f"   Density variation: {density.get('density_variation', 0):.2f}")
        
        connectivity = patterns.get('connectivity', {})
        print(f"   Connected components: {connectivity.get('components', 0)}")
        print(f"   Average node degree: {connectivity.get('average_degree', 0):.1f}")
        print(f"   Clustering coefficient: {connectivity.get('clustering_coefficient', 0):.3f}")
        
        path_complexity = patterns.get('path_complexity', {})
        print(f"   Path complexity (mean): {path_complexity.get('mean', 0):.2f}")
        
        completeness = patterns.get('completeness', {})
        completeness_pct = completeness.get('score', 0) * 100
        print(f"   Pattern completeness: {completeness_pct:.1f}%")
        
        fractal_dim = patterns.get('fractal_dimension', 1.0)
        print(f"   Fractal dimension: {fractal_dim:.2f}")
        
        with open(os.path.join(output_dir, "analysis.txt"), "a",encoding="utf-8") as f:
            f.write(f"   PATTERN COMPLEXITY:\n")
            f.write(f"   Pattern density: {density.get('overall', 0):.6f}\n")
            f.write(f"   Density variation: {density.get('density_variation', 0):.2f}\n")
            f.write(f"   Connected components: {connectivity.get('components', 0)}\n")
            f.write(f"   Average node degree: {connectivity.get('average_degree', 0):.1f}\n")
            f.write(f"   Clustering coefficient: {connectivity.get('clustering_coefficient', 0):.3f}\n")
            f.write(f"   Path complexity (mean): {path_complexity.get('mean', 0):.2f}\n")
            f.write(f"   Pattern completeness: {completeness_pct:.1f}%\n")
            f.write(f"   Fractal dimension: {fractal_dim:.2f}\n")
    
    # Human-readable summary
    human_rules = result.get('human_rules', '')
    print(f"\n💬 PATTERN SUMMARY:")
    print(f"{human_rules}")
    with open(os.path.join(output_dir, "analysis.txt"), "a",encoding="utf-8") as f:
        f.write(f"   PATTERN SUMMARY:\n")
        f.write(f"   {human_rules}\n")
    
    print("\n" + "=" * 60)

def create_analysis_visualization(original_img, result, base_name, output_dir):
    """Create visualization showing analysis results overlaid on original image"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.lines import Line2D
        import math


        # Use GridSpec for better control: top row (images) is much larger than bottom row (summary)
        import matplotlib.gridspec as gridspec
        fig = plt.figure(figsize=(22, 12))
        gs = gridspec.GridSpec(2, 3, height_ratios=[2.5, 1.8], hspace=0.18, wspace=0.18)
        fig.suptitle(f'Kolam Analysis: {base_name}', fontsize=20, fontweight='bold')

        # Convert BGR to RGB for matplotlib
        if len(original_img.shape) == 3:
            display_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        else:
            display_img = original_img

        # 1. Original image only (large)
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.imshow(display_img, cmap='gray')
        ax0.set_title("Original Image", fontsize=14, fontweight='bold')
        ax0.axis('off')

       # 2. Detected Dots (large)
        ax1 = fig.add_subplot(gs[0, 2])
        ax1.set_facecolor('white')
        if hasattr(display_img, 'shape'):
            ax1.set_xlim(0, display_img.shape[1])
            ax1.set_ylim(display_img.shape[0], 0)
        dots = result.get('dots', [])
        for dot in dots:
            circle = patches.Circle((dot['x'], dot['y']), dot['r'],
                                    fill=True, color='white', alpha=0.4, linewidth=0)
            ax1.add_patch(circle)
            outline = patches.Circle((dot['x'], dot['y']), dot['r'],
                                    fill=False, color='white', linewidth=1.5)
            ax1.add_patch(outline)
        ax1.axis('off')

        # 3. Graph structure (large)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor('white')
        ax2.set_title("Graph Structure", fontsize=14, fontweight='bold')
        graph = result.get('graph', {})
        nodes = graph.get('nodes', [])
        edges = graph.get('edges', [])
        for edge in edges:
            pixels = edge.get('pixels', [])
            if pixels and len(pixels) > 1:
                pixel_array = np.array(pixels)
                ax2.plot(pixel_array[:, 1], pixel_array[:, 0],
                         color='gray', alpha=1, linewidth=0.9, zorder=1)
        if edges:
            sorted_edges = sorted(edges, key=lambda e: e.get('euclidean_length', 0))
            for edge, color in zip([sorted_edges[0], sorted_edges[-1]], ['blue', 'green']):
                pixels = edge.get('pixels', [])
                if pixels and len(pixels) > 1:
                    pixel_array = np.array(pixels)
                    ax2.plot(pixel_array[:, 1], pixel_array[:, 0],
                             color=color, alpha=0.9, linewidth=2, zorder=2)
        if nodes:
            node_coords = np.array(nodes)
            ax2.scatter(node_coords[:, 1], node_coords[:, 0],
                        c='cyan', s=60, edgecolors='navy', linewidths=1.5, alpha=0.85, label='Nodes', zorder=3)
        ax2.axis('off')
        legend_elements = [
            Line2D([0], [0], color='gray', lw=2, label='Edges', alpha=0.5),
            Line2D([0], [0], color='blue', lw=2, label='Shortest Edge'),
            Line2D([0], [0], color='green', lw=2, label='Longest Edge'),
            Line2D([0], [0], marker='o', color='w', label='Nodes', markerfacecolor='cyan', markeredgecolor='navy', markersize=10)
        ]
        ax2.legend(handles=legend_elements, loc='upper right', fontsize=10)


        # 4. Cycles and symmetry (bottom left, still large)
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.imshow(display_img, cmap='gray', alpha=0.8)
        ax3.set_title("Symmetry", fontsize=14, fontweight='bold')
        cycles = result.get('cycles', [])
        cycle_colors = ['red', 'blue', 'green', 'orange', 'purple', 'magenta', 'lime', 'cyan']
        for i, cycle in enumerate(cycles[:5]):
            if 'node_ids' in cycle and 'coords' in cycle:
                coords = np.array(cycle['coords'])
                ax3.plot(coords[:, 0], coords[:, 1],
                         color=cycle_colors[i % len(cycle_colors)], lw=2, alpha=0.8, label=f'Cycle {i+1}')
                if 'centroid' in cycle:
                    centroid = cycle['centroid']
                    ax3.scatter(centroid[0], centroid[1], color=cycle_colors[i % len(cycle_colors)], s=60, marker='x')
        symmetry = result.get('symmetry', {})
        h, w = display_img.shape[:2]
        center = (w // 2, h // 2)
        rot_order = symmetry.get('rotational_order', 1)
        if rot_order > 1:
            circ = patches.Circle(center, min(h, w) * 0.12, fill=False, color='gold', lw=2, ls='--', alpha=0.7, label='Rotational Sym.')
            ax3.add_patch(circ)
        mirror_axes = symmetry.get('mirror_axes', 0)
        mirror_angles = symmetry.get('mirror_axis_angles', [])
        for i, angle in enumerate(mirror_angles[:mirror_axes]):
            theta = math.radians(angle)
            x0 = center[0] - int(0.45 * w * math.cos(theta))
            y0 = center[1] - int(0.45 * h * math.sin(theta))
            x1 = center[0] + int(0.45 * w * math.cos(theta))
            y1 = center[1] + int(0.45 * h * math.sin(theta))
            ax3.plot([x0, x1], [y0, y1], color='gold', lw=2, ls=':', alpha=0.7, label='Mirror Axis' if i == 0 else None)
        ax3.axis('off')
        legend_elements3 = [
            # Line2D([0], [0], color=cycle_colors[0], lw=2, label='Cycle 1'),
            # Line2D([0], [0], color=cycle_colors[1], lw=2, label='Cycle 2'),
            # Line2D([0], [0], color=cycle_colors[2], lw=2, label='Cycle 3'),
            Line2D([0], [0], color='gold', lw=2, ls='--', label='Rotational Sym.'),
            Line2D([0], [0], color='gold', lw=2, ls=':', label='Mirror Axis')
        ]
        ax3.legend(handles=legend_elements3, loc='upper right', fontsize=10)

        # 5. Pattern analysis summary (bottom center, spanning two columns)
        ax4 = fig.add_subplot(gs[1, 1:])
        ax4.axis('off')
        ax4.set_title("Analysis Summary", fontsize=14, fontweight='bold')
        summary_text = []
        summary_text.append(f"• Dots: {len(dots)}")
        lattice = result.get('lattice')
        if lattice:
            summary_text.append(f"• Lattice: {lattice['type'].capitalize()}  (Spacing: {lattice['spacing']:.1f}px)")
        summary_text.append(f"• Graph: {len(nodes)} nodes, {len(edges)} edges")
        summary_text.append(f"• Cycles: {len(cycles)}")
        if rot_order > 1:
            summary_text.append(f"• Rotational symmetry: {rot_order}-fold")
        if mirror_axes > 0:
            summary_text.append(f"• Mirror symmetries: {mirror_axes}")
        patterns = result.get('patterns', {})
        if patterns:
            completeness = patterns.get('completeness', {})
            comp_pct = completeness.get('score', 0) * 100
            summary_text.append(f"• Completeness: {comp_pct:.1f}%")
            fractal_dim = patterns.get('fractal_dimension', 1.0)
            summary_text.append(f"• Fractal dimension: {fractal_dim:.2f}")
        ax4.text(0.05, 0.95, '\n'.join(summary_text),
                 transform=ax4.transAxes, fontsize=13, fontfamily='monospace',
                 verticalalignment='top', bbox=dict(facecolor='whitesmoke', edgecolor='gray', boxstyle='round,pad=0.5'))
        human_rules = result.get('human_rules', '')
        ax4.text(0.05, 0.45, f"Summary:\n{human_rules}",
                 transform=ax4.transAxes, fontsize=11, fontfamily='monospace',
                 verticalalignment='top', wrap=True,
                 bbox=dict(facecolor='lavender', edgecolor='purple', boxstyle='round,pad=0.4'))

        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # Save visualization
        viz_output = os.path.join(output_dir, f"{base_name}_analysis_visualization.png")
        plt.savefig(viz_output, dpi=160, bbox_inches='tight')
        plt.close()

    except Exception as e:
        print(f"⚠  Could not create visualization: {e}")

def main():
    """Main function to analyze real kolam image"""
    print("REAL KOLAM ANALYZER")
    print("=" * 60)
    image_path = sys.argv[1] if len(sys.argv) > 1 else "image.png"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "public/results"
    analyze_real_kolam(image_path, output_dir)
if __name__ == "__main__":
    main()