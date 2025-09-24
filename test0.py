from recreate.manim_recreate import build_graph
app = build_graph()
result = app.invoke({
    "svg_path": "vectorized_output.svg",
    "output_dir": "media",
    "output_name": "kolam_example_0",
    "width": 1080,
    "height": 1080,
    "fps": 30,
    "duration": 5.0,
    "stroke_width": 4.0,  # Adjust this value to make strokes thicker or thinner
    "transparent": False,
    "save_last_frame": False,
    "strip_background_rects": True,
})
print("Render result:", result)