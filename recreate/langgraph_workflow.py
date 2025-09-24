"""
A thin wrapper around the video_converter module for backward compatibility.
All functionality has been moved to manim_recreate.video_converter.
"""
from manim_recreate.video_converter import (
    RenderState,
    render_kolam_node,
    build_graph,
    __all__ as video_converter_all
)

__all__ = [
    'RenderState',
    'render_kolam_node',
    'build_graph',
]

# Re-export all symbols from video_converter for backward compatibility
__all__.extend(video_converter_all)


