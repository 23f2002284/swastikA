# SwastikA - AI-Powered Kolam Art Analysis and Recreation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Note**: This project was developed as a solution for **SIH 2025 Problem Statement PS-25107**: "Develop computer programs (in any language, preferably Python) to identify the design principles behind the Kolam designs and recreate the kolams."

## Overview
SwastikA is an AI-powered platform that analyzes traditional Kolam art, understands its patterns, and can recreate or generate new variations while preserving the cultural essence. The system uses Vision LLM to process Kolam images, extract features, and generate Hypothesis and verfiy each with reasoning and conclude with evidences. also recreate variation of the kolam with its design principles along with some new principles. it uses **Manim community edition** to create animations of the kolam patterns. it also restore if the kolam is incomplete or some portion not in frame

## Demo
[![Watch on YouTube](https://img.shields.io/badge/youtube-red.svg)](https://youtu.be/2NMrc68UR5I)

## Features
- **Image Restoration**: Restore image if Kolam is incomplete or not in frame based on its design principles
- **Image Enhancement**: Preprocess and enhance Kolam images for better analysis
- **Pattern Recognition**: LLM powered analysis of Kolam patterns and structures
- **SVG Generation**: Convert hand-drawn Kolams into scalable vector graphics
- **Variation Creation**: Generate new Kolam variations while maintaining traditional aesthetics
- **Video Generation**: Create drawing animations of Kolam patterns with Manim community edition

## Prerequisites
- Python 3.8+
- FFmpeg (for video processing)
- Potrace (for vectorization)
- Google Cloud Account (for AI services)

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/SwastikA.git
cd SwastikA/swastikA-v0
```
### 2. Install system dependencies
#### Windows:
- Download and install [FFmpeg](https://ffmpeg.org/download.html)
- Download and install [Potrace](http://potrace.sourceforge.net/)
- Add both to your system PATH

#### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install -y ffmpeg potrace
```
### 3. Create and activate virtual environment
```bash
python -m venv venv
source .venv/Scripts/activate
```
### 4. Install Python dependencies
```bash
pip install -r requirements.txt
```
### 5. Set up environment variables
Create a `.env` file in the project root and add your Google Cloud credentials:
```env
GOOGLE_API_KEY=your_google_api_key
GCP_PROJECT=your_gcp_project_id
GCP_LOCATION=your_gcp_region
GOOGLE_GENAI_USE_VERTEXAI=true
```
## Quick Start
1. Place your Kolam images in the `swastikA/media/images/` directory
2. Change the `test0.py` file to your image path
3. Run the test script:
   ```bash
   python test0.py
   ```

## Technical Details
- **AI/ML**: Google's Gemini, LangChain
- **Vector Graphics**: CairoSVG, svglib, potrace
- **Animation**: Manim

## Project Structure
```
swastikA/
├── analysis/             # Pattern analysis and feature extraction
├── app/                  # FastAPI application
├── media/                # Media storage
│   ├── enhanced/         # Preprocessed images
│   ├── svg/              # Generated SVGs
│   └── videos/           # Output videos
│   └── test_kolam_1.png  
│   └── test_kolam_2.png  
│   └── test_kolam_3.png  # variation images
├── preprocessing/        # Image enhancement 
├── recreate/             # Pattern recreation
├── svg_converter/        # Image to SVG conversion
└── utils/                # Utility functions
```

## Acknowledgments
- Special thanks to the **Comfy** team for their invaluable assistance with SVG processing and optimization [ComfyUI-ToSVG](https://github.com/Yanick112/ComfyUI-ToSVG)
- The Potrace project for their excellent vectorization library
- Google AI for their generative models


## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## Contact
For any queries, please contact [pratyushliku29@gmail.com](mailto:pratyushliku29@gmail.com)
