# AI-ARTWORKS

A desktop application for AI-assisted image processing and creative tools.

## What Works

### Core Features
- ✅ Image loading and saving (JPEG, PNG, BMP, TIFF)
- ✅ Basic image editing (brightness, contrast, saturation)
- ✅ Image filters (blur, sharpen, edge detection)
- ✅ Image transformations (resize, rotate, flip, crop)
- ✅ Auto-enhance functionality
- ✅ Qt-based GUI interface

### Planned Features
- 🔄 AI-powered image generation
- 🔄 Style transfer
- 🔄 Voice commands
- 🔄 Document processing

## Quick Start

### Requirements
- Python 3.8 or higher
- 4GB RAM minimum
- Windows, macOS, or Linux

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ai-artworks.git
cd ai-artworks
```

2. Run the application:
```bash
python run.py
```

The script will automatically install required dependencies on first run.

### Manual Installation

If you prefer to install dependencies manually:

```bash
pip install -r requirements_core.txt
python -m ai_artworks.main
```

## Usage

1. Launch the application
2. Use File → Open to load an image
3. Apply edits using the toolbar and menus
4. Save your work with File → Save

## Project Structure

```
ai-artworks/
├── ai_artworks/
│   ├── core/           # Core processing modules
│   ├── ui/             # User interface
│   └── main.py         # Application entry point
├── run.py              # Simple run script
└── requirements_core.txt   # Minimal dependencies
```

## Development

To contribute:

1. Focus on core functionality
2. Test your changes thoroughly
3. Keep dependencies minimal
4. Document what actually works

## License

MIT License - see LICENSE file for details