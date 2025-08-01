# AI-ARTWORKS Neural Interface

A futuristic cyberpunk HUD system for AI-powered creative workflows with voice control and real-time agent visualization.

![AI-ARTWORKS HUD](assets/screenshot.png)

## Features

### 🔺 HUD Style GUI
- Parallax-scrolling animated UI with glass and pulse shaders
- Vulkan-accelerated rendering for maximum performance
- Full-screen cyberpunk aesthetic inspired by J.A.R.V.I.S. and Cyberpunk 2077

### 🔊 Live Voice Command System
- Real-time voice transcription using Whisper
- Animated waveform visualization
- Command pulse wave with translation and intent highlighting

### 🕷 Agent Swarm Visualization
- Real-time 3D display of agent network
- Neural network-style connection mapping
- Live status updates and task routing visualization

### 🧠 Consciousness Stream Inspector
- View Athena's internal thought processes
- Entropy level monitoring
- Decision-making transparency

### 📸 Neural Canvas
- AI rendering progress visualization
- Real-time neural activity display
- Task execution monitoring

## Quick Start

### Requirements
- Windows 10/11 (64-bit)
- 8GB RAM minimum (16GB recommended)
- NVIDIA GPU with CUDA support (recommended)
- Vulkan Runtime (optional, for best performance)

### Installation

1. Download the installer: `AIArtworks-Setup-1.0.0.exe`
2. Run the installer and follow the setup wizard
3. Launch from Start Menu or Desktop shortcut

### Manual Build

```bash
# Clone repository
git clone https://github.com/yourusername/ai-artworks.git
cd ai-artworks

# Install dependencies
pip install -r requirements.txt

# Run the application
python ai_artworks/main.py
```

## Voice Commands

Press **SPACE** to activate voice control. Examples:

- "Generate a cyberpunk cityscape"
- "Apply style transfer to current image"
- "Show agent status"
- "Analyze this document"

## Keyboard Shortcuts

- **SPACE** - Toggle voice listening
- **ESC** - Exit fullscreen / Quit application
- **F11** - Toggle fullscreen

## Architecture

### Backend Agents
- **Athena** - Central orchestration brain
- **RenderOps** - GPU-accelerated rendering
- **DataDaemon** - Analytics and logging
- **SecSentinel** - Security monitoring
- **VoiceNav** - Voice command processing
- **Autopilot** - Task planning and execution

### Technology Stack
- **GUI**: PySide6 + QML + QtQuick3D
- **Rendering**: Vulkan/OpenGL
- **AI Models**: Whisper, Stable Diffusion, GPT
- **Backend**: Multi-agent system with async processing
- **Database**: PostgreSQL, Neo4j, Redis

## Building from Source

### Build Executable
```bash
python build_exe.py
```

This will:
1. Create a standalone executable using PyInstaller
2. Generate an Inno Setup installer (Windows only)
3. Package all dependencies and assets

### Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Format code
black ai_artworks/
```

## System Requirements

### Minimum
- OS: Windows 10 64-bit
- Processor: Intel i5 or AMD equivalent
- Memory: 8GB RAM
- Graphics: DirectX 11 compatible GPU
- Storage: 10GB available space

### Recommended
- OS: Windows 11 64-bit
- Processor: Intel i7/i9 or AMD Ryzen 7/9
- Memory: 16GB+ RAM
- Graphics: NVIDIA RTX 3060+ with CUDA
- Storage: 20GB SSD space

## License

MIT License - see LICENSE file for details

## Support

- Documentation: [docs.ai-artworks.com](https://docs.ai-artworks.com)
- Issues: [GitHub Issues](https://github.com/yourusername/ai-artworks/issues)
- Discord: [Join our community](https://discord.gg/aiartworks) 
