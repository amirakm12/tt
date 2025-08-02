"""
Plugin Manager System
Handles dynamic loading, lifecycle management, and hot-reload of plugins
"""

import os
import sys
import importlib
import inspect
import json
from pathlib import Path
from typing import Dict, List, Optional, Type, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass
import logging

from PySide6.QtCore import QObject, Signal, Slot, QThread, QFileSystemWatcher

logger = logging.getLogger(__name__)


@dataclass
class PluginMetadata:
    """Plugin metadata structure"""
    id: str
    name: str
    version: str
    author: str
    description: str
    category: str  # 'tool', 'filter', 'agent', 'export'
    requires: List[str] = None
    icon: str = None
    shortcuts: Dict[str, str] = None
    
    @classmethod
    def from_dict(cls, data: dict):
        return cls(**data)


class PluginInterface(ABC):
    """Base interface for all plugins"""
    
    def __init__(self):
        self.metadata: Optional[PluginMetadata] = None
        self.enabled: bool = True
        self._settings: Dict[str, Any] = {}
        
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the plugin"""
        pass
        
    @abstractmethod
    def cleanup(self):
        """Clean up resources"""
        pass
        
    @abstractmethod
    def get_widget(self) -> Optional[QObject]:
        """Return the plugin's UI widget if any"""
        pass
        
    def get_settings(self) -> Dict[str, Any]:
        """Get plugin settings"""
        return self._settings
        
    def set_settings(self, settings: Dict[str, Any]):
        """Update plugin settings"""
        self._settings.update(settings)
        
    def save_settings(self):
        """Save settings to disk"""
        if self.metadata:
            settings_path = Path.home() / ".ai_artworks" / "plugins" / f"{self.metadata.id}.json"
            settings_path.parent.mkdir(parents=True, exist_ok=True)
            with open(settings_path, 'w') as f:
                json.dump(self._settings, f, indent=2)
                
    def load_settings(self):
        """Load settings from disk"""
        if self.metadata:
            settings_path = Path.home() / ".ai_artworks" / "plugins" / f"{self.metadata.id}.json"
            if settings_path.exists():
                with open(settings_path, 'r') as f:
                    self._settings = json.load(f)


class ToolPlugin(PluginInterface):
    """Base class for tool plugins (brushes, selection tools, etc.)"""
    
    @abstractmethod
    def activate(self):
        """Called when tool is selected"""
        pass
        
    @abstractmethod
    def deactivate(self):
        """Called when tool is deselected"""
        pass
        
    @abstractmethod
    def on_mouse_press(self, event):
        """Handle mouse press events"""
        pass
        
    @abstractmethod
    def on_mouse_move(self, event):
        """Handle mouse move events"""
        pass
        
    @abstractmethod
    def on_mouse_release(self, event):
        """Handle mouse release events"""
        pass


class FilterPlugin(PluginInterface):
    """Base class for filter/effect plugins"""
    
    @abstractmethod
    def apply(self, image, params: Dict[str, Any]):
        """Apply filter to image"""
        pass
        
    @abstractmethod
    def get_preview(self, image, params: Dict[str, Any]):
        """Generate preview of filter effect"""
        pass
        
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get filter parameters definition"""
        pass


class AgentPlugin(PluginInterface):
    """Base class for AI agent plugins"""
    
    @abstractmethod
    async def process_command(self, command: str, context: Dict[str, Any]):
        """Process a command from user"""
        pass
        
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this agent provides"""
        pass


class ExportPlugin(PluginInterface):
    """Base class for export format plugins"""
    
    @abstractmethod
    def export(self, image, path: str, options: Dict[str, Any]):
        """Export image to specified format"""
        pass
        
    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Return list of supported file extensions"""
        pass


class PluginManager(QObject):
    """Central plugin management system"""
    
    # Signals
    plugin_loaded = Signal(str)  # plugin_id
    plugin_unloaded = Signal(str)  # plugin_id
    plugin_error = Signal(str, str)  # plugin_id, error_message
    plugins_reloaded = Signal()
    
    def __init__(self):
        super().__init__()
        self.plugins: Dict[str, PluginInterface] = {}
        self.plugin_paths: List[Path] = []
        self.file_watcher = QFileSystemWatcher()
        self.file_watcher.fileChanged.connect(self._on_plugin_file_changed)
        self.file_watcher.directoryChanged.connect(self._on_plugin_dir_changed)
        
        # Default plugin directories
        self.add_plugin_path(Path(__file__).parent.parent / "plugins")
        self.add_plugin_path(Path.home() / ".ai_artworks" / "plugins")
        
    def add_plugin_path(self, path: Path):
        """Add a directory to search for plugins"""
        if path.exists() and path.is_dir():
            self.plugin_paths.append(path)
            self.file_watcher.addPath(str(path))
            logger.info(f"Added plugin path: {path}")
            
    def discover_plugins(self):
        """Discover all available plugins"""
        discovered = []
        
        for plugin_path in self.plugin_paths:
            if not plugin_path.exists():
                continue
                
            # Look for Python files
            for py_file in plugin_path.glob("*.py"):
                if py_file.name.startswith("_"):
                    continue
                    
                try:
                    # Check if it has plugin metadata
                    metadata = self._read_plugin_metadata(py_file)
                    if metadata:
                        discovered.append((py_file, metadata))
                except Exception as e:
                    logger.error(f"Error reading plugin {py_file}: {e}")
                    
            # Look for plugin directories
            for plugin_dir in plugin_path.iterdir():
                if plugin_dir.is_dir() and not plugin_dir.name.startswith("_"):
                    manifest_file = plugin_dir / "manifest.json"
                    if manifest_file.exists():
                        try:
                            with open(manifest_file, 'r') as f:
                                metadata = PluginMetadata.from_dict(json.load(f))
                                discovered.append((plugin_dir, metadata))
                        except Exception as e:
                            logger.error(f"Error reading plugin manifest {manifest_file}: {e}")
                            
        return discovered
        
    def _read_plugin_metadata(self, file_path: Path) -> Optional[PluginMetadata]:
        """Read plugin metadata from file"""
        # Try to extract metadata without importing
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Look for PLUGIN_METADATA dictionary
        if "PLUGIN_METADATA" in content:
            # Safe evaluation of metadata
            import ast
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "PLUGIN_METADATA":
                            if isinstance(node.value, ast.Dict):
                                metadata_dict = ast.literal_eval(node.value)
                                return PluginMetadata.from_dict(metadata_dict)
        return None
        
    def load_plugin(self, plugin_path: Path, metadata: PluginMetadata) -> bool:
        """Load a single plugin"""
        try:
            # Import the plugin module
            if plugin_path.is_file():
                spec = importlib.util.spec_from_file_location(metadata.id, plugin_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
            else:
                # Directory-based plugin
                sys.path.insert(0, str(plugin_path.parent))
                module = importlib.import_module(plugin_path.name)
                sys.path.pop(0)
                
            # Find the plugin class
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if (inspect.isclass(obj) and 
                    issubclass(obj, PluginInterface) and 
                    obj != PluginInterface and
                    not name.startswith("_")):
                    plugin_class = obj
                    break
                    
            if not plugin_class:
                raise ValueError(f"No plugin class found in {plugin_path}")
                
            # Instantiate the plugin
            plugin_instance = plugin_class()
            plugin_instance.metadata = metadata
            plugin_instance.load_settings()
            
            # Initialize the plugin
            if plugin_instance.initialize():
                self.plugins[metadata.id] = plugin_instance
                self.plugin_loaded.emit(metadata.id)
                logger.info(f"Loaded plugin: {metadata.name} v{metadata.version}")
                return True
            else:
                raise ValueError("Plugin initialization failed")
                
        except Exception as e:
            logger.error(f"Failed to load plugin {metadata.id}: {e}")
            self.plugin_error.emit(metadata.id, str(e))
            return False
            
    def unload_plugin(self, plugin_id: str):
        """Unload a plugin"""
        if plugin_id in self.plugins:
            try:
                plugin = self.plugins[plugin_id]
                plugin.save_settings()
                plugin.cleanup()
                del self.plugins[plugin_id]
                self.plugin_unloaded.emit(plugin_id)
                logger.info(f"Unloaded plugin: {plugin_id}")
            except Exception as e:
                logger.error(f"Error unloading plugin {plugin_id}: {e}")
                self.plugin_error.emit(plugin_id, str(e))
                
    def reload_plugin(self, plugin_id: str):
        """Reload a specific plugin"""
        if plugin_id in self.plugins:
            plugin = self.plugins[plugin_id]
            metadata = plugin.metadata
            self.unload_plugin(plugin_id)
            
            # Re-discover and load
            for plugin_path, discovered_metadata in self.discover_plugins():
                if discovered_metadata.id == plugin_id:
                    self.load_plugin(plugin_path, discovered_metadata)
                    break
                    
    def load_all_plugins(self):
        """Load all discovered plugins"""
        for plugin_path, metadata in self.discover_plugins():
            self.load_plugin(plugin_path, metadata)
            
    def get_plugins_by_category(self, category: str) -> List[PluginInterface]:
        """Get all plugins of a specific category"""
        return [p for p in self.plugins.values() 
                if p.metadata and p.metadata.category == category]
                
    def get_plugin(self, plugin_id: str) -> Optional[PluginInterface]:
        """Get a specific plugin by ID"""
        return self.plugins.get(plugin_id)
        
    @Slot(str)
    def _on_plugin_file_changed(self, file_path: str):
        """Handle plugin file changes for hot-reload"""
        # Find which plugin this file belongs to
        for plugin_id, plugin in self.plugins.items():
            if hasattr(plugin, "__module__"):
                module_file = inspect.getfile(plugin.__class__)
                if module_file == file_path:
                    logger.info(f"Plugin file changed, reloading: {plugin_id}")
                    self.reload_plugin(plugin_id)
                    break
                    
    @Slot(str)
    def _on_plugin_dir_changed(self, dir_path: str):
        """Handle plugin directory changes"""
        # Rescan for new plugins
        self.plugins_reloaded.emit()


# Example usage for testing
if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication
    
    app = QApplication([])
    manager = PluginManager()
    manager.load_all_plugins()
    
    print(f"Loaded {len(manager.plugins)} plugins")
    for plugin_id, plugin in manager.plugins.items():
        print(f"  - {plugin.metadata.name} ({plugin_id})")