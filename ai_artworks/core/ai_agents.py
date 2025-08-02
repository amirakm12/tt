"""
AI Agents System
Integrates Whisper for voice input and Mixtral for command processing
"""

import asyncio
import json
import logging
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass
import aiohttp
import sounddevice as sd
import queue
import threading

from PySide6.QtCore import QObject, Signal, Slot, QThread
import whisper
import torch

logger = logging.getLogger(__name__)


@dataclass
class AgentCommand:
    """Represents a command from the AI agent"""
    action: str
    parameters: Dict[str, Any]
    confidence: float
    raw_text: str


class WhisperThread(QThread):
    """Background thread for Whisper processing"""
    
    # Signals
    transcription_ready = Signal(str)
    error_occurred = Signal(str)
    
    def __init__(self, model_size: str = "base"):
        super().__init__()
        self.model_size = model_size
        self.model = None
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.sample_rate = 16000
        self.device = None
        
    def run(self):
        """Run the Whisper processing thread"""
        try:
            # Load Whisper model
            logger.info(f"Loading Whisper model: {self.model_size}")
            self.model = whisper.load_model(self.model_size)
            logger.info("Whisper model loaded successfully")
            
            # Process audio from queue
            while True:
                try:
                    audio_data = self.audio_queue.get(timeout=0.1)
                    if audio_data is None:  # Shutdown signal
                        break
                        
                    # Transcribe audio
                    result = self.model.transcribe(
                        audio_data,
                        language="en",
                        fp16=torch.cuda.is_available()
                    )
                    
                    text = result["text"].strip()
                    if text:
                        self.transcription_ready.emit(text)
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Transcription error: {e}")
                    self.error_occurred.emit(str(e))
                    
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.error_occurred.emit(f"Failed to load Whisper model: {e}")
            
    def start_recording(self):
        """Start recording audio"""
        if self.is_recording:
            return
            
        self.is_recording = True
        self.audio_buffer = []
        
        def audio_callback(indata, frames, time, status):
            if status:
                logger.warning(f"Audio callback status: {status}")
            if self.is_recording:
                self.audio_buffer.append(indata.copy())
                
        # Start audio stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            callback=audio_callback,
            device=self.device
        )
        self.stream.start()
        logger.info("Started recording audio")
        
    def stop_recording(self):
        """Stop recording and process audio"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        self.stream.stop()
        self.stream.close()
        
        # Combine audio buffer
        if self.audio_buffer:
            audio_data = np.concatenate(self.audio_buffer, axis=0)
            audio_data = audio_data.flatten()
            
            # Add to processing queue
            self.audio_queue.put(audio_data)
            
        logger.info("Stopped recording audio")
        
    def shutdown(self):
        """Shutdown the thread"""
        self.audio_queue.put(None)
        self.wait()


class MixtralAgent(QObject):
    """Mixtral AI agent for command processing"""
    
    # Signals
    response_ready = Signal(AgentCommand)
    error_occurred = Signal(str)
    processing_started = Signal()
    processing_finished = Signal()
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        super().__init__()
        self.ollama_url = ollama_url
        self.model_name = "mixtral"
        self.system_prompt = """You are an AI assistant for a creative image editing application. 
        Parse user commands and respond with structured JSON actions.
        
        Available actions:
        - load_image: Load an image from file
        - save_image: Save the current image
        - apply_filter: Apply a filter to the image
        - upscale: Upscale the image
        - generate: Generate new image with AI
        - enhance: Enhance image quality
        - remove_background: Remove image background
        - add_text: Add text to image
        - crop: Crop the image
        - resize: Resize the image
        
        Respond only with valid JSON in this format:
        {
            "action": "action_name",
            "parameters": {
                "param1": "value1",
                "param2": "value2"
            },
            "confidence": 0.95
        }
        """
        
    async def process_command(self, text: str, context: Dict[str, Any] = None):
        """Process a text command with Mixtral"""
        self.processing_started.emit()
        
        try:
            # Prepare the prompt
            prompt = f"{self.system_prompt}\n\nUser command: {text}"
            if context:
                prompt += f"\n\nContext: {json.dumps(context, indent=2)}"
                
            # Call Ollama API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_url}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "format": "json"
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        response_text = result.get("response", "")
                        
                        # Parse JSON response
                        try:
                            command_data = json.loads(response_text)
                            command = AgentCommand(
                                action=command_data.get("action", "unknown"),
                                parameters=command_data.get("parameters", {}),
                                confidence=command_data.get("confidence", 0.0),
                                raw_text=text
                            )
                            self.response_ready.emit(command)
                        except json.JSONDecodeError:
                            logger.error(f"Failed to parse AI response: {response_text}")
                            self.error_occurred.emit("Failed to parse AI response")
                    else:
                        error_text = await response.text()
                        logger.error(f"Ollama API error: {error_text}")
                        self.error_occurred.emit(f"AI service error: {response.status}")
                        
        except Exception as e:
            logger.error(f"Failed to process command: {e}")
            self.error_occurred.emit(str(e))
        finally:
            self.processing_finished.emit()
            
    @Slot(str)
    def process_command_sync(self, text: str):
        """Synchronous wrapper for process_command"""
        asyncio.create_task(self.process_command(text))


class CommandProcessor(QObject):
    """Processes commands from various sources"""
    
    # Signals
    command_executed = Signal(str, dict)  # action, result
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        self.handlers: Dict[str, Callable] = {}
        self.register_default_handlers()
        
    def register_handler(self, action: str, handler: Callable):
        """Register a command handler"""
        self.handlers[action] = handler
        logger.info(f"Registered handler for action: {action}")
        
    def register_default_handlers(self):
        """Register default command handlers"""
        # These will be connected to actual functionality
        default_actions = [
            "load_image", "save_image", "apply_filter", "upscale",
            "generate", "enhance", "remove_background", "add_text",
            "crop", "resize"
        ]
        
        for action in default_actions:
            self.handlers[action] = lambda params: {
                "status": "pending",
                "message": f"Handler for {action} not implemented"
            }
            
    async def execute_command(self, command: AgentCommand):
        """Execute a command"""
        try:
            if command.action in self.handlers:
                handler = self.handlers[command.action]
                
                # Execute handler (async if needed)
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(command.parameters)
                else:
                    result = handler(command.parameters)
                    
                self.command_executed.emit(command.action, result)
                return result
            else:
                error_msg = f"Unknown action: {command.action}"
                logger.warning(error_msg)
                self.error_occurred.emit(error_msg)
                return {"status": "error", "message": error_msg}
                
        except Exception as e:
            logger.error(f"Command execution error: {e}")
            self.error_occurred.emit(str(e))
            return {"status": "error", "message": str(e)}


class VoiceCommandSystem(QObject):
    """Complete voice command system"""
    
    # Signals
    listening_started = Signal()
    listening_stopped = Signal()
    transcription_ready = Signal(str)
    command_ready = Signal(AgentCommand)
    status_changed = Signal(str)
    
    def __init__(self, whisper_model: str = "base"):
        super().__init__()
        
        # Initialize components
        self.whisper_thread = WhisperThread(whisper_model)
        self.mixtral_agent = MixtralAgent()
        self.command_processor = CommandProcessor()
        
        # Connect signals
        self.whisper_thread.transcription_ready.connect(self._on_transcription)
        self.whisper_thread.error_occurred.connect(self._on_error)
        
        self.mixtral_agent.response_ready.connect(self._on_command_ready)
        self.mixtral_agent.error_occurred.connect(self._on_error)
        
        # Start Whisper thread
        self.whisper_thread.start()
        
    def start_listening(self):
        """Start voice recording"""
        self.whisper_thread.start_recording()
        self.listening_started.emit()
        self.status_changed.emit("Listening...")
        
    def stop_listening(self):
        """Stop voice recording and process"""
        self.whisper_thread.stop_recording()
        self.listening_stopped.emit()
        self.status_changed.emit("Processing...")
        
    @Slot(str)
    def _on_transcription(self, text: str):
        """Handle transcription result"""
        logger.info(f"Transcription: {text}")
        self.transcription_ready.emit(text)
        self.status_changed.emit("Understanding command...")
        
        # Process with AI agent
        self.mixtral_agent.process_command_sync(text)
        
    @Slot(AgentCommand)
    def _on_command_ready(self, command: AgentCommand):
        """Handle parsed command"""
        logger.info(f"Command ready: {command.action}")
        self.command_ready.emit(command)
        self.status_changed.emit("Executing command...")
        
        # Execute command
        asyncio.create_task(self.command_processor.execute_command(command))
        
    @Slot(str)
    def _on_error(self, error: str):
        """Handle errors"""
        logger.error(f"Voice command error: {error}")
        self.status_changed.emit(f"Error: {error}")
        
    def process_text_command(self, text: str):
        """Process a text command directly"""
        self.status_changed.emit("Processing command...")
        self.mixtral_agent.process_command_sync(text)
        
    def shutdown(self):
        """Shutdown the system"""
        self.whisper_thread.shutdown()
        
    def get_available_devices(self) -> List[Dict[str, Any]]:
        """Get available audio input devices"""
        devices = []
        for i, device in enumerate(sd.query_devices()):
            if device['max_input_channels'] > 0:
                devices.append({
                    'index': i,
                    'name': device['name'],
                    'channels': device['max_input_channels']
                })
        return devices
        
    def set_input_device(self, device_index: int):
        """Set the audio input device"""
        self.whisper_thread.device = device_index