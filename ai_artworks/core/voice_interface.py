"""
Voice Interface Controller
Real-time voice processing with Whisper and waveform visualization
"""

import numpy as np
import sounddevice as sd
import queue
import threading
from PySide6.QtCore import QObject, Signal, QThread
import whisper
import torch
import scipy.signal

class VoiceController(QObject):
    """Voice control system with real-time processing"""
    
    # Signals
    command_recognized = Signal(str)
    waveform_updated = Signal(list)
    listening_started = Signal()
    listening_stopped = Signal()
    error_occurred = Signal(str)
    
    def __init__(self):
        super().__init__()
        
        # Audio parameters
        self.sample_rate = 16000
        self.channels = 1
        self.blocksize = 1024
        self.audio_queue = queue.Queue()
        
        # Whisper model
        self.model = None
        self.load_model()
        
        # Processing thread
        self.processing_thread = None
        self.is_listening = False
        
        # Waveform buffer
        self.waveform_buffer = np.zeros(512)
        
    def load_model(self):
        """Load Whisper model"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = whisper.load_model("base", device=device)
        except Exception as e:
            self.error_occurred.emit(f"Failed to load Whisper model: {str(e)}")
            
    def start_listening(self):
        """Start voice listening"""
        if self.is_listening:
            return
            
        self.is_listening = True
        self.listening_started.emit()
        
        # Start audio stream
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            callback=self._audio_callback,
            blocksize=self.blocksize
        )
        self.stream.start()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop_listening(self):
        """Stop voice listening"""
        if not self.is_listening:
            return
            
        self.is_listening = False
        
        # Stop audio stream
        if hasattr(self, 'stream'):
            self.stream.stop()
            self.stream.close()
            
        self.listening_stopped.emit()
        
    def _audio_callback(self, indata, frames, time, status):
        """Audio stream callback"""
        if status:
            self.error_occurred.emit(str(status))
            
        # Add to queue for processing
        self.audio_queue.put(indata.copy())
        
        # Update waveform
        self._update_waveform(indata)
        
    def _update_waveform(self, audio_data):
        """Update waveform visualization data"""
        # Downsample for visualization
        downsampled = scipy.signal.resample(audio_data.flatten(), 128)
        
        # Normalize
        max_val = np.max(np.abs(downsampled))
        if max_val > 0:
            normalized = downsampled / max_val
        else:
            normalized = downsampled
            
        # Emit waveform data
        self.waveform_updated.emit(normalized.tolist())
        
    def _process_audio(self):
        """Process audio in background thread"""
        audio_buffer = []
        silence_threshold = 0.01
        silence_duration = 0
        max_silence = 1.5  # seconds
        
        while self.is_listening:
            try:
                # Get audio chunk
                chunk = self.audio_queue.get(timeout=0.1)
                audio_buffer.append(chunk)
                
                # Check for silence
                if np.max(np.abs(chunk)) < silence_threshold:
                    silence_duration += len(chunk) / self.sample_rate
                else:
                    silence_duration = 0
                    
                # Process if we have enough audio and detected end of speech
                buffer_duration = len(audio_buffer) * self.blocksize / self.sample_rate
                
                if buffer_duration > 0.5 and silence_duration > max_silence:
                    # Concatenate audio
                    audio = np.concatenate(audio_buffer)
                    
                    # Process with Whisper
                    self._transcribe_audio(audio)
                    
                    # Clear buffer
                    audio_buffer = []
                    silence_duration = 0
                    
            except queue.Empty:
                continue
            except Exception as e:
                self.error_occurred.emit(f"Processing error: {str(e)}")
                
    def _transcribe_audio(self, audio):
        """Transcribe audio using Whisper"""
        if self.model is None:
            return
            
        try:
            # Ensure audio is float32
            audio = audio.astype(np.float32).flatten()
            
            # Transcribe
            result = self.model.transcribe(
                audio,
                language="en",
                task="transcribe",
                temperature=0.0
            )
            
            text = result["text"].strip()
            
            if text:
                self.command_recognized.emit(text)
                
        except Exception as e:
            self.error_occurred.emit(f"Transcription error: {str(e)}")
            
    def process_command(self, command: str):
        """Process a text command directly"""
        # This can be called from QML for testing
        self.command_recognized.emit(command)