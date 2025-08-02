"""
Neural Processing Engine
Advanced AI model management with optimized inference pipeline
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
from pathlib import Path
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import logging
from queue import Queue, PriorityQueue
import time

from PySide6.QtCore import QObject, Signal, QThread
from transformers import pipeline, AutoModel, AutoTokenizer
import onnxruntime as ort

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuration for AI models"""
    name: str
    model_type: str  # 'pytorch', 'onnx', 'transformers'
    path: str
    device: str  # 'cuda', 'cpu', 'mps'
    precision: str  # 'fp32', 'fp16', 'int8'
    batch_size: int
    max_memory_mb: int
    priority: int = 5  # 1-10, higher = more important


@dataclass
class InferenceRequest:
    """Request for model inference"""
    id: str
    model_name: str
    input_data: Any
    callback: Callable
    priority: int = 5
    timestamp: float = 0.0
    
    def __lt__(self, other):
        # Higher priority first, then older requests
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.timestamp < other.timestamp


class ModelOptimizer:
    """Optimizes models for inference"""
    
    @staticmethod
    def optimize_pytorch_model(model: nn.Module, example_input: torch.Tensor) -> nn.Module:
        """Optimize PyTorch model for inference"""
        model.eval()
        
        # Try different optimization techniques
        optimized = model
        
        try:
            # TorchScript optimization
            traced = torch.jit.trace(model, example_input)
            traced = torch.jit.optimize_for_inference(traced)
            optimized = traced
            logger.info("Applied TorchScript optimization")
        except:
            logger.warning("TorchScript optimization failed, using original model")
            
        # Disable gradients
        for param in optimized.parameters():
            param.requires_grad = False
            
        return optimized
        
    @staticmethod
    def quantize_model(model: nn.Module, calibration_data: List[torch.Tensor]) -> nn.Module:
        """Apply dynamic quantization"""
        try:
            quantized = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            logger.info("Applied int8 quantization")
            return quantized
        except:
            logger.warning("Quantization failed, using original model")
            return model
            
    @staticmethod
    def convert_to_onnx(model: nn.Module, example_input: torch.Tensor, output_path: str):
        """Convert PyTorch model to ONNX"""
        torch.onnx.export(
            model,
            example_input,
            output_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        logger.info(f"Exported model to ONNX: {output_path}")


class ModelCache:
    """LRU cache for loaded models"""
    
    def __init__(self, max_memory_gb: float = 4.0):
        self.max_memory = max_memory_gb * 1024  # Convert to MB
        self.models: Dict[str, Any] = {}
        self.memory_usage: Dict[str, float] = {}
        self.last_used: Dict[str, float] = {}
        self.lock = threading.Lock()
        
    def get(self, model_name: str) -> Optional[Any]:
        """Get model from cache"""
        with self.lock:
            if model_name in self.models:
                self.last_used[model_name] = time.time()
                return self.models[model_name]
        return None
        
    def put(self, model_name: str, model: Any, memory_mb: float):
        """Add model to cache"""
        with self.lock:
            # Check if we need to free memory
            while self._total_memory() + memory_mb > self.max_memory:
                if not self._evict_lru():
                    logger.warning("Cannot free enough memory for new model")
                    return
                    
            self.models[model_name] = model
            self.memory_usage[model_name] = memory_mb
            self.last_used[model_name] = time.time()
            
    def _total_memory(self) -> float:
        """Calculate total memory usage"""
        return sum(self.memory_usage.values())
        
    def _evict_lru(self) -> bool:
        """Evict least recently used model"""
        if not self.models:
            return False
            
        lru_model = min(self.last_used.items(), key=lambda x: x[1])[0]
        del self.models[lru_model]
        del self.memory_usage[lru_model]
        del self.last_used[lru_model]
        
        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info(f"Evicted model from cache: {lru_model}")
        return True


class InferenceEngine(QThread):
    """High-performance inference engine"""
    
    # Signals
    result_ready = Signal(str, object)  # request_id, result
    error_occurred = Signal(str, str)  # request_id, error
    
    def __init__(self, max_workers: int = 4):
        super().__init__()
        self.request_queue = PriorityQueue()
        self.models: Dict[str, Any] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.model_cache = ModelCache()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = True
        
        # Performance tracking
        self.inference_times: Dict[str, List[float]] = {}
        
    def register_model(self, config: ModelConfig):
        """Register a model configuration"""
        self.model_configs[config.name] = config
        logger.info(f"Registered model: {config.name}")
        
    def load_model(self, model_name: str) -> Any:
        """Load model with caching"""
        # Check cache first
        cached_model = self.model_cache.get(model_name)
        if cached_model is not None:
            return cached_model
            
        config = self.model_configs.get(model_name)
        if not config:
            raise ValueError(f"Model not registered: {model_name}")
            
        logger.info(f"Loading model: {model_name}")
        
        if config.model_type == 'pytorch':
            model = self._load_pytorch_model(config)
        elif config.model_type == 'onnx':
            model = self._load_onnx_model(config)
        elif config.model_type == 'transformers':
            model = self._load_transformers_model(config)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
            
        # Cache the model
        self.model_cache.put(model_name, model, config.max_memory_mb)
        
        return model
        
    def _load_pytorch_model(self, config: ModelConfig) -> nn.Module:
        """Load PyTorch model"""
        model = torch.load(config.path, map_location=config.device)
        model.eval()
        
        if config.precision == 'fp16' and config.device == 'cuda':
            model = model.half()
            
        return model.to(config.device)
        
    def _load_onnx_model(self, config: ModelConfig) -> ort.InferenceSession:
        """Load ONNX model"""
        providers = []
        if config.device == 'cuda':
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        return ort.InferenceSession(config.path, session_options, providers=providers)
        
    def _load_transformers_model(self, config: ModelConfig) -> Any:
        """Load Transformers model"""
        return pipeline(
            task=config.name.split('_')[0],  # Extract task from name
            model=config.path,
            device=0 if config.device == 'cuda' else -1,
            torch_dtype=torch.float16 if config.precision == 'fp16' else torch.float32
        )
        
    def submit_request(self, request: InferenceRequest):
        """Submit inference request"""
        request.timestamp = time.time()
        self.request_queue.put(request)
        
    def run(self):
        """Process inference requests"""
        while self.running:
            try:
                # Get request with timeout
                request = self.request_queue.get(timeout=0.1)
                
                # Process in thread pool
                future = self.executor.submit(self._process_request, request)
                future.add_done_callback(lambda f: self._handle_result(f, request))
                
            except:
                continue
                
    def _process_request(self, request: InferenceRequest) -> Any:
        """Process single inference request"""
        start_time = time.time()
        
        try:
            # Load model
            model = self.load_model(request.model_name)
            config = self.model_configs[request.model_name]
            
            # Run inference based on model type
            if config.model_type == 'pytorch':
                result = self._inference_pytorch(model, request.input_data, config)
            elif config.model_type == 'onnx':
                result = self._inference_onnx(model, request.input_data, config)
            elif config.model_type == 'transformers':
                result = self._inference_transformers(model, request.input_data, config)
            else:
                raise ValueError(f"Unknown model type: {config.model_type}")
                
            # Track performance
            inference_time = time.time() - start_time
            if request.model_name not in self.inference_times:
                self.inference_times[request.model_name] = []
            self.inference_times[request.model_name].append(inference_time)
            
            return result
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise
            
    def _inference_pytorch(self, model: nn.Module, input_data: Any, config: ModelConfig) -> Any:
        """Run PyTorch inference"""
        with torch.no_grad():
            if isinstance(input_data, np.ndarray):
                input_tensor = torch.from_numpy(input_data).to(config.device)
            else:
                input_tensor = input_data.to(config.device)
                
            if config.precision == 'fp16' and config.device == 'cuda':
                input_tensor = input_tensor.half()
                
            output = model(input_tensor)
            
            if isinstance(output, torch.Tensor):
                return output.cpu().numpy()
            return output
            
    def _inference_onnx(self, session: ort.InferenceSession, input_data: Any, config: ModelConfig) -> Any:
        """Run ONNX inference"""
        if isinstance(input_data, torch.Tensor):
            input_data = input_data.numpy()
            
        input_name = session.get_inputs()[0].name
        output = session.run(None, {input_name: input_data})
        
        return output[0] if len(output) == 1 else output
        
    def _inference_transformers(self, pipeline: Any, input_data: Any, config: ModelConfig) -> Any:
        """Run Transformers inference"""
        return pipeline(input_data)
        
    def _handle_result(self, future, request: InferenceRequest):
        """Handle inference result"""
        try:
            result = future.result()
            self.result_ready.emit(request.id, result)
            if request.callback:
                request.callback(result)
        except Exception as e:
            self.error_occurred.emit(request.id, str(e))
            
    def get_performance_stats(self, model_name: str) -> Dict[str, float]:
        """Get performance statistics for a model"""
        if model_name not in self.inference_times:
            return {}
            
        times = self.inference_times[model_name]
        if not times:
            return {}
            
        return {
            'avg_inference_ms': np.mean(times) * 1000,
            'min_inference_ms': np.min(times) * 1000,
            'max_inference_ms': np.max(times) * 1000,
            'std_inference_ms': np.std(times) * 1000,
            'total_inferences': len(times)
        }
        
    def optimize_all_models(self):
        """Optimize all loaded models"""
        for model_name, model in self.models.items():
            config = self.model_configs[model_name]
            if config.model_type == 'pytorch':
                logger.info(f"Optimizing model: {model_name}")
                # Create example input
                example_input = torch.randn(1, 3, 224, 224).to(config.device)
                optimized = ModelOptimizer.optimize_pytorch_model(model, example_input)
                self.models[model_name] = optimized
                
    def shutdown(self):
        """Shutdown the engine"""
        self.running = False
        self.executor.shutdown(wait=True)
        self.wait()


class NeuralProcessor(QObject):
    """High-level neural processing interface"""
    
    # Signals
    processing_started = Signal(str)
    processing_completed = Signal(str, object)
    processing_failed = Signal(str, str)
    
    def __init__(self):
        super().__init__()
        self.engine = InferenceEngine()
        self.engine.result_ready.connect(self.processing_completed)
        self.engine.error_occurred.connect(self.processing_failed)
        self.engine.start()
        
        # Register default models
        self._register_default_models()
        
    def _register_default_models(self):
        """Register default AI models"""
        # Image enhancement model
        self.engine.register_model(ModelConfig(
            name="image_enhance",
            model_type="pytorch",
            path="models/enhance_net.pth",
            device="cuda" if torch.cuda.is_available() else "cpu",
            precision="fp16" if torch.cuda.is_available() else "fp32",
            batch_size=1,
            max_memory_mb=512
        ))
        
        # Style transfer model
        self.engine.register_model(ModelConfig(
            name="style_transfer",
            model_type="onnx",
            path="models/style_transfer.onnx",
            device="cuda" if torch.cuda.is_available() else "cpu",
            precision="fp32",
            batch_size=1,
            max_memory_mb=256
        ))
        
        # Text-to-image model
        self.engine.register_model(ModelConfig(
            name="text2img",
            model_type="transformers",
            path="stabilityai/stable-diffusion-2-1",
            device="cuda" if torch.cuda.is_available() else "cpu",
            precision="fp16",
            batch_size=1,
            max_memory_mb=4096
        ))
        
    def process_image(self, image: np.ndarray, model_name: str, **kwargs) -> str:
        """Process image with specified model"""
        request_id = f"{model_name}_{time.time()}"
        
        request = InferenceRequest(
            id=request_id,
            model_name=model_name,
            input_data=image,
            callback=None,
            priority=kwargs.get('priority', 5)
        )
        
        self.processing_started.emit(request_id)
        self.engine.submit_request(request)
        
        return request_id
        
    def process_batch(self, images: List[np.ndarray], model_name: str, **kwargs) -> List[str]:
        """Process batch of images"""
        request_ids = []
        
        for i, image in enumerate(images):
            request_id = f"{model_name}_batch_{i}_{time.time()}"
            request = InferenceRequest(
                id=request_id,
                model_name=model_name,
                input_data=image,
                callback=None,
                priority=kwargs.get('priority', 5)
            )
            
            self.engine.submit_request(request)
            request_ids.append(request_id)
            
        return request_ids
        
    def get_model_stats(self) -> Dict[str, Any]:
        """Get statistics for all models"""
        stats = {}
        
        for model_name in self.engine.model_configs:
            stats[model_name] = self.engine.get_performance_stats(model_name)
            
        return stats
        
    def optimize_models(self):
        """Optimize all models for inference"""
        self.engine.optimize_all_models()
        
    def shutdown(self):
        """Shutdown the processor"""
        self.engine.shutdown()