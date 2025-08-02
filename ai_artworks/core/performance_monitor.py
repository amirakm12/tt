"""
Advanced Performance Monitoring System
Real-time performance tracking with automatic optimization
"""

import time
import psutil
import GPUtil
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import threading
import logging

from PySide6.QtCore import QObject, Signal, QTimer, QThread
import torch

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    timestamp: float
    cpu_percent: float
    cpu_per_core: List[float]
    memory_percent: float
    memory_used_gb: float
    memory_available_gb: float
    gpu_percent: float
    gpu_memory_percent: float
    gpu_temperature: float
    gpu_power_draw: float
    fps: float
    frame_time_ms: float
    draw_calls: int
    texture_memory_mb: float
    vertex_count: int
    
    # AI-specific metrics
    model_inference_ms: float = 0.0
    model_memory_mb: float = 0.0
    batch_size: int = 1
    
    # Network metrics
    network_latency_ms: float = 0.0
    bandwidth_mbps: float = 0.0


class PerformanceOptimizer(QObject):
    """Automatic performance optimization"""
    
    # Signals
    optimization_applied = Signal(str, dict)
    warning_issued = Signal(str, str)
    
    def __init__(self):
        super().__init__()
        self.thresholds = {
            'cpu_high': 80.0,
            'memory_high': 85.0,
            'gpu_high': 90.0,
            'gpu_temp_high': 85.0,
            'fps_low': 30.0,
            'frame_time_high': 33.0  # 30 FPS threshold
        }
        
        self.optimization_history = deque(maxlen=100)
        self.last_optimization = {}
        
    def analyze_metrics(self, metrics: PerformanceMetrics) -> Dict[str, Any]:
        """Analyze metrics and suggest optimizations"""
        optimizations = {}
        
        # CPU optimization
        if metrics.cpu_percent > self.thresholds['cpu_high']:
            optimizations['cpu'] = {
                'issue': 'High CPU usage',
                'action': 'reduce_thread_count',
                'params': {'reduction': 0.2}
            }
            
        # Memory optimization
        if metrics.memory_percent > self.thresholds['memory_high']:
            optimizations['memory'] = {
                'issue': 'High memory usage',
                'action': 'clear_cache',
                'params': {'aggressive': True}
            }
            
        # GPU optimization
        if metrics.gpu_percent > self.thresholds['gpu_high']:
            optimizations['gpu'] = {
                'issue': 'High GPU usage',
                'action': 'reduce_quality',
                'params': {'level': 1}
            }
            
        # Temperature throttling
        if metrics.gpu_temperature > self.thresholds['gpu_temp_high']:
            optimizations['temperature'] = {
                'issue': 'GPU overheating',
                'action': 'enable_throttling',
                'params': {'limit_fps': 60}
            }
            
        # FPS optimization
        if metrics.fps < self.thresholds['fps_low'] and metrics.fps > 0:
            optimizations['fps'] = {
                'issue': 'Low FPS',
                'action': 'reduce_resolution',
                'params': {'scale': 0.75}
            }
            
        return optimizations
        
    def apply_optimizations(self, optimizations: Dict[str, Any]):
        """Apply performance optimizations"""
        for category, optimization in optimizations.items():
            if category not in self.last_optimization or \
               time.time() - self.last_optimization.get(category, 0) > 10:
                
                self.optimization_applied.emit(
                    optimization['action'],
                    optimization['params']
                )
                
                self.last_optimization[category] = time.time()
                self.optimization_history.append({
                    'timestamp': time.time(),
                    'category': category,
                    'optimization': optimization
                })
                
                logger.info(f"Applied optimization: {optimization['action']}")


class GPUMonitor(QThread):
    """Dedicated GPU monitoring thread"""
    
    metrics_updated = Signal(dict)
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.gpus = GPUtil.getGPUs()
        self.cuda_available = torch.cuda.is_available()
        
    def run(self):
        """Monitor GPU metrics"""
        while self.running:
            try:
                metrics = {}
                
                if self.gpus:
                    gpu = self.gpus[0]  # Primary GPU
                    metrics['gpu_percent'] = gpu.load * 100
                    metrics['gpu_memory_percent'] = gpu.memoryUtil * 100
                    metrics['gpu_temperature'] = gpu.temperature
                    metrics['gpu_memory_used'] = gpu.memoryUsed
                    metrics['gpu_memory_total'] = gpu.memoryTotal
                    
                if self.cuda_available:
                    # PyTorch CUDA metrics
                    metrics['cuda_memory_allocated'] = torch.cuda.memory_allocated() / 1024**2
                    metrics['cuda_memory_reserved'] = torch.cuda.memory_reserved() / 1024**2
                    
                self.metrics_updated.emit(metrics)
                
            except Exception as e:
                logger.error(f"GPU monitoring error: {e}")
                
            time.sleep(0.5)  # 2Hz update rate
            
    def stop(self):
        """Stop monitoring"""
        self.running = False
        self.wait()


class PerformanceMonitor(QObject):
    """Comprehensive performance monitoring system"""
    
    # Signals
    metrics_updated = Signal(PerformanceMetrics)
    performance_report = Signal(dict)
    optimization_suggested = Signal(dict)
    
    def __init__(self):
        super().__init__()
        
        # Monitoring components
        self.optimizer = PerformanceOptimizer()
        self.gpu_monitor = GPUMonitor()
        
        # Metrics storage
        self.metrics_history = deque(maxlen=1000)
        self.current_metrics = None
        
        # Frame timing
        self.frame_times = deque(maxlen=120)
        self.last_frame_time = time.time()
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_metrics)
        self.update_timer.start(100)  # 10Hz update
        
        # Report timer
        self.report_timer = QTimer()
        self.report_timer.timeout.connect(self.generate_report)
        self.report_timer.start(5000)  # 5 second reports
        
        # Connect GPU monitor
        self.gpu_monitor.metrics_updated.connect(self._on_gpu_metrics)
        self.gpu_monitor.start()
        
        # Performance counters
        self.draw_calls = 0
        self.vertex_count = 0
        self.texture_memory = 0
        
    def update_metrics(self):
        """Update all performance metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_per_core = psutil.cpu_percent(interval=0.1, percpu=True)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / 1024**3
            memory_available_gb = memory.available / 1024**3
            
            # Frame timing
            current_time = time.time()
            frame_time = current_time - self.last_frame_time
            self.last_frame_time = current_time
            self.frame_times.append(frame_time)
            
            # Calculate FPS
            if len(self.frame_times) > 0:
                avg_frame_time = np.mean(self.frame_times)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                frame_time_ms = avg_frame_time * 1000
            else:
                fps = 0
                frame_time_ms = 0
                
            # Create metrics object
            metrics = PerformanceMetrics(
                timestamp=current_time,
                cpu_percent=cpu_percent,
                cpu_per_core=cpu_per_core,
                memory_percent=memory_percent,
                memory_used_gb=memory_used_gb,
                memory_available_gb=memory_available_gb,
                gpu_percent=getattr(self, '_gpu_percent', 0),
                gpu_memory_percent=getattr(self, '_gpu_memory_percent', 0),
                gpu_temperature=getattr(self, '_gpu_temperature', 0),
                gpu_power_draw=getattr(self, '_gpu_power_draw', 0),
                fps=fps,
                frame_time_ms=frame_time_ms,
                draw_calls=self.draw_calls,
                texture_memory_mb=self.texture_memory,
                vertex_count=self.vertex_count
            )
            
            self.current_metrics = metrics
            self.metrics_history.append(metrics)
            self.metrics_updated.emit(metrics)
            
            # Check for optimizations
            optimizations = self.optimizer.analyze_metrics(metrics)
            if optimizations:
                self.optimization_suggested.emit(optimizations)
                
        except Exception as e:
            logger.error(f"Metrics update error: {e}")
            
    def _on_gpu_metrics(self, gpu_data: dict):
        """Handle GPU metrics update"""
        self._gpu_percent = gpu_data.get('gpu_percent', 0)
        self._gpu_memory_percent = gpu_data.get('gpu_memory_percent', 0)
        self._gpu_temperature = gpu_data.get('gpu_temperature', 0)
        self._gpu_power_draw = gpu_data.get('gpu_power_draw', 0)
        
    def record_frame(self):
        """Record frame rendering"""
        self.frame_times.append(time.time() - self.last_frame_time)
        self.last_frame_time = time.time()
        
    def record_draw_call(self, vertices: int = 0):
        """Record a draw call"""
        self.draw_calls += 1
        self.vertex_count += vertices
        
    def record_texture_memory(self, size_mb: float):
        """Record texture memory usage"""
        self.texture_memory += size_mb
        
    def reset_frame_counters(self):
        """Reset per-frame counters"""
        self.draw_calls = 0
        self.vertex_count = 0
        
    def generate_report(self):
        """Generate performance report"""
        if not self.metrics_history:
            return
            
        # Calculate statistics
        recent_metrics = list(self.metrics_history)[-60:]  # Last 6 seconds
        
        report = {
            'timestamp': time.time(),
            'duration_seconds': 6.0,
            'avg_fps': np.mean([m.fps for m in recent_metrics]),
            'min_fps': np.min([m.fps for m in recent_metrics]),
            'max_fps': np.max([m.fps for m in recent_metrics]),
            'fps_stability': np.std([m.fps for m in recent_metrics]),
            'avg_cpu': np.mean([m.cpu_percent for m in recent_metrics]),
            'avg_memory': np.mean([m.memory_percent for m in recent_metrics]),
            'avg_gpu': np.mean([m.gpu_percent for m in recent_metrics]),
            'avg_gpu_temp': np.mean([m.gpu_temperature for m in recent_metrics]),
            'total_draw_calls': sum([m.draw_calls for m in recent_metrics]),
            'total_vertices': sum([m.vertex_count for m in recent_metrics]),
            'performance_score': self._calculate_performance_score(recent_metrics)
        }
        
        self.performance_report.emit(report)
        
    def _calculate_performance_score(self, metrics: List[PerformanceMetrics]) -> float:
        """Calculate overall performance score (0-100)"""
        if not metrics:
            return 0
            
        # Weight different factors
        fps_score = min(100, (np.mean([m.fps for m in metrics]) / 60) * 100)
        cpu_score = max(0, 100 - np.mean([m.cpu_percent for m in metrics]))
        gpu_score = max(0, 100 - np.mean([m.gpu_percent for m in metrics]))
        memory_score = max(0, 100 - np.mean([m.memory_percent for m in metrics]))
        
        # Weighted average
        score = (
            fps_score * 0.4 +
            cpu_score * 0.2 +
            gpu_score * 0.3 +
            memory_score * 0.1
        )
        
        return round(score, 1)
        
    def get_optimization_suggestions(self) -> List[str]:
        """Get human-readable optimization suggestions"""
        if not self.current_metrics:
            return []
            
        suggestions = []
        
        if self.current_metrics.fps < 30:
            suggestions.append("• Reduce canvas resolution or disable effects for better FPS")
            
        if self.current_metrics.gpu_memory_percent > 80:
            suggestions.append("• Clear unused layers to free GPU memory")
            
        if self.current_metrics.cpu_percent > 70:
            suggestions.append("• Disable background plugins to reduce CPU load")
            
        if self.current_metrics.gpu_temperature > 80:
            suggestions.append("• Enable V-Sync or FPS limiting to reduce GPU temperature")
            
        return suggestions
        
    def shutdown(self):
        """Shutdown monitoring"""
        self.update_timer.stop()
        self.report_timer.stop()
        self.gpu_monitor.stop()