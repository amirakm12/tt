// Voice Visualizer Component
// Audio waveform visualization with Web Audio API

class VoiceVisualizer {
    constructor(container) {
        this.container = container;
        this.canvas = null;
        this.ctx = null;
        this.audioContext = null;
        this.analyser = null;
        this.microphone = null;
        this.dataArray = null;
        this.animationId = null;
        this.isActive = false;
        
        this.config = {
            fftSize: 256,
            smoothing: 0.8,
            minDecibels: -90,
            maxDecibels: -10,
            barWidth: 3,
            barGap: 1,
            waveColor: '#00d4ff',
            peakColor: '#ff00ff'
        };
        
        this.init();
    }
    
    init() {
        // Create canvas
        this.canvas = document.createElement('canvas');
        this.canvas.className = 'voice-canvas';
        this.container.appendChild(this.canvas);
        
        this.ctx = this.canvas.getContext('2d');
        
        // Set canvas size
        this.resize();
        
        // Initialize audio context
        this.initAudio();
    }
    
    async initAudio() {
        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.analyser = this.audioContext.createAnalyser();
            this.analyser.fftSize = this.config.fftSize;
            this.analyser.smoothingTimeConstant = this.config.smoothing;
            this.analyser.minDecibels = this.config.minDecibels;
            this.analyser.maxDecibels = this.config.maxDecibels;
            
            const bufferLength = this.analyser.frequencyBinCount;
            this.dataArray = new Uint8Array(bufferLength);
            
        } catch (error) {
            console.error('Failed to initialize audio context:', error);
        }
    }
    
    async start() {
        if (this.isActive) return;
        
        try {
            // Get microphone access
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                } 
            });
            
            this.microphone = this.audioContext.createMediaStreamSource(stream);
            this.microphone.connect(this.analyser);
            
            this.isActive = true;
            this.animate();
            
        } catch (error) {
            console.error('Failed to access microphone:', error);
            throw error;
        }
    }
    
    stop() {
        if (!this.isActive) return;
        
        this.isActive = false;
        
        if (this.microphone) {
            this.microphone.disconnect();
            this.microphone = null;
        }
        
        if (this.animationId) {
            cancelAnimationFrame(this.animationId);
            this.animationId = null;
        }
        
        // Clear canvas
        this.clearCanvas();
    }
    
    animate() {
        if (!this.isActive) return;
        
        this.animationId = requestAnimationFrame(() => this.animate());
        
        // Get frequency data
        this.analyser.getByteFrequencyData(this.dataArray);
        
        // Clear canvas
        this.clearCanvas();
        
        // Draw visualization
        this.drawWaveform();
        this.drawFrequencyBars();
        this.drawPeakIndicator();
    }
    
    clearCanvas() {
        // Create gradient background
        const gradient = this.ctx.createLinearGradient(0, 0, 0, this.canvas.height);
        gradient.addColorStop(0, 'rgba(10, 10, 10, 0.8)');
        gradient.addColorStop(1, 'rgba(10, 10, 10, 0.95)');
        
        this.ctx.fillStyle = gradient;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
    }
    
    drawWaveform() {
        const width = this.canvas.width;
        const height = this.canvas.height;
        const centerY = height / 2;
        
        // Get time domain data for waveform
        const waveData = new Uint8Array(this.analyser.fftSize);
        this.analyser.getByteTimeDomainData(waveData);
        
        // Draw waveform
        this.ctx.beginPath();
        this.ctx.strokeStyle = this.config.waveColor;
        this.ctx.lineWidth = 2;
        
        const sliceWidth = width / waveData.length;
        let x = 0;
        
        for (let i = 0; i < waveData.length; i++) {
            const v = waveData[i] / 128.0;
            const y = v * centerY;
            
            if (i === 0) {
                this.ctx.moveTo(x, y);
            } else {
                this.ctx.lineTo(x, y);
            }
            
            x += sliceWidth;
        }
        
        this.ctx.stroke();
    }
    
    drawFrequencyBars() {
        const barCount = this.dataArray.length / 2; // Use only lower frequencies
        const barWidth = this.config.barWidth;
        const barGap = this.config.barGap;
        const totalWidth = (barWidth + barGap) * barCount;
        const startX = (this.canvas.width - totalWidth) / 2;
        const maxHeight = this.canvas.height * 0.7;
        
        for (let i = 0; i < barCount; i++) {
            const value = this.dataArray[i];
            const percent = value / 255;
            const height = maxHeight * percent;
            const x = startX + i * (barWidth + barGap);
            const y = this.canvas.height - height;
            
            // Create gradient for bars
            const gradient = this.ctx.createLinearGradient(x, y, x, y + height);
            gradient.addColorStop(0, this.config.peakColor);
            gradient.addColorStop(1, this.config.waveColor);
            
            this.ctx.fillStyle = gradient;
            this.ctx.fillRect(x, y, barWidth, height);
            
            // Add glow effect for loud frequencies
            if (percent > 0.7) {
                this.ctx.shadowBlur = 10;
                this.ctx.shadowColor = this.config.peakColor;
                this.ctx.fillRect(x, y, barWidth, height);
                this.ctx.shadowBlur = 0;
            }
        }
    }
    
    drawPeakIndicator() {
        // Calculate overall volume
        let sum = 0;
        for (let i = 0; i < this.dataArray.length; i++) {
            sum += this.dataArray[i];
        }
        const average = sum / this.dataArray.length;
        const normalizedVolume = average / 255;
        
        // Draw peak indicator
        const indicatorSize = 20;
        const x = this.canvas.width - indicatorSize - 20;
        const y = 20;
        
        // Outer circle
        this.ctx.beginPath();
        this.ctx.arc(x + indicatorSize/2, y + indicatorSize/2, indicatorSize/2, 0, Math.PI * 2);
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
        this.ctx.lineWidth = 2;
        this.ctx.stroke();
        
        // Inner circle (volume indicator)
        this.ctx.beginPath();
        this.ctx.arc(x + indicatorSize/2, y + indicatorSize/2, (indicatorSize/2 - 4) * normalizedVolume, 0, Math.PI * 2);
        this.ctx.fillStyle = normalizedVolume > 0.7 ? this.config.peakColor : this.config.waveColor;
        this.ctx.fill();
        
        // Add glow for high volume
        if (normalizedVolume > 0.7) {
            this.ctx.shadowBlur = 20;
            this.ctx.shadowColor = this.config.peakColor;
            this.ctx.fill();
            this.ctx.shadowBlur = 0;
        }
    }
    
    resize() {
        const rect = this.container.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        
        // Redraw if active
        if (this.isActive) {
            this.clearCanvas();
        }
    }
    
    updateConfig(newConfig) {
        Object.assign(this.config, newConfig);
        
        if (this.analyser) {
            this.analyser.fftSize = this.config.fftSize;
            this.analyser.smoothingTimeConstant = this.config.smoothing;
            this.analyser.minDecibels = this.config.minDecibels;
            this.analyser.maxDecibels = this.config.maxDecibels;
            
            // Recreate data array with new size
            const bufferLength = this.analyser.frequencyBinCount;
            this.dataArray = new Uint8Array(bufferLength);
        }
    }
    
    getVolumeLevel() {
        if (!this.dataArray) return 0;
        
        let sum = 0;
        for (let i = 0; i < this.dataArray.length; i++) {
            sum += this.dataArray[i];
        }
        
        return (sum / this.dataArray.length) / 255;
    }
    
    getFrequencyData() {
        if (!this.dataArray) return [];
        
        return Array.from(this.dataArray);
    }
    
    destroy() {
        this.stop();
        
        if (this.audioContext) {
            this.audioContext.close();
        }
        
        if (this.canvas && this.canvas.parentNode) {
            this.canvas.parentNode.removeChild(this.canvas);
        }
    }
}

// Export for use
window.VoiceVisualizer = VoiceVisualizer;