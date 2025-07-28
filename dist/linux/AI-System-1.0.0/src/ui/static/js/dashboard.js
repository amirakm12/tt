// AI System Dashboard JavaScript

class Dashboard {
    constructor() {
        this.websocket = null;
        this.charts = {};
        this.isConnected = false;
        this.voiceRecognition = null;
        this.isListening = false;
        
        this.init();
    }
    
    init() {
        this.setupNavigation();
        this.setupWebSocket();
        this.setupVoiceInterface();
        this.setupCharts();
        this.setupEventListeners();
        this.startDataUpdates();
    }
    
    setupNavigation() {
        const navLinks = document.querySelectorAll('.nav-link');
        const sections = document.querySelectorAll('.dashboard-section');
        
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                
                // Remove active class from all links and sections
                navLinks.forEach(nl => nl.classList.remove('active'));
                sections.forEach(section => section.classList.remove('active'));
                
                // Add active class to clicked link
                link.classList.add('active');
                
                // Show corresponding section
                const targetId = link.getAttribute('href').substring(1);
                const targetSection = document.getElementById(targetId);
                if (targetSection) {
                    targetSection.classList.add('active');
                }
            });
        });
    }
    
    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        
        this.websocket = new WebSocket(wsUrl);
        
        this.websocket.onopen = () => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.updateConnectionStatus(true);
        };
        
        this.websocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                this.handleWebSocketMessage(data);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };
        
        this.websocket.onclose = () => {
            console.log('WebSocket disconnected');
            this.isConnected = false;
            this.updateConnectionStatus(false);
            
            // Attempt to reconnect after 5 seconds
            setTimeout(() => {
                this.setupWebSocket();
            }, 5000);
        };
        
        this.websocket.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'system_status':
                this.updateSystemStatus(data.payload);
                break;
            case 'agent_status':
                this.updateAgentStatus(data.payload);
                break;
            case 'performance_metrics':
                this.updatePerformanceMetrics(data.payload);
                break;
            case 'sensor_data':
                this.updateSensorData(data.payload);
                break;
            case 'log_entry':
                this.addLogEntry(data.payload);
                break;
            case 'security_event':
                this.handleSecurityEvent(data.payload);
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }
    
    updateConnectionStatus(connected) {
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            if (connected) {
                statusElement.innerHTML = '<i class="fas fa-circle"></i> Connected';
                statusElement.classList.remove('disconnected');
            } else {
                statusElement.innerHTML = '<i class="fas fa-circle"></i> Disconnected';
                statusElement.classList.add('disconnected');
            }
        }
    }
    
    updateSystemStatus(status) {
        // Update system health
        const healthElement = document.getElementById('system-health');
        if (healthElement) {
            healthElement.textContent = status.health || 'Unknown';
        }
        
        // Update uptime
        const uptimeElement = document.getElementById('system-uptime');
        if (uptimeElement) {
            uptimeElement.textContent = this.formatUptime(status.uptime || 0);
        }
        
        // Update active tasks
        const tasksElement = document.getElementById('active-tasks');
        if (tasksElement) {
            tasksElement.textContent = status.active_tasks || 0;
        }
        
        // Update CPU usage
        const cpuElement = document.getElementById('cpu-usage');
        if (cpuElement) {
            cpuElement.textContent = `${Math.round(status.cpu_usage || 0)}%`;
        }
    }
    
    updateAgentStatus(agents) {
        // Update triage agent
        if (agents.triage) {
            const statusEl = document.getElementById('triage-status');
            const processedEl = document.getElementById('triage-processed');
            const successEl = document.getElementById('triage-success');
            
            if (statusEl) statusEl.textContent = agents.triage.status;
            if (processedEl) processedEl.textContent = agents.triage.processed || 0;
            if (successEl) successEl.textContent = `${agents.triage.success_rate || 100}%`;
        }
        
        // Update research agent
        if (agents.research) {
            const statusEl = document.getElementById('research-status');
            const queriesEl = document.getElementById('research-queries');
            const kbSizeEl = document.getElementById('kb-size');
            
            if (statusEl) statusEl.textContent = agents.research.status;
            if (queriesEl) queriesEl.textContent = agents.research.queries || 0;
            if (kbSizeEl) kbSizeEl.textContent = `${agents.research.kb_size || 0} docs`;
        }
        
        // Update orchestration agent
        if (agents.orchestration) {
            const statusEl = document.getElementById('orchestration-status');
            const activeEl = document.getElementById('active-workflows');
            const completedEl = document.getElementById('completed-workflows');
            
            if (statusEl) statusEl.textContent = agents.orchestration.status;
            if (activeEl) activeEl.textContent = agents.orchestration.active_workflows || 0;
            if (completedEl) completedEl.textContent = agents.orchestration.completed_workflows || 0;
        }
    }
    
    updatePerformanceMetrics(metrics) {
        // Update performance chart
        if (this.charts.performance) {
            const chart = this.charts.performance;
            const now = new Date();
            
            // Add new data point
            chart.data.labels.push(now.toLocaleTimeString());
            chart.data.datasets[0].data.push(metrics.cpu_usage || 0);
            chart.data.datasets[1].data.push(metrics.memory_usage || 0);
            
            // Keep only last 20 data points
            if (chart.data.labels.length > 20) {
                chart.data.labels.shift();
                chart.data.datasets[0].data.shift();
                chart.data.datasets[1].data.shift();
            }
            
            chart.update('none');
        }
        
        // Update activity chart
        if (this.charts.activity && metrics.agent_activity) {
            const chart = this.charts.activity;
            chart.data.datasets[0].data = [
                metrics.agent_activity.triage || 0,
                metrics.agent_activity.research || 0,
                metrics.agent_activity.orchestration || 0
            ];
            chart.update('none');
        }
    }
    
    updateSensorData(sensors) {
        // Update CPU sensor
        if (sensors.cpu) {
            const valueEl = document.getElementById('cpu-sensor');
            const qualityEl = document.getElementById('cpu-quality');
            
            if (valueEl) valueEl.textContent = `${Math.round(sensors.cpu.value)}%`;
            if (qualityEl) {
                qualityEl.textContent = sensors.cpu.quality;
                qualityEl.className = `sensor-quality ${sensors.cpu.quality.toLowerCase()}`;
            }
        }
        
        // Update Memory sensor
        if (sensors.memory) {
            const valueEl = document.getElementById('memory-sensor');
            const qualityEl = document.getElementById('memory-quality');
            
            if (valueEl) valueEl.textContent = `${Math.round(sensors.memory.value)}%`;
            if (qualityEl) {
                qualityEl.textContent = sensors.memory.quality;
                qualityEl.className = `sensor-quality ${sensors.memory.quality.toLowerCase()}`;
            }
        }
        
        // Update Network sensor
        if (sensors.network) {
            const valueEl = document.getElementById('network-sensor');
            const qualityEl = document.getElementById('network-quality');
            
            if (valueEl) valueEl.textContent = `${sensors.network.value.toFixed(1)} MB/s`;
            if (qualityEl) {
                qualityEl.textContent = sensors.network.quality;
                qualityEl.className = `sensor-quality ${sensors.network.quality.toLowerCase()}`;
            }
        }
        
        // Update Disk sensor
        if (sensors.disk) {
            const valueEl = document.getElementById('disk-sensor');
            const qualityEl = document.getElementById('disk-quality');
            
            if (valueEl) valueEl.textContent = `${Math.round(sensors.disk.value)}%`;
            if (qualityEl) {
                qualityEl.textContent = sensors.disk.quality;
                qualityEl.className = `sensor-quality ${sensors.disk.quality.toLowerCase()}`;
            }
        }
    }
    
    addLogEntry(logEntry) {
        const logsDisplay = document.getElementById('logs-display');
        if (logsDisplay) {
            const logElement = document.createElement('div');
            logElement.className = `log-entry log-${logEntry.level.toLowerCase()}`;
            logElement.innerHTML = `
                <span class="log-timestamp">${new Date(logEntry.timestamp).toLocaleTimeString()}</span>
                <span class="log-level">[${logEntry.level}]</span>
                <span class="log-message">${logEntry.message}</span>
            `;
            
            logsDisplay.appendChild(logElement);
            logsDisplay.scrollTop = logsDisplay.scrollHeight;
            
            // Keep only last 100 log entries
            while (logsDisplay.children.length > 100) {
                logsDisplay.removeChild(logsDisplay.firstChild);
            }
        }
    }
    
    handleSecurityEvent(event) {
        // Update threat count
        const threatCountEl = document.getElementById('threat-count');
        if (threatCountEl && event.threat_count !== undefined) {
            threatCountEl.textContent = event.threat_count;
        }
        
        // Update anomaly count
        const anomalyCountEl = document.getElementById('anomaly-count');
        if (anomalyCountEl && event.anomaly_count !== undefined) {
            anomalyCountEl.textContent = event.anomaly_count;
        }
        
        // Update last scan time
        const lastScanEl = document.getElementById('last-scan');
        if (lastScanEl && event.last_scan) {
            lastScanEl.textContent = new Date(event.last_scan).toLocaleString();
        }
    }
    
    setupVoiceInterface() {
        const voiceToggle = document.getElementById('voice-toggle');
        const voiceModal = document.getElementById('voice-modal');
        const closeModal = document.getElementById('close-voice-modal');
        const startListening = document.getElementById('start-listening');
        const stopListening = document.getElementById('stop-listening');
        
        // Check if browser supports speech recognition
        if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            this.voiceRecognition = new SpeechRecognition();
            
            this.voiceRecognition.continuous = true;
            this.voiceRecognition.interimResults = true;
            this.voiceRecognition.lang = 'en-US';
            
            this.voiceRecognition.onstart = () => {
                this.isListening = true;
                this.updateVoiceStatus('Listening...');
                startListening.disabled = true;
                stopListening.disabled = false;
            };
            
            this.voiceRecognition.onend = () => {
                this.isListening = false;
                this.updateVoiceStatus('Click to start listening');
                startListening.disabled = false;
                stopListening.disabled = true;
            };
            
            this.voiceRecognition.onresult = (event) => {
                let transcript = '';
                for (let i = event.resultIndex; i < event.results.length; i++) {
                    transcript += event.results[i][0].transcript;
                }
                this.updateVoiceTranscript(transcript);
                
                // Send to server if final result
                if (event.results[event.results.length - 1].isFinal) {
                    this.sendVoiceCommand(transcript);
                }
            };
        }
        
        // Event listeners
        if (voiceToggle) {
            voiceToggle.addEventListener('click', () => {
                voiceModal.classList.add('active');
            });
        }
        
        if (closeModal) {
            closeModal.addEventListener('click', () => {
                voiceModal.classList.remove('active');
                if (this.isListening) {
                    this.voiceRecognition.stop();
                }
            });
        }
        
        if (startListening && this.voiceRecognition) {
            startListening.addEventListener('click', () => {
                this.voiceRecognition.start();
            });
        }
        
        if (stopListening && this.voiceRecognition) {
            stopListening.addEventListener('click', () => {
                this.voiceRecognition.stop();
            });
        }
        
        // Close modal when clicking outside
        if (voiceModal) {
            voiceModal.addEventListener('click', (e) => {
                if (e.target === voiceModal) {
                    voiceModal.classList.remove('active');
                    if (this.isListening) {
                        this.voiceRecognition.stop();
                    }
                }
            });
        }
    }
    
    updateVoiceStatus(status) {
        const statusElement = document.querySelector('#voice-status p');
        if (statusElement) {
            statusElement.textContent = status;
        }
        
        const icon = document.querySelector('#voice-status i');
        if (icon) {
            if (this.isListening) {
                icon.className = 'fas fa-microphone';
            } else {
                icon.className = 'fas fa-microphone-slash';
            }
        }
    }
    
    updateVoiceTranscript(transcript) {
        const transcriptElement = document.getElementById('voice-transcript');
        if (transcriptElement) {
            transcriptElement.textContent = transcript;
        }
    }
    
    sendVoiceCommand(command) {
        if (this.websocket && this.isConnected) {
            this.websocket.send(JSON.stringify({
                type: 'voice_command',
                payload: { command: command }
            }));
        }
    }
    
    setupCharts() {
        // Performance Chart
        const performanceCtx = document.getElementById('performance-chart');
        if (performanceCtx) {
            this.charts.performance = new Chart(performanceCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'CPU Usage (%)',
                        data: [],
                        borderColor: '#2563eb',
                        backgroundColor: 'rgba(37, 99, 235, 0.1)',
                        tension: 0.4
                    }, {
                        label: 'Memory Usage (%)',
                        data: [],
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 100
                        }
                    },
                    plugins: {
                        legend: {
                            position: 'top'
                        }
                    },
                    animation: {
                        duration: 0
                    }
                }
            });
        }
        
        // Activity Chart
        const activityCtx = document.getElementById('activity-chart');
        if (activityCtx) {
            this.charts.activity = new Chart(activityCtx, {
                type: 'doughnut',
                data: {
                    labels: ['Triage Agent', 'Research Agent', 'Orchestration Agent'],
                    datasets: [{
                        data: [0, 0, 0],
                        backgroundColor: ['#2563eb', '#10b981', '#f59e0b'],
                        borderWidth: 2,
                        borderColor: '#ffffff'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        }
                    }
                }
            });
        }
    }
    
    setupEventListeners() {
        // Settings forms
        const generalSettings = document.getElementById('general-settings');
        if (generalSettings) {
            generalSettings.addEventListener('submit', (e) => {
                e.preventDefault();
                this.saveGeneralSettings();
            });
        }
        
        const aiSettings = document.getElementById('ai-settings');
        if (aiSettings) {
            aiSettings.addEventListener('submit', (e) => {
                e.preventDefault();
                this.saveAISettings();
            });
        }
        
        // Temperature slider
        const temperatureSlider = document.getElementById('temperature');
        const temperatureValue = document.getElementById('temperature-value');
        if (temperatureSlider && temperatureValue) {
            temperatureSlider.addEventListener('input', (e) => {
                temperatureValue.textContent = e.target.value;
            });
        }
        
        // Log controls
        const clearLogs = document.getElementById('clear-logs');
        if (clearLogs) {
            clearLogs.addEventListener('click', () => {
                const logsDisplay = document.getElementById('logs-display');
                if (logsDisplay) {
                    logsDisplay.innerHTML = '';
                }
            });
        }
        
        const exportLogs = document.getElementById('export-logs');
        if (exportLogs) {
            exportLogs.addEventListener('click', () => {
                this.exportLogs();
            });
        }
    }
    
    saveGeneralSettings() {
        const logLevel = document.getElementById('system-log-level').value;
        const enableVoice = document.getElementById('enable-voice').checked;
        
        const settings = {
            log_level: logLevel,
            enable_voice: enableVoice
        };
        
        if (this.websocket && this.isConnected) {
            this.websocket.send(JSON.stringify({
                type: 'update_settings',
                payload: { general: settings }
            }));
        }
        
        this.showNotification('General settings saved successfully', 'success');
    }
    
    saveAISettings() {
        const model = document.getElementById('ai-model').value;
        const temperature = parseFloat(document.getElementById('temperature').value);
        
        const settings = {
            model: model,
            temperature: temperature
        };
        
        if (this.websocket && this.isConnected) {
            this.websocket.send(JSON.stringify({
                type: 'update_settings',
                payload: { ai: settings }
            }));
        }
        
        this.showNotification('AI settings saved successfully', 'success');
    }
    
    exportLogs() {
        const logsDisplay = document.getElementById('logs-display');
        if (logsDisplay) {
            const logs = logsDisplay.textContent;
            const blob = new Blob([logs], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            
            const a = document.createElement('a');
            a.href = url;
            a.download = `ai-system-logs-${new Date().toISOString().split('T')[0]}.txt`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }
    }
    
    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.textContent = message;
        
        // Style the notification
        notification.style.position = 'fixed';
        notification.style.top = '20px';
        notification.style.right = '20px';
        notification.style.padding = '1rem 1.5rem';
        notification.style.borderRadius = '0.5rem';
        notification.style.color = 'white';
        notification.style.fontWeight = '500';
        notification.style.zIndex = '9999';
        notification.style.transform = 'translateX(100%)';
        notification.style.transition = 'transform 0.3s ease';
        
        // Set background color based on type
        switch (type) {
            case 'success':
                notification.style.backgroundColor = '#10b981';
                break;
            case 'error':
                notification.style.backgroundColor = '#ef4444';
                break;
            case 'warning':
                notification.style.backgroundColor = '#f59e0b';
                break;
            default:
                notification.style.backgroundColor = '#2563eb';
        }
        
        // Add to page
        document.body.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.style.transform = 'translateX(0)';
        }, 100);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.style.transform = 'translateX(100%)';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }
    
    startDataUpdates() {
        // Request initial data
        if (this.websocket && this.isConnected) {
            this.websocket.send(JSON.stringify({
                type: 'request_status',
                payload: {}
            }));
        }
        
        // Set up periodic updates
        setInterval(() => {
            if (this.websocket && this.isConnected) {
                this.websocket.send(JSON.stringify({
                    type: 'request_status',
                    payload: {}
                }));
            }
        }, 5000); // Update every 5 seconds
    }
    
    formatUptime(seconds) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = Math.floor(seconds % 60);
        
        return `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new Dashboard();
});