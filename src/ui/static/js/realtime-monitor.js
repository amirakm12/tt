// Real-time System Monitor
// Advanced monitoring with Chart.js and WebSocket data streaming

class RealtimeMonitor {
    constructor() {
        this.charts = {};
        this.updateInterval = 1000; // 1 second
        this.maxDataPoints = 60; // Show last 60 seconds
        
        this.metrics = {
            cpu: [],
            memory: [],
            gpu: [],
            network: { in: [], out: [] },
            aiOperations: []
        };
    }
    
    initialize() {
        // Initialize performance chart
        this.initPerformanceChart();
        
        // Initialize network chart
        this.initNetworkChart();
        
        // Initialize AI operations chart
        this.initAIOperationsChart();
        
        // Start real-time updates
        this.startUpdates();
    }
    
    initPerformanceChart() {
        const ctx = document.getElementById('performance-chart');
        if (!ctx) return;
        
        this.charts.performance = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.generateTimeLabels(),
                datasets: [{
                    label: 'CPU',
                    data: this.metrics.cpu,
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    tension: 0.4
                }, {
                    label: 'Memory',
                    data: this.metrics.memory,
                    borderColor: '#00ff88',
                    backgroundColor: 'rgba(0, 255, 136, 0.1)',
                    tension: 0.4
                }, {
                    label: 'GPU',
                    data: this.metrics.gpu,
                    borderColor: '#ff00ff',
                    backgroundColor: 'rgba(255, 0, 255, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#ffffff',
                            font: {
                                family: 'Inter'
                            }
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(0, 0, 0, 0.8)',
                        titleColor: '#ffffff',
                        bodyColor: '#ffffff',
                        borderColor: '#00d4ff',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        display: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#666666'
                        }
                    },
                    y: {
                        display: true,
                        min: 0,
                        max: 100,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#666666',
                            callback: function(value) {
                                return value + '%';
                            }
                        }
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        });
    }
    
    initNetworkChart() {
        const ctx = document.getElementById('network-chart');
        if (!ctx) return;
        
        this.charts.network = new Chart(ctx, {
            type: 'line',
            data: {
                labels: this.generateTimeLabels(),
                datasets: [{
                    label: 'Incoming',
                    data: this.metrics.network.in,
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    tension: 0.4
                }, {
                    label: 'Outgoing',
                    data: this.metrics.network.out,
                    borderColor: '#ff00ff',
                    backgroundColor: 'rgba(255, 0, 255, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            color: '#ffffff'
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            color: '#666666',
                            callback: function(value) {
                                return value + ' MB/s';
                            }
                        }
                    }
                }
            }
        });
    }
    
    initAIOperationsChart() {
        const ctx = document.getElementById('neural-activity-chart');
        if (!ctx) return;
        
        this.charts.aiOperations = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Triage', 'Research', 'Orchestration', 'RAG', 'Decoder'],
                datasets: [{
                    label: 'Operations/sec',
                    data: [0, 0, 0, 0, 0],
                    backgroundColor: [
                        'rgba(0, 255, 136, 0.8)',
                        'rgba(255, 0, 255, 0.8)',
                        'rgba(0, 212, 255, 0.8)',
                        'rgba(255, 170, 0, 0.8)',
                        'rgba(255, 0, 68, 0.8)'
                    ],
                    borderColor: [
                        '#00ff88',
                        '#ff00ff',
                        '#00d4ff',
                        '#ffaa00',
                        '#ff0044'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#666666'
                        }
                    },
                    x: {
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#ffffff'
                        }
                    }
                }
            }
        });
    }
    
    generateTimeLabels() {
        const labels = [];
        for (let i = this.maxDataPoints; i > 0; i--) {
            labels.push(`-${i}s`);
        }
        return labels;
    }
    
    startUpdates() {
        // Subscribe to WebSocket updates
        if (window.AISystemDashboard && window.AISystemDashboard.ws) {
            // Updates will come through WebSocket
        }
        
        // Also fetch periodic updates
        setInterval(() => this.fetchMetrics(), this.updateInterval);
    }
    
    async fetchMetrics() {
        try {
            const response = await fetch('/api/system/metrics');
            const data = await response.json();
            
            if (data.realtime) {
                this.updateCharts(data.realtime);
            }
        } catch (error) {
            console.error('Failed to fetch metrics:', error);
        }
    }
    
    updateCharts(data) {
        // Update performance chart
        if (data.cpu && this.charts.performance) {
            this.addDataPoint(this.metrics.cpu, data.cpu[data.cpu.length - 1].value);
            this.addDataPoint(this.metrics.memory, data.memory[data.memory.length - 1].value);
            this.addDataPoint(this.metrics.gpu, data.gpu[data.gpu.length - 1].value);
            
            this.charts.performance.data.labels = this.generateTimeLabels();
            this.charts.performance.update('none'); // No animation for smooth updates
        }
        
        // Update network chart
        if (data.network && this.charts.network) {
            // Add network data points
            this.charts.network.update('none');
        }
        
        // Update AI operations
        if (data.ai_operations && this.charts.aiOperations) {
            // Update bar chart data
            this.charts.aiOperations.update('none');
        }
    }
    
    addDataPoint(array, value) {
        array.push(value);
        if (array.length > this.maxDataPoints) {
            array.shift();
        }
    }
    
    handleRealtimeData(data) {
        // Handle real-time WebSocket data
        if (data.cpu !== undefined) {
            this.addDataPoint(this.metrics.cpu, data.cpu);
        }
        if (data.memory !== undefined) {
            this.addDataPoint(this.metrics.memory, data.memory);
        }
        if (data.gpu !== undefined) {
            this.addDataPoint(this.metrics.gpu, data.gpu);
        }
        
        // Update charts
        if (this.charts.performance) {
            this.charts.performance.update('none');
        }
    }
    
    resize() {
        Object.values(this.charts).forEach(chart => {
            if (chart) {
                chart.resize();
            }
        });
    }
    
    destroy() {
        Object.values(this.charts).forEach(chart => {
            if (chart) {
                chart.destroy();
            }
        });
    }
}

// Initialize when dashboard loads
window.RealtimeMonitor = RealtimeMonitor;