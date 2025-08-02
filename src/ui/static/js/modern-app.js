// Modern AI System Dashboard - Main Application
// Neural Command Center with React, Three.js, and WebSocket integration

class ModernDashboard {
    constructor() {
        this.state = {
            currentView: 'overview',
            theme: 'dark',
            metrics: {},
            agents: {},
            notifications: [],
            isConnected: false,
            user: {
                name: 'Neural Operator',
                role: 'Administrator'
            }
        };
        
        this.ws = null;
        this.chatWs = null;
        this.voiceWs = null;
        this.visualizers = {};
        this.charts = {};
    }
    
    async initialize() {
        console.log('🚀 Initializing Neural Command Center...');
        
        // Hide loading screen with fade effect
        setTimeout(() => {
            const loadingScreen = document.getElementById('loading-screen');
            loadingScreen.style.opacity = '0';
            setTimeout(() => loadingScreen.style.display = 'none', 500);
        }, 1000);
        
        // Initialize React app
        this.renderApp();
        
        // Connect to WebSocket
        this.connectWebSocket();
        
        // Load initial data
        await this.loadInitialData();
        
        // Initialize 3D visualizations
        this.init3DVisualizations();
        
        // Setup event listeners
        this.setupEventListeners();
        
        // Start real-time updates
        this.startRealtimeUpdates();
        
        console.log('✅ Neural Command Center initialized');
    }
    
    renderApp() {
        const { useState, useEffect, useRef } = React;
        
        const App = () => {
            const [view, setView] = useState('overview');
            const [theme, setTheme] = useState('dark');
            const [sidebarOpen, setSidebarOpen] = useState(true);
            const [notifications, setNotifications] = useState([]);
            
            return React.createElement('div', { className: 'neural-dashboard' },
                // Header
                React.createElement(Header, { 
                    onViewChange: setView,
                    currentView: view,
                    theme: theme,
                    onThemeToggle: () => setTheme(theme === 'dark' ? 'light' : 'dark')
                }),
                
                // Main Content
                React.createElement('div', { className: 'dashboard-layout' },
                    // Sidebar
                    sidebarOpen && React.createElement(Sidebar, {
                        onViewChange: setView,
                        currentView: view
                    }),
                    
                    // Content Area
                    React.createElement('main', { className: 'dashboard-content' },
                        view === 'overview' && React.createElement(OverviewDashboard),
                        view === 'agents' && React.createElement(AgentsView),
                        view === 'neural' && React.createElement(NeuralNetworkView),
                        view === 'chat' && React.createElement(ChatInterfaceWrapper),
                        view === 'voice' && React.createElement(VoiceInterface),
                        view === 'settings' && React.createElement(SettingsView)
                    )
                ),
                
                // Floating Action Buttons
                React.createElement(FloatingActions),
                
                // Notifications
                React.createElement(NotificationStack, { notifications })
            );
        };
        
        // Render the app
        ReactDOM.render(React.createElement(App), document.getElementById('root'));
    }
    
    connectWebSocket() {
        const wsUrl = `ws://${window.location.host}/ws/realtime`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('✅ WebSocket connected');
            this.state.isConnected = true;
            this.showNotification('Connected to Neural Command Center', 'success');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.showNotification('Connection error', 'error');
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.state.isConnected = false;
            this.showNotification('Disconnected from server', 'warning');
            
            // Attempt to reconnect after 5 seconds
            setTimeout(() => this.connectWebSocket(), 5000);
        };
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'metrics':
                this.updateMetrics(data.data);
                break;
            case 'agent_update':
                this.updateAgentStatus(data.agent);
                break;
            case 'notification':
                this.showNotification(data.message, data.level);
                break;
            case 'theme_update':
                this.updateTheme(data.settings);
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }
    
    async loadInitialData() {
        try {
            // Load system status
            const statusResponse = await fetch('/api/system/status');
            const status = await statusResponse.json();
            this.updateSystemStatus(status);
            
            // Load agents
            const agentsResponse = await fetch('/api/agents/status');
            const agents = await agentsResponse.json();
            this.state.agents = agents;
            
            // Load metrics
            const metricsResponse = await fetch('/api/system/metrics');
            const metrics = await metricsResponse.json();
            this.state.metrics = metrics;
            
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.showNotification('Failed to load system data', 'error');
        }
    }
    
    init3DVisualizations() {
        // Initialize Three.js visualizations
        const neuralContainer = document.getElementById('neural-network-viz');
        if (neuralContainer) {
            this.visualizers.neural = new NeuralNetworkVisualizer(neuralContainer);
            this.visualizers.neural.animate();
        }
        
        const topologyContainer = document.getElementById('system-topology-viz');
        if (topologyContainer) {
            this.visualizers.topology = new SystemTopologyVisualizer(topologyContainer);
            this.visualizers.topology.animate();
        }
    }
    
    setupEventListeners() {
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + K for command palette
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                this.openCommandPalette();
            }
            
            // Ctrl/Cmd + / for help
            if ((e.ctrlKey || e.metaKey) && e.key === '/') {
                e.preventDefault();
                this.showHelp();
            }
        });
        
        // Window resize
        window.addEventListener('resize', () => {
            this.handleResize();
        });
    }
    
    startRealtimeUpdates() {
        // Update charts every second
        setInterval(() => {
            this.updateCharts();
        }, 1000);
        
        // Update 3D visualizations
        if (this.visualizers.neural) {
            this.visualizers.neural.startAnimation();
        }
        
        if (this.visualizers.topology) {
            this.visualizers.topology.startAnimation();
        }
    }
    
    updateMetrics(metrics) {
        this.state.metrics = { ...this.state.metrics, ...metrics };
        
        // Update metric displays
        Object.keys(metrics).forEach(key => {
            const element = document.querySelector(`[data-metric="${key}"]`);
            if (element) {
                element.textContent = this.formatMetricValue(metrics[key]);
                
                // Add animation class
                element.classList.add('metric-updated');
                setTimeout(() => element.classList.remove('metric-updated'), 300);
            }
        });
        
        // Update charts
        this.updateCharts();
    }
    
    updateCharts() {
        // Update all active charts with new data
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.update === 'function') {
                chart.update();
            }
        });
    }
    
    updateAgentStatus(agent) {
        this.state.agents[agent.id] = agent;
        
        // Update agent visualization
        if (this.visualizers.topology) {
            this.visualizers.topology.updateAgent(agent);
        }
        
        // Update UI
        const agentElement = document.querySelector(`[data-agent-id="${agent.id}"]`);
        if (agentElement) {
            this.updateAgentUI(agentElement, agent);
        }
    }
    
    updateSystemStatus(status) {
        // Update status indicators
        document.querySelectorAll('[data-status]').forEach(element => {
            const statusType = element.dataset.status;
            const value = this.getNestedValue(status, statusType);
            if (value !== undefined) {
                element.textContent = value;
            }
        });
    }
    
    showNotification(message, level = 'info') {
        const notification = {
            id: Date.now(),
            message,
            level,
            timestamp: new Date()
        };
        
        this.state.notifications.push(notification);
        
        // Create notification element
        const notificationEl = this.createNotificationElement(notification);
        const container = document.getElementById('notification-container');
        if (container) {
            container.appendChild(notificationEl);
            
            // Auto-remove after 5 seconds
            setTimeout(() => {
                notificationEl.style.opacity = '0';
                setTimeout(() => notificationEl.remove(), 300);
            }, 5000);
        }
    }
    
    createNotificationElement(notification) {
        const div = document.createElement('div');
        div.className = `notification notification-${notification.level}`;
        div.innerHTML = `
            <div class="notification-content">
                <span class="notification-message">${notification.message}</span>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        return div;
    }
    
    openCommandPalette() {
        // Create and show command palette
        const palette = new CommandPalette(this);
        palette.show();
    }
    
    showHelp() {
        // Show help modal
        const helpModal = new HelpModal();
        helpModal.show();
    }
    
    handleResize() {
        // Update visualizations on resize
        Object.values(this.visualizers).forEach(viz => {
            if (viz && typeof viz.resize === 'function') {
                viz.resize();
            }
        });
        
        // Update charts
        Object.values(this.charts).forEach(chart => {
            if (chart && typeof chart.resize === 'function') {
                chart.resize();
            }
        });
    }
    
    updateTheme(settings) {
        this.state.theme = settings.mode;
        document.body.className = `theme-${settings.mode}`;
        
        // Update CSS variables
        if (settings.accent_color) {
            document.documentElement.style.setProperty('--neural-primary', settings.accent_color);
        }
    }
    
    formatMetricValue(value) {
        if (typeof value === 'number') {
            if (value > 1000000) {
                return (value / 1000000).toFixed(1) + 'M';
            } else if (value > 1000) {
                return (value / 1000).toFixed(1) + 'K';
            }
            return value.toFixed(1);
        }
        return value;
    }
    
    getNestedValue(obj, path) {
        return path.split('.').reduce((acc, part) => acc && acc[part], obj);
    }
    
    updateAgentUI(element, agent) {
        // Update agent status
        const statusEl = element.querySelector('.agent-status');
        if (statusEl) {
            statusEl.className = `agent-status status-${agent.status}`;
            statusEl.textContent = agent.status;
        }
        
        // Update metrics
        element.querySelectorAll('[data-agent-metric]').forEach(metricEl => {
            const metric = metricEl.dataset.agentMetric;
            if (agent.performance && agent.performance[metric]) {
                metricEl.textContent = agent.performance[metric];
            }
        });
    }
}

// React Components
const Header = ({ onViewChange, currentView, theme, onThemeToggle }) => {
    return React.createElement('header', { className: 'neural-header' },
        React.createElement('div', { className: 'header-content' },
            // Logo
            React.createElement('div', { className: 'logo-section' },
                React.createElement('div', { className: 'neural-logo' }),
                React.createElement('h1', { className: 'logo-text' }, 'Neural Command Center')
            ),
            
            // Navigation
            React.createElement('nav', { className: 'neural-nav' },
                ['overview', 'agents', 'neural', 'chat', 'voice', 'settings'].map(view =>
                    React.createElement('a', {
                        key: view,
                        className: `nav-item ${currentView === view ? 'active' : ''}`,
                        onClick: () => onViewChange(view)
                    }, view.charAt(0).toUpperCase() + view.slice(1))
                )
            ),
            
            // Actions
            React.createElement('div', { className: 'header-actions' },
                React.createElement('button', {
                    className: 'theme-toggle',
                    onClick: onThemeToggle
                }, React.createElement('i', { 
                    className: `fas fa-${theme === 'dark' ? 'sun' : 'moon'}` 
                }))
            )
        )
    );
};

const Sidebar = ({ onViewChange, currentView }) => {
    const menuItems = [
        { id: 'overview', icon: 'fa-th-large', label: 'Overview' },
        { id: 'agents', icon: 'fa-robot', label: 'AI Agents' },
        { id: 'neural', icon: 'fa-brain', label: 'Neural Network' },
        { id: 'chat', icon: 'fa-comments', label: 'AI Chat' },
        { id: 'voice', icon: 'fa-microphone', label: 'Voice Control' },
        { id: 'settings', icon: 'fa-cog', label: 'Settings' }
    ];
    
    return React.createElement('aside', { className: 'neural-sidebar' },
        React.createElement('ul', { className: 'sidebar-menu' },
            menuItems.map(item =>
                React.createElement('li', {
                    key: item.id,
                    className: `sidebar-item ${currentView === item.id ? 'active' : ''}`,
                    onClick: () => onViewChange(item.id)
                },
                    React.createElement('i', { className: `fas ${item.icon}` }),
                    React.createElement('span', null, item.label)
                )
            )
        )
    );
};

const OverviewDashboard = () => {
    return React.createElement('div', { className: 'overview-dashboard' },
        // Metrics Grid
        React.createElement('div', { className: 'neural-grid grid-4' },
            React.createElement(MetricCard, {
                title: 'CPU Usage',
                value: '45.2%',
                change: '+2.3%',
                icon: 'fa-microchip'
            }),
            React.createElement(MetricCard, {
                title: 'Memory',
                value: '8.5 GB',
                change: '-0.5 GB',
                icon: 'fa-memory'
            }),
            React.createElement(MetricCard, {
                title: 'AI Operations',
                value: '125K/s',
                change: '+15%',
                icon: 'fa-brain'
            }),
            React.createElement(MetricCard, {
                title: 'Active Agents',
                value: '3',
                change: '0',
                icon: 'fa-robot'
            })
        ),
        
        // Charts Section
        React.createElement('div', { className: 'neural-grid grid-2' },
            React.createElement('div', { className: 'neural-card' },
                React.createElement('div', { className: 'card-header' },
                    React.createElement('h3', { className: 'card-title' }, 'System Performance'),
                    React.createElement('i', { className: 'fas fa-chart-line card-icon' })
                ),
                React.createElement('div', { 
                    id: 'performance-chart',
                    className: 'chart-container',
                    style: { height: '300px' }
                })
            ),
            React.createElement('div', { className: 'neural-card' },
                React.createElement('div', { className: 'card-header' },
                    React.createElement('h3', { className: 'card-title' }, 'Neural Activity'),
                    React.createElement('i', { className: 'fas fa-brain card-icon' })
                ),
                React.createElement('div', { 
                    id: 'neural-activity-chart',
                    className: 'chart-container',
                    style: { height: '300px' }
                })
            )
        ),
        
        // 3D Visualization
        React.createElement('div', { className: 'neural-card' },
            React.createElement('div', { className: 'card-header' },
                React.createElement('h3', { className: 'card-title' }, 'System Topology'),
                React.createElement('div', { className: 'viz-controls' },
                    React.createElement('button', { className: 'viz-control-btn' },
                        React.createElement('i', { className: 'fas fa-expand' })
                    ),
                    React.createElement('button', { className: 'viz-control-btn' },
                        React.createElement('i', { className: 'fas fa-redo' })
                    )
                )
            ),
            React.createElement('div', { 
                id: 'system-topology-viz',
                className: 'visualization-3d'
            })
        )
    );
};

const MetricCard = ({ title, value, change, icon }) => {
    const isPositive = change && change.startsWith('+');
    
    return React.createElement('div', { className: 'neural-card metric-card' },
        React.createElement('div', { className: 'card-header' },
            React.createElement('span', { className: 'metric-label' }, title),
            React.createElement('i', { className: `fas ${icon} card-icon` })
        ),
        React.createElement('div', { className: 'metric-value' }, value),
        change && React.createElement('div', { 
            className: `metric-change ${isPositive ? 'positive' : 'negative'}` 
        },
            React.createElement('i', { 
                className: `fas fa-arrow-${isPositive ? 'up' : 'down'}` 
            }),
            ' ', change
        )
    );
};

const FloatingActions = () => {
    return React.createElement('div', { className: 'fab-container' },
        React.createElement('button', { className: 'fab fab-primary' },
            React.createElement('i', { className: 'fas fa-plus' })
        ),
        React.createElement('button', { className: 'fab fab-secondary' },
            React.createElement('i', { className: 'fas fa-terminal' })
        )
    );
};

const NotificationStack = ({ notifications }) => {
    return React.createElement('div', { 
        id: 'notification-container',
        className: 'notification-stack' 
    });
};

// Additional React Components

const AgentsView = () => {
    const [agents, setAgents] = React.useState({});
    
    React.useEffect(() => {
        fetchAgents();
    }, []);
    
    const fetchAgents = async () => {
        try {
            const response = await fetch('/api/agents/status');
            const data = await response.json();
            setAgents(data);
        } catch (error) {
            console.error('Failed to fetch agents:', error);
        }
    };
    
    return React.createElement('div', { className: 'agents-view' },
        React.createElement('h2', { className: 'view-title' }, 'AI Agents'),
        React.createElement('div', { className: 'neural-grid grid-3' },
            Object.values(agents).map(agent =>
                React.createElement('div', { 
                    key: agent.id,
                    className: 'neural-card agent-card',
                    'data-agent-id': agent.id
                },
                    React.createElement('div', { className: 'card-header' },
                        React.createElement('h3', { className: 'card-title' }, agent.name),
                        React.createElement('span', { 
                            className: `status-indicator status-${agent.status}` 
                        },
                            React.createElement('span', { className: 'status-dot' }),
                            agent.status
                        )
                    ),
                    React.createElement('div', { className: 'agent-metrics' },
                        React.createElement('div', { className: 'metric' },
                            React.createElement('span', { className: 'metric-label' }, 'Speed'),
                            React.createElement('div', { className: 'progress-bar' },
                                React.createElement('div', { 
                                    className: 'progress-fill',
                                    style: { width: `${agent.performance?.speed || 0}%` }
                                })
                            )
                        ),
                        React.createElement('div', { className: 'metric' },
                            React.createElement('span', { className: 'metric-label' }, 'Accuracy'),
                            React.createElement('div', { className: 'progress-bar' },
                                React.createElement('div', { 
                                    className: 'progress-fill',
                                    style: { width: `${agent.performance?.accuracy || 0}%` }
                                })
                            )
                        )
                    ),
                    React.createElement('div', { className: 'agent-task' },
                        React.createElement('p', null, agent.current_task)
                    )
                )
            )
        )
    );
};

const NeuralNetworkView = () => {
    React.useEffect(() => {
        // Initialize 3D visualization
        const container = document.getElementById('neural-network-container');
        if (container && !container.hasChildNodes()) {
            const visualizer = new NeuralNetworkVisualizer(container);
            visualizer.animate();
        }
    }, []);
    
    return React.createElement('div', { className: 'neural-network-view' },
        React.createElement('h2', { className: 'view-title' }, 'Neural Network Visualization'),
        React.createElement('div', { className: 'neural-card full-height' },
            React.createElement('div', { 
                id: 'neural-network-container',
                className: 'visualization-3d',
                style: { height: 'calc(100vh - 200px)' }
            })
        )
    );
};

const ChatInterfaceWrapper = () => {
    const chatRef = React.useRef(null);
    
    React.useEffect(() => {
        if (!chatRef.current) {
            chatRef.current = new ChatInterface();
            chatRef.current.initialize();
        }
    }, []);
    
    if (chatRef.current) {
        return chatRef.current.render();
    }
    
    return React.createElement('div', { className: 'loading' }, 'Loading chat...');
};

const VoiceInterface = () => {
    const [isListening, setIsListening] = React.useState(false);
    const [transcript, setTranscript] = React.useState('');
    
    return React.createElement('div', { className: 'voice-interface' },
        React.createElement('h2', { className: 'view-title' }, 'Voice Control'),
        React.createElement('div', { className: 'neural-card' },
            React.createElement('div', { className: 'voice-visualizer' },
                React.createElement('div', { className: 'voice-bars' },
                    Array.from({ length: 20 }).map((_, i) =>
                        React.createElement('div', { 
                            key: i,
                            className: 'voice-bar',
                            style: { 
                                animationDelay: `${i * 0.05}s`,
                                height: isListening ? '100%' : '20%'
                            }
                        })
                    )
                )
            ),
            React.createElement('button', {
                className: `voice-control-btn ${isListening ? 'listening' : ''}`,
                onClick: () => setIsListening(!isListening)
            },
                React.createElement('i', { 
                    className: `fas fa-microphone${isListening ? '-slash' : ''}` 
                }),
                isListening ? ' Stop Listening' : ' Start Listening'
            ),
            transcript && React.createElement('div', { className: 'transcript' },
                React.createElement('p', null, transcript)
            )
        )
    );
};

const SettingsView = () => {
    const [settings, setSettings] = React.useState({
        theme: 'dark',
        notifications: true,
        autoUpdate: true,
        animationSpeed: 1
    });
    
    const updateSetting = (key, value) => {
        setSettings(prev => ({ ...prev, [key]: value }));
        // Save to backend
        fetch('/api/theme/update', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ [key]: value })
        });
    };
    
    return React.createElement('div', { className: 'settings-view' },
        React.createElement('h2', { className: 'view-title' }, 'Settings'),
        React.createElement('div', { className: 'neural-card' },
            React.createElement('div', { className: 'settings-section' },
                React.createElement('h3', null, 'Appearance'),
                React.createElement('div', { className: 'setting-item' },
                    React.createElement('label', null, 'Theme'),
                    React.createElement('select', {
                        value: settings.theme,
                        onChange: (e) => updateSetting('theme', e.target.value)
                    },
                        React.createElement('option', { value: 'dark' }, 'Dark'),
                        React.createElement('option', { value: 'light' }, 'Light'),
                        React.createElement('option', { value: 'auto' }, 'Auto')
                    )
                )
            ),
            React.createElement('div', { className: 'settings-section' },
                React.createElement('h3', null, 'Notifications'),
                React.createElement('div', { className: 'setting-item' },
                    React.createElement('label', null, 
                        React.createElement('input', {
                            type: 'checkbox',
                            checked: settings.notifications,
                            onChange: (e) => updateSetting('notifications', e.target.checked)
                        }),
                        ' Enable notifications'
                    )
                )
            )
        )
    );
};

// Initialize dashboard when DOM is ready
window.ModernDashboard = ModernDashboard;