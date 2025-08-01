// AI Chat Interface Component
// Modern chat UI with streaming responses, markdown rendering, and code highlighting

class ChatInterface {
    constructor() {
        this.messages = [];
        this.isStreaming = false;
        this.currentStreamMessage = null;
        this.ws = null;
        this.messageIdCounter = 0;
    }
    
    render() {
        return React.createElement('div', { className: 'chat-interface' },
            React.createElement('div', { className: 'chat-header' },
                React.createElement('h2', { className: 'chat-title' }, 
                    React.createElement('i', { className: 'fas fa-brain' }),
                    ' Neural Assistant'
                ),
                React.createElement('div', { className: 'chat-status' },
                    React.createElement('span', { 
                        className: `status-indicator ${this.ws ? 'status-active' : 'status-inactive'}` 
                    },
                        React.createElement('span', { className: 'status-dot' }),
                        this.ws ? 'Connected' : 'Disconnected'
                    )
                )
            ),
            
            React.createElement(ChatMessages, { 
                messages: this.messages,
                isStreaming: this.isStreaming 
            }),
            
            React.createElement(ChatInput, {
                onSend: this.sendMessage.bind(this),
                disabled: this.isStreaming
            })
        );
    }
    
    async initialize() {
        // Connect to chat WebSocket
        this.connectWebSocket();
        
        // Load chat history if available
        await this.loadChatHistory();
        
        // Send welcome message
        this.addMessage({
            type: 'ai',
            content: "Hello! I'm your Neural Assistant. I can help you with system monitoring, AI agent management, and answering questions about your AI System. How can I assist you today?",
            timestamp: new Date()
        });
    }
    
    connectWebSocket() {
        const wsUrl = `ws://${window.location.host}/ws/chat`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('Chat WebSocket connected');
            this.updateStatus('connected');
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleWebSocketMessage(data);
        };
        
        this.ws.onerror = (error) => {
            console.error('Chat WebSocket error:', error);
            this.updateStatus('error');
        };
        
        this.ws.onclose = () => {
            console.log('Chat WebSocket disconnected');
            this.updateStatus('disconnected');
            
            // Attempt to reconnect after 3 seconds
            setTimeout(() => this.connectWebSocket(), 3000);
        };
    }
    
    handleWebSocketMessage(data) {
        switch (data.type) {
            case 'stream':
                this.handleStreamMessage(data);
                break;
            case 'complete':
                this.handleCompleteMessage(data);
                break;
            case 'error':
                this.handleErrorMessage(data);
                break;
            case 'suggestion':
                this.handleSuggestion(data);
                break;
        }
    }
    
    handleStreamMessage(data) {
        if (!this.currentStreamMessage) {
            this.currentStreamMessage = {
                id: `msg-${++this.messageIdCounter}`,
                type: 'ai',
                content: '',
                timestamp: new Date(),
                isStreaming: true
            };
            this.messages.push(this.currentStreamMessage);
        }
        
        this.currentStreamMessage.content += data.content;
        this.updateUI();
        
        if (data.finished) {
            this.currentStreamMessage.isStreaming = false;
            this.isStreaming = false;
            this.currentStreamMessage = null;
        }
    }
    
    async sendMessage(content) {
        if (!content.trim() || this.isStreaming) return;
        
        // Add user message
        this.addMessage({
            type: 'user',
            content: content,
            timestamp: new Date()
        });
        
        this.isStreaming = true;
        
        // Send via WebSocket if connected
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                type: 'message',
                content: content
            }));
        } else {
            // Fallback to HTTP API
            try {
                const response = await fetch('/api/ai/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: content })
                });
                
                const data = await response.json();
                this.addMessage({
                    type: 'ai',
                    content: data.message,
                    timestamp: new Date()
                });
            } catch (error) {
                this.addMessage({
                    type: 'error',
                    content: 'Failed to send message. Please try again.',
                    timestamp: new Date()
                });
            }
            
            this.isStreaming = false;
        }
    }
    
    addMessage(message) {
        message.id = `msg-${++this.messageIdCounter}`;
        this.messages.push(message);
        this.updateUI();
        
        // Save to local storage
        this.saveChatHistory();
    }
    
    async loadChatHistory() {
        try {
            const saved = localStorage.getItem('ai-chat-history');
            if (saved) {
                const history = JSON.parse(saved);
                this.messages = history.messages || [];
                this.messageIdCounter = history.lastId || 0;
            }
        } catch (error) {
            console.error('Failed to load chat history:', error);
        }
    }
    
    saveChatHistory() {
        try {
            const history = {
                messages: this.messages.slice(-50), // Keep last 50 messages
                lastId: this.messageIdCounter
            };
            localStorage.setItem('ai-chat-history', JSON.stringify(history));
        } catch (error) {
            console.error('Failed to save chat history:', error);
        }
    }
    
    updateStatus(status) {
        // Update connection status in UI
        const statusEl = document.querySelector('.chat-status .status-indicator');
        if (statusEl) {
            statusEl.className = `status-indicator status-${status}`;
        }
    }
    
    updateUI() {
        // Trigger React re-render
        if (window.AISystemDashboard) {
            window.AISystemDashboard.forceUpdate();
        }
    }
}

// Chat Messages Component
const ChatMessages = ({ messages, isStreaming }) => {
    const messagesEndRef = React.useRef(null);
    
    React.useEffect(() => {
        // Scroll to bottom when new messages arrive
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);
    
    return React.createElement('div', { className: 'chat-messages' },
        messages.map(message => 
            React.createElement(ChatMessage, { 
                key: message.id, 
                message: message 
            })
        ),
        React.createElement('div', { ref: messagesEndRef })
    );
};

// Individual Chat Message Component
const ChatMessage = ({ message }) => {
    const [isExpanded, setIsExpanded] = React.useState(false);
    
    const renderContent = () => {
        // Parse markdown and render formatted content
        const content = message.content;
        
        // Simple markdown parsing (in production, use a proper markdown library)
        const formatted = content
            .replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
                return `<pre class="code-block" data-lang="${lang || 'text'}"><code>${escapeHtml(code.trim())}</code></pre>`;
            })
            .replace(/`([^`]+)`/g, '<code class="inline-code">$1</code>')
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
            
        return { __html: formatted };
    };
    
    return React.createElement('div', { 
        className: `chat-message message-${message.type}` 
    },
        React.createElement('div', { className: 'message-avatar' },
            React.createElement('i', { 
                className: `fas fa-${message.type === 'user' ? 'user' : 'robot'}` 
            })
        ),
        React.createElement('div', { className: 'message-content' },
            React.createElement('div', { 
                className: 'message-header' 
            },
                React.createElement('span', { className: 'message-sender' },
                    message.type === 'user' ? 'You' : 'Neural Assistant'
                ),
                React.createElement('span', { className: 'message-time' },
                    formatTime(message.timestamp)
                )
            ),
            React.createElement('div', { 
                className: 'message-text',
                dangerouslySetInnerHTML: renderContent()
            }),
            message.isStreaming && React.createElement('div', { 
                className: 'streaming-indicator' 
            },
                React.createElement('span', { className: 'typing-dot' }),
                React.createElement('span', { className: 'typing-dot' }),
                React.createElement('span', { className: 'typing-dot' })
            ),
            message.suggestions && React.createElement('div', { 
                className: 'message-suggestions' 
            },
                message.suggestions.map((suggestion, index) =>
                    React.createElement('button', {
                        key: index,
                        className: 'suggestion-chip',
                        onClick: () => window.AISystemDashboard.chatInterface.sendMessage(suggestion)
                    }, suggestion)
                )
            )
        ),
        message.type === 'ai' && React.createElement('div', { 
            className: 'message-actions' 
        },
            React.createElement('button', { 
                className: 'action-btn',
                title: 'Copy'
            },
                React.createElement('i', { className: 'fas fa-copy' })
            ),
            React.createElement('button', { 
                className: 'action-btn',
                title: 'Regenerate'
            },
                React.createElement('i', { className: 'fas fa-redo' })
            ),
            React.createElement('button', { 
                className: 'action-btn',
                title: message.isExpanded ? 'Collapse' : 'Expand',
                onClick: () => setIsExpanded(!isExpanded)
            },
                React.createElement('i', { 
                    className: `fas fa-chevron-${isExpanded ? 'up' : 'down'}` 
                })
            )
        )
    );
};

// Chat Input Component
const ChatInput = ({ onSend, disabled }) => {
    const [input, setInput] = React.useState('');
    const [isRecording, setIsRecording] = React.useState(false);
    const textareaRef = React.useRef(null);
    
    const handleSubmit = (e) => {
        e.preventDefault();
        if (input.trim() && !disabled) {
            onSend(input);
            setInput('');
            textareaRef.current.style.height = 'auto';
        }
    };
    
    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    };
    
    const handleInput = (e) => {
        setInput(e.target.value);
        
        // Auto-resize textarea
        e.target.style.height = 'auto';
        e.target.style.height = e.target.scrollHeight + 'px';
    };
    
    const toggleVoiceInput = () => {
        setIsRecording(!isRecording);
        
        if (!isRecording) {
            // Start voice recording
            startVoiceRecording();
        } else {
            // Stop voice recording
            stopVoiceRecording();
        }
    };
    
    return React.createElement('form', { 
        className: 'chat-input-container',
        onSubmit: handleSubmit
    },
        React.createElement('div', { className: 'input-wrapper' },
            React.createElement('textarea', {
                ref: textareaRef,
                className: 'chat-input',
                placeholder: 'Ask me anything about your AI system...',
                value: input,
                onChange: handleInput,
                onKeyDown: handleKeyDown,
                disabled: disabled,
                rows: 1,
                maxLength: 2000
            }),
            React.createElement('div', { className: 'input-actions' },
                React.createElement('button', {
                    type: 'button',
                    className: 'action-btn',
                    onClick: toggleVoiceInput,
                    title: 'Voice input'
                },
                    React.createElement('i', { 
                        className: `fas fa-microphone ${isRecording ? 'recording' : ''}` 
                    })
                ),
                React.createElement('button', {
                    type: 'button',
                    className: 'action-btn',
                    title: 'Attach file'
                },
                    React.createElement('i', { className: 'fas fa-paperclip' })
                )
            )
        ),
        React.createElement('button', {
            type: 'submit',
            className: 'chat-send-btn',
            disabled: disabled || !input.trim()
        },
            disabled ? 
                React.createElement('i', { className: 'fas fa-spinner fa-spin' }) :
                React.createElement('i', { className: 'fas fa-paper-plane' })
        )
    );
};

// Helper functions
function formatTime(timestamp) {
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;
    
    if (diff < 60000) {
        return 'Just now';
    } else if (diff < 3600000) {
        return `${Math.floor(diff / 60000)}m ago`;
    } else if (diff < 86400000) {
        return `${Math.floor(diff / 3600000)}h ago`;
    } else {
        return date.toLocaleDateString();
    }
}

function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#039;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// Voice recording functions
async function startVoiceRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        // Implementation for voice recording
        console.log('Voice recording started');
    } catch (error) {
        console.error('Failed to start voice recording:', error);
    }
}

function stopVoiceRecording() {
    // Implementation for stopping voice recording
    console.log('Voice recording stopped');
}

// Export for use in main app
window.ChatInterface = ChatInterface;