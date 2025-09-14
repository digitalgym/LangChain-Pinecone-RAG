class ChatBot {
    constructor() {
        this.sessionId = 'default';
        this.messageInput = document.getElementById('message-input');
        this.sendButton = document.getElementById('send-button');
        this.clearButton = document.getElementById('clear-button');
        this.chatMessages = document.getElementById('chat-messages');
        this.loading = document.getElementById('loading');
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        this.sendButton.addEventListener('click', () => this.sendMessage());
        this.clearButton.addEventListener('click', () => this.clearChat());
        
        this.messageInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        this.messageInput.addEventListener('input', () => {
            this.sendButton.disabled = this.messageInput.value.trim() === '';
        });
    }
    
    async sendMessage() {
        const message = this.messageInput.value.trim();
        if (!message) return;
        
        // Add user message to chat
        this.addMessage(message, 'user');
        
        // Clear input and disable send button
        this.messageInput.value = '';
        this.sendButton.disabled = true;
        
        // Show loading indicator
        this.showLoading(true);
        
        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    session_id: this.sessionId
                })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Add assistant response to chat
            this.addMessage(data.response, 'assistant');
            
        } catch (error) {
            console.error('Error sending message:', error);
            this.addMessage('Sorry, I encountered an error. Please try again.', 'assistant', true);
        } finally {
            this.showLoading(false);
        }
    }
    
    addMessage(content, sender, isError = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${sender}-message`;
        
        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        if (sender === 'user') {
            messageContent.innerHTML = `<strong>You:</strong> ${this.escapeHtml(content)}`;
        } else {
            messageContent.innerHTML = `<strong>Assistant:</strong> ${this.escapeHtml(content)}`;
            if (isError) {
                messageContent.style.color = '#ef4444';
            }
        }
        
        messageDiv.appendChild(messageContent);
        this.chatMessages.appendChild(messageDiv);
        
        // Scroll to bottom
        this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
    }
    
    async clearChat() {
        try {
            await fetch(`/api/chat/${this.sessionId}`, {
                method: 'DELETE'
            });
            
            // Clear chat messages except the initial greeting
            this.chatMessages.innerHTML = `
                <div class="message assistant-message">
                    <div class="message-content">
                        <strong>Assistant:</strong> Hello! I'm your RAG-powered assistant. Ask me anything about the documents in the knowledge base.
                    </div>
                </div>
            `;
            
        } catch (error) {
            console.error('Error clearing chat:', error);
        }
    }
    
    showLoading(show) {
        if (show) {
            this.loading.classList.remove('hidden');
        } else {
            this.loading.classList.add('hidden');
        }
    }
    
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
}

// Initialize the chatbot when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new ChatBot();
});
