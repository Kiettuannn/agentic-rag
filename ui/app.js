document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');

    // Tự động điều chỉnh chiều cao của ô nhập liệu
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    // Bắt sự kiện phím Enter (Gửi tin) / Shift+Enter (Xuống dòng)
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', sendMessage);

    async function sendMessage() {
        const text = userInput.value.trim();
        if (!text) return;

        // 1. Hiển thị tin nhắn của User
        appendMessage('user', text);
        
        // Reset ô nhập
        userInput.value = '';
        userInput.style.height = 'auto';
        sendBtn.disabled = true;

        // 2. Hiển thị Loading (Bot đang suy nghĩ)
        const loadingId = appendLoading();

        try {
            // 3. Gọi API FastAPI Backend
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: text })
            });

            const data = await response.json();
            
            // Xóa Loading
            removeMessage(loadingId);

            if (data.error) {
                appendMessage('bot', `**Lỗi:** ${data.error}`);
            } else {
                // 4. Hiển thị tin nhắn của Bot (có chứa Strategy và Retry)
                appendBotMessageWithMeta(data.answer, data.strategy, data.retry_count);
            }

        } catch (error) {
            removeMessage(loadingId);
            appendMessage('bot', `**Lỗi kết nối:** Không thể gọi tới Server API. Vui lòng kiểm tra lại.`);
        } finally {
            sendBtn.disabled = false;
            userInput.focus();
        }
    }

    // Hàm tạo Bong bóng chat thông thường
    function appendMessage(sender, text) {
        const div = document.createElement('div');
        div.className = `message ${sender}-message`;
        
        const avatar = sender === 'user' ? '👤' : '🤖';
        
        // Dùng thư viện marked để parse Markdown sang HTML
        const htmlContent = sender === 'bot' ? marked.parse(text) : escapeHTML(text);

        div.innerHTML = `
            <div class="avatar">${avatar}</div>
            <div class="message-content">${htmlContent}</div>
        `;
        chatBox.appendChild(div);
        scrollToBottom();
    }

    // Hàm tạo Bong bóng chat có gắn nhãn Strategy & Retry cho BOT
    function appendBotMessageWithMeta(text, strategy, retryCount) {
        const div = document.createElement('div');
        div.className = `message bot-message`;
        
        const htmlContent = marked.parse(text);
        
        // Tạo khối Badges
        const badgesHtml = `
            <div class="meta-badges">
                <span class="badge badge-strategy">🧠 Chiến lược: ${strategy}</span>
                <span class="badge badge-retry">🔄 Lần thử: ${retryCount}</span>
            </div>
        `;

        div.innerHTML = `
            <div class="avatar">🤖</div>
            <div class="message-content">
                ${badgesHtml}
                ${htmlContent}
            </div>
        `;
        chatBox.appendChild(div);
        scrollToBottom();
    }

    // Hàm tạo Dấu ba chấm Loading
    function appendLoading() {
        const id = 'loading-' + Date.now();
        const div = document.createElement('div');
        div.className = `message bot-message`;
        div.id = id;
        div.innerHTML = `
            <div class="avatar">🤖</div>
            <div class="message-content">
                <div class="typing-indicator">
                    <span></span><span></span><span></span>
                </div>
            </div>
        `;
        chatBox.appendChild(div);
        scrollToBottom();
        return id;
    }

    // Tiện ích xóa tin nhắn (dùng để xóa Loading)
    function removeMessage(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    // Tự động cuộn xuống cuối
    function scrollToBottom() {
        chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Lọc ký tự HTML độc hại từ người dùng (Chống XSS)
    function escapeHTML(str) {
        return str.replace(/[&<>'"]/g, 
            tag => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                "'": '&#39;',
                '"': '&quot;'
            }[tag])
        );
    }
});