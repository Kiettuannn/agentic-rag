document.addEventListener('DOMContentLoaded', () => {
    const chatBox = document.getElementById('chat-box');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    let chatHistory = []; // Lưu trữ lịch sử chat

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

        chatHistory.push({ role: 'user', content: text }); // Lưu vào lịch sử chat

        if(chatHistory.length > 6) {
            chatHistory = chatHistory.slice(chatHistory.length - 6); // Giữ lại 6 tin nhắn gần nhất
        }

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
                body: JSON.stringify({ 
                    query: text,
                    history: chatHistory
                 })
            });

            const data = await response.json();
            
            // Xóa Loading
            removeMessage(loadingId);

            if (data.error) {
                appendMessage('bot', `**Lỗi:** ${data.error}`);
            } else {
                // 4. Hiển thị tin nhắn của Bot (có chứa Strategy và Retry)
                appendBotMessageWithMeta(
                    data.answer, data.strategy, data.retry_count, data.search_query,
                    data.strategy_reason, data.reflection_reason, data.sources || []
                );

                chatHistory.push({ role: 'bot', content: data.answer }); // Lưu vào lịch sử chat
                if(chatHistory.length > 6) {
                    chatHistory = chatHistory.slice(chatHistory.length - 6); // Giữ lại 6 tin nhắn gần nhất
                }
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
    function appendBotMessageWithMeta(text, strategy, retryCount, searchQuery, strategyReason, reflectionReason, sources=[]) {
        text = text || "";
        strategy = strategy || "";
        searchQuery = searchQuery || "";
        
        const div = document.createElement('div');
        div.className = `message bot-message`;
        
        const htmlContent = marked.parse(text);
        const sourcesHtml = buildSourcesHtml(sources)


        // Tạo khối Badges
        const badgesHtml = `
            <div class="meta-badges">
                <span class="badge badge-strategy">🧠 Chiến lược: ${strategy}</span>
                <span class="badge badge-retry">🔄 Lần thử: ${retryCount}</span>
                <span class="badge badge-strategy" style="background: rgba(59, 130, 246, 0.2); color: #93c5fd; border-color: rgba(59, 130, 246, 0.4);">
                    📝 Đã hiểu ý: "${searchQuery}"
                </span>
            </div>
        `;

        let thoughtsHtml = '';
        if (strategyReason || reflectionReason) {
            thoughtsHtml = `
                <details class="thought-process">
                    <summary class="thought-summary">🔍 Nhấn để xem quá trình phân tích...</summary>
                    <div class="thought-content">
                        ${strategyReason ? `<strong>Phân tích Chiến lược:</strong><br>${escapeHTML(strategyReason)}<br><br>` : ''}
                        ${reflectionReason ? `<strong>Trạm kiểm duyệt (Reflection):</strong><br>${escapeHTML(reflectionReason)}` : ''}
                    </div>
                </details>
            `;
        }


        div.innerHTML = `
            <div class="avatar">🤖</div>
            <div class="message-content">
                ${badgesHtml}
                ${thoughtsHtml}
                ${sourcesHtml}
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
        if (!str) return "";
        return String(str).replace(/[&<>'"]/g, 
            tag => ({
                '&': '&amp;',
                '<': '&lt;',
                '>': '&gt;',
                "'": '&#39;',
                '"': '&quot;'
            }[tag])
        );
    }

      // =====================================================================
    // [MỚI] Hàm xây dựng HTML cho section Citations / Nguồn tham khảo
    //
    // @param {Array} sources - Mảng object từ API, mỗi object có dạng:
    //   { doc_id, title, doc_type, authority, issue_date, status }
    //
    // @returns {string} - Chuỗi HTML hoàn chỉnh, hoặc '' nếu không có nguồn
    // =====================================================================
    function buildSourcesHtml(sources) {
        // Guard: nếu không có sources hoặc mảng rỗng → không render gì cả
        if (!sources || sources.length === 0) return '';

        // Hàm nội bộ: chọn emoji icon dựa theo loại văn bản
        // Mục đích: phân biệt trực quan giữa các loại văn bản pháp luật
        function getDocIcon(docType) {
            if (!docType) return '📄';
            const type = docType.toLowerCase();
            if (type.includes('luật'))         return '⚖️';
            if (type.includes('nghị định'))    return '📋';
            if (type.includes('thông tư'))     return '📝';
            if (type.includes('quyết định'))   return '📌';
            if (type.includes('nghị quyết'))   return '🏛️';
            if (type.includes('chỉ thị'))      return '📢';
            return '📄'; // fallback mặc định
        }

        // Hàm nội bộ: xác định class CSS của badge trạng thái hiệu lực
        // Màu xanh = còn hiệu lực, màu xám = hết / không rõ
        function getStatusClass(status) {
            if (!status) return 'source-badge inactive';
            const s = status.toLowerCase();
            // Kiểm tra các từ khóa "còn hiệu lực" phổ biến trong dữ liệu pháp luật VN
            if (s.includes('còn') || s.includes('hiệu lực') || s.includes('có hiệu lực')) {
                return 'source-badge active';   // → màu xanh emerald
            }
            return 'source-badge inactive';     // → màu xám muted
        }

        // Tạo HTML cho từng source card
        // Dùng escapeHTML() (đã có sẵn trong app.js) để tránh XSS
        const cardsHtml = sources.map(src => {
            const icon       = getDocIcon(src.doc_type);
            const statusClass = getStatusClass(src.status);
            const title      = escapeHTML(src.title || src.doc_id || 'Không rõ tên');
            const authority  = escapeHTML(src.authority || '');
            const issueDate  = escapeHTML(src.issue_date || '');
            const status     = escapeHTML(src.status || 'Không rõ');

            // Tạo dòng meta: "Cơ quan · Ngày" (bỏ qua nếu field rỗng)
            const metaParts = [authority, issueDate].filter(Boolean);
            const metaText  = metaParts.join(' · ');

            return `
                <div class="source-card">
                    <div class="source-icon">${icon}</div>
                    <div class="source-info">
                        <div class="source-title">${title}</div>
                        ${metaText ? `<div class="source-meta">${metaText}</div>` : ''}
                    </div>
                    <span class="${statusClass}">${status}</span>
                </div>
            `;
        }).join('');

        // Render toàn bộ section citations
        return `
            <div class="sources-section">
                <div class="sources-label">📎 Nguồn tham khảo (${sources.length} văn bản)</div>
                <div class="sources-list">
                    ${cardsHtml}
                </div>
            </div>
        `;
    }
});

