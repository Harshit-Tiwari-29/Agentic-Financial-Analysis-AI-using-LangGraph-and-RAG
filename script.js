// static/script.js
document.addEventListener('DOMContentLoaded', () => {
    let sessionId = null;

    const uploadArea = document.getElementById('upload-area');
    const pdfUpload = document.getElementById('pdf-upload');
    const uploadText = document.getElementById('upload-text');
    const uploadStatus = document.getElementById('upload-status');
    const chatHistory = document.getElementById('chat-history');
    const userQueryInput = document.getElementById('user-query');
    const sendQueryBtn = document.getElementById('send-query');

    // --- Initialize Session ---
    const createSession = async () => {
        try {
            const response = await fetch('/session', { method: 'POST' });
            if (!response.ok) {
                const errorText = await response.text();
                throw new Error(`Failed to create session: ${errorText}`);
            }
            const data = await response.json();
            sessionId = data.session_id;
            console.log('Session created:', sessionId);
        } catch (error) {
            console.error('Session creation error:', error);
            addMessageToChat(`Error: Could not establish a session with the server. ${error.message}`, 'agent');
        }
    };

    // --- File Upload Logic ---
    uploadArea.addEventListener('click', () => pdfUpload.click());
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('border-blue-500');
    });
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('border-blue-500');
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('border-blue-500');
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            pdfUpload.files = files;
            handleFileUpload(files[0]);
        }
    });
    pdfUpload.addEventListener('change', () => {
        if (pdfUpload.files.length > 0) {
            handleFileUpload(pdfUpload.files[0]);
        }
    });

    const handleFileUpload = async (file) => {
        if (!sessionId) {
            alert('Session not initialized. Please refresh the page.');
            return;
        }
        if (file.type !== 'application/pdf') {
            alert('Only PDF files are supported.');
            return;
        }

        uploadText.textContent = `Uploading ${file.name}...`;
        uploadStatus.textContent = 'Processing...';
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('session_id', sessionId);

        try {
            const response = await fetch('/upload_report', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorText = await response.text();
                try {
                    const errorJson = JSON.parse(errorText);
                    throw new Error(errorJson.detail || 'File upload failed.');
                } catch (e) {
                    throw new Error(`Server error: ${response.status}. Details: ${errorText}`);
                }
            }
            
            const result = await response.json();
            uploadStatus.textContent = result.message;
            addMessageToChat(`System: ${result.message}`, 'system');
        } catch (error) {
            console.error('Upload error:', error);
            uploadStatus.textContent = `Error: ${error.message}`;
        } finally {
            uploadText.textContent = 'Drag & drop another PDF, or click to select';
        }
    };

    // --- Chat Logic ---
    const sendQuery = async () => {
        const query = userQueryInput.value.trim();
        if (!query || !sessionId) return;

        addMessageToChat(query, 'user');
        userQueryInput.value = '';
        sendQueryBtn.disabled = true;
        addMessageToChat('Thinking...', 'agent', true); // Add thinking indicator

        const formData = new FormData();
        formData.append('user_query', query);
        formData.append('session_id', sessionId);
        
        try {
            const response = await fetch('/query_agent', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorText = await response.text();
                try {
                    const errorJson = JSON.parse(errorText);
                    throw new Error(errorJson.detail || 'Query failed.');
                } catch (e) {
                     throw new Error(`Server error: ${response.status}. Details: ${errorText}`);
                }
            }

            const result = await response.json();
            const formattedAnswer = formatAnswer(result.answer);
            updateLastMessage(formattedAnswer, 'agent');

        } catch (error) {
            console.error('Query error:', error);
            updateLastMessage(`Error: ${error.message}`, 'agent');
        } finally {
            sendQueryBtn.disabled = false;
        }
    };

    const addMessageToChat = (message, sender, isThinking = false) => {
        const messageWrapper = document.createElement('div');
        messageWrapper.className = `message ${sender}-message`;

        const avatar = document.createElement('div');
        avatar.className = 'avatar';
        avatar.innerHTML = sender === 'user' 
            ? `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-user"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>`
            : `<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="feather feather-cpu"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>`;

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';
        
        if (isThinking) {
            messageContent.innerHTML = `<div class="thinking"><span></span><span></span><span></span></div>`;
        } else {
             messageContent.innerHTML = `<p>${message}</p>`;
        }
        
        messageWrapper.appendChild(avatar);
        messageWrapper.appendChild(messageContent);
        chatHistory.appendChild(messageWrapper);
        chatHistory.scrollTop = chatHistory.scrollHeight;
    };
    
    const updateLastMessage = (newMessage, sender) => {
        const lastMessage = chatHistory.querySelector('.message:last-child .message-content');
        if (lastMessage) {
             lastMessage.innerHTML = `<p>${newMessage}</p>`;
        }
        chatHistory.scrollTop = chatHistory.scrollHeight;
    };
    
    const formatAnswer = (answer) => {
        if (!answer) return "I'm sorry, I couldn't generate a response.";
        
        // Sanitize and format JSON-like strings from tool outputs
        if (answer.trim().startsWith('{') && answer.trim().endsWith('}')) {
             try {
                // More robust JSON parsing
                const jsonString = answer.replace(/'/g, '"')
                                       .replace(/(\w+)"\s*:/g, '"$1":')
                                       .replace(/:\s*None/g, ': null');
                const obj = JSON.parse(jsonString);
                let formatted = '<div class="data-block">';
                for (const key in obj) {
                    const value = Array.isArray(obj[key]) ? obj[key].join(', ') : (obj[key] === null ? 'N/A' : obj[key]);
                    const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                    formatted += `<p><strong>${formattedKey}:</strong> ${value}</p>`;
                }
                formatted += '</div>';
                return formatted;
             } catch(e) {
                // Fallback for malformed JSON
                return answer.replace(/\n/g, '<br>');
             }
        }
        // Format plain text answers
        return answer.replace(/\n/g, '<br>');
    };

    sendQueryBtn.addEventListener('click', sendQuery);
    userQueryInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            sendQuery();
        }
    });

    // --- Start the app ---
    createSession();
});
