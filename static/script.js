let sessionId = null;

// Elements
const uploadArea = document.getElementById("upload-area");
const pdfInput = document.getElementById("pdf-upload");
const uploadStatus = document.getElementById("upload-status");
const sendBtn = document.getElementById("send-query");
const userInput = document.getElementById("user-query");
const chatHistory = document.getElementById("chat-history");

// 1️⃣ Create session on page load
async function createSession() {
    const res = await fetch("/session", { method: "POST" });
    const data = await res.json();
    sessionId = data.session_id;
    console.log("Session created:", sessionId);
}

createSession();

// 2️⃣ Upload PDF
uploadArea.addEventListener("click", () => pdfInput.click());

pdfInput.addEventListener("change", async () => {
    if (!pdfInput.files.length) return;

    const formData = new FormData();
    formData.append("file", pdfInput.files[0]);
    formData.append("session_id", sessionId);

    uploadStatus.innerText = "Uploading...";

    const res = await fetch("/upload_report", {
        method: "POST",
        body: formData
    });

    const data = await res.json();
    uploadStatus.innerText = data.message || "Upload complete";
});

// 3️⃣ Send query
sendBtn.addEventListener("click", async () => {
    const query = userInput.value.trim();
    if (!query) return;

    chatHistory.innerHTML += `
        <div class="message user-message">
            <div class="message-content">${query}</div>
        </div>
    `;

    userInput.value = "";

    const formData = new FormData();
    formData.append("session_id", sessionId);
    formData.append("user_query", query);

    const res = await fetch("/query_agent", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    chatHistory.innerHTML += `
        <div class="message agent-message">
            <div class="message-content">${data.answer}</div>
        </div>
    `;

    chatHistory.scrollTop = chatHistory.scrollHeight;
});
