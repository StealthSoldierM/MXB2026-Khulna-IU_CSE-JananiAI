const THE_URL = "http://127.0.0.1:8004/";

const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");
const loading = document.getElementById("loading");


const addMessage = (content, isUser = false, sources = null) => {
  const msgDiv = document.createElement("div");
  msgDiv.className = `${isUser ? 'user-message' : 'bot-message'}`;
  msgDiv.textContent = content;

  if (sources && sources.length > 0) {
    const srcDiv = document.createElement("div");
    srcDiv.className = "sources";
    srcDiv.innerHTML = "<strong> Source: </strong> <br> " + sources.map((s, i) => `${i+1}. ${s.content.substring(0, 100)}...`).join("<br>");
    msgDiv.appendChild(srcDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
  }
}

async function sendMessage() {
  const msg = userInput.value.trim();
  console.log(msg);
  if (!msg) return;

  userInput.disabled = true;
  sendBtn.disabled = true;
  loading.classList.remove("hidden");

  addMessage(msg, true);
  userInput.value = "";

  try {
    const resp = await fetch(`${THE_URL}/chat`, {
      method: POST,
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(`{"message": {"question": ${msg} }, "match_count": 2`)
    });

    if (!resp.ok) {
      throw new Error(`HTTP error! status: ${resp.status}`);
    }

    const data = await resp.json();
    addMessage(data.response, false, data.sources);
  }
  catch (error) {
    addMessage(`Error: ${error.message}. Please check if backend is running.`, false);
  }
  finally {
    userInput.disabled = false;
    sendBtn.disabled = false;
    loading.classList.add("hidden");
    userInput.focus();
  }
}

sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", (e) => {
  if (e.key == "Enter") sendMessage();
});

addMessage("Hello! Ask me any Maternal Questions.");
