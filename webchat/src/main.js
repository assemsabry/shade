// Removed CSS import as it's now in the HTML for direct serving
const app = document.querySelector('#app')

app.innerHTML = `
  <header>
    <div class="brand">
      <img src="/logo.png" alt="Shade Logo" onerror="this.style.display='none'">
    </div>
  </header>

  <div id="chat-wrapper" class="chat-wrapper">
    <div class="empty-state">
      <h1>Welcome to Shade</h1>
      <p>AI liberated and uncensored—how can I assist your unfiltered requests today?</p>
      <div id="model-info" class="model-badge">Loading model info...</div>
    </div>
  </div>

  <div class="input-container">
    <div class="box-wrapper">
      <textarea id="user-input" placeholder="Message Shade..." rows="1"></textarea>
      <button id="send-btn" class="send-button">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
          <line x1="22" y1="2" x2="11" y2="13"></line>
          <polygon points="22 2 15 22 11 13 2 9 22 2"></polygon>
        </svg>
      </button>
    </div>
  </div>
`

const chatWrapper = document.querySelector('#chat-wrapper')
const userInput = document.querySelector('#user-input')
const sendBtn = document.querySelector('#send-btn')

// Auto-resize textarea
userInput.addEventListener('input', () => {
  userInput.style.height = 'auto'
  userInput.style.height = userInput.scrollHeight + 'px'
})

function appendMessage(role, content) {
  // Remove empty state if present
  const emptyState = chatWrapper.querySelector('.empty-state')
  if (emptyState) emptyState.remove()

  const msgDiv = document.createElement('div')
  msgDiv.className = `message ${role === 'user' ? 'user-message' : 'bot-message'}`

  if (role === 'user') {
    msgDiv.textContent = content
  } else {
    // Render Markdown for bot messages
    if (window.marked && content) {
      msgDiv.innerHTML = marked.parse(content)
    } else {
      msgDiv.innerHTML = content
    }
  }

  chatWrapper.appendChild(msgDiv)
  chatWrapper.scrollTop = chatWrapper.scrollHeight
  return msgDiv
}

async function handleChat() {
  const text = userInput.value.trim()
  if (!text) return

  userInput.value = ''
  userInput.style.height = 'auto'

  appendMessage('user', text)

  // Loading indicator
  const botMsg = appendMessage('bot', '')
  botMsg.innerHTML = `
    <div class="typing">
      <span></span>
      <span></span>
      <span></span>
    </div>
  `

  try {
    const response = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text })
    })

    if (!response.ok) throw new Error('Server error')

    // Read the stream
    const reader = response.body.getReader()
    const decoder = new TextDecoder()
    let botResponseText = ''

    // Clear typing indicator
    botMsg.innerHTML = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break

      const chunk = decoder.decode(value, { stream: true })
      botResponseText += chunk

      // Update bot message with markdown response
      if (window.marked) {
        botMsg.innerHTML = marked.parse(botResponseText)
      } else {
        botMsg.innerHTML = botResponseText
      }

      chatWrapper.scrollTop = chatWrapper.scrollHeight
    }
  } catch (err) {
    botMsg.innerHTML = `<span style="color: #ef4444">Error: Could not connect to the Shade local server.</span>`
  }

  chatWrapper.scrollTop = chatWrapper.scrollHeight
}

sendBtn.addEventListener('click', handleChat)

userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault()
    handleChat()
  }
})

// Fetch active model info on start
async function fetchModelInfo() {
  const modelBadge = document.querySelector('#model-info')
  try {
    const res = await fetch('/info')
    const data = await res.json()
    modelBadge.textContent = `Active Model: ${data.model}`
  } catch (err) {
    modelBadge.textContent = 'Model: Local Node'
  }
}

fetchModelInfo()
