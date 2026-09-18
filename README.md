# YaHo-chat

YaHo-chat is a smart Telegram userbot (AI Friend) built on Telethon. The main “brain” is any LLM via OpenRouter (Moonshot Kimi by default), and Google Gemini TTS is used for voice messages. The bot talks like a real person, remembers conversation context (via RAG and the Qdrant vector database), reads and forwards posts from Telegram channels it follows, searches the web, and can even read the contents of web pages.

## Main features

- **OpenRouter (LLM) and Gemini (TTS) integration:** Uses powerful language models (Claude, DeepSeek, Kimi, and others) to generate meaningful, contextual, and natural replies.
- **Long-term memory (Memory Graph & RAG):** A background process analyzes conversations with LangGraph, extracts facts about the user, and stores them in the Qdrant vector database.
- **Tool use (ReAct agent):**
  - Web search (via the high-precision Tavily API).
  - Reading web page content (text parsing).
  - Working with PDF documents (via Playwright).
  - Python code interpreter: the bot can run Python on the fly for calculations and data processing.
  - Reminders: the bot can set timers and message the user proactively to remind them about tasks.
- **Natural conversation:**
  - Voice messages (Gemini TTS): the bot can generate realistic audio with emotions (laughter, whisper, sadness) via Google TTS.
  - Dynamic typing wait: the bot will not interrupt you — it watches the Telegram `typing...` status and waits until you finish your thought before answering.
  - Simulated typing (`typing...`) and voice recording.
  - Splitting long replies into several shorter messages.
  - “Night silence” (the bot does not reply instantly at night, simulating sleep).
- **Monitoring and metrics:** Built-in Prometheus and Grafana support for bot health and message statistics.
- **Containerization:** Fully ready to run with Docker and Docker Compose.

## Tech stack

- **Language:** Python 3.11+
- **Telegram API:** Telethon
- **Agent architecture:** LangChain and LangGraph (ReAct StateGraph)
- **LLM:** `langchain-openai` (OpenRouter API), `google-genai` (Gemini TTS)
- **Vector DB:** Qdrant (`qdrant-client`)
- **Embeddings:** `sentence-transformers` (model `deepvk/USER-bge-m3`)
- **Search and parsing:** `httpx` (Tavily API), `ddgs` (YouTube), `playwright`
- **Monitoring:** Prometheus, Grafana, Loki, Promtail

## Install and run (local Docker)

1. **Clone the repository:**
  ```bash
    git clone https://github.com/48bebrikov/YaHo-chat.git
    cd YaHo-chat
  ```
2. **Set environment variables:**
  Create a `.env` file in the project root and fill in your credentials (see the example in `DEPLOY.md`):
    *You can get* `API_ID` *and* `API_HASH` *at [my.telegram.org](https://my.telegram.org).*
    *Pick the chat LLM with* `OPENROUTER_MODEL_ID` *(see [Choosing models](#choosing-models)).*
3. **Create an empty session file:**
  ```bash
    touch userbot_session.session
  ```
4. **First run (authorization):**
  Start the containers in interactive mode so you can enter your phone number and the Telegram code:
    Follow the console prompts. After a successful login, stop the process (`Ctrl+C`).
5. **Run in the background:**
  ```bash
    docker compose up -d
  ```

## Choosing models

The chat LLM is set in `.env` (defaults live in `config.py`):

```env
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL_ID=moonshotai/kimi-k2.8
BOT_LOCALE=ru
TTS_VOICE=Achernar
```

- **Chat, tools, memory, and proactive messages** all use `OPENROUTER_MODEL_ID`. Use any model id from [OpenRouter](https://openrouter.ai/models), for example `anthropic/claude-sonnet-4.5` or `deepseek/deepseek-chat`. If the variable is omitted, the default is `moonshotai/kimi-k2.6`.
- **TTS voice:** `TTS_VOICE` in `.env` (default `Achernar`; other Gemini voices include `Aoede`, `Callirrhoe`, `Kore`, `Charon`). The TTS model itself is hardcoded in `ai/tts.py` (`gemini-3.1-flash-tts-preview`).
- **Embeddings:** change the SentenceTransformer name in `ai/embedder.py`.

After changing `.env`, restart the bot: `docker compose up -d`.

## Language and copy

Prompts, fallback replies, reminders, and TTS style live in `i18n/` so you can switch language without editing agent code.

Set `BOT_LOCALE` in `.env`:

```env
BOT_LOCALE=en
```

- `ru` (default) — Russian persona and user-facing lines
- `en` — English pack

Edit the packs themselves in `i18n/ru.py` and `i18n/en.py` (`BotCopy` fields in `i18n/schema.py`). Restart after a change.

## Deploying on a server

Detailed instructions for VPS deployment and GitLab CI/CD are in [DEPLOY.md](DEPLOY.md).