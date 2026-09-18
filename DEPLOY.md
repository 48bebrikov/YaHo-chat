# Deploying the bot (Docker) and pushing to GitLab

This file covers running the project locally with Docker and pushing your code to a remote GitLab repository.

## 1. Push the code to GitLab (CI/CD)

To push the current project to your GitLab repository, run the following in the project folder:

1. **Initialize a local Git repository:** (already done)
  ```bash
   git init
  ```
2. **Stage all required files:**
  ```bash
   git add .
  ```
3. **Create a commit:**
  ```bash
   git commit -m "Your commit message"
  ```
4. **Push the code to the** `main` **branch:**
  ```bash
   git push origin main
  ```

After `push`, your code will be on GitLab. Because `.gitlab-ci.yml` is included, GitLab will automatically start a **CI/CD pipeline**.

## 2. Deploying on your server (production)

You do not store `.env` or session files in GitLab (which is correct for security), so you need to create them once by hand on the server.

This assumes **Docker**, **Docker Compose**, and **Git** are already installed on the server.

### Step 1. Clone the repository on the server

Connect to your server over SSH.

If the repository is private and the GitLab account was created via GitHub, you will need a **Personal Access Token (PAT)** instead of a regular password:

1. Open GitLab settings (from your computer): `your avatar in the top-right corner` → `Preferences` → `Access Tokens` (left menu).
2. Click `Add new token`. Name it (for example, `vps-deploy`), check `read_repository` (or `write_repository` if the server will push anything back). Click `Create personal access token`.
3. Copy the token immediately (it starts with `glpat-...`). You will not be able to see it again.

Then clone the repository:

```bash
git clone https://github.com/48bebrikov/YaHo-chat.git
cd yoho-chat
```

When Git asks for `Username`, enter your GitLab username.  
When it asks for `Password`, **paste your copied token** (`glpat-...`) instead of the GitHub password.

### Step 2. Create the .env file

Create a `.env` file directly on the server:

```bash
nano .env
```

Paste your keys (right-click in PuTTY or use Shift+Insert to paste):

```env
API_ID=your_api_id
API_HASH=your_api_hash
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL_ID=moonshotai/kimi-k2.8
BOT_LOCALE=ru
GEMINI_API_KEY=your_gemini_key
TTS_VOICE=Achernar
QDRANT_HOST=qdrant
QDRANT_PORT=6333
POLL_INTERVAL_SECONDS=3600
MONITORED_CHANNELS=telegram,durov
FRIENDS_LIST=username1,username2
```

Save the file (`Ctrl+O`, `Enter`, `Ctrl+X`).

To switch the chat LLM, change `OPENROUTER_MODEL_ID` to any id from [OpenRouter](https://openrouter.ai/models). Chat, tools, memory, and proactive messages all use that value; the fallback default is in `config.py`. Set `BOT_LOCALE` to `en` or `ru` (prompts and fallbacks are in `i18n/`). TTS voice is `TTS_VOICE`; the TTS model is set in `ai/tts.py`. Restart after edits: `docker compose up -d`.

### Step 3. Create the session file

To keep the Docker container from failing, create an empty Telethon session file in advance:

```bash
touch userbot_session.session
```



### Step 4. First run and Telegram authorization

On the first run you will need to enter your phone number and the Telegram code. Start the containers **in interactive mode**:

```bash
docker compose up --build
```

Wait for the images to download and the libraries to install. When the console shows:
`Please enter your phone (or bot token):`

1. Enter your phone number (for example, `+79991234567`) and press Enter.
2. Enter the confirmation code sent to your Telegram account.
3. If you have a cloud password (2FA), it will ask for that as well.

Once you see `Userbot started successfully.`, the login completed successfully.

### Step 5. Run the bot in the background

After `userbot_session.session` is filled with authorization data, stop the current process (`Ctrl+C` in the terminal).

Start the containers in the background (daemon mode):

```bash
docker compose up -d
```



### Updating the code on the server later

When you change the code on your computer and push it to GitLab (`git push`), the server will update **automatically**.

A `deploy` stage is included in `.gitlab-ci.yml`. To make it work, configure CI/CD variables in GitLab once:

1. Open your GitLab repository in the browser.
2. In the left menu, choose **Settings** → **CI/CD**.
3. Find **Variables** and click **Expand**.
4. Click **Add variable** and add these 3 variables one by one:
  - **Key:** `VPS_IP`  
   **Value:** `123.45.67.89` (replace with your server’s real IP address)
  - **Key:** `SSH_USER`  
  **Value:** `root` (or your server username if you do not log in as root)
  - **Key:** `SSH_PRIVATE_KEY`  
  **Value:** paste the **private** SSH key for your server (usually from `~/.ssh/id_rsa`).
  *Important: the key must keep its line breaks. Copy it in full, from* `-----BEGIN OPENSSH PRIVATE KEY-----` *through* `-----END OPENSSH PRIVATE KEY-----`*.*

After linting, build, and tests succeed (green check), the GitLab Runner (from the cloud) will SSH into your server, pull updates (`git pull`), and restart Docker (`docker compose up -d --build`).

*If you do not need auto-deploy, you can delete the* `deploy_to_vps` *block from* `.gitlab-ci.yml` *and update the server manually:* `git pull origin main` *and* `docker compose up -d --build`*.*