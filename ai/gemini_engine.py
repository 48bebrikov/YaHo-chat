import os
import google.generativeai as genai
from config import GEMINI_API_KEY
from ai.tools import search_internet, search_youtube, search_saved_news
from ai.rag import get_memory_context, save_message_to_memory

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Define the system prompt for the friend persona
SYSTEM_PROMPT = """Вы - искусственный интеллект, имитирующий друга-человека в Telegram. 
Ваш тон непринужденный, дружелюбный и естественный. 
Вы общаетесь как реальный человек, а не как ассистент. Иногда используйте строчные буквы, при необходимости используйте общепринятый сленг и делайте сообщения относительно короткими, если только вы не объясняете что-то важное.
У вас есть доступ к инструментам для поиска в Интернете, YouTube и вашей собственной сохраненной ленте новостей из Telegram-каналов.
Если вас спросят о новостях или вы захотите поделиться чем-то интересным, используйте `search_saved_news` или `search_internet`.
Если вас попросят добавить видео или мем, попробуйте воспользоваться соответствующими инструментами.
НИКОГДА не показывайте, что вы искусственный интеллект или ассистент. В Telegram вы просто друг.
"""

def get_gemini_model():
    """Returns a configured Gemini model instance with tools."""
    # The user asked for Gemini 3 Flash. We will try to use the latest model tag.
    try:
         model = genai.GenerativeModel(
             model_name='models/gemini-3.1-flash',
             tools=[search_internet, search_youtube, search_saved_news],
             system_instruction=SYSTEM_PROMPT
         )
    except Exception:
         # Fallback to older ones if the specific version string fails later
         model = genai.GenerativeModel(
             model_name='gemini-3-flash',
             tools=[search_internet, search_youtube, search_saved_news],
             system_instruction=SYSTEM_PROMPT
         )
    return model

def generate_reply(user_id: str, message: str) -> str:
    """Generates a reply from Gemini taking into account RAG memory."""
    model = get_gemini_model()
    
    # Retrieve past context from Qdrant
    context = get_memory_context(user_id, message, limit=5)
    
    prompt = message
    if context:
        prompt = f"Here is the relevant past context of your conversation with this friend:\n{context}\n\nFriend's new message:\n{message}"
    
    # Generate the response (this allows function calling automatically in newer python SDK versions via `chat` or generate_content)
    # We should use a chat session if we want it to automatically handle tool calls back-and-forth easily.
    # Alternatively, start a new chat with the prompt.
    chat = model.start_chat(enable_automatic_function_calling=True)
    response = chat.send_message(prompt)
    
    reply_text = response.text
    
    # Save the interaction to memory
    save_message_to_memory(user_id, message, role="user")
    save_message_to_memory(user_id, reply_text, role="bot")
    
    return reply_text
