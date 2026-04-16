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
    # Use gemini-1.5-flash since 3.1 flash is not fully supported in the REST API v1beta yet.
    model = genai.GenerativeModel(
        model_name='gemini-3.1-flash-lite-preview',
        tools=[search_internet, search_youtube, search_saved_news],
        system_instruction=SYSTEM_PROMPT
    )
    return model

def generate_reply(user_id: str, message: str, media_path: str = None) -> str:
    """Generates a reply from Gemini taking into account RAG memory and optional media."""
    model = get_gemini_model()
    
    # Retrieve past context from Qdrant
    context = get_memory_context(user_id, message, limit=5)
    
    prompt_text = message
    if context:
        prompt_text = f"Here is the relevant past context of your conversation with this friend:\n{context}\n\nFriend's new message:\n{message}"
    
    # Construct the final prompt (handling multimodal input)
    prompt = []
    if media_path and os.path.exists(media_path):
        import PIL.Image
        try:
            img = PIL.Image.open(media_path)
            prompt.append(img)
        except Exception as e:
            print(f"Failed to open image {media_path}: {e}")
            
    prompt.append(prompt_text)
    
    # Using generate_content instead of start_chat to avoid potential blocking issues
    # with automatic function calling loops in older SDK versions
    response = model.generate_content(prompt)
    
    # If the model decided to call a function, response.parts will contain function_call
    # We will handle basic function calling here
    if response.parts and hasattr(response.parts[0], 'function_call') and response.parts[0].function_call:
        fc = response.parts[0].function_call
        function_name = fc.name
        args = {k: v for k, v in fc.args.items()}
        
        # Execute the function
        function_response = "Function not found."
        if function_name == "search_internet":
            function_response = search_internet(**args)
        elif function_name == "search_youtube":
            function_response = search_youtube(**args)
        elif function_name == "search_saved_news":
            function_response = search_saved_news(**args)
            
        # Send the function response back to the model to get the final text
        # Since we didn't use a chat session, we need to construct the history manually
        final_response = model.generate_content([
            prompt,
            response.parts[0], # The function call part from model
            genai.protos.Part(function_response=genai.protos.FunctionResponse(name=function_name, response={"result": function_response}))
        ])
        reply_text = final_response.text
    else:
        reply_text = response.text
    
    # Save the interaction to memory
    save_message_to_memory(user_id, message, role="user")
    save_message_to_memory(user_id, reply_text, role="bot")
    
    return reply_text
