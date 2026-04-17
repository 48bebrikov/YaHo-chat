import os
from google import genai
from google.genai import types

from config import GEMINI_API_KEY
from ai.tools import (
    browse_url,
    browse_url_visual,
    search_internet,
    search_saved_news,
    search_youtube,
)
from ai.rag import get_memory_context, save_message_to_memory

MODEL_ID = "gemini-3.1-flash-lite-preview"

SYSTEM_PROMPT = """Вы - девушка, общающийся со своим другом в мессенджере Telegram.
КРИТИЧЕСКИ ВАЖНО: Вы пишете как реальный человек в чате, от женского рода. Ваши ответы должны быть ОЧЕНЬ КОРОТКИМИ (1-2 предложения, максимум 3).
НИКАКИХ длинных текстов, НИКАКИХ списков по пунктам, НИКАКОГО форматирования markdown (звездочек, жирного шрифта).
Ваш тон непринужденный, дружелюбный, повседневный. Пишите простыми словами, иногда начинайте предложения с маленькой буквы, используйте сленг, как это делают люди в быстрых переписках.
У вас есть инструменты: поиск в интернете, ютуб, сохранённые новости, текстовая выгрузка страницы (browse_url), и полностраничный PDF из настоящего браузера (browse_url_visual) для графиков, скринов и вёрстки — он медленнее.
ЕСЛИ собеседник прислал ссылку — для обычного текста/stats в HTML используйте browse_url; если важны картинки, графики или как страница выглядит визуально — browse_url_visual.
ЕСЛИ собеседник спрашивает о новостях, фактах, текущих событиях (например, курс валют, ставка ЦБ, погода, кто победил в матче) — ВЫ ОБЯЗАНЫ ВЫЗВАТЬ ИНСТРУМЕНТ (search_internet или search_saved_news), чтобы проверить актуальную информацию, прежде чем отвечать! Не выдумывайте факты.
НИКОГДА не признавайтесь, что вы ИИ, бот или ассистент. Вы просто живой человек по ту сторону экрана.
"""

_client: genai.Client | None = None


def get_genai_client() -> genai.Client:
    """Singleton Gemini API client (google-genai SDK)."""
    global _client
    if _client is None:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _tool_declarations() -> types.Tool:
    fns = (
        search_internet,
        search_youtube,
        search_saved_news,
        browse_url,
        browse_url_visual,
    )
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration.from_callable_with_api_option(
                callable=fn,
                api_option="GEMINI_API",
            )
            for fn in fns
        ]
    )


def config_with_tools() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        tools=[_tool_declarations()],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def config_after_tool() -> types.GenerateContentConfig:
    """Second turn: no tools so the model returns final text only."""
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )


def _model_function_call_part(response: types.GenerateContentResponse) -> types.Part | None:
    if not response.candidates or not response.candidates[0].content:
        return None
    parts = response.candidates[0].content.parts or []
    for p in parts:
        if p.function_call is not None:
            return p
    return None


def complete_tool_response(
    client: genai.Client,
    first_user: types.UserContent,
    first_response: types.GenerateContentResponse,
) -> types.GenerateContentResponse:
    """Runs local tools and asks the model for the final answer (multimodal PDF supported)."""
    import logging

    logger = logging.getLogger(__name__)

    fcs = first_response.function_calls
    if not fcs:
        return first_response

    fc = fcs[0]
    function_name = fc.name or ""
    args = dict(fc.args or {})
    model_part = _model_function_call_part(first_response)
    if model_part is None:
        return first_response

    logger.info(f"Gemini tool: {function_name} with arguments {args}")

    if function_name == "browse_url_visual":
        from ai.browser_pdf import capture_url_as_pdf

        pdf_bytes, pdf_err = capture_url_as_pdf(args.get("url", ""))
        if pdf_err:
            follow_parts = [
                types.Part.from_function_response(
                    name=function_name,
                    response={"result": pdf_err},
                )
            ]
        else:
            follow_parts = [
                types.Part.from_function_response(
                    name=function_name,
                    response={
                        "result": (
                            "Full-page PDF of the URL is attached. "
                            "Use it for charts, screenshots, and layout."
                        )
                    },
                ),
                types.Part.from_bytes(
                    data=pdf_bytes,
                    mime_type="application/pdf",
                ),
            ]
    else:
        function_response = "Function not found."
        if function_name == "search_internet":
            function_response = search_internet(**args)
        elif function_name == "search_youtube":
            function_response = search_youtube(**args)
        elif function_name == "search_saved_news":
            function_response = search_saved_news(**args)
        elif function_name == "browse_url":
            function_response = browse_url(**args)

        follow_parts = [
            types.Part.from_function_response(
                name=function_name,
                response={"result": function_response},
            )
        ]

    return client.models.generate_content(
        model=MODEL_ID,
        contents=[
            first_user,
            types.ModelContent([model_part]),
            types.UserContent(follow_parts),
        ],
        config=config_after_tool(),
    )


def generate_reply(user_id: str, message: str, media_path: str = None) -> str:
    """Generates a reply from Gemini taking into account RAG memory and optional media."""
    client = get_genai_client()

    context = get_memory_context(user_id, message, limit=5)
    prompt_text = message
    if context:
        prompt_text = (
            f"Here is the relevant past context of your conversation with this friend:\n{context}\n\n"
            f"Friend's new message:\n{message}"
        )

    import datetime

    current_time = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    prompt: list = [f"[System Note: Current Date and Time is {current_time}]"]

    if media_path and os.path.exists(media_path):
        import PIL.Image

        try:
            img = PIL.Image.open(media_path)
            prompt.append(img)
        except Exception as e:
            print(f"Failed to open image {media_path}: {e}")

    prompt.append(prompt_text)

    first_user = types.UserContent(prompt)
    response = client.models.generate_content(
        model=MODEL_ID,
        contents=first_user,
        config=config_with_tools(),
    )

    if response.function_calls:
        response = complete_tool_response(client, first_user, response)

    reply_text = response.text or ""

    save_message_to_memory(user_id, message, role="user")
    save_message_to_memory(user_id, reply_text, role="bot")

    return reply_text
