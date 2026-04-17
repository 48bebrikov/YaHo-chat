import logging
import os
import random
import time

from google import genai
from google.genai import errors, types

from config import GEMINI_API_KEY, RECENT_CHAT_WINDOW_MESSAGES

from ai.tools import (
    browse_url,
    browse_url_visual,
    search_internet,
    search_saved_news,
    search_youtube,
)
from ai.rag import format_recent_chat_block, get_memory_context, persist_conversation_turn

logger = logging.getLogger(__name__)

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


def _is_transient_api_error(exc: BaseException) -> bool:
    if isinstance(exc, errors.APIError):
        code = getattr(exc, "code", None)
        if code in (429, 500, 502, 503, 504):
            return True
        status = (getattr(exc, "status", None) or "")
        if isinstance(status, str) and status.upper() in (
            "UNAVAILABLE",
            "RESOURCE_EXHAUSTED",
            "DEADLINE_EXCEEDED",
        ):
            return True
    text = str(exc).lower()
    return "503" in text or "429" in text


async def generate_content_with_retry(
    client: genai.Client,
    *,
    model: str,
    contents,
    config,
) -> types.GenerateContentResponse:
    """Retries on overload / rate limits (503, 429, etc.) with exponential backoff."""
    import asyncio
    from ai.metrics import GEMINI_REQUESTS_TOTAL, GEMINI_REQUEST_LATENCY, GEMINI_TOKENS_TOTAL, Timer
    max_attempts = 6
    delay = 1.5
    for attempt in range(1, max_attempts + 1):
        try:
            with Timer(GEMINI_REQUEST_LATENCY, model=model):
                response = await client.aio.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )
            
            GEMINI_REQUESTS_TOTAL.labels(model=model, status="success").inc()
            
            # Log token usage if available
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                metadata = response.usage_metadata
                if hasattr(metadata, "prompt_token_count") and metadata.prompt_token_count:
                    GEMINI_TOKENS_TOTAL.labels(model=model, token_type="prompt").inc(metadata.prompt_token_count)
                if hasattr(metadata, "candidates_token_count") and metadata.candidates_token_count:
                    GEMINI_TOKENS_TOTAL.labels(model=model, token_type="completion").inc(metadata.candidates_token_count)
            
            return response
            
        except Exception as e:
            if attempt == max_attempts or not _is_transient_api_error(e):
                GEMINI_REQUESTS_TOTAL.labels(model=model, status="error_fatal").inc()
                raise
            
            GEMINI_REQUESTS_TOTAL.labels(model=model, status="error_transient").inc()
            jitter = random.uniform(0, delay * 0.25)
            wait = min(delay + jitter, 45.0)
            logger.warning(
                "Gemini generate_content failed (attempt %s/%s): %s — retry in %.1fs",
                attempt,
                max_attempts,
                e,
                wait,
            )
            await asyncio.sleep(wait)
            delay = min(delay * 2, 30.0)
    raise RuntimeError("unreachable")


def generate_content_with_retry_sync(
    client: genai.Client,
    *,
    model: str,
    contents,
    config,
) -> types.GenerateContentResponse:
    """Same retry policy as generate_content_with_retry, using the synchronous client API."""
    from ai.metrics import GEMINI_REQUESTS_TOTAL, GEMINI_REQUEST_LATENCY, GEMINI_TOKENS_TOTAL, Timer

    max_attempts = 6
    delay = 1.5
    for attempt in range(1, max_attempts + 1):
        try:
            with Timer(GEMINI_REQUEST_LATENCY, model=model):
                response = client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )

            GEMINI_REQUESTS_TOTAL.labels(model=model, status="success").inc()

            if hasattr(response, "usage_metadata") and response.usage_metadata:
                metadata = response.usage_metadata
                if hasattr(metadata, "prompt_token_count") and metadata.prompt_token_count:
                    GEMINI_TOKENS_TOTAL.labels(model=model, token_type="prompt").inc(
                        metadata.prompt_token_count
                    )
                if hasattr(metadata, "candidates_token_count") and metadata.candidates_token_count:
                    GEMINI_TOKENS_TOTAL.labels(model=model, token_type="completion").inc(
                        metadata.candidates_token_count
                    )

            return response

        except Exception as e:
            if attempt == max_attempts or not _is_transient_api_error(e):
                GEMINI_REQUESTS_TOTAL.labels(model=model, status="error_fatal").inc()
                raise

            GEMINI_REQUESTS_TOTAL.labels(model=model, status="error_transient").inc()
            jitter = random.uniform(0, delay * 0.25)
            wait = min(delay + jitter, 45.0)
            logger.warning(
                "Gemini generate_content (sync) failed (attempt %s/%s): %s — retry in %.1fs",
                attempt,
                max_attempts,
                e,
                wait,
            )
            time.sleep(wait)
            delay = min(delay * 2, 30.0)
    raise RuntimeError("unreachable")


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


async def complete_tool_response(
    client: genai.Client,
    first_user: types.UserContent,
    first_response: types.GenerateContentResponse,
) -> types.GenerateContentResponse:
    """Runs local tools and asks the model for the final answer. Loops up to 3 times."""
    response = first_response
    contents = [first_user]
    import asyncio
    
    for _ in range(3):
        fcs = response.function_calls
        if not fcs:
            return response
            
        fc = fcs[0]
        function_name = fc.name or ""
        args = dict(fc.args or {})
        model_part = _model_function_call_part(response)
        if model_part is None:
            return response

        contents.append(types.ModelContent([model_part]))
        logger.info(f"Gemini tool: {function_name} with arguments {args}")

        if function_name == "browse_url_visual":
            from ai.browser_pdf import capture_url_as_pdf

            pdf_bytes, pdf_err = await capture_url_as_pdf(args.get("url", ""))
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
                function_response = await asyncio.to_thread(search_internet, **args)
            elif function_name == "search_youtube":
                function_response = await asyncio.to_thread(search_youtube, **args)
            elif function_name == "search_saved_news":
                function_response = await asyncio.to_thread(search_saved_news, **args)
            elif function_name == "browse_url":
                function_response = await asyncio.to_thread(browse_url, **args)

            follow_parts = [
                types.Part.from_function_response(
                    name=function_name,
                    response={"result": function_response},
                )
            ]
            
        contents.append(types.UserContent(follow_parts))
        
        response = await generate_content_with_retry(
            client,
            model=MODEL_ID,
            contents=contents,
            config=config_with_tools(),
        )

    return response


async def generate_reply(user_id: str, message: str, media_path: str = None) -> str:
    """Generates a reply from Gemini taking into account RAG memory and optional media."""
    client = get_genai_client()

    recent_block = format_recent_chat_block(user_id, message, RECENT_CHAT_WINDOW_MESSAGES)
    rag_context = get_memory_context(user_id, message, limit_facts=5, limit_dialogue=5)

    if rag_context:
        prompt_text = (
            f"{recent_block}\n\n"
            "Here is long-term memory from the past (RAG; may overlap with the lines above):\n"
            f"{rag_context}\n\n"
            "Reply naturally. The friend's latest message is the last line in the recent block above."
        )
    else:
        prompt_text = (
            f"{recent_block}\n\n"
            "Reply naturally. The friend's latest message is the last line in the recent block above."
        )

    import datetime

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    current_time = now_utc.strftime("%Y-%m-%d %H:%M:%S UTC")

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
    response = await generate_content_with_retry(
        client,
        model=MODEL_ID,
        contents=first_user,
        config=config_with_tools(),
    )

    if response.function_calls:
        response = await complete_tool_response(client, first_user, response)

    reply_text = response.text or ""

    now_save = datetime.datetime.now(datetime.timezone.utc)
    import asyncio
    asyncio.create_task(
        asyncio.to_thread(persist_conversation_turn,
            user_id,
            message,
            reply_text,
            sort_ts=now_save.timestamp(),
            event_utc_iso=now_save.isoformat(),
        )
    )

    from database.sqlite_db import append_friend_chat_turn

    asyncio.create_task(
        asyncio.to_thread(append_friend_chat_turn, user_id, message, reply_text)
    )

    return reply_text
