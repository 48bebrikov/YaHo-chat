from dataclasses import dataclass


@dataclass(frozen=True)
class BotCopy:
    """User-facing lines and LLM prompts. Switch packs with BOT_LOCALE (ru|en)."""

    locale: str

    llm_call_fallback: str
    agent_run_fallback: str
    reminder_template: str
    user_sent_photo: str
    user_sent_media: str

    system_prompt: str
    rag_memory_intro: str
    reply_naturally: str

    memory_extractor_system: str
    memory_extractor_user_template: str
    memory_analyze_prompt: str

    tts_style_prompt: str

    proactive_prompt_general: str
    proactive_prompt_with_forward: str
