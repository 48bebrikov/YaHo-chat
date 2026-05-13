import asyncio
from google import genai
from google.genai import types
from config import GEMINI_API_KEY

async def test_tts_voice(file_path: str):
    client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        config = types.GenerateContentConfig(
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Aoede"
                    )
                )
            )
        )
        response = await client.aio.models.generate_content(
            model="gemini-3.1-flash-tts-preview",
            contents="Скажи драмматически: [sigh] Ой, да ладно тебе! [laughs] Как ты это сделал?",
            config=config
        )
        print("Success! Got response.")
        
        audio_bytes = None
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    audio_bytes = part.inline_data.data
                    break
        
        if audio_bytes:
            # Add WAV header (required for Telethon to recognize the raw PCM data correctly)
            from ai.tts import convert_to_wav
            mime_type = "audio/L16;rate=24000"
            if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                 for part in response.candidates[0].content.parts:
                    if part.inline_data:
                         mime_type = part.inline_data.mime_type
                         break
            wav_data = convert_to_wav(audio_bytes, mime_type)
            with open(file_path, "wb") as f:
                f.write(wav_data)
            print(f"Saved audio to {file_path}")
        else:
            print("No audio data found in response.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    file_path = "test_voice.wav"
    asyncio.run(test_tts_voice(file_path))