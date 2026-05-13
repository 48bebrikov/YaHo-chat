import os
import logging
from google import genai
from google.genai import types
from config import GEMINI_API_KEY, TTS_VOICE

logger = logging.getLogger(__name__)

import struct

def parse_audio_mime_type(mime_type: str) -> dict:
    """Parses bits per sample and rate from an audio MIME type string."""
    bits_per_sample = 16
    rate = 24000

    parts = mime_type.split(";")
    for param in parts:
        param = param.strip()
        if param.lower().startswith("rate="):
            try:
                rate_str = param.split("=", 1)[1]
                rate = int(rate_str)
            except (ValueError, IndexError):
                pass
        elif param.startswith("audio/L"):
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass

    return {"bits_per_sample": bits_per_sample, "rate": rate}

def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Generates a WAV file header for the given audio data and parameters."""
    parameters = parse_audio_mime_type(mime_type)
    bits_per_sample = parameters["bits_per_sample"]
    sample_rate = parameters["rate"]
    num_channels = 1
    data_size = len(audio_data)
    bytes_per_sample = bits_per_sample // 8
    block_align = num_channels * bytes_per_sample
    byte_rate = sample_rate * block_align
    chunk_size = 36 + data_size

    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",          # ChunkID
        chunk_size,       # ChunkSize (total file size - 8 bytes)
        b"WAVE",          # Format
        b"fmt ",          # Subchunk1ID
        16,               # Subchunk1Size (16 for PCM)
        1,                # AudioFormat (1 for PCM)
        num_channels,     # NumChannels
        sample_rate,      # SampleRate
        byte_rate,        # ByteRate
        block_align,      # BlockAlign
        bits_per_sample,  # BitsPerSample
        b"data",          # Subchunk2ID
        data_size         # Subchunk2Size (size of audio data)
    )
    return header + audio_data

async def generate_voice_message(text: str, filepath: str = "voice.wav", voice: str = TTS_VOICE, style_prompt: str = None):
    """
    Calls Gemini 3.1 Flash TTS API to generate a voice message.
    It expects the text to contain emotional tags like [laughs], [sigh].
    """
    if not GEMINI_API_KEY:
        logger.error("No API key for TTS")
        return None
        
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Optional styling prompt for the TTS model
    if not style_prompt:
        style_prompt = "Ты - 20-летняя девушка Катя. Говори натурально, весело, эмоционально."
        
    # The API accepts prompt instructions prepended to the text
    full_text = f"{style_prompt}\n\nText: {text}"
    
    try:
        config = types.GenerateContentConfig(
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice
                    )
                )
            )
        )
        
        response = await client.aio.models.generate_content(
            model="gemini-3.1-flash-tts-preview",
            contents=full_text,
            config=config
        )
        
        audio_bytes = None
        mime_type = ""
        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data:
                    audio_bytes = part.inline_data.data
                    mime_type = part.inline_data.mime_type
                    break
        
        if audio_bytes:
            # We save the file as raw audio.
            # If Telegram complains, we might need to add the WAV header like in the docs.
            wav_data = convert_to_wav(audio_bytes, mime_type or "audio/L16;rate=24000")
            
            is_ogg = filepath.endswith('.ogg')
            temp_wav_path = filepath if not is_ogg else filepath.rsplit('.', 1)[0] + '.wav'
                
            with open(temp_wav_path, "wb") as f:
                f.write(wav_data)

            if is_ogg:
                import subprocess
                try:
                    # Convert to OGG Opus for Telegram voice notes
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", temp_wav_path, "-c:a", "libopus", filepath],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                    )
                    os.remove(temp_wav_path)
                    return filepath
                except Exception as e:
                    logger.error(f"FFmpeg conversion to ogg failed: {e}")
                    return temp_wav_path
                    
            return filepath
        else:
            logger.error("No inline_data found in TTS response")
            return None
            
    except Exception as e:
        logger.error(f"Exception during TTS generation: {e}")
        return None
