import pytest
from ai.tts import parse_audio_mime_type, convert_to_wav

def test_parse_audio_mime_type():
    res1 = parse_audio_mime_type("audio/L16;rate=24000")
    assert res1["bits_per_sample"] == 16
    assert res1["rate"] == 24000

    res2 = parse_audio_mime_type("audio/ogg")
    assert res2["bits_per_sample"] == 16  # fallback default
    assert res2["rate"] == 24000

    res3 = parse_audio_mime_type("audio/L8; rate=48000")
    assert res3["bits_per_sample"] == 8
    assert res3["rate"] == 48000

def test_convert_to_wav():
    raw_audio = b"some fake audio data"
    wav = convert_to_wav(raw_audio, "audio/L16;rate=24000")
    
    # Check headers
    assert wav.startswith(b"RIFF")
    assert b"WAVEfmt " in wav
    assert b"data" in wav
    
    # Check that original audio is preserved at the end
    assert wav.endswith(raw_audio)
