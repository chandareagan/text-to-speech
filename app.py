# To run this code you need to install the following dependencies:
# pip install google-genai

import base64
import mimetypes
import os
import re
import struct
import time
from google import genai
from google.genai import types


def save_binary_file(file_name, data):
    """Save the generated audio data to a file."""
    try:
        with open(file_name, "wb") as f:
            f.write(data)
        print(f"✅ File saved to: {file_name}")
    except PermissionError:
        print(f"⚠️ Permission denied. Please close '{file_name}' if it’s open in another app.")


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Convert raw audio data to a WAV format if needed."""
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
        b"RIFF",
        chunk_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        num_channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b"data",
        data_size
    )
    return header + audio_data


def parse_audio_mime_type(mime_type: str) -> dict:
    """Extract bits per sample and sample rate from audio MIME type."""
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
        elif "audio/L" in param:
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except (ValueError, IndexError):
                pass

    return {"bits_per_sample": bits_per_sample, "rate": rate}


def generate(voice_choice="male"):
    """Generate audio from text using Gemini TTS."""

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    model = "gemini-2.5-flash-preview-tts"

    # Voice selection
    if voice_choice.lower() == "female":
        voice_name = "Aoede"
        tone_instruction = "Speak with a clear, natural female voice, speak as if you are Zambian, like a lecturer who wants the students to understand, also add weight to the voice."
    else:
        voice_name = "Enceladus"
        tone_instruction = "Speak with a deep African male voice. speak as if you are Zambian, like a lecturer who wants the students to understand,"

    # The spoken content
    text_to_speak = f"""{tone_instruction} 
As you speak, pronounce any Zambian local language terms, such as Cibemba, correctly as they are said in Zambia.
Do not exaggerate the tone. 
Say the next words:
'Mining is the process of extracting valuable minerals or resources from the Earth, involving exploration, excavation, processing, and refining to obtain useful raw materials. 
Bwalya works at this mine which is: Kansanshi Mine in Zambia, this is an important mine as it is operated by First Quantum Minerals, is one of Africa’s largest copper mines. 
It extracts and processes copper and gold ore, contributing significantly to Zambia’s economy through mineral exports, employment, and infrastructure development in the North-Western Province.'"""

    contents = [
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=text_to_speak)],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        temperature=1,
        response_modalities=["audio"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        ),
    )

    for chunk in client.models.generate_content_stream(
        model=model,
        contents=contents,
        config=generate_content_config,
    ):
        if (
            not chunk.candidates
            or not chunk.candidates[0].content
            or not chunk.candidates[0].content.parts
        ):
            continue

        part = chunk.candidates[0].content.parts[0]
        if part.inline_data and part.inline_data.data:
            timestamp = int(time.time())
            file_name = f"speech_{voice_choice}_{timestamp}"

            inline_data = part.inline_data
            data_buffer = inline_data.data

            file_extension = mimetypes.guess_extension(inline_data.mime_type)
            if file_extension is None:
                file_extension = ".wav"
                data_buffer = convert_to_wav(inline_data.data, inline_data.mime_type)

            save_binary_file(f"{file_name}{file_extension}", data_buffer)
        else:
            print(chunk.text)


if __name__ == "__main__":
    if not os.environ.get("GEMINI_API_KEY"):
        print("⚠️ Please set your Gemini API key first:")
        print("   set GEMINI_API_KEY=your_api_key_here  (on Windows)")
        print("   export GEMINI_API_KEY=your_api_key_here  (on macOS/Linux)")
    else:
        print("Select voice type:")
        print("1. Male") #Enceladus
        print("2. Female") #Aoede
        choice = input("Enter 1 or 2: ").strip()

        voice_choice = "female" if choice == "2" else "male"
        generate(voice_choice)
