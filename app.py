# app.py
# -------------------------------
# Flask backend for Gemini Text-to-Speech (TTS)
# Includes full tone/speaking instructions
# -------------------------------

from flask import Flask, request, jsonify, send_from_directory
import os
import time
import mimetypes
import struct
from google import genai
from google.genai import types

app = Flask(__name__, static_folder='.')

# -------------------------------
# Helper functions
# -------------------------------

def save_binary_file(file_name, data):
    """Save generated audio data to a file."""
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"✅ File saved: {file_name}")

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
                rate = int(param.split("=", 1)[1])
            except:
                pass
        elif "audio/L" in param:
            try:
                bits_per_sample = int(param.split("L", 1)[1])
            except:
                pass
    return {"bits_per_sample": bits_per_sample, "rate": rate}

# -------------------------------
# Routes
# -------------------------------

@app.route('/')
def home():
    return app.send_static_file('tts_interface.html')


@app.route("/generate_speech", methods=["POST"])
def generate_speech():
    """Generate speech from text using Gemini."""
    try:
        api_key = "AIzaSyDEabvFms5_MLzM1EwHox2UB_vI6uG3wPk"
        client = genai.Client(api_key=api_key)
        model = "gemini-2.5-flash-preview-tts"

        data = request.get_json()
        text_to_speak = data.get("text", "")
        voice_choice = data.get("voice", "male")

        if not text_to_speak.strip():
            return jsonify({"error": "Text input is empty"}), 400

        # Voice instructions
        if voice_choice.lower() == "female":
            voice_name = "Aoede"
            tone_instruction = (
                "Speak with a clear, natural female voice, speak as if you are Zambian, "
                "like a lecturer who wants the students to understand, also add weight to the voice. "
                "As you speak, pronounce any Zambian local language terms, such as Cibemba, correctly. "
                "Do not exaggerate the tone."
            )
        else:
            voice_name = "Enceladus"
            tone_instruction = (
                "Speak with a deep African male voice, as if you are Zambian, "
                "like a lecturer who wants the students to understand. "
                "As you speak, pronounce any Zambian local language terms, such as Cibemba, correctly. "
                "Do not exaggerate the tone."
            )

        final_text = f"{tone_instruction}\nNow say the following words:\n{text_to_speak}"

        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=final_text)],
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

                full_path = f"{file_name}{file_extension}"
                save_binary_file(full_path, data_buffer)

                return jsonify({
                    "success": True,
                    "file": full_path
                })

        return jsonify({"error": "No audio generated"}), 500

    except Exception as e:
        # Handle quota errors gracefully
        error_message = str(e)
        if "RESOURCE_EXHAUSTED" in error_message or "quota exceeded" in error_message:
            return jsonify({
                "error": "Your daily TTS quota has been reached. Please contact the administrator to extend your quota."
            }), 429
        return jsonify({"error": error_message}), 500


@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)


# -------------------------------
# Run the app
# -------------------------------
if __name__ == "__main__":
    print("🚀 Starting Gemini TTS Server at http://127.0.0.1:5000")
    app.run(debug=True)
