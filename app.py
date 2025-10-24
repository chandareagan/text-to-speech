# app.py
# -------------------------------
# Flask backend for Gemini Text-to-Speech (TTS)
# Includes full tone/speaking instructions
# Adds optional extra instructions (≤15 words)
# Uses backup API key if primary quota fails
# Automatically serves audio as downloadable file
# -------------------------------

from flask import Flask, request, jsonify, send_from_directory
import os
import time
import mimetypes
import struct
from google import genai
from google.genai import types
import glob

app = Flask(__name__, static_folder='.')

# -------------------------------
# Configuration
# -------------------------------
PRIMARY_API_KEY = "AIzaSyDEabvFms5_MLzM1EwHox2UB_vI6uG3wPk"
BACKUP_API_KEY = "AIzaSyCDJnYLIn0VODW1DNjZYwPiCKRnK2QpuIQ"
MODEL_NAME = "gemini-2.5-flash-preview-tts"


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


def get_client():
    """Try primary API key first, then backup if quota fails."""
    try:
        client = genai.Client(api_key=PRIMARY_API_KEY)
        client.models.list()  # quick sanity check
        return client
    except Exception as e:
        print(f"⚠️ Primary API key failed: {e}")
        print("🔁 Switching to backup API key...")
        return genai.Client(api_key=BACKUP_API_KEY)


# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def home():
    return app.send_static_file('tts_interface.html')


@app.route("/generate_speech", methods=["POST"])
def generate_speech():
    """Generate speech from text using Gemini and send it as a downloadable file."""
    try:
        # Cleanup old files (older than 3 months)
        three_months = 90 * 24 * 60 * 60
        for f in glob.glob("speech_*.wav"):
            if time.time() - os.path.getmtime(f) > three_months:
                os.remove(f)

        data = request.get_json()
        text_to_speak = data.get("text", "").strip()
        voice_choice = data.get("voice", "male")
        additional_instructions = data.get("additionalInstructions", "").strip()

        if not text_to_speak:
            return jsonify({"error": "Text input is empty"}), 400

        # Initialize client
        client = get_client()

        # Voice configuration
        if voice_choice.lower() == "female":
            voice_name = "Aoede"
            tone_instruction = (
                "Speak with a clear, natural female voice, as if you are Zambian, "
                "like a lecturer who wants students to understand, with balanced tone. "
                "Pronounce local Zambian language terms correctly, like Cibemba or Tonga."
            )
        else:
            voice_name = "Enceladus"
            tone_instruction = (
                "Speak with a deep African male voice, as if you are Zambian, "
                "like a lecturer who wants students to understand. "
                "Pronounce local Zambian language terms correctly, such as Cibemba or Nyanja."
            )

        # Append user extra instruction (≤15 words)
        if additional_instructions:
            word_count = len(additional_instructions.split())
            if word_count > 15:
                return jsonify({"error": "Additional instructions must be 15 words or fewer."}), 400
            tone_instruction += " " + additional_instructions

        final_text = f"{tone_instruction}\nNow say the following words:\n{text_to_speak}"

        contents = [
            types.Content(role="user", parts=[types.Part.from_text(text=final_text)]),
        ]

        config = types.GenerateContentConfig(
            temperature=1,
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                )
            ),
        )

        for chunk in client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=contents,
            config=config,
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
                data_buffer = part.inline_data.data
                ext = mimetypes.guess_extension(part.inline_data.mime_type) or ".wav"

                full_path = f"{file_name}{ext}"
                save_binary_file(full_path, data_buffer)
                return send_from_directory('.', full_path, as_attachment=True)

        return jsonify({"error": "No audio generated"}), 500

    except Exception as e:
        msg = str(e)
        if "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
            return jsonify({
                "error": "Your daily TTS quota has been reached. Backup API will be used automatically."
            }), 429
        return jsonify({"error": msg}), 500


@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)


# -------------------------------
# Run the app
# -------------------------------
if __name__ == "__main__":
    print("🚀 Server running at: http://127.0.0.1:5000")
    app.run(debug=True)

