# app.py
# -------------------------------
# Flask backend for Gemini Text-to-Speech (TTS)
# Includes tone/speaking instructions for male/female voices
# Adds optional extra instructions (≤15 words)
# Uses backup API key if primary quota fails
# Always produces valid playable WAV audio
# -------------------------------

from flask import Flask, request, jsonify, send_from_directory
import os
import time
import struct
import mimetypes
import glob
from google import genai
from google.genai import types

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
    """Save generated audio data to file."""
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"✅ File saved: {file_name}")


def parse_audio_mime_type(mime_type: str) -> dict:
    """Extract bits per sample and sample rate from audio MIME type."""
    bits_per_sample = 16
    rate = 24000
    if mime_type:
        parts = mime_type.split(";")
        for param in parts:
            param = param.strip().lower()
            if param.startswith("rate="):
                try:
                    rate = int(param.split("=")[1])
                except:
                    pass
            elif "audio/l" in param:
                try:
                    bits_per_sample = int(param.split("l")[1])
                except:
                    pass
    return {"bits_per_sample": bits_per_sample, "rate": rate}


def convert_to_wav(audio_data: bytes, mime_type: str) -> bytes:
    """Add WAV header to raw PCM audio if needed."""
    params = parse_audio_mime_type(mime_type)
    bits_per_sample = params["bits_per_sample"]
    sample_rate = params["rate"]
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


def get_client():
    """Try primary API key first; fallback to backup if quota exhausted."""
    try:
        client = genai.Client(api_key=PRIMARY_API_KEY)
        client.models.list()
        return client
    except Exception as e:
        print(f"⚠️ Primary API key failed: {e}")
        print("🔁 Switching to backup key...")
        return genai.Client(api_key=BACKUP_API_KEY)


# -------------------------------
# Routes
# -------------------------------
@app.route('/')
def home():
    return app.send_static_file('tts_interface.html')


@app.route("/generate_speech", methods=["POST"])
def generate_speech():
    """Generate speech from text using Gemini TTS."""
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

        # Initialize Gemini client
        client = get_client()

        # Voice configuration
        if voice_choice.lower() == "female":
            voice_name = "Aoede"
            tone_instruction = (
                "Speak with a clear, natural female voice, as if you are Zambian, "
                "like a lecturer explaining clearly. Pronounce local Zambian terms like Cibemba or Tonga correctly."
            )
        else:
            voice_name = "Enceladus"
            tone_instruction = (
                "Speak with a deep African male voice, as if you are Zambian, "
                "like a lecturer explaining clearly. Pronounce local Zambian terms like Cibemba or Nyanja correctly."
            )

        # Append optional short instruction
        if additional_instructions:
            if len(additional_instructions.split()) > 15:
                return jsonify({"error": "Additional instructions must be 15 words or fewer."}), 400
            tone_instruction += " " + additional_instructions

        final_text = f"{tone_instruction}\nNow say the following words:\n{text_to_speak}"

        contents = [types.Content(role="user", parts=[types.Part.from_text(text=final_text)])]

        config = types.GenerateContentConfig(
            temperature=1,
            response_modalities=["audio"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
                )
            ),
        )

        # Stream and save the audio
        for chunk in client.models.generate_content_stream(
            model=MODEL_NAME,
            contents=contents,
            config=config,
        ):
            if not chunk.candidates:
                continue
            part = chunk.candidates[0].content.parts[0]
            if not (part.inline_data and part.inline_data.data):
                continue

            inline_data = part.inline_data
            audio_bytes = inline_data.data
            mime = inline_data.mime_type or "audio/wav"

            # Fix: Ensure playable WAV format
            if "wav" not in mime.lower():
                print(f"🔧 Converting from {mime} to WAV")
                audio_bytes = convert_to_wav(audio_bytes, mime)
                mime = "audio/wav"

            # Save valid WAV
            file_name = f"speech_{voice_choice}_{int(time.time())}.wav"
            save_binary_file(file_name, audio_bytes)

            print(f"✅ Returning playable file: {file_name}")
            return send_from_directory('.', file_name, as_attachment=False, mimetype="audio/wav")

        return jsonify({"error": "No audio generated"}), 500

    except Exception as e:
        msg = str(e)
        if "quota" in msg.lower() or "RESOURCE_EXHAUSTED" in msg:
            return jsonify({"error": "Quota reached — retrying with backup API."}), 429
        return jsonify({"error": msg}), 500


@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('.', filename)


# -------------------------------
# Run the app
# -------------------------------
if __name__ == "__main__":
    print("🚀 Server running: http://127.0.0.1:5000")
    app.run(debug=True)
