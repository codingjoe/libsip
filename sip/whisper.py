"""Whisper-based audio transcription for incoming SIP calls."""

from __future__ import annotations

import asyncio
import pathlib
import tempfile
import wave

import whisper

from .calls import IncomingCall

__all__ = ["WhisperCall"]

_SAMPLE_RATE = 8000
_CHUNK_BYTES = _SAMPLE_RATE * 30  # 30 seconds of 8 kHz, 8-bit mono audio


class WhisperCall(IncomingCall):
    """Incoming SIP call that transcribes the audio stream using OpenAI Whisper."""

    def __init__(self, *args, model: str = "base", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._whisper_model = whisper.load_model(model)
        self._buffer = bytearray()

    def handle(self, audio: bytes) -> None:
        """Buffer audio and schedule transcription when a full chunk is available."""
        self._buffer.extend(audio)
        if len(self._buffer) >= _CHUNK_BYTES:
            asyncio.create_task(self._transcribe_chunk())

    async def _transcribe_chunk(self) -> None:
        chunk = bytes(self._buffer[:_CHUNK_BYTES])
        del self._buffer[:_CHUNK_BYTES]
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self._run_transcription, chunk)
        self.transcription_received(text.strip())

    def _run_transcription(self, audio_bytes: bytes) -> str:
        with tempfile.TemporaryDirectory() as tmp_dir:
            audio_path = pathlib.Path(tmp_dir) / "audio.wav"
            with wave.open(str(audio_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(1)
                wav_file.setframerate(_SAMPLE_RATE)
                wav_file.writeframes(audio_bytes)
            return self._whisper_model.transcribe(str(audio_path))["text"]

    def transcription_received(self, text: str) -> None:
        """Handle a transcription result. Override in subclasses."""
        return NotImplemented
