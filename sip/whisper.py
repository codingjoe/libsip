"""Whisper-based transcription for Opus-encoded incoming SIP calls."""

from __future__ import annotations

import asyncio

import numpy as np

import whisper

from .calls import IncomingCall

__all__ = ["WhisperCall"]


class WhisperCall(IncomingCall):
    """Inbound SIP call that decodes Opus audio and transcribes it with OpenAI Whisper."""

    #: Opus clock rate (Hz) as specified by RFC 7587 §4.
    opus_sample_rate = 48000
    #: Opus frame size in samples for a standard 20 ms frame at 48 kHz.
    opus_frame_size = 960
    #: Audio buffered (in seconds) before each transcription is triggered.
    chunk_duration = 30

    def __init__(self, *args, model: str = "base", **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._whisper_model = whisper.load_model(model)
        try:
            import opuslib  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "WhisperCall requires libopus. "
                "Install the system library and run: pip install opuslib"
            ) from exc
        try:
            self._decoder = opuslib.Decoder(self.opus_sample_rate, 1)
        except Exception as exc:
            # opuslib raises a plain Exception when the libopus C library is missing.
            raise ImportError(
                "WhisperCall requires libopus. "
                "Install the system library and run: pip install opuslib"
            ) from exc
        self._pcm_buffer = np.empty(0, dtype=np.float32)

    def audio_received(self, data: bytes) -> None:
        """Decode an Opus RTP payload, buffer the PCM, and transcribe when ready."""
        pcm_bytes = self._decoder.decode(data, self.opus_frame_size)
        pcm = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        # Resample from 48 kHz to 16 kHz by taking every third sample.
        self._pcm_buffer = np.append(self._pcm_buffer, pcm[::3])
        chunk_samples = whisper.audio.SAMPLE_RATE * self.chunk_duration
        if len(self._pcm_buffer) >= chunk_samples:
            asyncio.create_task(self._transcribe_chunk())

    async def _transcribe_chunk(self) -> None:
        chunk_samples = whisper.audio.SAMPLE_RATE * self.chunk_duration
        chunk = self._pcm_buffer[:chunk_samples]
        self._pcm_buffer = self._pcm_buffer[chunk_samples:]
        loop = asyncio.get_running_loop()
        text = await loop.run_in_executor(None, self._run_transcription, chunk)
        self.transcription_received(text.strip())

    def _run_transcription(self, audio: np.ndarray) -> str:
        return self._whisper_model.transcribe(audio)["text"]

    def transcription_received(self, text: str) -> None:
        """Handle a transcription result. Override in subclasses."""
        return NotImplemented
