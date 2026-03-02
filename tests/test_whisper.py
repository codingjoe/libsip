"""Tests for Whisper-based audio transcription."""

import asyncio
import pathlib
import wave
from unittest.mock import MagicMock, patch

from sip.calls import IncomingCall
from sip.messages import Request
from sip.whisper import _CHUNK_BYTES, WhisperCall


def make_invite() -> Request:
    """Return an INVITE request with default headers."""
    return Request(
        method="INVITE",
        uri="sip:alice@atlanta.com",
        headers={"From": "sip:bob@biloxi.com"},
    )


def make_whisper_call(model_mock: MagicMock) -> WhisperCall:
    """Return a WhisperCall with a mocked Whisper model."""
    with patch("sip.whisper.whisper.load_model", return_value=model_mock):
        return WhisperCall(make_invite(), ("192.0.2.1", 5060), MagicMock())


class TestWhisperCall:
    def test_whisper_call__is_incoming_call(self):
        """WhisperCall is a subclass of IncomingCall."""
        assert issubclass(WhisperCall, IncomingCall)

    def test_handle__buffers_audio(self):
        """Accumulate audio bytes without transcribing below chunk threshold."""
        model_mock = MagicMock()
        call = make_whisper_call(model_mock)
        call.handle(b"audio")
        assert call._buffer == bytearray(b"audio")
        model_mock.transcribe.assert_not_called()

    def test_handle__triggers_transcription_at_chunk_boundary(self):
        """Schedule transcription when the buffer reaches the chunk size."""
        transcriptions = []
        model_mock = MagicMock()
        model_mock.transcribe.return_value = {"text": "hello world"}

        class Capture(WhisperCall):
            def transcription_received(self, text: str) -> None:
                transcriptions.append(text)

        async def run() -> None:
            with patch("sip.whisper.whisper.load_model", return_value=model_mock):
                call = Capture(make_invite(), ("192.0.2.1", 5060), MagicMock())
            call.handle(b"\x00" * _CHUNK_BYTES)
            await asyncio.sleep(0.1)  # allow the executor task to complete

        asyncio.run(run())
        assert transcriptions == ["hello world"]

    def test_handle__clears_transcribed_chunk_from_buffer(self):
        """Remove transcribed audio from the buffer after transcription."""
        model_mock = MagicMock()
        model_mock.transcribe.return_value = {"text": ""}

        async def run() -> None:
            with patch("sip.whisper.whisper.load_model", return_value=model_mock):
                call = WhisperCall(make_invite(), ("192.0.2.1", 5060), MagicMock())
            remainder = b"\x01\x02"
            call.handle(b"\x00" * _CHUNK_BYTES + remainder)
            await asyncio.sleep(0.1)
            assert call._buffer == bytearray(remainder)

        asyncio.run(run())

    def test_run_transcription__writes_wav_and_calls_model(self):
        """Write audio to a WAV file and pass the path to the Whisper model."""
        model_mock = MagicMock()
        model_mock.transcribe.return_value = {"text": "test"}
        call = make_whisper_call(model_mock)
        result = call._run_transcription(b"\x00" * 100)
        assert result == "test"
        model_mock.transcribe.assert_called_once()
        path_arg = model_mock.transcribe.call_args[0][0]
        assert pathlib.Path(path_arg).suffix == ".wav"

    def test_run_transcription__wav_is_valid(self):
        """Write a readable WAV file with correct metadata."""
        model_mock = MagicMock()
        model_mock.transcribe.return_value = {"text": ""}
        call = make_whisper_call(model_mock)

        # Capture the wav path before it's deleted
        written_paths = []

        def capture(path):
            written_paths.append(path)
            # Read the wav before returning so we can inspect it
            with wave.open(path, "rb") as wav:
                written_paths.append(
                    (wav.getnchannels(), wav.getsampwidth(), wav.getframerate())
                )
            return {"text": ""}

        model_mock.transcribe.side_effect = capture
        call._run_transcription(b"\x00" * 80)
        assert written_paths[1] == (1, 1, 8000)

    def test_transcription_received__returns_not_implemented(self):
        """Return NotImplemented for unhandled transcriptions."""
        model_mock = MagicMock()
        call = make_whisper_call(model_mock)
        assert call.transcription_received("hello") is NotImplemented

    def test_transcription_received__strips_whitespace(self):
        """Strip leading and trailing whitespace from the transcription."""
        transcriptions = []
        model_mock = MagicMock()
        model_mock.transcribe.return_value = {"text": "  hello world  "}

        class Capture(WhisperCall):
            def transcription_received(self, text: str) -> None:
                transcriptions.append(text)

        async def run() -> None:
            with patch("sip.whisper.whisper.load_model", return_value=model_mock):
                call = Capture(make_invite(), ("192.0.2.1", 5060), MagicMock())
            call.handle(b"\x00" * _CHUNK_BYTES)
            await asyncio.sleep(0.1)

        asyncio.run(run())
        assert transcriptions == ["hello world"]
