"""Tests for Whisper-based audio transcription."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

np = pytest.importorskip("numpy")
av = pytest.importorskip("av")
pytest.importorskip("faster_whisper")

from voip.audio import WhisperCall, _build_ogg_opus  # noqa: E402
from voip.rtp import RTP, RTPPayloadType  # noqa: E402
from voip.sdp.types import MediaDescription, RTPPayloadFormat  # noqa: E402


def _make_media(fmt: str, rtpmap: str | None = None) -> MediaDescription:
    """Build a single-codec MediaDescription for use in tests."""
    if rtpmap:
        payload_format = RTPPayloadFormat.parse(rtpmap)
    else:
        payload_format = RTPPayloadFormat(payload_type=int(fmt))
    return MediaDescription(
        media="audio", port=0, proto="RTP/AVP", fmt=[payload_format]
    )


OPUS_MEDIA = _make_media("111", "111 opus/48000/2")
PCMA_MEDIA = _make_media("8", "8 PCMA/8000")
PCMU_MEDIA = _make_media("0")  # static PT, no rtpmap
G722_MEDIA = _make_media("9", "9 G722/8000")


def make_whisper_call(
    model_mock: MagicMock, call_class=None, media: MediaDescription | None = None
) -> WhisperCall:
    """Return a WhisperCall with a mocked Whisper model."""
    cls = call_class or WhisperCall
    med = media if media is not None else OPUS_MEDIA
    with patch("voip.audio.WhisperModel", return_value=model_mock):
        return cls(caller="sip:bob@biloxi.com", media=med)


class TestWhisperCall:
    def test_whisper_call__is_rtp(self):
        """WhisperCall is a subclass of RTP."""
        assert issubclass(WhisperCall, RTP)

    def test_class_attrs__chunk_duration(self):
        """chunk_duration controls how many seconds are buffered before transcription."""
        assert WhisperCall.chunk_duration == 5

    def test_init__packet_threshold__opus(self):
        """_packet_threshold is 250 for Opus with chunk_duration=5 (50 pkt/s × 5 s)."""
        call = make_whisper_call(MagicMock(), media=OPUS_MEDIA)
        assert call._packet_threshold == 250

    def test_init__packet_threshold__g722(self):
        """_packet_threshold is 250 for G.722 with chunk_duration=5 (50 pkt/s × 5 s)."""
        call = make_whisper_call(MagicMock(), media=G722_MEDIA)
        assert call._packet_threshold == 250

    def test_init__packet_threshold__pcma(self):
        """_packet_threshold is 250 for PCMA with chunk_duration=5 (50 pkt/s × 5 s)."""
        call = make_whisper_call(MagicMock(), media=PCMA_MEDIA)
        assert call._packet_threshold == 250

    def test_init__packet_threshold__pcmu(self):
        """_packet_threshold is 250 for PCMU with chunk_duration=5 (50 pkt/s × 5 s)."""
        call = make_whisper_call(MagicMock(), media=PCMU_MEDIA)
        assert call._packet_threshold == 250

    def test_init__stores_media(self):
        """Media is stored and accessible as self.media."""
        call = make_whisper_call(MagicMock())
        assert call.media is OPUS_MEDIA

    def test_init__derives_payload_type_from_opus_media(self):
        """payload_type is 111 (Opus) when given OPUS_MEDIA."""
        call = make_whisper_call(MagicMock())
        assert call.payload_type == RTPPayloadType.OPUS

    def test_init__derives_payload_type_from_pcma_media(self):
        """payload_type is 8 (PCMA) when given PCMA_MEDIA."""
        call = make_whisper_call(MagicMock(), media=PCMA_MEDIA)
        assert call.payload_type == RTPPayloadType.PCMA

    def test_audio_received__does_not_buffer_on_whisper_call(self):
        """audio_received on WhisperCall schedules a task, not a packet buffer."""
        call = make_whisper_call(MagicMock())
        assert not hasattr(call, "_audio_packets")

    def test_audio_received__below_threshold_no_transcription(self):
        """Don't trigger transcription until the packet threshold is reached."""
        model_mock = MagicMock()
        call = make_whisper_call(model_mock)
        # Feed one packet to the base-class buffer (threshold=1500, so no emit)
        call._audio_buffer.append(b"opus_packet")
        model_mock.transcribe.assert_not_called()

    async def test_audio_received__triggers_transcription_when_buffer_full(self):
        """Transcription fires when audio_received is emitted by the base-class buffer."""
        transcriptions = []
        model_mock = MagicMock()
        seg = MagicMock()
        seg.text = "hello"
        model_mock.transcribe.return_value = ([seg], MagicMock())

        class SmallChunkCall(WhisperCall):
            chunk_duration = 1  # 1 s @ 48 kHz / 960 samples = 50 packets

            def transcription_received(self, text: str) -> None:
                transcriptions.append(text)

        call = make_whisper_call(model_mock, SmallChunkCall)
        pcm_samples = np.zeros(16000, dtype=np.float32)
        with patch.object(call, "_decode_audio", return_value=pcm_samples):
            # Simulate the base class emitting audio_received with a full chunk
            call.audio_received([b"x"] * call._packet_threshold)
            await asyncio.sleep(0.1)
        assert transcriptions == ["hello"]

    async def test_transcribe_chunk__strips_whitespace(self):
        """Strip leading and trailing whitespace from the transcription text."""
        transcriptions = []
        model_mock = MagicMock()
        seg = MagicMock()
        seg.text = "  hello world  "
        model_mock.transcribe.return_value = ([seg], MagicMock())

        class Capture(WhisperCall):
            def transcription_received(self, text: str) -> None:
                transcriptions.append(text)

        call = make_whisper_call(model_mock, Capture)
        with patch.object(
            call, "_decode_audio", return_value=np.zeros(16000, dtype=np.float32)
        ):
            await call._transcribe_chunk([b"x"])
        assert transcriptions == ["hello world"]

    def test_run_transcription__passes_numpy_array_directly(self):
        """Pass a numpy float32 array to the Whisper model without file I/O."""
        model_mock = MagicMock()
        seg = MagicMock()
        seg.text = "test"
        model_mock.transcribe.return_value = ([seg], MagicMock())
        call = make_whisper_call(model_mock)
        audio = np.zeros(16000, dtype=np.float32)
        assert call._run_transcription(audio) == "test"
        model_mock.transcribe.assert_called_once_with(audio)

    def test_run_transcription__no_file_written(self):
        """The transcription path must not write any files to disk."""
        model_mock = MagicMock()
        seg = MagicMock()
        seg.text = ""
        model_mock.transcribe.return_value = ([seg], MagicMock())
        call = make_whisper_call(model_mock)
        with patch(
            "builtins.open", side_effect=AssertionError("open() must not be called")
        ):
            call._run_transcription(np.zeros(16000, dtype=np.float32))

    def test_decode_via_av__opus(self):
        """Pipe Ogg Opus data through PyAV and return a float32 PCM array."""
        model_mock = MagicMock()
        call = make_whisper_call(model_mock)
        pcm_array = np.zeros(16000, dtype=np.float32)
        mock_frame = MagicMock()
        mock_frame.to_ndarray.return_value = pcm_array
        mock_resampled = [mock_frame]
        with patch("voip.audio.av") as mock_av:
            mock_resampler = MagicMock()
            mock_resampler.resample.side_effect = [mock_resampled, []]
            mock_av.audio.resampler.AudioResampler.return_value = mock_resampler
            mock_container = MagicMock()
            mock_container.__enter__ = lambda s: s
            mock_container.__exit__ = MagicMock(return_value=False)
            mock_container.decode.return_value = [MagicMock()]
            mock_av.open.return_value = mock_container
            mock_av.AVError = Exception
            result = call._decode_via_av(
                b"fake_ogg_data", input_format="ogg", input_sample_rate=None
            )
        mock_av.open.assert_called_once()
        assert result.dtype == np.float32

    def test_decode_via_av__raises_on_av_error(self):
        """Raise RuntimeError when PyAV raises AVError."""
        call = make_whisper_call(MagicMock())
        with (
            patch("voip.audio.av") as mock_av,
            pytest.raises(RuntimeError, match="Audio decoding failed"),
        ):
            mock_av.AVError = RuntimeError
            mock_av.audio.resampler.AudioResampler.return_value = MagicMock()
            mock_av.open.side_effect = RuntimeError("av error")
            call._decode_via_av(b"bad_data", input_format="ogg", input_sample_rate=None)

    def test_decode_audio__opus__wraps_in_ogg(self):
        """Decode Opus packets by wrapping them in an Ogg container before calling PyAV."""
        call = make_whisper_call(MagicMock(), media=OPUS_MEDIA)
        assert call.payload_type == RTPPayloadType.OPUS
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_audio([b"pkt"])
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_format") == "ogg"

    def test_decode_audio__pcma__uses_alaw_format(self):
        """Decode PCMA packets using the alaw PyAV input format."""
        call = make_whisper_call(MagicMock(), media=PCMA_MEDIA)
        assert call.payload_type == RTPPayloadType.PCMA
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_audio([b"pkt"])
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_format") == "alaw"

    def test_decode_audio__pcmu__uses_mulaw_format(self):
        """Decode PCMU packets using the mulaw PyAV input format."""
        call = make_whisper_call(MagicMock(), media=PCMU_MEDIA)
        assert call.payload_type == RTPPayloadType.PCMU
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_audio([b"pkt"])
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_format") == "mulaw"

    def test_decode_audio__g722__uses_g722_format(self):
        """Decode G.722 packets using the g722 PyAV input format."""
        call = make_whisper_call(MagicMock(), media=G722_MEDIA)
        assert call.payload_type == RTPPayloadType.G722
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_audio([b"pkt"])
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_format") == "g722"

    def test_decode_audio__unknown__raises(self):
        """Raise NotImplementedError for unsupported encoding names."""
        unknown_media = _make_media("99")
        call = make_whisper_call(MagicMock(), media=unknown_media)
        with pytest.raises(NotImplementedError, match="Unsupported"):
            call._decode_audio([b"pkt"])

    def test_decode_audio__uses_sample_rate_from_media(self):
        """Pass the sample rate from the MediaDescription to _decode_via_av."""
        wideband_pcma = _make_media("8", "8 PCMA/16000")
        call = make_whisper_call(MagicMock(), media=wideband_pcma)
        assert call.sample_rate == 16000
        with patch.object(
            call, "_decode_via_av", return_value=np.zeros(16000, dtype=np.float32)
        ) as mock_decode:
            call._decode_audio([b"pkt"])
        kwargs = mock_decode.call_args[1]
        assert kwargs.get("input_sample_rate") == 16000

    def test_audio_received__logs_debug(self, caplog):
        """Log a debug message for each received audio frame."""
        import logging

        call = make_whisper_call(MagicMock())
        with caplog.at_level(logging.DEBUG, logger="voip.audio"):
            with patch("asyncio.create_task"):
                call.audio_received([b"opus_packet"])
        assert any("Audio frame" in r.message for r in caplog.records)


class TestBuildOggOpus:
    def test_build_ogg_opus__starts_with_ogg_magic(self):
        """The resulting container starts with the Ogg capture pattern 'OggS'."""
        assert _build_ogg_opus([b"packet"]).startswith(b"OggS")

    def test_build_ogg_opus__contains_opus_head(self):
        """The resulting container includes the OpusHead identification header."""
        assert b"OpusHead" in _build_ogg_opus([b"packet"])

    def test_build_ogg_opus__contains_opus_tags(self):
        """The resulting container includes the OpusTags comment header."""
        assert b"OpusTags" in _build_ogg_opus([b"packet"])

    def test_build_ogg_opus__non_empty_for_single_packet(self):
        """Produce a non-empty Ogg container for a single Opus packet."""
        assert len(_build_ogg_opus([b"x" * 100])) > 100

    def test_build_ogg_opus__empty_packets_list(self):
        """Produce a valid Ogg container even with an empty packet list."""
        result = _build_ogg_opus([])
        assert b"OggS" in result

    def test_build_ogg_opus__large_packet_uses_255_lacing(self):
        """Produce a valid Ogg container when a packet exceeds 254 bytes (lacing spans 255)."""
        large_packet = b"x" * 256
        result = _build_ogg_opus([large_packet])
        assert b"OggS" in result
        assert len(result) > 256

    def test_build_ogg_opus__multiple_pages(self):
        """Produce multiple Ogg pages when more than 50 packets are provided."""
        packets = [b"x" * 10] * 55
        result = _build_ogg_opus(packets)
        assert result.count(b"OggS") > 3
