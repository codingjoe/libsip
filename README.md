<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/codingjoe/VoIP/raw/main/docs/images/logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/codingjoe/VoIP/raw/main/docs/images/logo-light.svg">
    <img alt="Python VoIP" src="https://github.com/codingjoe/VoIP/raw/main/docs/images/logo-light.svg">
  </picture>
<br>
  <a href="https://github.com/codingjoe">Documentation</a> |
  <a href="https://github.com/codingjoe/VoIP/issues/new/choose">Issues</a> |
  <a href="https://github.com/codingjoe/VoIP/releases">Changelog</a> |
  <a href="https://github.com/sponsors/codingjoe">Funding</a> ♥
</p>

# Python VoIP library

> [!WARNING]
> This library is in early development and may contain breaking changes. Use with caution.

Python asyncio library for SIP telephony ([RFC 3261](https://tools.ietf.org/html/rfc3261)).

All signalling uses **SIP over TLS** (SIPS, RFC 3261 §26) and all media is
protected with **SRTP** ([RFC 3711](https://tools.ietf.org/html/rfc3711))
using the `AES_CM_128_HMAC_SHA1_80` cipher suite with SDES key exchange
([RFC 4568](https://tools.ietf.org/html/rfc4568)).

## Setup

```console
pip install voip[audio,cli,pygments]
```

## Usage

### CLI

Answer calls and transcribe them live from the terminal:

```console
voip sip transcribe sips:alice@sip.example.com --password secret
```

### Python API

Subclass `WhisperCall` and override `transcription_received` to handle results.
Pass it as `call_class` when answering an incoming call:

```python
import asyncio
import ssl
from voip.audio import WhisperCall
from voip.sip.protocol import SIP


class MyCall(WhisperCall):
    def transcription_received(self, text: str) -> None:
        print(f"[{self.caller}] {text}")


class MySession(SIP):
    def call_received(self, request) -> None:
        asyncio.create_task(self.answer(request=request, call_class=MyCall))


async def main():
    loop = asyncio.get_running_loop()
    ssl_context = ssl.create_default_context()
    await loop.create_connection(
        lambda: MySession(
            aor="sips:alice@example.com",
            username="alice",
            password="secret",
        ),
        host="sip.example.com",
        port=5061,
        ssl=ssl_context,
    )
    await asyncio.Future()


asyncio.run(main())
```

For raw audio access without transcription, subclass `AudioCall` and override
`audio_received(self, audio: np.ndarray)` instead.

## Roadmap

| RFC | Title | Status |
|-----|-------|--------|
| [RFC 3261](https://datatracker.ietf.org/doc/html/rfc3261) | SIP: Session Initiation Protocol | 🚧 Partial |
| [RFC 3824](https://datatracker.ietf.org/doc/html/rfc3824) | Using E.164 Numbers with SIP | ❌ Planned |
| [RFC 3966](https://datatracker.ietf.org/doc/html/rfc3966) | The tel URI for Telephone Numbers | ❌ Planned |
| [RFC 6116](https://datatracker.ietf.org/doc/html/rfc6116) | The E.164 to URI DDDS Application (ENUM) | ❌ Planned |
| [RFC 4733](https://datatracker.ietf.org/doc/html/rfc4733) | RTP Payload for DTMF Digits, Telephony Tones, and Telephony Signals | ❌ Planned |
| [RFC 2805](https://datatracker.ietf.org/doc/html/rfc2805) | Media Gateway Control Protocol (MGCP) Architecture and Requirements | ❌ Planned |
| [RFC 3435](https://datatracker.ietf.org/doc/html/rfc3435) | Media Gateway Control Protocol (MGCP) Version 1.0 | ❌ Planned |
| [RFC 3660](https://datatracker.ietf.org/doc/html/rfc3660) | Basic Media Gateway Control Protocol (MGCP) Packages | ❌ Planned |
| [RFC 3661](https://datatracker.ietf.org/doc/html/rfc3661) | Media Gateway Control Protocol (MGCP) Return Code Usage | ❌ Planned |
| [RFC 3991](https://datatracker.ietf.org/doc/html/rfc3991) | MGCP Redirect and Reset Package | ❌ Planned |
| [RFC 6230](https://datatracker.ietf.org/doc/html/rfc6230) | Media Control Channel Framework | ❌ Planned |
| [RFC 6231](https://datatracker.ietf.org/doc/html/rfc6231) | An Interactive Voice Response (IVR) Control Package for the Media Control Channel Framework | ❌ Planned |
| [RFC 4458](https://datatracker.ietf.org/doc/html/rfc4458) | SIP URIs for Applications such as Voicemail and IVR | ❌ Planned |
| [RFC 3880](https://datatracker.ietf.org/doc/html/rfc3880) | Call Processing Language (CPL) | ❌ Planned |
| [RFC 3801](https://datatracker.ietf.org/doc/html/rfc3801) | Voice Profile for Internet Mail – version 2 (VPIMv2) | ❌ Planned |
| [RFC 4239](https://datatracker.ietf.org/doc/html/rfc4239) | Internet Voice Messaging (IVM) | ❌ Planned |
| [RFC 2871](https://datatracker.ietf.org/doc/html/rfc2871) | A Framework for Telephony Routing over IP | ❌ Planned |
| [RFC 3219](https://datatracker.ietf.org/doc/html/rfc3219) | Telephony Routing over IP (TRIP) | ❌ Planned |
| [RFC 5115](https://datatracker.ietf.org/doc/html/rfc5115) | Telephony Routing over IP (TRIP) Attribute for Resource Priority | ❌ Planned |
