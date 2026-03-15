# Call Handling

The `voip.rtp` module provides `RTPCall`, the base class for all call leg handlers.

## RTPCall

`RTPCall` is the entry point for implementing custom call logic. Override `packet_received`
to process incoming media and use `send_packet` to transmit outbound media.

::: voip.rtp.RTPCall

## Audio Handling

::: voip.audio.AudioCall

## AI Calls

::: voip.ai.TranscribeCall

::: voip.ai.AgentCall
