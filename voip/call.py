"""Backward-compatibility re-export of `RTPCall` as `Call`.

.. deprecated::
    Import `RTPCall` from `voip.rtp` directly.
"""

from voip.rtp import RTPCall as Call

__all__ = ["Call"]
