"""
Voice alerts module.
Optional hook for text-to-speech trade notifications.

Enabled by setting ``voice_alerts: true`` in config/settings.yaml.
Uses the stdlib ``pyttsx3`` library (offline TTS, no API key needed).
Gracefully degrades — if pyttsx3 is unavailable, falls back to logger only.
"""

from typing import Optional

from src.utils.logger import logger

# Lazy import so the module loads even when pyttsx3 is not installed
_engine: Optional[object] = None
_tts_available: bool = False


def _get_tts_engine() -> Optional[object]:
    """Return a cached pyttsx3 engine, or None if unavailable."""
    global _engine, _tts_available  # noqa: PLW0603
    if _engine is not None:
        return _engine
    try:
        import pyttsx3  # type: ignore[import]
        _engine = pyttsx3.init()
        _tts_available = True
        logger.info("Voice alerts: pyttsx3 TTS engine initialised.")
        return _engine
    except (ImportError, RuntimeError) as exc:
        logger.warning(
            "Voice alerts: pyttsx3 unavailable (%s). "
            "Alerts will be logged only.",
            exc,
        )
        _tts_available = False
        return None


def speak(message: str, enabled: bool = False) -> None:
    """Speak a message aloud via TTS if enabled and pyttsx3 is available.

    Always logs the message via the structured logger regardless of
    whether TTS is enabled.

    Args:
        message: The alert text to speak / log.
        enabled: Pass ``True`` to activate TTS output.  Should be read
                 from ``config['voice_alerts']`` by the caller.
    """
    logger.info("VOICE ALERT: %s", message)

    if not enabled:
        return

    tts = _get_tts_engine()
    if tts is None:
        return

    try:
        import pyttsx3  # type: ignore[import]
        engine: pyttsx3.Engine = tts  # type: ignore[assignment]
        engine.say(message)
        engine.runAndWait()
    except RuntimeError as exc:
        # runAndWait can fail if called from a non-main thread
        logger.warning("Voice alert TTS error: %s", exc)


def alert_entry(symbol: str, side: str, price: float, enabled: bool = False) -> None:
    """Speak an order-entry alert.

    Args:
        symbol: Trading pair, e.g. ``'BTC/USDT'``.
        side: ``'buy'`` or ``'sell'``.
        price: Execution price.
        enabled: Whether TTS is active.
    """
    msg = f"Order filled: {side} {symbol} at {price:.2f}"
    speak(msg, enabled=enabled)


def alert_exit(
    symbol: str,
    reason: str,
    price: float,
    pnl: float,
    enabled: bool = False,
) -> None:
    """Speak an exit alert with PnL.

    Args:
        symbol: Trading pair.
        reason: Exit reason — one of ``'signal'``, ``'stop_loss'``,
                ``'take_profit'``, ``'risk_halt'``.
        price: Exit price.
        pnl: Net profit / loss for the trade.
        enabled: Whether TTS is active.
    """
    direction = "profit" if pnl >= 0 else "loss"
    msg = (
        f"Position closed ({reason}): {symbol} at {price:.2f}, "
        f"net {direction} {abs(pnl):.2f}"
    )
    speak(msg, enabled=enabled)


def alert_halt(reason: str, enabled: bool = False) -> None:
    """Speak a trading-halt alert.

    Args:
        reason: Human-readable reason for the halt.
        enabled: Whether TTS is active.
    """
    msg = f"Trading halted: {reason}"
    speak(msg, enabled=enabled)
