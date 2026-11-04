"""Procedural, memory-only effect sounds; playback never blocks the camera.

CPython's winsound rejects SND_MEMORY | SND_ASYNC. A single daemon worker uses
SND_MEMORY | SND_SYNC instead, giving the caller asynchronous playback without
creating temporary audio files. On non-Windows systems the effect is silent.
"""
from __future__ import annotations

import io
import math
import queue
import threading
import wave

import numpy as np

try:
    import winsound
except ImportError:
    winsound = None

SAMPLE_RATE = 44100


def _wav(samples: np.ndarray) -> bytes:
    values = np.nan_to_num(samples, nan=0.0, posinf=0.0, neginf=0.0)
    pcm = (np.clip(values, -1.0, 1.0) * 32767.0).astype('<i2')
    output = io.BytesIO()
    with wave.open(output, 'wb') as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(SAMPLE_RATE)
        stream.writeframes(pcm.tobytes())
    return output.getvalue()


def synthesize_goku_teleport() -> bytes:
    t = np.arange(round(0.22 * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    # Integrating a linear frequency ramp produces a continuous-phase chirp.
    phase = 2 * math.pi * (1800 * t + (220 - 1800) * t * t / (2 * 0.22))
    envelope = np.exp(-8.0 * t) * np.minimum(1.0, t / 0.003)
    tail = np.minimum(1.0, np.maximum(0.0, (0.22 - t) / 0.015))
    pop = 0.30 * np.exp(-t * 600) * np.sin(2 * math.pi * 2600 * t)
    return _wav((0.7 * np.sin(phase) * envelope + pop) * tail)


def synthesize_thanos_snap() -> bytes:
    t = np.arange(round(1.5 * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    rng = np.random.default_rng(2027)
    noise = rng.uniform(-1, 1, len(t))
    # Running average is a cheap low-pass filter; no SciPy or file is needed.
    wind = np.convolve(noise, np.ones(19) / 19.0, mode='same')
    wind *= (1.0 - np.exp(-22 * t)) * np.exp(-2.3 * t)
    click = np.exp(-180 * t) * (0.72 * np.sin(2 * math.pi * 3800 * t)
                                + 0.28 * np.sin(2 * math.pi * 6200 * t))
    tail = np.minimum(1.0, np.maximum(0.0, (1.5 - t) / 0.04))
    return _wav((0.9 * click + 0.55 * wind) * tail)


def synthesize_naruto_smoke() -> bytes:
    t = np.arange(round(0.55 * SAMPLE_RATE), dtype=np.float64) / SAMPLE_RATE
    rng = np.random.default_rng(892)
    noise = rng.uniform(-1, 1, len(t))
    low = np.convolve(noise, np.ones(29) / 29.0, mode='same')
    boom = np.sin(2 * math.pi * (125 * t - 55 * t * t)) * np.exp(-20 * t)
    env = (1.0 - np.exp(-100 * t)) * np.exp(-7 * t)
    tail = np.minimum(1.0, np.maximum(0.0, (0.55 - t) / 0.025))
    return _wav((0.35 * boom + 1.9 * low * env) * tail)


class ProceduralSoundPlayer:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled and winsound is not None
        self.pending: queue.Queue[bytes | None] = queue.Queue(maxsize=2)
        self.cache: dict[str, bytes] = {}
        if self.enabled:
            self.cache = {'goku': synthesize_goku_teleport(),
                          'thanos': synthesize_thanos_snap(),
                          'naruto': synthesize_naruto_smoke()}
        self.thread: threading.Thread | None = None
        self.lock = threading.Lock()
        self.closed = False
        if enabled and winsound is None:
            print('[Audio] winsound is unavailable on this OS; continuing silently.')

    def _run(self) -> None:
        while True:
            data = self.pending.get()
            try:
                if data is None:
                    return
                try:
                    winsound.PlaySound(data, winsound.SND_MEMORY | winsound.SND_SYNC)
                except (RuntimeError, OSError, TypeError) as error:
                    print(f'[Audio] Playback unavailable: {error}')
            finally:
                self.pending.task_done()

    def play(self, effect: str) -> None:
        if not self.enabled or self.closed:
            return
        generators = {'goku': synthesize_goku_teleport,
                      'thanos': synthesize_thanos_snap,
                      'naruto': synthesize_naruto_smoke}
        if effect not in generators:
            return
        # Cache at startup rather than allocate audio in the video render loop.
        with self.lock:
            if self.thread is None:
                self.thread = threading.Thread(target=self._run, name='vanish-audio', daemon=True)
                self.thread.start()
        try:
            self.pending.put_nowait(self.cache[effect])
        except queue.Full:
            # Never stall video; discard this sound rather than queue stale cues.
            pass

    def close(self) -> None:
        self.closed = True
        if self.thread is not None:
            try:
                self.pending.put_nowait(None)
            except queue.Full:
                try:
                    self.pending.get_nowait()
                    self.pending.task_done()
                    self.pending.put_nowait(None)
                except queue.Empty:
                    pass


_default_player = ProceduralSoundPlayer(enabled=False)


def _default() -> ProceduralSoundPlayer:
    global _default_player
    if not _default_player.enabled and not _default_player.closed and winsound is not None:
        _default_player = ProceduralSoundPlayer()
    return _default_player


def play_goku_teleport_sound() -> None:
    _default().play('goku')


def play_thanos_snap_sound() -> None:
    _default().play('thanos')


def play_naruto_smoke_sound() -> None:
    _default().play('naruto')
