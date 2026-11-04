"""Two-hand trigger and smoke/log compositing for body substitution."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import sys


EFFECT_SECONDS = 1.45
def asset_path(name: str) -> Path:
    roots = [Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent,
             Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))]
    for root in roots:
        path = root / "assets" / name
        if path.is_file():
            return path
    return roots[0] / "assets" / name


LOG_PATH = asset_path("substitution_log.png")


def substitution_anchor(mask: np.ndarray) -> tuple[int, int, int]:
    """Place the log at the person's torso instead of at a fixed screen spot."""
    height, width = mask.shape
    binary = (mask > 0.45).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if count <= 1:
        return width // 2, height // 2, int(height * 0.57)
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, _ = stats[largest]
    size = int(np.clip(h * 0.58, height * 0.24, height * 0.70))
    center_x = int(np.clip(x + w * 0.5, size // 3, width - size // 3))
    center_y = int(np.clip(y + h * 0.52, size // 3, height - size // 3))
    return center_x, center_y, size


def load_log() -> np.ndarray:
    if not LOG_PATH.is_file():
        print("[Naruto] Missing substitution_log.png; using procedural log fallback.")
        sprite = np.zeros((300, 150, 4), dtype=np.uint8)
        cv2.rectangle(sprite, (25, 40), (125, 255), (49, 80, 119, 255), -1)
        cv2.ellipse(sprite, (75, 255), (50, 21), 0, 0, 180, (45, 70, 104, 255), -1)
        cv2.ellipse(sprite, (75, 40), (50, 24), 0, 0, 360, (127, 172, 205, 255), -1)
        for radius in (11, 25, 40):
            cv2.ellipse(sprite, (75, 40), (radius, max(4, radius // 2)), 0, 0, 360,
                        (75, 114, 150, 255), 2)
        for x in (35, 53, 92, 116):
            cv2.line(sprite, (x, 65), (x - 5, 244), (35, 58, 89, 255), 3)
        return sprite
    sprite = cv2.imread(str(LOG_PATH), cv2.IMREAD_UNCHANGED)
    if sprite is None or sprite.ndim != 3 or sprite.shape[2] != 4:
        raise RuntimeError(f"Sprite del tronco non disponibile: {LOG_PATH}")
    return sprite


def overlay_log(frame: np.ndarray, sprite: np.ndarray, anchor: tuple[int, int, int],
                opacity: float = 1.0, scale: float = 1.0,
                cache: dict[int, np.ndarray] | None = None) -> None:
    if opacity <= 0:
        return
    cx, cy, target_height = anchor
    height = max(1, int(target_height * scale))
    width = max(1, int(height * sprite.shape[1] / sprite.shape[0]))
    sized = cache.get(height) if cache is not None else None
    if sized is None:
        sized = cv2.resize(sprite, (width, height), interpolation=cv2.INTER_AREA)
        if cache is not None:
            cache[height] = sized
    x1, y1 = cx - width // 2, cy - height // 2
    x2, y2 = x1 + width, y1 + height
    left, top = max(0, x1), max(0, y1)
    right, bottom = min(frame.shape[1], x2), min(frame.shape[0], y2)
    if left >= right or top >= bottom:
        return
    part = sized[top - y1:bottom - y1, left - x1:right - x1]
    alpha = (part[:, :, 3:4].astype(np.float32) / 255.0) * opacity
    destination = frame[top:bottom, left:right]
    destination[:] = np.clip(
        part[:, :, :3].astype(np.float32) * alpha
        + destination.astype(np.float32) * (1.0 - alpha),
        0, 255,
    ).astype(np.uint8)


def landed_log_anchor(anchor: tuple[int, int, int], frame_height: int) -> tuple[int, int, int]:
    cx, cy, size = anchor
    return cx, max(cy, frame_height - round(size * 0.48)), size


def falling_log_anchor(anchor: tuple[int, int, int], frame_height: int,
                       age: float) -> tuple[int, int, int]:
    """Accelerate downwards, touch the frame floor, then settle with a small bounce."""
    cx, cy, size = anchor
    floor_y = landed_log_anchor(anchor, frame_height)[1]
    distance = floor_y - cy
    if distance <= 0 or age <= 0.18:
        return anchor
    elapsed = age - 0.18
    gravity = frame_height * 3.2
    fall = gravity * elapsed * elapsed * 0.5
    if fall < distance:
        return cx, cy + round(fall), size
    landed_at = math.sqrt(2.0 * distance / gravity)
    bounce_age = elapsed - landed_at
    bounce = size * 0.075 * math.exp(-5.0 * bounce_age) * abs(math.sin(11.0 * bounce_age))
    return cx, floor_y - round(bounce), size


def _smoke_layer(frame: np.ndarray, anchor: tuple[int, int, int],
                 age: float, foreground: bool) -> None:
    if age >= EFFECT_SECONDS:
        return
    cx, cy, size = anchor
    rng = np.random.default_rng(9127 if foreground else 1934)
    reveal = min(1.0, age / 0.16)
    fade = max(0.0, 1.0 - max(0.0, age - 0.33) / 1.12)
    if fade <= 0:
        return
    # Smoke only occupies the area around the log; avoid allocating and
    # blending a full-size RGBA video frame twice per animation frame.
    margin = int(size * 1.35)
    left, top = max(0, cx - margin), max(0, cy - margin)
    right = min(frame.shape[1], cx + margin + 1)
    bottom = min(frame.shape[0], cy + margin + 1)
    if left >= right or top >= bottom:
        return
    layer = np.zeros((bottom - top, right - left, 4), dtype=np.uint8)
    radius = size * (0.27 + 0.54 * min(age, 1.0))
    for _ in range(22 if foreground else 18):
        angle = rng.uniform(0, 2 * math.pi)
        distance = radius * rng.uniform(0.25, 1.0)
        x = int(cx + math.cos(angle) * distance) - left
        y = int(cy + math.sin(angle) * distance * 0.63 - age * size * 0.13) - top
        puff = max(2, int(size * rng.uniform(0.09, 0.20) * (0.7 + age * 0.7)))
        color = (229, 235, 235, 190) if foreground else (152, 162, 166, 125)
        cv2.ellipse(layer, (x, y), (puff, int(puff * 0.68)),
                    rng.uniform(-25, 25), 0, 360, color, -1, cv2.LINE_AA)
    # Long white winding smoke ribbons echo the supplied anime reference.
    for direction in (-1, 1):
        center = (int(cx + direction * size * 0.13) - left,
                  int(cy - size * 0.13 - age * size * 0.09) - top)
        axes = (max(2, int(size * (0.30 + age * 0.27))),
                max(2, int(size * (0.09 + age * 0.12))))
        cv2.ellipse(layer, center, axes, direction * 15, 20, 230,
                    (244, 247, 245, 220), max(2, int(size * 0.025)), cv2.LINE_AA)
    occupied = cv2.findNonZero(layer[:, :, 3])
    if occupied is None:
        return
    x, y, width, height = cv2.boundingRect(occupied)
    destination = frame[top + y:top + y + height, left + x:left + x + width]
    layer = layer[y:y + height, x:x + width]
    alpha = (layer[:, :, 3:4].astype(np.float32) / 255.0) * reveal * fade
    destination[:] = np.clip(
        layer[:, :, :3].astype(np.float32) * alpha
        + destination.astype(np.float32) * (1.0 - alpha),
        0, 255,
    ).astype(np.uint8)


def render_substitution(frame: np.ndarray, background: np.ndarray, mask: np.ndarray,
                        sprite: np.ndarray, anchor: tuple[int, int, int], age: float,
                        log_cache: dict[int, np.ndarray] | None = None) -> np.ndarray:
    """Quick body replacement, expanding smoke, then a persistent wooden log."""
    if age >= EFFECT_SECONDS:
        output = background.copy()
        overlay_log(output, sprite, landed_log_anchor(anchor, output.shape[0]), cache=log_cache)
        return output
    if age >= 0.13:
        output = background.copy()
    else:
        person_alpha = mask * max(0.0, 1.0 - age / 0.13)
        output = (
            frame.astype(np.float32) * person_alpha[:, :, None]
            + background.astype(np.float32) * (1.0 - person_alpha[:, :, None])
        ).astype(np.uint8)
    _smoke_layer(output, anchor, age, foreground=False)
    log_opacity = min(1.0, max(0.0, (age - 0.05) / 0.15))
    moving_anchor = falling_log_anchor(anchor, output.shape[0], age)
    overlay_log(output, sprite, moving_anchor, log_opacity,
                scale=1.15 - min(age, 0.25) * 0.6, cache=log_cache)
    _smoke_layer(output, anchor, age, foreground=True)
    return output


def render_reappearance(frame: np.ndarray, background: np.ndarray, mask: np.ndarray,
                        sprite: np.ndarray, anchor: tuple[int, int, int], age: float,
                        log_cache: dict[int, np.ndarray] | None = None) -> np.ndarray:
    """Smoke hides removal of the old log and reveals the current live person."""
    if age >= EFFECT_SECONDS:
        return frame.copy()
    alpha = mask * float(np.clip((age - 0.04) / 0.24, 0, 1))
    output = (frame.astype(np.float32) * alpha[:, :, None]
              + background.astype(np.float32) * (1 - alpha[:, :, None])).astype(np.uint8)
    landed = landed_log_anchor(anchor, frame.shape[0])
    _smoke_layer(output, landed, age, foreground=False)
    overlay_log(output, sprite, landed, opacity=max(0, 1 - age / 0.14), cache=log_cache)
    _smoke_layer(output, landed, age, foreground=True)
    _smoke_layer(output, substitution_anchor(mask), age, foreground=True)
    return output

# All renderers are deliberately kept in this file. They never alter their inputs.

def alpha_composite(person: np.ndarray, background: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
    return np.clip(person.astype(np.float32) * alpha
                   + background.astype(np.float32) * (1.0 - alpha), 0, 255).astype(np.uint8)


class NormalFade:
    duration = 1.5

    def render(self, live: np.ndarray, background: np.ndarray, mask: np.ndarray,
               age: float, appearing: bool) -> np.ndarray:
        progress = float(np.clip(age / self.duration, 0, 1))
        progress = progress * progress * (3 - 2 * progress)
        alpha = progress if appearing else 1.0 - progress
        return alpha_composite(live, background, mask * alpha)


class GokuTeleport:
    duration = 0.25
    sound_time = 0.10

    def __init__(self, frame: np.ndarray, mask: np.ndarray):
        self.frame = frame.copy()
        self.mask = np.clip(mask, 0, 1).astype(np.float32).copy()
        self.binary = (mask > 0.4).astype(np.uint8)
        self.contours, _ = cv2.findContours(self.binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ys, xs = np.nonzero(self.binary)
        if xs.size:
            self.bounds = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
        else:
            self.bounds = (0, 0, frame.shape[1], frame.shape[0])
        self.outline = np.zeros(self.binary.shape, np.uint8)
        cv2.drawContours(self.outline, self.contours, -1, 255, 2, cv2.LINE_AA)
        self.halo = cv2.GaussianBlur(self.outline, (0, 0), 5.0).astype(np.float32) / 255.0
        self.rng = np.random.default_rng(706)
        h = max(1, self.bounds[3] - self.bounds[1])
        self.stripes = [(int(self.rng.uniform(0.08, 0.90) * h),
                         max(2, int(self.rng.uniform(0.015, 0.05) * h)),
                         int(self.rng.choice([-1, 1]) * self.rng.uniform(4, 14)))
                        for _ in range(5)]

    def render(self, background: np.ndarray, age: float, appearing: bool = False) -> np.ndarray:
        p = float(np.clip(age / self.duration, 0, 1))
        if appearing:
            p = 1.0 - p
        x1, y1, x2, y2 = self.bounds
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if p >= 0.4:
            # Optical flare contracts at the position where the person was.
            output = background.copy()
            decay = max(0.0, 1.0 - (p - 0.4) / 0.6)
            radius = int(max(1, min(x2 - x1, y2 - y1) * 0.20 * decay))
            if decay > 0:
                layer = np.zeros_like(output)
                cv2.circle(layer, (cx, cy), radius, (255, 220, 0), 2, cv2.LINE_AA)
                cv2.line(layer, (cx - radius * 2, cy), (cx + radius * 2, cy), (255, 255, 255), 1, cv2.LINE_AA)
                output = cv2.addWeighted(output, 1.0, layer, decay, 0)
            return output
        output = alpha_composite(self.frame, background, self.mask)
        # Offset horizontal slices of the captured silhouette, not the entire background.
        shifted = self.frame.copy()
        for index, (relative_y, thickness, offset) in enumerate(self.stripes):
            top = min(background.shape[0] - 1, max(0, y1 + relative_y))
            bottom = min(background.shape[0], top + thickness)
            amount = offset if (int(age * 40) + index) % 2 else -offset
            shifted[top:bottom] = np.roll(self.frame[top:bottom], amount, axis=1)
            shifted_mask = np.roll(self.binary[top:bottom], amount, axis=1).astype(bool)
            output[top:bottom][shifted_mask] = shifted[top:bottom][shifted_mask]
            # Bright horizontal speedlines stay clipped inside the silhouette.
            line_y = min(bottom - 1, top + thickness // 2)
            line = self.binary[line_y].astype(bool)
            output[line_y, line] = (255, 235, 120) if index % 2 else (255, 255, 255)
        strength = float(np.clip(0.65 + 0.35 * math.sin(age * 65), 0, 1))
        glow = (self.halo * strength * 0.8)[:, :, None]
        neon = np.array([255, 220, 0], dtype=np.float32)  # BGR: RGB (0,220,255)
        output = np.clip(output.astype(np.float32) * (1 - glow) + neon * glow, 0, 255).astype(np.uint8)
        cv2.drawContours(output, self.contours, -1, (255, 230, 35), 3, cv2.LINE_AA)
        cv2.drawContours(output, self.contours, -1, (255, 255, 255), 1, cv2.LINE_AA)
        return output


class ThanosDust:
    vanish_duration = 1.85
    appear_duration = 1.2

    def __init__(self, frame: np.ndarray, mask: np.ndarray, seed: int = 294):
        self.frame = frame.copy()
        self.mask = np.clip(mask, 0, 1).astype(np.float32).copy()
        self.height, self.width = mask.shape
        rng = np.random.default_rng(seed)
        foreground = mask > 0.4
        yy, xx = np.indices(mask.shape, dtype=np.float32)
        ys, xs = np.nonzero(foreground)
        top, bottom = (int(ys.min()), int(ys.max())) if ys.size else (0, 1)
        # Body erodes top-to-bottom; this same field schedules particle emission.
        self.delay = np.clip((yy - top) / max(1, bottom - top), 0, 1) * 0.82
        self.delay += rng.random(mask.shape, dtype=np.float32) * 0.055
        if xs.size:
            count = min(3000, max(1500, xs.size // 18), xs.size)
            chosen = rng.choice(xs.size, count, replace=False)
            px, py = xs[chosen], ys[chosen]
        else:
            px, py = xs, ys
        self.origin = np.column_stack((px, py)).astype(np.float32)
        self.spawn_delay = self.delay[py, px]
        source = frame[py, px].astype(np.float32)
        ash = np.array([102, 116, 128], dtype=np.float32)
        ember = np.array([43, 123, 225], dtype=np.float32)
        colors = source * 0.76 + ash * 0.24
        warm = rng.random(len(px)) < 0.09
        colors[warm] = colors[warm] * 0.6 + ember * 0.4
        self.colors = np.clip(colors, 0, 255)
        scale = max(0.4, self.width / 960.0)
        self.life = rng.uniform(0.65, 0.94, len(px)).astype(np.float32)
        # Dust exits towards the right edge; reversing the same paths gathers it
        # back from that edge into the newly captured silhouette.
        horizontal = (self.width - px + rng.uniform(0, self.width * 0.08, len(px))) / self.life
        self.velocity = np.column_stack((horizontal, rng.uniform(-100, -24, len(px)) * scale)).astype(np.float32)
        self.phase = rng.uniform(0, 2 * math.pi, len(px)).astype(np.float32)
        self.turbulence = rng.uniform(8, 25, len(px)).astype(np.float32) * scale
        self.offsets = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int32)

    @property
    def particle_count(self) -> int:
        return len(self.origin)

    def render(self, background: np.ndarray, age: float, appearing: bool = False) -> np.ndarray:
        if appearing:
            dust_age = (1.0 - float(np.clip(age / self.appear_duration, 0, 1))) * self.vanish_duration
        else:
            dust_age = max(0.0, min(self.vanish_duration, age))
        if dust_age >= self.vanish_duration:
            return background.copy()
        survival = np.clip(1.0 - (dust_age - self.delay) / 0.10, 0, 1)
        output = alpha_composite(self.frame, background, self.mask * survival)
        elapsed = dust_age - self.spawn_delay
        active = (elapsed >= 0) & (elapsed < self.life)
        t = elapsed[active]
        position = self.origin[active] + self.velocity[active] * t[:, None]
        position[:, 0] += self.turbulence[active] * np.sin(t * 9 + self.phase[active]) * t
        position[:, 1] += self.turbulence[active] * np.cos(t * 7 + self.phase[active]) * t * 0.4
        alpha = (1.0 - t / self.life[active]) ** 1.3
        pixels = (np.rint(position).astype(np.int32)[:, None, :] + self.offsets).reshape(-1, 2)
        colors = np.repeat(self.colors[active], len(self.offsets), axis=0)
        alphas = np.repeat(alpha, len(self.offsets))
        valid = ((pixels[:, 0] >= 0) & (pixels[:, 0] < self.width)
                 & (pixels[:, 1] >= 0) & (pixels[:, 1] < self.height))
        pixels, colors, alphas = pixels[valid], colors[valid], alphas[valid]
        flat, first = np.unique(pixels[:, 1] * self.width + pixels[:, 0], return_index=True)
        data = output.reshape(-1, 3)
        a = alphas[first, None]
        data[flat] = np.clip(colors[first] * a + data[flat] * (1 - a), 0, 255).astype(np.uint8)
        return output
