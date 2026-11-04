"""Four-effect webcam: continuous MediaPipe input and gesture-based toggling.

Controls in the preview window: 1 Normal, 2 Naruto, 3 Thanos, 4 Goku,
V manual toggle, R immediately visible, Space recapture background, F fullscreen,
Q/Escape quit. No labels are drawn onto the video sent to the virtual camera.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import platform
import sys
import time

import cv2
import numpy as np

from naruto_effect import (EFFECT_SECONDS, GokuTeleport, NormalFade, ThanosDust,
                           landed_log_anchor, load_log, overlay_log,
                           render_reappearance, render_substitution, substitution_anchor)
from vanish_sound import ProceduralSoundPlayer

WINDOW_NAME = 'Webcam - Anteprima'
MODES = ('normal', 'naruto', 'thanos', 'goku')
BACKGROUND_SECONDS = 4.0
SAMPLE_SECONDS = 0.55
COOLDOWN_SECONDS = 1.5
RELEASE_SECONDS = 0.8
MAX_FRAME_GAP = 0.30
MODE_FRAMES = {'normal': 5, 'naruto': 3, 'thanos': 4, 'goku': 3}
PROJECT_DIR = Path(__file__).resolve().parent
BUNDLE_DIR = Path(getattr(sys, '_MEIPASS', PROJECT_DIR))
EXTERNAL_DIR = Path(sys.executable).resolve().parent if getattr(sys, 'frozen', False) else PROJECT_DIR


def _resource(name: str, *, landmark: bool = False) -> Path:
    """Root landmark models in source; allow old spec's bundled models/ path."""
    roots = (EXTERNAL_DIR, BUNDLE_DIR, PROJECT_DIR)
    for root in roots:
        candidate = root / name
        if candidate.is_file():
            return candidate
    # The previous PyInstaller spec bundled its data under _internal/models.
    # In a source project face/hand remain directly next to vanish_cam.py.
    for root in roots:
        candidate = root / 'models' / name
        if candidate.is_file() and (not landmark or getattr(sys, 'frozen', False)):
            return candidate
    return EXTERNAL_DIR / name


def _existing_models(segmenter_override: str | None = None) -> tuple[Path, Path, Path]:
    segmenter = Path(segmenter_override).expanduser().resolve() if segmenter_override else _resource('selfie_segmenter_landscape.tflite')
    hand = _resource('hand_landmarker.task', landmark=True)
    face = _resource('face_landmarker.task', landmark=True)
    paths = (segmenter, hand, face)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError('Missing model(s): ' + ', '.join(missing)
                           + '. Place hand_landmarker.task and face_landmarker.task in the project root; '
                           'place the selfie segmenter there or pass --model PATH. '
                           'No files are downloaded or created by this version.')
    return paths


def positive_fps(value: str) -> float:
    fps = float(value)
    if not np.isfinite(fps) or not 1 <= fps <= 120:
        raise argparse.ArgumentTypeError('FPS must be between 1 and 120')
    return fps


def positive_dimension(value: str) -> int:
    dimension = int(value)
    if not 64 <= dimension <= 4096:
        raise argparse.ArgumentTypeError('Video dimensions must be between 64 and 4096')
    return dimension


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=MODES, help='Skip the startup menu')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--width', type=positive_dimension, default=960)
    parser.add_argument('--height', type=positive_dimension, default=540)
    parser.add_argument('--fps', type=positive_fps, default=30.0)
    parser.add_argument('--model', help='Path to the selfie segmenter .tflite model')
    parser.add_argument('--fullscreen', action='store_true')
    parser.add_argument('--no-mirror', action='store_true')
    parser.add_argument('--virtual-camera', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--virtual', dest='virtual_camera', action='store_const', const=True,
                        help='Require virtual camera output (same as --virtual-camera)')
    parser.add_argument('--obs', action='store_true', help='Require OBS virtual camera backend')
    parser.add_argument('--virtual-backend', choices=('obs', 'unitycapture', 'v4l2loopback'))
    parser.add_argument('--virtual-device')
    parser.add_argument('--gesture-debug', action='store_true')
    parser.add_argument('--no-sound', action='store_true')
    parser.add_argument('--no-download', action='store_true', help='Retained for compatibility; this version never downloads')
    parser.add_argument('--download-models', action='store_true', help='Check installed models; no files are created')
    parser.add_argument('--self-test', action='store_true', help='Run real model inference on blank frames without a webcam')
    args = parser.parse_args(argv)
    if args.obs:
        if args.virtual_camera is False or (args.virtual_backend and args.virtual_backend != 'obs'):
            parser.error('--obs conflicts with --no-virtual-camera or another virtual backend')
        args.virtual_camera = True
        args.virtual_backend = 'obs'
    return args


def choose_mode() -> str | None:
    title = 'Scegli effetto'
    selected = [None]
    image = np.full((600, 740, 3), (26, 29, 34), dtype=np.uint8)
    cv2.putText(image, 'CHOOSE YOUR VANISH', (70, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (236, 239, 242), 2, cv2.LINE_AA)
    labels = (('NORMAL VANISH', 'Open palm / fade'),
              ('NARUTO VANISH', 'Two-hand seal / smoke + log'),
              ('THANOS', 'Thumb-middle pinch / ash particles'),
              ('GOKU', 'Two fingers to forehead / neon flash'))
    for i, (heading, detail) in enumerate(labels):
        top = 95 + i * 112
        cv2.rectangle(image, (60, top), (680, top + 94), (48, 54, 62), -1)
        cv2.rectangle(image, (60, top), (680, top + 94), (111, 142, 153), 2)
        cv2.putText(image, f'{i + 1}  {heading}', (84, top + 37), cv2.FONT_HERSHEY_SIMPLEX,
                    0.72, (240, 245, 247), 2, cv2.LINE_AA)
        cv2.putText(image, detail, (84, top + 69), cv2.FONT_HERSHEY_SIMPLEX,
                    0.49, (179, 195, 201), 1, cv2.LINE_AA)

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONUP and 60 <= x <= 680:
            index = (y - 95) // 112
            if 0 <= index < len(MODES) and 0 <= y - (95 + index * 112) <= 94:
                selected[0] = MODES[index]

    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title, on_mouse)
    try:
        while selected[0] is None:
            cv2.imshow(title, image)
            key = cv2.waitKey(30) & 0xff
            if ord('1') <= key <= ord('4'):
                selected[0] = MODES[key - ord('1')]
            elif key in (ord('q'), ord('Q'), 27):
                return None
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                return None
        return selected[0]
    finally:
        cv2.destroyWindow(title)


def open_camera(index: int, width: int, height: int):
    if platform.system() == 'Windows':
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(index)
    else:
        capture = cv2.VideoCapture(index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f'Cannot open webcam {index}. Close other camera apps or try --camera 1.')
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def open_virtual_camera(args: argparse.Namespace, width: int, height: int):
    if args.virtual_camera is False:
        return None
    try:
        import pyvirtualcam
        camera = pyvirtualcam.Camera(width=width, height=height, fps=args.fps,
                                     fmt=pyvirtualcam.PixelFormat.BGR,
                                     backend=args.virtual_backend, device=args.virtual_device)
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        message = f'Virtual camera unavailable: {error}. Install/enable the requested driver and free the device.'
        if args.virtual_camera is True or args.virtual_backend or args.virtual_device:
            raise RuntimeError(message) from error
        print('[Webcam] ' + message + ' Continuing with local preview.')
        return None
    print(f'[Webcam] Virtual output: {camera.device} ({width}x{height}, {args.fps:g} FPS target)')
    return camera


def send_virtual_frame(camera, frame: np.ndarray) -> None:
    if frame.shape[:2] != (camera.height, camera.width):
        frame = cv2.resize(frame, (camera.width, camera.height), interpolation=cv2.INTER_LINEAR)
    camera.send(np.ascontiguousarray(frame, dtype=np.uint8))


def set_fullscreen(enabled: bool) -> None:
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN if enabled else cv2.WINDOW_NORMAL)


def points_from_landmarks(landmarks, width: int, height: int) -> np.ndarray:
    return np.asarray([(p.x * width, p.y * height, p.z * width) for p in landmarks], np.float32)


def palm_size(p: np.ndarray) -> float:
    return max(float(np.linalg.norm(p[5, :2] - p[17, :2])),
               float(np.linalg.norm(p[0, :2] - p[9, :2])), 1e-6)


def finger_flags(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Estimate straight/curled fingers independent of global hand rotation."""
    fingers = p[np.arange(5, 21).reshape(4, 4), :2]
    lengths = np.linalg.norm(np.diff(fingers, axis=1), axis=2).sum(axis=1)
    reach = np.linalg.norm(fingers[:, 3] - fingers[:, 0], axis=1)
    ratio = reach / np.maximum(lengths, 1e-6)
    tip = np.linalg.norm(fingers[:, 3] - p[0, :2], axis=1)
    pip = np.linalg.norm(fingers[:, 1] - p[0, :2], axis=1)
    extended = (ratio >= 0.78) & (tip > pip * 1.08) & (lengths > 1e-5)
    folded = (ratio < 0.76) | (tip <= pip * 1.05)
    return extended, folded


def thumb_extended(p: np.ndarray) -> bool:
    thumb = p[1:5, :2]
    length = float(np.linalg.norm(np.diff(thumb, axis=0), axis=1).sum())
    straight = float(np.linalg.norm(thumb[-1] - thumb[0])) / max(length, 1e-6)
    # Require the thumb to project away from the index base, rejecting fists.
    away = float(np.linalg.norm(p[4, :2] - p[5, :2])) / palm_size(p)
    return straight > 0.72 and away > 0.55


def thumb_tucked(p: np.ndarray) -> bool:
    palm = np.mean(p[[0, 5, 9, 13, 17], :2], axis=0)
    return (np.linalg.norm(p[4, :2] - palm) < palm_size(p) * 1.15
            or np.linalg.norm(p[4, :2] - p[5, :2]) < palm_size(p) * 0.55)


def open_palm(hands, width: int, height: int) -> bool:
    for hand in hands:
        p = points_from_landmarks(hand, width, height)
        if p.shape != (21, 3) or not np.isfinite(p).all() or palm_size(p) < 10:
            continue
        extended, _ = finger_flags(p)
        width_ratio = np.linalg.norm(p[5, :2] - p[17, :2]) / max(1e-6, np.linalg.norm(p[0, :2] - p[9, :2]))
        if extended.all() and thumb_extended(p) and width_ratio > 0.30:
            return True
    return False


def goku_gesture(hands, faces, width: int, height: int) -> bool:
    for face in faces:
        if len(face) < 468:
            continue
        f = points_from_landmarks(face, width, height)
        if not np.isfinite(f).all():
            continue
        up = f[10, :2] - f[152, :2]
        face_height = float(np.linalg.norm(up))
        if face_height < 25:
            continue
        # Slightly above the brows, roll-aware; landmarks 9/151 also bound the target.
        brow = (f[105, :2] + f[334, :2]) / 2
        forehead = brow + up * 0.10
        tolerance = min(face_height * 0.24, max(width, height) * 0.12)
        for hand in hands:
            p = points_from_landmarks(hand, width, height)
            if p.shape != (21, 3) or not np.isfinite(p).all() or palm_size(p) < 10:
                continue
            extended, folded = finger_flags(p)
            tips = p[[8, 12], :2]
            if (extended[0] and extended[1] and folded[2] and folded[3] and thumb_tucked(p)
                    and np.linalg.norm(tips[0] - tips[1]) < palm_size(p) * 0.58
                    and np.linalg.norm(tips.mean(axis=0) - forehead) < tolerance
                    and np.max(np.linalg.norm(tips - forehead, axis=1)) < tolerance * 1.25):
                return True
    return False


def naruto_seal(hands, width: int, height: int) -> bool:
    pts = [points_from_landmarks(hand, width, height) for hand in hands if len(hand) == 21]
    pts = [p for p in pts if np.isfinite(p).all() and palm_size(p) >= 10]
    # Two-hand tiger/crossed seal: hands approach with indices extended and
    # parallel or crossing; reject two arbitrary open palms held side by side.
    if len(pts) >= 2:
        a, b = pts[:2]
        sa, sb = palm_size(a), palm_size(b)
        ea, _ = finger_flags(a)
        eb, _ = finger_flags(b)
        va, vb = a[8, :2] - a[5, :2], b[8, :2] - b[5, :2]
        direction = abs(float(np.dot(va, vb) / max(1e-6, np.linalg.norm(va) * np.linalg.norm(vb))))
        close_palms = np.linalg.norm(a[[0, 5, 9, 13, 17], :2].mean(axis=0)
                                    - b[[0, 5, 9, 13, 17], :2].mean(axis=0)) < (sa + sb) * 0.85
        close_indices = np.linalg.norm(a[8, :2] - b[8, :2]) < (sa + sb) * 0.85
        if ea[0] and eb[0] and direction > 0.45 and close_palms and close_indices:
            return True
    # Single-hand tiger: index and middle upright, together, with ring/pinky
    # tucked and thumb across the palm. Keep away from the Goku forehead pose.
    for p in pts:
        ext, fold = finger_flags(p)
        if (ext[0] and ext[1] and fold[2] and fold[3] and thumb_tucked(p)
                and np.linalg.norm(p[8, :2] - p[12, :2]) < palm_size(p) * 0.35):
            return True
    return False


@dataclass
class SnapTrack:
    wrist: np.ndarray
    scale: float
    last_seen: float
    pinch_started: float | None = None
    last_pinch: float | None = None
    pinch_frames: int = 0
    last_gap: float = 0.0


class SnapDetector:
    def __init__(self):
        self.tracks: list[SnapTrack] = []

    def reset(self) -> None:
        self.tracks.clear()

    def update(self, hands, width: int, height: int, now: float) -> tuple[bool, bool]:
        available, current = list(self.tracks), []
        pinching = flick = False
        for hand in hands[:2]:
            p = points_from_landmarks(hand, width, height)
            if p.shape != (21, 3) or not np.isfinite(p).all():
                continue
            scale = palm_size(p)
            if scale < 10:
                continue
            track = None
            if available:
                distance = [float(np.linalg.norm(p[0, :2] - t.wrist)) / max(scale, t.scale) for t in available]
                index = int(np.argmin(distance))
                candidate = available[index]
                if distance[index] < 1.0 and 0 <= now - candidate.last_seen <= 0.20:
                    track = available.pop(index)
            if track is None:
                track = SnapTrack(p[0, :2].copy(), scale, now)
            _, folded = finger_flags(p)
            gap_pixels = min(float(np.linalg.norm(p[4, :2] - p[12, :2])),
                             float(np.linalg.norm(p[4, :2] - p[8, :2])))
            # Match 0.045 normalized and adapt to hand size for different distances.
            gap_limit = min(max(width, height) * 0.045, scale * 0.35)
            gap = gap_pixels / scale
            contact = bool(folded[2] and folded[3] and gap_pixels < gap_limit)
            if contact:
                if track.pinch_started is None:
                    track.pinch_started, track.pinch_frames = now, 0
                track.pinch_frames += 1
                track.last_pinch = now
                track.last_gap = gap
                pinching = True
            elif track.pinch_started is not None:
                age = now - track.pinch_started
                if (track.pinch_frames >= 2 and age <= 0.35 and folded[2] and folded[3]
                        and gap > max(0.55, track.last_gap + 0.25)):
                    flick = True
                track.pinch_started = track.last_pinch = None
                track.pinch_frames = 0
            track.wrist, track.scale, track.last_seen = p[0, :2].copy(), scale, now
            current.append(track)
        self.tracks = current
        return pinching, flick


@dataclass(frozen=True)
class Observation:
    active: bool = False
    pulse: bool = False
    label: str = 'none'


class VisionPipeline:
    """Both landmark tasks observe EVERY real capture in all modes/states."""
    def __init__(self, segmenter_path: Path, hand_path: Path, face_path: Path):
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        self.mp = mp
        self.resources = ExitStack()
        self.timestamp = -1
        self.image = self.frame = None
        self.cached_mask = None
        self.mask_timestamp = -1
        self.snap = SnapDetector()
        try:
            self.hands = self.resources.enter_context(vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(hand_path)),
                    running_mode=vision.RunningMode.VIDEO, num_hands=2,
                    min_hand_detection_confidence=0.35, min_hand_presence_confidence=0.35,
                    min_tracking_confidence=0.45)))
            self.face = self.resources.enter_context(vision.FaceLandmarker.create_from_options(
                vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(face_path)),
                    running_mode=vision.RunningMode.VIDEO, num_faces=1,
                    min_face_detection_confidence=0.4, min_face_presence_confidence=0.4,
                    min_tracking_confidence=0.4)))
            self.segmenter = self.resources.enter_context(vision.ImageSegmenter.create_from_options(
                vision.ImageSegmenterOptions(base_options=python.BaseOptions(model_asset_path=str(segmenter_path)),
                    running_mode=vision.RunningMode.VIDEO, output_confidence_masks=True,
                    output_category_mask=False)))
        except Exception:
            self.resources.close()
            raise

    def observe(self, raw_frame: np.ndarray, mode: str, now: float) -> Observation:
        self.frame = raw_frame
        rgb = np.ascontiguousarray(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB))
        self.image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        self.timestamp = max(self.timestamp + 1, int(now * 1000))
        # Both calls intentionally occur even during calibration, hidden frames,
        # and every transition; neither is fed the rendered/background frame.
        hand_result = self.hands.detect_for_video(self.image, self.timestamp)
        face_result = self.face.detect_for_video(self.image, self.timestamp)
        hands, faces = hand_result.hand_landmarks, face_result.face_landmarks
        height, width = raw_frame.shape[:2]
        if mode == 'normal':
            active = open_palm(hands, width, height)
            return Observation(active, False, 'open palm' if active else 'none')
        if mode == 'naruto':
            active = naruto_seal(hands, width, height)
            return Observation(active, False, 'seal' if active else 'none')
        if mode == 'goku':
            active = goku_gesture(hands, faces, width, height)
            return Observation(active, False, 'forehead' if active else 'none')
        active, flick = self.snap.update(hands, width, height, now)
        return Observation(active, flick, 'snap pinch' if active else 'snap flick' if flick else 'none')

    def mask(self) -> np.ndarray:
        if self.image is None:
            raise RuntimeError('Call observe before mask')
        if self.mask_timestamp != self.timestamp:
            result = self.segmenter.segment_for_video(self.image, self.timestamp)
            if not result.confidence_masks or len(result.confidence_masks) > 2:
                raise RuntimeError('Expected a binary selfie segmentation confidence mask')
            index = 1 if len(result.confidence_masks) == 2 else 0
            mask = result.confidence_masks[index].numpy_view().copy().astype(np.float32).squeeze()
            h, w = self.frame.shape[:2]
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.GaussianBlur(mask, (0, 0), 1.5)
            self.cached_mask = np.clip((mask - 0.12) / 0.68, 0, 1)
            self.mask_timestamp = self.timestamp
        return self.cached_mask

    def reset_gestures(self) -> None:
        self.snap.reset()

    def close(self) -> None:
        self.resources.close()


class Phase(str, Enum):
    VISIBLE = 'visible'
    VANISHING = 'vanishing'
    VANISHED = 'vanished'
    APPEARING = 'appearing'


class GestureGate:
    def __init__(self):
        self.gesture_released_required = False
        self.cooldown_timer = 0.0
        self.release_started: float | None = None
        self.last_sample: float | None = None
        self.stable_frames = 0

    def lock(self, now: float, cooldown: bool = True) -> None:
        self.gesture_released_required = True
        self.cooldown_timer = now + (COOLDOWN_SECONDS if cooldown else 0.0)
        self.release_started = self.last_sample = None
        self.stable_frames = 0

    def update(self, observation: Observation, mode: str, now: float) -> bool:
        if self.last_sample is not None and now - self.last_sample > MAX_FRAME_GAP:
            self.release_started = None
            self.stable_frames = 0
        self.last_sample = now
        if not observation.active and not observation.pulse:
            self.stable_frames = 0
            if self.release_started is None:
                self.release_started = now
            if now - self.release_started >= RELEASE_SECONDS:
                self.gesture_released_required = False
            return False
        self.release_started = None
        if self.gesture_released_required or now < self.cooldown_timer:
            self.stable_frames = 0
            return False
        if observation.pulse:
            self.lock(now, cooldown=False)
            return True
        if observation.active:
            self.stable_frames += 1
            if self.stable_frames >= MODE_FRAMES[mode]:
                self.lock(now, cooldown=False)
                return True
        else:
            self.stable_frames = 0
        return False


class EffectController:
    def __init__(self, mode: str, background: np.ndarray, sound: ProceduralSoundPlayer):
        self.mode = mode
        self.background = background.copy()
        self.sound = sound
        self.phase = Phase.VISIBLE
        self.gate = GestureGate()
        self.started = self.duration = 0.0
        self.effect = None
        self.goku_sound_played = False
        self.log_sprite = None
        self.log_anchor = None
        self.log_cache: dict[int, np.ndarray] = {}
        self.hidden_frame = None

    @property
    def is_vanished(self) -> bool:
        return self.phase in (Phase.VANISHED, Phase.APPEARING)

    @property
    def busy(self) -> bool:
        return self.phase in (Phase.VANISHING, Phase.APPEARING)

    def switch_mode(self, mode: str, now: float) -> None:
        if mode == self.mode:
            return
        hidden = self.is_vanished
        self.mode = mode
        self.phase = Phase.VANISHED if hidden else Phase.VISIBLE
        self.effect = self.hidden_frame = self.log_anchor = None
        self.log_cache.clear()
        self.gate.lock(now)
        print(f'[Mode] {mode}; current state: {self.phase.value}. Release the pose before triggering.')

    def force_visible(self, now: float) -> None:
        self.phase = Phase.VISIBLE
        self.effect = self.hidden_frame = self.log_anchor = None
        self.log_cache.clear()
        self.gate.lock(now)
        print('[Webcam] Immediately visible.')

    def toggle(self, frame: np.ndarray, mask: np.ndarray, now: float) -> bool:
        if self.busy:
            return False
        if mask.shape != frame.shape[:2] or not np.isfinite(mask).all() or np.count_nonzero(mask > 0.4) < 64:
            print('[Gesture] Person mask unavailable; step into view, release, then try again.')
            self.gate.lock(now)
            return False
        appearing = self.is_vanished
        self.phase = Phase.APPEARING if appearing else Phase.VANISHING
        self.started = now
        self.hidden_frame = None
        self.goku_sound_played = False
        if self.mode == 'normal':
            self.effect = NormalFade()
            self.duration = self.effect.duration
        elif self.mode == 'naruto':
            if self.log_sprite is None:
                self.log_sprite = load_log()
            if not appearing or self.log_anchor is None:
                self.log_anchor = substitution_anchor(mask)
            self.duration = EFFECT_SECONDS
            self.sound.play('naruto')
        elif self.mode == 'goku':
            self.effect = GokuTeleport(frame, mask)
            self.duration = self.effect.duration
            if appearing:
                self.sound.play('goku')
                self.goku_sound_played = True
        else:
            self.effect = ThanosDust(frame, mask)
            self.duration = self.effect.appear_duration if appearing else self.effect.vanish_duration
            if not appearing:
                self.sound.play('thanos')
        self.gate.lock(now, cooldown=False)
        print(f'[{self.mode}] {"Appearing" if appearing else "Vanishing"}')
        return True

    def process(self, frame: np.ndarray, observation: Observation, now: float, mask_provider) -> np.ndarray:
        if not self.busy and self.gate.update(observation, self.mode, now):
            self.toggle(frame, mask_provider(), now)
        if self.busy:
            age = max(0.0, now - self.started)
            if self.mode == 'goku' and not self.goku_sound_played and age >= GokuTeleport.sound_time:
                self.sound.play('goku')
                self.goku_sound_played = True
            if age >= self.duration:
                self.phase = Phase.VANISHED if self.phase == Phase.VANISHING else Phase.VISIBLE
                self.gate.lock(now)
                self.effect = None
                self.hidden_frame = None
                if self.phase == Phase.VISIBLE:
                    self.log_anchor = None
                    self.log_cache.clear()
                print(f'[{self.mode}] {self.phase.value}; release gesture for 0.8s; cooldown 1.5s.')
        if self.phase == Phase.VISIBLE:
            return frame.copy()
        if self.phase == Phase.VANISHED:
            if self.hidden_frame is None:
                self.hidden_frame = self.background.copy()
                if self.mode == 'naruto' and self.log_anchor is not None and self.log_sprite is not None:
                    overlay_log(self.hidden_frame, self.log_sprite,
                                landed_log_anchor(self.log_anchor, frame.shape[0]), cache=self.log_cache)
            return self.hidden_frame.copy()
        age = max(0.0, now - self.started)
        appearing = self.phase == Phase.APPEARING
        if self.mode in ('goku', 'thanos'):
            return self.effect.render(self.background, age, appearing)
        mask = mask_provider()
        if self.mode == 'normal':
            return self.effect.render(frame, self.background, mask, age, appearing)
        renderer = render_reappearance if appearing else render_substitution
        return renderer(frame, self.background, mask, self.log_sprite, self.log_anchor, age, self.log_cache)


def run_camera(args: argparse.Namespace, resources: ExitStack, mode: str) -> int:
    paths = _existing_models(args.model)
    capture = open_camera(args.camera, args.width, args.height)
    resources.callback(capture.release)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    vision = VisionPipeline(*paths)
    resources.callback(vision.close)
    sound = ProceduralSoundPlayer(enabled=not args.no_sound)
    resources.callback(sound.close)
    resources.callback(cv2.destroyAllWindows)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, args.width, args.height)
    fullscreen = args.fullscreen
    if fullscreen:
        set_fullscreen(True)
    controller = None
    samples: deque[np.ndarray] = deque(maxlen=18)
    calibration_started = time.perf_counter()
    calibration_remaining = None
    capture_shape = None
    virtual_camera = None
    virtual_initialized = False
    pending_toggle = False
    last_debug = float('-inf')
    print('[Webcam] Leave the view for 4 seconds to capture the clean background.')
    print('[Keys] 1 normal, 2 naruto, 3 thanos, 4 goku; V toggle; R visible; Space background; F fullscreen; Q quit.')
    print('[Gesture] Release for 0.8s; cooldown 1.5s after each completed effect.')
    while True:
        loop_started = time.perf_counter()
        ok, raw_frame = capture.read()
        if not ok or raw_frame is None:
            raise RuntimeError('The webcam stopped returning frames')
        now = time.perf_counter()
        height, width = raw_frame.shape[:2]
        if capture_shape is not None and capture_shape != raw_frame.shape:
            controller = None
            samples.clear()
            pending_toggle = False
            calibration_started, calibration_remaining = now, None
            vision.reset_gestures()
            print('[Webcam] Resolution changed; recapturing the background.')
        capture_shape = raw_frame.shape
        if not virtual_initialized:
            virtual_camera = open_virtual_camera(args, width, height)
            virtual_initialized = True
            if virtual_camera is not None:
                resources.callback(virtual_camera.close)
        # Raw camera pixels always reach BOTH task models: no compositing,
        # background substitution or preview mirroring before inference.
        observation = vision.observe(raw_frame, mode, now)
        if controller is None:
            elapsed = now - calibration_started
            remaining = max(0, int(np.ceil(BACKGROUND_SECONDS - elapsed)))
            if remaining != calibration_remaining:
                print(f'[Background] {remaining}s')
                calibration_remaining = remaining
            if elapsed >= BACKGROUND_SECONDS - SAMPLE_SECONDS:
                samples.append(raw_frame.copy())
            if elapsed >= BACKGROUND_SECONDS:
                background = np.median(np.stack(tuple(samples)), axis=0).astype(np.uint8) if samples else raw_frame.copy()
                samples.clear()
                controller = EffectController(mode, background, sound)
                vision.reset_gestures()
                print('[Background] Ready. Step into view.')
            display = raw_frame.copy()
        else:
            if pending_toggle:
                controller.toggle(raw_frame, vision.mask(), now)
                vision.reset_gestures()
                pending_toggle = False
            display = controller.process(raw_frame, observation, now, vision.mask)
        if args.gesture_debug and now - last_debug >= 0.5:
            phase = controller.phase.value if controller else 'calibrating'
            released = not controller.gate.gesture_released_required if controller else False
            print(f'[Gesture] {observation.label}; mode={mode}; state={phase}; released={released}')
            last_debug = now
        # Mirroring is applied only to final output, leaving inference pixels raw.
        if not args.no_mirror:
            display = cv2.flip(display, 1)
        if virtual_camera is not None:
            send_virtual_frame(virtual_camera, display)
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(1) & 0xff
        if key in (ord('q'), ord('Q'), 27) or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord(' '):
            controller = None
            pending_toggle = False
            samples.clear()
            vision.reset_gestures()
            calibration_started, calibration_remaining = time.perf_counter(), None
            print('[Webcam] Leave the view for 4 seconds for a new background.')
        elif ord('1') <= key <= ord('4'):
            new_mode = MODES[key - ord('1')]
            if new_mode != mode:
                mode = new_mode
                vision.reset_gestures()
                pending_toggle = False
                if controller is not None:
                    controller.switch_mode(mode, time.perf_counter())
                else:
                    print(f'[Mode] {mode}')
        elif key in (ord('r'), ord('R')) and controller is not None:
            controller.force_visible(time.perf_counter())
            vision.reset_gestures()
            pending_toggle = False
        elif key in (ord('v'), ord('V')) and controller is not None and not controller.busy:
            pending_toggle = True
        elif key in (ord('f'), ord('F')):
            fullscreen = not fullscreen
            set_fullscreen(fullscreen)
        if virtual_camera is not None:
            virtual_camera.sleep_until_next_frame()
        else:
            time.sleep(max(0.0, 1.0 / args.fps - (time.perf_counter() - loop_started)))
    return 0


def self_test(model: str | None = None) -> int:
    paths = _existing_models(model)
    vision = VisionPipeline(*paths)
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    try:
        for i, mode in enumerate(MODES):
            for j in range(2):
                observation = vision.observe(blank, mode, 1 + i * 0.1 + j * 0.03)
                mask = vision.mask()
                if mask.shape != blank.shape[:2] or not np.isfinite(mask).all():
                    raise RuntimeError(f'{mode}: invalid segmentation mask')
                if observation.active or observation.pulse:
                    raise RuntimeError(f'{mode}: gesture falsely detected in blank test image')
            print(f'[Self-test] {mode}: face, hand and segmenter processed real model frames.')
        print('[Self-test] Passed; webcam, gesture accuracy, audio and OBS still require hardware testing.')
        return 0
    finally:
        vision.close()


def run(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.download_models:
        paths = _existing_models(args.model)
        print('[Models] Required models exist: ' + ', '.join(str(p) for p in paths))
        print('[Models] No downloads: this version does not create files.')
        return 0
    if args.self_test:
        return self_test(args.model)
    mode = args.mode or choose_mode()
    if mode is None:
        return 0
    with ExitStack() as resources:
        return run_camera(args, resources, mode)


def main() -> int:
    try:
        return run()
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f'\n[VanishCam] ERROR: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
