"""Policy wrapper that intercepts observations to stream camera frames.

Wraps any policy object so that :meth:`get_action` captures camera images from
the observation dict and sends them to a :class:`FrameStreamer` before
delegating to the real policy.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from olmo.eval.frame_streamer import FrameStreamer

logger = logging.getLogger(__name__)

# Camera name aliases used by SynthVLAPolicy._populate_action_buffer
# (configure_molmo_spaces.py lines 101-105).  The sim may use a different
# sensor name than the one listed in camera_names.
_CAMERA_ALIASES: dict[str, list[str]] = {
    "exo_camera_1": ["droid_shoulder_light_randomization", "exo_camera_1"],
    "wrist_camera": ["wrist_camera_zed_mini", "wrist_camera"],
}


def _resolve_camera_key(name: str, obs_keys: set[str]) -> str | None:
    """Return the actual observation key for *name*, respecting aliases."""
    candidates = _CAMERA_ALIASES.get(name, [name])
    for candidate in candidates:
        if candidate in obs_keys:
            return candidate
    return None


def _tile_horizontal(images: list[np.ndarray]) -> np.ndarray:
    """Tile images horizontally, padding shorter ones with black."""
    if len(images) == 1:
        return images[0]
    max_h = max(img.shape[0] for img in images)
    padded = []
    for img in images:
        if img.shape[0] < max_h:
            pad = np.zeros((max_h - img.shape[0], img.shape[1], 3), dtype=img.dtype)
            img = np.concatenate([img, pad], axis=0)
        padded.append(img)
    return np.concatenate(padded, axis=1)


class StreamingPolicyWrapper:
    """Transparent wrapper that streams observation frames via WebSocket.

    Parameters
    ----------
    inner_policy:
        The real policy to delegate to.
    streamer:
        A running :class:`FrameStreamer` instance.
    camera_names:
        Camera names to extract from observations (e.g. ``["exo_camera_1", "wrist_camera"]``).
    """

    def __init__(
        self,
        inner_policy: Any,
        streamer: FrameStreamer,
        camera_names: list[str],
    ) -> None:
        self._inner = inner_policy
        self._streamer = streamer
        self._camera_names = camera_names
        self._step_count = 0
        self._episode_count = 0

    # ------------------------------------------------------------------
    # Core policy interface
    # ------------------------------------------------------------------

    def get_action(self, observation: Any) -> Any:
        self._capture_and_stream(observation)
        self._step_count += 1
        return self._inner.get_action(observation)

    def reset(self) -> None:
        self._step_count = 0
        self._episode_count += 1
        return self._inner.reset()

    def prepare_model(self) -> None:
        return self._inner.prepare_model()

    # ------------------------------------------------------------------
    # Attribute delegation
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> Any:
        # Only called for attributes not found on *this* instance.
        return getattr(self._inner, name)

    # ------------------------------------------------------------------
    # Frame capture
    # ------------------------------------------------------------------

    def _capture_and_stream(self, observation: Any) -> None:
        try:
            obs = observation[0] if isinstance(observation, list) else observation
            obs_keys = set(obs.keys())

            frames: list[np.ndarray] = []
            cam_labels: list[str] = []
            for cam_name in self._camera_names:
                key = _resolve_camera_key(cam_name, obs_keys)
                if key is None:
                    continue
                frame = obs[key]
                if isinstance(frame, np.ndarray) and frame.ndim == 3:
                    frames.append(frame)
                    cam_labels.append(cam_name)

            if not frames:
                return

            tiled = _tile_horizontal(frames)
            metadata = {
                "step": self._step_count,
                "episode": self._episode_count,
                "cameras": cam_labels,
            }
            self._streamer.send_frame(tiled, metadata)
        except Exception:
            logger.debug("Frame capture failed", exc_info=True)
