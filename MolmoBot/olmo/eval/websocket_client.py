"""Synchronous WebSocket client for the MolmoBot policy server.

Mirrors the protocol in websocket_server.py:
  1. On connect, server sends one metadata message (msgpack-packed dict).
  2. Client sends msgpack-packed observations, server replies with msgpack-packed actions.
  3. Connection close triggers server-side policy.reset().

Usage:
    with MolmoBotClient("localhost", 8000) as client:
        metadata = client.metadata
        for step in range(100):
            obs = build_rby1_obs(
                head_camera=img, wrist_camera_l=img, wrist_camera_r=img,
                base=np.zeros(3), left_arm=np.zeros(7), left_gripper=np.zeros(1),
                right_arm=np.zeros(7), right_gripper=np.zeros(1), torso=np.zeros(6),
                task="pick the phone",
            )
            action = client.get_action(obs)
"""

from __future__ import annotations

import logging
from typing import Any

import msgpack_numpy
import numpy as np
from websockets.sync.client import connect as ws_connect

logger = logging.getLogger(__name__)

# Expected shapes for RBY1 multitask obs (from real_robot_molmobot_rby1_multitask.py)
_RBY1_QPOS_SPEC: dict[str, int] = {
    "base": 3,
    "left_arm": 7,
    "left_gripper": 1,
    "right_arm": 7,
    "right_gripper": 1,
    "torso": 6,
}

# Expected action shapes returned by server
RBY1_ACTION_SPEC: dict[str, int] = {
    "base": 3,
    "left_arm": 7,
    "left_gripper": 1,
    "right_arm": 7,
    "right_gripper": 1,
    "torso": 1,
}


class MolmoBotClient:
    """Synchronous WebSocket client for the MolmoBot policy server.

    Uses websockets.sync.client and msgpack_numpy — same libraries as the server.
    """

    def __init__(self, host: str = "localhost", port: int = 8000):
        self._uri = f"ws://{host}:{port}"
        self._ws = None
        self._packer = msgpack_numpy.Packer()
        self._metadata: dict | None = None

    @property
    def metadata(self) -> dict | None:
        """Server metadata received on connection (model_name, etc.)."""
        return self._metadata

    @property
    def connected(self) -> bool:
        return self._ws is not None

    def connect(self) -> dict:
        """Open connection and receive server metadata.

        Returns:
            Server metadata dict (contains at least 'model_name').
        """
        if self._ws is not None:
            logger.warning("Already connected, closing existing connection first")
            self.close()

        self._ws = ws_connect(self._uri, compression=None, max_size=None)
        raw = self._ws.recv()
        self._metadata = msgpack_numpy.unpackb(raw)
        logger.info(f"Connected to {self._uri}, metadata: {self._metadata}")
        return self._metadata

    def get_action(self, obs: dict) -> dict:
        """Send observation and receive action.

        Args:
            obs: Observation dict matching the server's expected format.

        Returns:
            Action dict with move group arrays and server_timing.

        Raises:
            RuntimeError: If not connected.
        """
        if self._ws is None:
            raise RuntimeError("Not connected. Call connect() first.")

        self._ws.send(self._packer.pack(obs))
        raw = self._ws.recv()
        action = msgpack_numpy.unpackb(raw)
        return action

    def reset(self) -> dict:
        """Reset the server-side policy by reconnecting.

        Returns:
            Fresh server metadata dict.
        """
        self.close()
        return self.connect()

    def close(self) -> None:
        """Close the WebSocket connection."""
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None
            self._metadata = None

    def __enter__(self) -> MolmoBotClient:
        self.connect()
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def build_rby1_obs(
    *,
    head_camera: np.ndarray,
    wrist_camera_l: np.ndarray,
    wrist_camera_r: np.ndarray,
    base: np.ndarray,
    left_arm: np.ndarray,
    left_gripper: np.ndarray,
    right_arm: np.ndarray,
    right_gripper: np.ndarray,
    torso: np.ndarray,
    task: str,
    reset: bool = False,
    object_image_points: dict[str, Any] | None = None,
) -> dict:
    """Build an RBY1 multitask observation dict matching the server's expected format.

    All camera images must be (H, W, 3) uint8 arrays.
    Joint arrays must match the shapes in _RBY1_QPOS_SPEC.

    Args:
        head_camera: Head camera RGB image.
        wrist_camera_l: Left wrist camera RGB image.
        wrist_camera_r: Right wrist camera RGB image.
        base: Base position (3,).
        left_arm: Left arm joint positions (7,).
        left_gripper: Left gripper position (1,).
        right_arm: Right arm joint positions (7,).
        right_gripper: Right gripper position (1,).
        torso: Torso joint positions (6,). Server extracts indices [1,2,3].
        task: Task description string.
        reset: If True, triggers server-side policy reset.
        object_image_points: Optional conditioning points for first frame.

    Returns:
        Observation dict ready for MolmoBotClient.get_action().

    Raises:
        ValueError: If any array has an unexpected shape.
    """
    # Validate cameras
    for name, img in [
        ("head_camera", head_camera),
        ("wrist_camera_l", wrist_camera_l),
        ("wrist_camera_r", wrist_camera_r),
    ]:
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"{name} must be (H, W, 3), got {img.shape}")

    # Validate qpos
    qpos_arrays = {
        "base": base,
        "left_arm": left_arm,
        "left_gripper": left_gripper,
        "right_arm": right_arm,
        "right_gripper": right_gripper,
        "torso": torso,
    }
    for name, arr in qpos_arrays.items():
        expected = _RBY1_QPOS_SPEC[name]
        if arr.shape[0] < expected:
            raise ValueError(
                f"qpos['{name}'] needs at least {expected} elements, got {arr.shape}"
            )

    obs: dict[str, Any] = {
        "head_camera": np.asarray(head_camera, dtype=np.uint8),
        "wrist_camera_l": np.asarray(wrist_camera_l, dtype=np.uint8),
        "wrist_camera_r": np.asarray(wrist_camera_r, dtype=np.uint8),
        "qpos": {
            "base": np.asarray(base, dtype=np.float32),
            "left_arm": np.asarray(left_arm, dtype=np.float32),
            "left_gripper": np.asarray(left_gripper, dtype=np.float32),
            "right_arm": np.asarray(right_arm, dtype=np.float32),
            "right_gripper": np.asarray(right_gripper, dtype=np.float32),
            "torso": np.asarray(torso, dtype=np.float32),
        },
        "task": task,
    }

    if reset:
        obs["reset"] = True

    if object_image_points is not None:
        obs["object_image_points"] = object_image_points

    return obs
