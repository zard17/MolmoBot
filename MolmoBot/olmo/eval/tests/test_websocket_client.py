"""Unit tests for MolmoBotClient and build_rby1_obs.

Uses a lightweight mock WebSocket server — no GPU or MolmoBot model needed.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any

import msgpack_numpy
import numpy as np
import pytest
import websockets.asyncio.server as ws_server

from olmo.eval.websocket_client import (
    MolmoBotClient,
    RBY1_ACTION_SPEC,
    build_rby1_obs,
)

# ---------------------------------------------------------------------------
# Mock server
# ---------------------------------------------------------------------------

_MOCK_METADATA = {"model_name": "mock-rby1-multitask", "version": "test"}


def _make_dummy_action() -> dict[str, Any]:
    """Return an action dict matching RBY1_ACTION_SPEC."""
    action = {k: np.zeros(v, dtype=np.float32) for k, v in RBY1_ACTION_SPEC.items()}
    action["server_timing"] = {"total_ms": 42}
    return action


async def _mock_handler(websocket):
    """Minimal handler mirroring websocket_server.py protocol."""
    packer = msgpack_numpy.Packer()

    # Step 1: send metadata
    await websocket.send(packer.pack(_MOCK_METADATA))

    # Step 2: obs → action loop
    try:
        async for raw in websocket:
            _obs = msgpack_numpy.unpackb(raw)
            action = _make_dummy_action()
            await websocket.send(packer.pack(action))
    except websockets.exceptions.ConnectionClosed:
        pass


class MockServer:
    """Runs a mock MolmoBot WebSocket server in a background thread."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self._host = host
        self._port = port
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server = None
        self.actual_port: int | None = None

    def start(self) -> int:
        started = threading.Event()

        def _run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._serve(started))

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        started.wait(timeout=5)
        return self.actual_port

    async def _serve(self, started: threading.Event):
        self._server = await ws_server.serve(
            _mock_handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        )
        # Grab the actual port (useful when port=0)
        for sock in self._server.sockets:
            self.actual_port = sock.getsockname()[1]
            break
        started.set()
        await self._server.serve_forever()

    def stop(self):
        if self._server and self._loop:
            self._loop.call_soon_threadsafe(self._server.close)
        if self._thread:
            self._thread.join(timeout=2)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mock_server():
    server = MockServer()
    port = server.start()
    time.sleep(0.1)  # let server fully bind
    yield port
    server.stop()


def _dummy_obs() -> dict:
    """Build a valid RBY1 obs using build_rby1_obs."""
    return build_rby1_obs(
        head_camera=np.zeros((64, 64, 3), dtype=np.uint8),
        wrist_camera_l=np.zeros((64, 64, 3), dtype=np.uint8),
        wrist_camera_r=np.zeros((64, 64, 3), dtype=np.uint8),
        base=np.zeros(3),
        left_arm=np.zeros(7),
        left_gripper=np.zeros(1),
        right_arm=np.zeros(7),
        right_gripper=np.zeros(1),
        torso=np.zeros(6),
        task="pick the phone",
    )


# ---------------------------------------------------------------------------
# Tests: build_rby1_obs
# ---------------------------------------------------------------------------

class TestBuildRby1Obs:
    def test_valid_obs_keys(self):
        obs = _dummy_obs()
        assert "head_camera" in obs
        assert "wrist_camera_l" in obs
        assert "wrist_camera_r" in obs
        assert "qpos" in obs
        assert "task" in obs
        assert obs["task"] == "pick the phone"

    def test_qpos_keys_and_shapes(self):
        obs = _dummy_obs()
        qpos = obs["qpos"]
        assert qpos["base"].shape == (3,)
        assert qpos["left_arm"].shape == (7,)
        assert qpos["left_gripper"].shape == (1,)
        assert qpos["right_arm"].shape == (7,)
        assert qpos["right_gripper"].shape == (1,)
        assert qpos["torso"].shape == (6,)

    def test_dtypes(self):
        obs = _dummy_obs()
        assert obs["head_camera"].dtype == np.uint8
        assert obs["qpos"]["base"].dtype == np.float32

    def test_reset_flag(self):
        obs = build_rby1_obs(
            head_camera=np.zeros((64, 64, 3), dtype=np.uint8),
            wrist_camera_l=np.zeros((64, 64, 3), dtype=np.uint8),
            wrist_camera_r=np.zeros((64, 64, 3), dtype=np.uint8),
            base=np.zeros(3),
            left_arm=np.zeros(7),
            left_gripper=np.zeros(1),
            right_arm=np.zeros(7),
            right_gripper=np.zeros(1),
            torso=np.zeros(6),
            task="reset test",
            reset=True,
        )
        assert obs["reset"] is True

    def test_no_reset_flag_by_default(self):
        obs = _dummy_obs()
        assert "reset" not in obs

    def test_object_image_points(self):
        pts = {"phone": {"head_camera": {"points": np.array([[0.5, 0.5]]), "num_points": np.array([1])}}}
        obs = build_rby1_obs(
            head_camera=np.zeros((64, 64, 3), dtype=np.uint8),
            wrist_camera_l=np.zeros((64, 64, 3), dtype=np.uint8),
            wrist_camera_r=np.zeros((64, 64, 3), dtype=np.uint8),
            base=np.zeros(3),
            left_arm=np.zeros(7),
            left_gripper=np.zeros(1),
            right_arm=np.zeros(7),
            right_gripper=np.zeros(1),
            torso=np.zeros(6),
            task="pick",
            object_image_points=pts,
        )
        assert "object_image_points" in obs

    def test_invalid_camera_shape(self):
        with pytest.raises(ValueError, match="head_camera"):
            build_rby1_obs(
                head_camera=np.zeros((64, 64), dtype=np.uint8),  # missing channel dim
                wrist_camera_l=np.zeros((64, 64, 3), dtype=np.uint8),
                wrist_camera_r=np.zeros((64, 64, 3), dtype=np.uint8),
                base=np.zeros(3),
                left_arm=np.zeros(7),
                left_gripper=np.zeros(1),
                right_arm=np.zeros(7),
                right_gripper=np.zeros(1),
                torso=np.zeros(6),
                task="fail",
            )

    def test_invalid_qpos_shape(self):
        with pytest.raises(ValueError, match="left_arm"):
            build_rby1_obs(
                head_camera=np.zeros((64, 64, 3), dtype=np.uint8),
                wrist_camera_l=np.zeros((64, 64, 3), dtype=np.uint8),
                wrist_camera_r=np.zeros((64, 64, 3), dtype=np.uint8),
                base=np.zeros(3),
                left_arm=np.zeros(5),  # should be 7
                left_gripper=np.zeros(1),
                right_arm=np.zeros(7),
                right_gripper=np.zeros(1),
                torso=np.zeros(6),
                task="fail",
            )


# ---------------------------------------------------------------------------
# Tests: MolmoBotClient
# ---------------------------------------------------------------------------

class TestMolmoBotClient:
    def test_connect_and_metadata(self, mock_server):
        client = MolmoBotClient("127.0.0.1", mock_server)
        metadata = client.connect()
        assert metadata["model_name"] == "mock-rby1-multitask"
        assert client.connected
        client.close()

    def test_context_manager(self, mock_server):
        with MolmoBotClient("127.0.0.1", mock_server) as client:
            assert client.connected
            assert client.metadata is not None
        assert not client.connected

    def test_get_action_returns_correct_keys(self, mock_server):
        with MolmoBotClient("127.0.0.1", mock_server) as client:
            obs = _dummy_obs()
            action = client.get_action(obs)
            for key in RBY1_ACTION_SPEC:
                assert key in action, f"Missing action key: {key}"
            assert "server_timing" in action

    def test_get_action_shapes(self, mock_server):
        with MolmoBotClient("127.0.0.1", mock_server) as client:
            action = client.get_action(_dummy_obs())
            for key, expected_dim in RBY1_ACTION_SPEC.items():
                assert action[key].shape == (expected_dim,), (
                    f"action['{key}'] shape mismatch: {action[key].shape} != ({expected_dim},)"
                )

    def test_multiple_steps(self, mock_server):
        with MolmoBotClient("127.0.0.1", mock_server) as client:
            for _ in range(5):
                action = client.get_action(_dummy_obs())
                assert "base" in action

    def test_reset_reconnects(self, mock_server):
        with MolmoBotClient("127.0.0.1", mock_server) as client:
            action1 = client.get_action(_dummy_obs())
            assert action1 is not None

            metadata = client.reset()
            assert metadata["model_name"] == "mock-rby1-multitask"

            action2 = client.get_action(_dummy_obs())
            assert action2 is not None

    def test_get_action_before_connect_raises(self):
        client = MolmoBotClient("127.0.0.1", 9999)
        with pytest.raises(RuntimeError, match="Not connected"):
            client.get_action(_dummy_obs())

    def test_close_is_idempotent(self, mock_server):
        client = MolmoBotClient("127.0.0.1", mock_server)
        client.connect()
        client.close()
        client.close()  # should not raise
        assert not client.connected
