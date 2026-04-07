#!/usr/bin/env python3
"""Standalone test runner for MolmoBotClient.

Mock mode (no server/GPU needed):
    python -m olmo.eval.test_websocket_client --mock

Live mode (requires a running MolmoBot server):
    python -m olmo.eval.test_websocket_client --host localhost --port 8000 \\
        --task "pick the phone" --steps 20
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import threading
import time

import msgpack_numpy
import numpy as np
import websockets.asyncio.server as ws_server

from olmo.eval.websocket_client import (
    MolmoBotClient,
    RBY1_ACTION_SPEC,
    build_rby1_obs,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mock server (reused from unit tests, self-contained here for standalone use)
# ---------------------------------------------------------------------------

_MOCK_METADATA = {"model_name": "mock-rby1-multitask", "version": "standalone-test"}


def _make_dummy_action() -> dict:
    action = {k: np.random.randn(v).astype(np.float32) for k, v in RBY1_ACTION_SPEC.items()}
    action["server_timing"] = {"total_ms": 10}
    return action


async def _mock_handler(websocket):
    packer = msgpack_numpy.Packer()
    await websocket.send(packer.pack(_MOCK_METADATA))
    try:
        async for raw in websocket:
            _obs = msgpack_numpy.unpackb(raw)
            await websocket.send(packer.pack(_make_dummy_action()))
    except Exception:
        pass


def _start_mock_server(port: int = 0) -> tuple[threading.Thread, int]:
    """Start mock server in background, return (thread, actual_port)."""
    actual_port_holder = [None]
    started = threading.Event()

    async def _serve():
        srv = await ws_server.serve(
            _mock_handler, "127.0.0.1", port, compression=None, max_size=None,
        )
        for sock in srv.sockets:
            actual_port_holder[0] = sock.getsockname()[1]
            break
        started.set()
        await srv.serve_forever()

    def _run():
        asyncio.run(_serve())

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    started.wait(timeout=5)
    return t, actual_port_holder[0]


# ---------------------------------------------------------------------------
# Test routines
# ---------------------------------------------------------------------------

def _make_synthetic_obs(task: str, img_size: int = 64) -> dict:
    return build_rby1_obs(
        head_camera=np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8),
        wrist_camera_l=np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8),
        wrist_camera_r=np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8),
        base=np.zeros(3, dtype=np.float32),
        left_arm=np.zeros(7, dtype=np.float32),
        left_gripper=np.zeros(1, dtype=np.float32),
        right_arm=np.zeros(7, dtype=np.float32),
        right_gripper=np.zeros(1, dtype=np.float32),
        torso=np.zeros(6, dtype=np.float32),
        task=task,
    )


def run_test(host: str, port: int, task: str, steps: int) -> bool:
    """Run a multi-step test against a server. Returns True on success."""
    logger.info(f"Connecting to ws://{host}:{port} ...")

    with MolmoBotClient(host, port) as client:
        logger.info(f"Connected. Metadata: {client.metadata}")

        for step in range(steps):
            obs = _make_synthetic_obs(task)
            t0 = time.monotonic()
            action = client.get_action(obs)
            elapsed_ms = (time.monotonic() - t0) * 1000

            # Validate action keys and shapes
            for key, expected_dim in RBY1_ACTION_SPEC.items():
                if key not in action:
                    logger.error(f"Step {step}: missing action key '{key}'")
                    return False
                if action[key].shape != (expected_dim,):
                    logger.error(
                        f"Step {step}: action['{key}'] shape {action[key].shape} "
                        f"!= ({expected_dim},)"
                    )
                    return False

            timing = action.get("server_timing", {})
            server_ms = timing.get("total_ms", "?")
            logger.info(
                f"  Step {step:3d}/{steps}  "
                f"round-trip={elapsed_ms:6.1f}ms  "
                f"server={server_ms}ms  "
                f"base={action['base']}"
            )

        # Test reset
        logger.info("Testing reset (reconnect) ...")
        metadata = client.reset()
        logger.info(f"Reset OK. Metadata: {metadata}")

        action = client.get_action(_make_synthetic_obs(task))
        logger.info(f"Post-reset action OK. base={action['base']}")

    logger.info("All checks passed.")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test MolmoBotClient")
    parser.add_argument("--mock", action="store_true", help="Use built-in mock server")
    parser.add_argument("--host", default="localhost", help="Server host (live mode)")
    parser.add_argument("--port", type=int, default=8000, help="Server port (live mode)")
    parser.add_argument("--task", default="pick the phone and place it in the slot")
    parser.add_argument("--steps", type=int, default=10, help="Number of obs/action steps")
    args = parser.parse_args()

    if args.mock:
        logger.info("Starting mock server ...")
        _thread, port = _start_mock_server()
        logger.info(f"Mock server on port {port}")
        success = run_test("127.0.0.1", port, args.task, args.steps)
    else:
        success = run_test(args.host, args.port, args.task, args.steps)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
