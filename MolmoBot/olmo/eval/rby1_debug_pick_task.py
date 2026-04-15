"""RBY1 pick-task diagnostics used by the local freeze-test benchmark."""

from __future__ import annotations

from typing import Any

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.tasks.pick_task import PickTask as BasePickTask


class PickTask(BasePickTask):
    """PickTask variant that records RBY1-specific TCP and contact diagnostics."""

    def _create_sensor_suite_from_config(self, config: MlSpacesExpConfig):
        from molmo_spaces.env.abstract_sensors import SensorSuite
        from molmo_spaces.env.rby1_sensors import RBY1TCPPoseSensor
        from molmo_spaces.env.sensors import get_core_sensors

        sensors = get_core_sensors(config)
        sensor_uuids = {sensor.uuid for sensor in sensors}
        if "left_tcp_pose" not in sensor_uuids:
            sensors.append(RBY1TCPPoseSensor(uuid="left_tcp_pose", arm_side="left"))
        if "right_tcp_pose" not in sensor_uuids:
            sensors.append(RBY1TCPPoseSensor(uuid="right_tcp_pose", arm_side="right"))
        return SensorSuite(sensors)

    def get_info(self) -> list[dict[str, Any]]:
        metrics = super().get_info()
        for batch_index, metric in enumerate(metrics):
            metric.update(self._get_rby1_contact_debug(batch_index))
        return metrics

    def _get_rby1_contact_debug(self, batch_index: int) -> dict[str, Any]:
        import mujoco
        from molmo_spaces.env.data_views import MlSpacesObject
        from molmo_spaces.utils.mj_model_and_data_utils import descendant_geoms

        data = self._env.mj_datas[batch_index]
        model = data.model
        pickup_obj = MlSpacesObject(
            data=data,
            object_name=self.config.task_config.pickup_obj_name,
        )
        object_geoms = set(descendant_geoms(model, pickup_obj.body_id, visual_only=False))
        robot_root_body_id = self.env.current_robot.robot_view.base.root_body_id

        contact_pairs = []
        object_contact_count = 0
        robot_object_contact_count = 0
        left_finger_contact = False
        right_finger_contact = False

        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            if (contact.geom1 in object_geoms) == (contact.geom2 in object_geoms):
                continue

            object_contact_count += 1
            other_geom = contact.geom2 if contact.geom1 in object_geoms else contact.geom1
            other_body = model.geom_bodyid[other_geom]
            other_root_body = model.body_rootid[other_body]
            other_body_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other_body) or ""
            )
            other_geom_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_geom) or ""
            )

            if other_root_body == robot_root_body_id:
                robot_object_contact_count += 1
            if "finger_l" in other_body_name or (
                "left" in other_body_name and "finger" in other_body_name
            ):
                left_finger_contact = True
            if "finger_r" in other_body_name or (
                "right" in other_body_name and "finger" in other_body_name
            ):
                right_finger_contact = True

            if len(contact_pairs) < 12:
                contact_pairs.append(
                    {
                        "body": other_body_name,
                        "geom": other_geom_name,
                        "dist": float(contact.dist),
                    }
                )

        return {
            "object_contact_count": object_contact_count,
            "robot_object_contact_count": robot_object_contact_count,
            "left_finger_contact": left_finger_contact,
            "right_finger_contact": right_finger_contact,
            "object_contact_pairs": contact_pairs,
            **self._get_rby1_finger_body_debug(model, data, pickup_obj.body_id),
        }

    def _get_rby1_finger_body_debug(self, model, data, object_body_id: int) -> dict[str, Any]:
        import mujoco
        import numpy as np

        object_pos = np.asarray(data.xpos[object_body_id], dtype=float)
        debug: dict[str, Any] = {"pickup_obj_body_pos": object_pos.tolist()}
        finger_body_names = {
            "left_l1": "robot_0/ee_finger_l1",
            "left_l2": "robot_0/ee_finger_l2",
            "right_r1": "robot_0/ee_finger_r1",
            "right_r2": "robot_0/ee_finger_r2",
        }

        for label, body_name in finger_body_names.items():
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                continue
            pos = np.asarray(data.xpos[body_id], dtype=float)
            debug[f"{label}_body_pos"] = pos.tolist()
            debug[f"{label}_body_obj_dist"] = float(np.linalg.norm(pos - object_pos))

        if "left_l1_body_pos" in debug and "left_l2_body_pos" in debug:
            midpoint = (
                np.asarray(debug["left_l1_body_pos"])
                + np.asarray(debug["left_l2_body_pos"])
            ) / 2.0
            debug["left_finger_midpoint_pos"] = midpoint.tolist()
            debug["left_finger_midpoint_obj_dist"] = float(
                np.linalg.norm(midpoint - object_pos)
            )
        if "right_r1_body_pos" in debug and "right_r2_body_pos" in debug:
            midpoint = (
                np.asarray(debug["right_r1_body_pos"])
                + np.asarray(debug["right_r2_body_pos"])
            ) / 2.0
            debug["right_finger_midpoint_pos"] = midpoint.tolist()
            debug["right_finger_midpoint_obj_dist"] = float(
                np.linalg.norm(midpoint - object_pos)
            )

        return debug
