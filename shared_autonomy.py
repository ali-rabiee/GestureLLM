import math
import time
from typing import Optional, Tuple, Dict, Any

import numpy as np

from config import (
    ASSIST_MODE,
    INTENT_DIST_THRESH,
    INTENT_CONF_THRESH,
    HIGH_CONF_AUTO_THRESH,
    ALIGN_YAW_ERR_THRESH,
    ALIGN_YAW_STEP,
    PREGRASP_HEIGHT,
    GRASP_APPROACH_HEIGHT,
    LIFT_HEIGHT,
    MAX_SKILL_RUNTIME_S,
)


class SharedAutonomyManager:
    """Rule-based shared autonomy manager (v0.1).

    Non-invasive: reads sim state, proposes or runs small skills by directly
    adjusting sim.pos/sim.orn and calling sim._update_robot_state().
    """

    def __init__(self, sim, mode: Optional[str] = None):
        self.sim = sim
        self.mode = mode or ASSIST_MODE

        # Prompt state
        self.pending_prompt: Optional[Dict[str, Any]] = None
        self.last_prompt_time: float = 0.0

        # Running skill state
        self.active_skill: Optional[str] = None
        self.skill_goal_obj_id: Optional[int] = None
        self.skill_phase: str = ""
        self.skill_start_time: float = 0.0
        self.skill_saved_user_mode: Optional[int] = None

    # ------------------ Public API ------------------
    def update(self, user_action: int, user_mode: int) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
        """Main entry. Returns (final_action, prompt_dict).

        If a skill is active, it will progress it. If not, it may propose a prompt
        or return the user action unchanged.
        """
        # If a skill is running, allow immediate override by any active user action
        if self.active_skill is not None and user_action not in (-1, None):
            self._cancel_skill(reason="user_override")

        # Progress a running skill if any
        if self.active_skill is not None:
            self._run_active_skill_step()
            return None, None

        # No skill running; compute intent and possibly propose/auto-run
        state = self.sim.get_state()
        intent = self._infer_intent(state)

        # Offer or act based on simple arbitration
        if user_mode == 1:  # Orientation mode
            if self._detect_struggle():
                if self._should_auto(HIGH_CONF_AUTO_THRESH):
                    self._start_skill("auto_align", intent.get("goal_id"))
                    return None, None
                else:
                    self._maybe_prompt({
                        "type": "auto_align",
                        "text": "You seem to be struggling with orientation. Auto-align gripper? (G accept / H decline)",
                        "goal_id": intent.get("goal_id"),
                    })
        elif user_mode == 2:  # Gripper mode
            if intent.get("near") and intent.get("confidence", 0.0) >= INTENT_CONF_THRESH and self.sim.gripper_state in ("open",):
                if self._should_auto(intent.get("confidence", 0.0)):
                    self._start_skill("auto_grasp", intent.get("goal_id"))
                    return None, None
                else:
                    self._maybe_prompt({
                        "type": "auto_grasp",
                        "text": "I can complete the grasp on the nearest object. Proceed? (G accept / H decline)",
                        "goal_id": intent.get("goal_id"),
                    })

        # No special action; pass through user action
        return user_action, self.pending_prompt

    def accept(self):
        if not self.pending_prompt:
            return
        prompt = self.pending_prompt
        self.pending_prompt = None
        self.last_prompt_time = time.time()
        self._start_skill(prompt.get("type"), prompt.get("goal_id"))

    def decline(self):
        if not self.pending_prompt:
            return
        self.pending_prompt = None
        self.last_prompt_time = time.time()

    # ------------------ Intent & heuristics ------------------
    def _infer_intent(self, state: Dict[str, Any]) -> Dict[str, Any]:
        robot_pos = np.array(state["robot_state"]["position"])  # EE position
        objects = state["environment"].get("objects", [])
        if not objects:
            return {"goal_id": None, "near": False, "confidence": 0.0}

        # Nearest object in XY plane
        min_d = 1e9
        goal_id = None
        goal_pos = None
        for obj in objects:
            pos = np.array(obj["position"])  # object base pos
            d = np.linalg.norm((robot_pos[:2] - pos[:2]))
            if d < min_d:
                min_d = d
                goal_id = obj["id"]
                goal_pos = pos

        near = min_d <= INTENT_DIST_THRESH
        # Simple confidence based on distance (clipped) and forward progress could be added later
        conf = float(np.clip(1.0 - (min_d / (INTENT_DIST_THRESH + 1e-6)), 0.0, 1.0))
        return {"goal_id": goal_id, "near": near, "confidence": conf, "goal_pos": goal_pos}

    def _detect_struggle(self) -> bool:
        # Heuristic: many direction flips in orientation actions in last 2s
        hist = self.sim.get_action_history(within_seconds=2.0)
        if len(hist) < 4:
            return False
        seq = [h.get("state", -1) for h in hist if h.get("state", -1) in (6, 7, 8, 9, 10, 11)]
        if len(seq) < 4:
            return False
        flips = 0
        for a, b in zip(seq[:-1], seq[1:]):
            # opposite pairs: (6,7), (8,9), (10,11)
            if (a, b) in ((6, 7), (7, 6), (8, 9), (9, 8), (10, 11), (11, 10)):
                flips += 1
        return flips >= 2

    def _should_auto(self, confidence: float) -> bool:
        if self.mode == "auto_high_conf" and confidence >= HIGH_CONF_AUTO_THRESH:
            return True
        return False

    def _maybe_prompt(self, prompt: Dict[str, Any]) -> None:
        # Throttle prompts to avoid spamming
        now = time.time()
        if self.pending_prompt is not None:
            return
        if (now - self.last_prompt_time) < 1.0:
            return
        self.pending_prompt = prompt

    # ------------------ Skills ------------------
    def _start_skill(self, skill_type: str, goal_obj_id: Optional[int]) -> None:
        if goal_obj_id is None:
            return
        self.active_skill = skill_type
        self.skill_goal_obj_id = goal_obj_id
        self.skill_phase = "start"
        self.skill_start_time = time.time()

    def _cancel_skill(self, reason: str = "") -> None:
        self.active_skill = None
        self.skill_goal_obj_id = None
        self.skill_phase = ""
        self.skill_start_time = 0.0

    def _run_active_skill_step(self) -> None:
        if self.active_skill == "auto_align":
            aligned = self._step_auto_align(inline=False)
            if aligned:
                # finish standalone align
                self._cancel_skill("done")
        elif self.active_skill == "auto_grasp":
            self._step_auto_grasp()
        else:
            self._cancel_skill()

    def _get_object_pose(self, obj_id: int):
        try:
            import pybullet as p
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            return np.array(pos), np.array(orn)
        except Exception:
            return None, None

    def _step_auto_align(self, inline: bool = False) -> bool:
        # Yaw-align EE X axis to face object horizontally
        if (time.time() - self.skill_start_time) > MAX_SKILL_RUNTIME_S:
            if not inline:
                self._cancel_skill("timeout")
            return True

        obj_pos, _ = self._get_object_pose(self.skill_goal_obj_id)
        if obj_pos is None:
            if not inline:
                self._cancel_skill("obj_missing")
            return True

        ee_pos = np.array(self.sim.pos)
        vec = obj_pos[:2] - ee_pos[:2]
        if np.linalg.norm(vec) < 1e-6:
            if not inline:
                self._cancel_skill("degenerate")
            return True
        target_yaw = math.atan2(vec[1], vec[0])

        # Current yaw from orientation quaternion: project EE X axis onto XY
        from scipy.spatial.transform import Rotation as R
        Rm = R.from_quat(self.sim.orn).as_matrix()
        x_axis = Rm[:, 0]
        current_yaw = math.atan2(x_axis[1], x_axis[0])
        yaw_err = (target_yaw - current_yaw + math.pi) % (2 * math.pi) - math.pi

        if abs(yaw_err) <= ALIGN_YAW_ERR_THRESH:
            # aligned
            return True

        delta = np.clip(yaw_err, -ALIGN_YAW_STEP, ALIGN_YAW_STEP)
        Rz = np.array([[math.cos(delta), -math.sin(delta), 0.0],
                       [math.sin(delta),  math.cos(delta), 0.0],
                       [0.0,              0.0,             1.0]])
        new_R = Rz @ Rm
        new_q = R.from_matrix(new_R).as_quat()
        self.sim.orn = list(new_q)
        self.sim.newPosInput = 1
        self.sim._update_robot_state()
        return False

    def _step_auto_grasp(self) -> None:
        now = time.time()
        if (now - self.skill_start_time) > MAX_SKILL_RUNTIME_S:
            return self._cancel_skill("timeout")

        obj_pos, _ = self._get_object_pose(self.skill_goal_obj_id)
        if obj_pos is None:
            return self._cancel_skill("obj_missing")

        # Ensure yaw roughly aligned along approach
        if self.skill_phase in ("start", "align"):
            self.skill_phase = "align"
            aligned = self._step_auto_align(inline=True)
            if aligned:
                # proceed when alignment achieved
                self.skill_phase = "pregrasp"
            return

        # Pregrasp positioning: go above object
        if self.skill_phase == "pregrasp":
            target = np.array(self.sim.pos)
            target[:2] = obj_pos[:2]
            target[2] = max(target[2], obj_pos[2] + PREGRASP_HEIGHT)
            self._move_towards(target, step=0.01)
            if np.linalg.norm(np.array(self.sim.pos) - target) < 0.01:
                self.skill_phase = "descend"
            return self.sim._update_robot_state()

        # Descend to grasp height
        if self.skill_phase == "descend":
            target = np.array(self.sim.pos)
            target[2] = max(obj_pos[2] + GRASP_APPROACH_HEIGHT, self.sim.min_z_height)
            self._move_towards(target, step=0.008)
            if abs(self.sim.pos[2] - target[2]) < 0.008:
                # Close gripper
                self.sim.process_command(13, 2)  # close
                self.skill_phase = "lift"
            return self.sim._update_robot_state()

        # Lift after grasp
        if self.skill_phase == "lift":
            target = np.array(self.sim.pos)
            target[2] = min(self.sim.wu[2], obj_pos[2] + LIFT_HEIGHT)
            self._move_towards(target, step=0.01)
            if abs(self.sim.pos[2] - target[2]) < 0.01:
                return self._cancel_skill("done")
            return self.sim._update_robot_state()

    # ------------------ Utils ------------------
    def _move_towards(self, target: np.ndarray, step: float = 0.01) -> None:
        cur = np.array(self.sim.pos)
        delta = target - cur
        dist = float(np.linalg.norm(delta))
        if dist < 1e-6:
            return
        direction = delta / dist
        new_pos = cur + direction * min(step, dist)
        # Assign and let sim enforce workspace limits
        self.sim.pos = list(new_pos)
        self.sim.newPosInput = 1

