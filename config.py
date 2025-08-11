# config.py

# Debug mode for gesture recognition and AI model
DEBUG_MODE = False

# User ID or name for personalized gesture data/model
USER_ID = "Ali3"

# Random seed for reproducible object spawning (set to None for random each time)
RANDOM_SEED = 40

# List of all 14 actions: (action_id, description)
ACTIONS = [
    (0, "Forw"),
    (1, "Backw"),
    (2, "Left"),
    (3, "Right"),
    (4, "Up"),
    (5, "Down"),
    (6, "X+"),
    (7, "X-"),
    (8, "Y+"),
    (9, "Y-"),
    (10, "Z+"),
    (11, "Z-"),
    (12, "Open"),
    (13, "Close"),
]


# Shared autonomy defaults (v0.1)
# Modes: "off", "suggest", "auto_high_conf"
ASSIST_MODE = "suggest"

# Intent/assist thresholds
INTENT_DIST_THRESH = 0.18       # meters, consider object near
INTENT_CONF_THRESH = 0.55       # simple heuristic confidence
HIGH_CONF_AUTO_THRESH = 0.8     # auto-run if above and mode is auto_high_conf

# Auto-align parameters
ALIGN_YAW_ERR_THRESH = 0.15     # radians, within ~9 deg
ALIGN_YAW_STEP = 0.05           # radians per control tick

# Auto-grasp parameters
PREGRASP_HEIGHT = 0.10          # meters above object
GRASP_APPROACH_HEIGHT = 0.02    # meters above object before close
LIFT_HEIGHT = 0.10              # meters to lift after grasp
MAX_SKILL_RUNTIME_S = 6.0       # safety timeout for skills

