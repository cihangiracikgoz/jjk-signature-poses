import json
import sys
from src.logger import logger

DEFAULT_CONFIG_PATH = "config.json"


def load_config(path=DEFAULT_CONFIG_PATH):
    try:
        with open(path, "r") as f:
            raw = json.load(f)
    except FileNotFoundError:
        logger.error("Config file not found: %s", path)
        logger.error("Copy config.example.json to config.json and customize it.")
        sys.exit(1)
    except json.JSONDecodeError as e:
        logger.error("Invalid JSON in config file: %s", e)
        sys.exit(1)

    gestures = raw.get("gestures", [])
    if not gestures:
        logger.error("No gestures defined in config file")
        sys.exit(1)

    label_names = {0: raw.get("idle_label", "IDLE")}
    gif_mapping = {}

    for i, gesture in enumerate(gestures, start=1):
        label_names[i] = gesture["label"]
        gif_mapping[i] = (gesture["label"].lower(), gesture["gif"])

    return {
        "app_name": raw.get("app_name", "Gesture Recognition"),
        "label_names": label_names,
        "gif_mapping": gif_mapping,
    }
