"""Single-page space-shooter demo server with a local djev API bridge."""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


DEMO_DIR = Path(__file__).resolve().parent
HTML_PATH = DEMO_DIR / "space-shooter.html"
DEFAULT_DECISION = {
    "enemy_type": "scout",
    "bullet_pattern": "aimed",
    "difficulty": "normal",
    "mode": "recenter",
    "movement": "hold",
    "urgency": "observe",
    "confidence": 0.0,
    "reason": "No nearby collision is imminent; move a small step toward the center corridor.",
    "directive": "normal aimed wave · recenter / hold",
}
ALLOWED = {
    "enemy_type": {"scout", "tank", "swarm"},
    "bullet_pattern": {"aimed", "spray", "sweep"},
    "difficulty": {"easy", "normal", "hard", "extreme"},
    "mode": {"recenter", "hold_center", "adjust", "evade"},
    "movement": {"hold", "left", "right", "up", "down"},
    "urgency": {"observe", "adjust", "evade"},
}


def _load_env_file() -> None:
    env_path = DEMO_DIR.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def build_decision_request(state: dict[str, Any]) -> dict[str, Any]:
    raw_threats = state.get("nearby_threats", {}) or {}
    if not isinstance(raw_threats, dict):
        raw_threats = {}
    raw_bullets = state.get("nearby_bullets", raw_threats.get("bullets", [])) or []
    raw_enemies = state.get("nearby_enemies", raw_threats.get("enemies", [])) or []

    def compact_threats(items: Any, limit: int, fields: tuple[str, ...]) -> list[dict[str, Any]]:
        valid = [item for item in items if isinstance(item, dict)]
        def risk_value(item: dict[str, Any]) -> float:
            try:
                return float(item.get("risk", 0) or 0)
            except (TypeError, ValueError):
                return 0.0

        valid.sort(key=risk_value, reverse=True)
        return [{field: item.get(field) for field in fields if field in item} for item in valid[:limit]]

    nearby_threats = {
        "radius_px": int(raw_threats.get("radius_px", 300) or 300),
        "bullets": compact_threats(
            raw_bullets,
            8,
            ("distance", "time_to_closest", "risk", "x", "y", "vx", "vy", "closing_speed", "pattern"),
        ),
        "enemies": compact_threats(
            raw_enemies,
            6,
            ("distance", "time_to_contact", "risk", "x", "y", "vx", "vy", "closing_speed", "type", "hp"),
        ),
    }
    preferred_center = state.get("preferred_center", {"x": 480, "y": 420})
    if not isinstance(preferred_center, dict):
        preferred_center = {"x": 480, "y": 420}
    control_phase = str(state.get("control_phase", "recenter"))
    raw_summary = state.get("battlefield_summary", {}) or {}
    if not isinstance(raw_summary, dict):
        raw_summary = {}
    raw_lanes = state.get("escape_lanes", raw_summary.get("escape_lanes", {})) or {}
    if not isinstance(raw_lanes, dict):
        raw_lanes = {}

    def bounded_number(value: Any, default: float = 0.0) -> float:
        try:
            return round(float(value), 3)
        except (TypeError, ValueError):
            return default

    battlefield_summary = {
        "scan_radius_px": nearby_threats["radius_px"],
        "bullets_visible": len([item for item in raw_bullets if isinstance(item, dict)]),
        "enemies_visible": len([item for item in raw_enemies if isinstance(item, dict)]),
        "max_risk": bounded_number(raw_summary.get("max_risk", 0)),
        "nearest_bullet_tti": bounded_number(raw_summary.get("nearest_bullet_tti", 1.2), 1.2),
        "nearest_enemy_tti": bounded_number(raw_summary.get("nearest_enemy_tti", 1.2), 1.2),
        "imminent_bullets": max(0, int(raw_summary.get("imminent_bullets", 0) or 0)),
        "imminent_enemies": max(0, int(raw_summary.get("imminent_enemies", 0) or 0)),
        "no_survival": bool(raw_summary.get("no_survival", False)),
        "escape_lanes": {
            lane: bounded_number(raw_lanes.get(lane, 1.0), 1.0)
            for lane in ("left", "right", "up", "down")
        },
        "safe_target": state.get("safe_target", {"x": 0, "y": 0}),
    }
    return {
        "model": "jev-latest",
        "controller": {
            "role": "conservative real-time space-shooter director",
            "execution_model": "parallel_local_control",
            "local_control_tick_ms": 40,
            "decision_horizon_ms": 1200,
            "replan_interval_ms": 450,
            "near_threat_radius_px": nearby_threats["radius_px"],
            "local_reaction": "The browser autopilot reacts every frame; this decision only sets a small tactical bias.",
            "movement_contract": "Prefer hold and small displacement. Allow a larger move only when a near collision is imminent and no small step can survive.",
            "output_contract": "Return exactly one enemy profile, bullet pattern, difficulty, movement direction, and urgency.",
            "decision_tree": [
                "evade: a nearby bullet/enemy is likely to collide within 0.8 seconds",
                "adjust: a nearby threat exists but collision is not imminent; make a small correction",
                "recenter: no nearby threat is dangerous; move a small step toward preferred_center for maneuvering room",
                "hold_center: already inside the center corridor; preserve position and wait for new information",
            ],
        },
        "state": {
            "wave": int(state.get("wave", 1)),
            "score": int(state.get("score", 0)),
            "lives": int(state.get("lives", 3)),
            "threat": str(state.get("threat", "steady")),
            "player_position": state.get("player_position", {"x": 0, "y": 0}),
            "player_velocity": state.get("player_velocity", {"x": 0, "y": 0}),
            "autopilot": bool(state.get("autopilot", True)),
            "local_evasion_count": int(state.get("local_evasion_count", 0) or 0),
            "preferred_center": preferred_center,
            "center_delta": state.get("center_delta", {"x": 0, "y": 0}),
            "center_error_px": float(state.get("center_error_px", 0) or 0),
            "maneuver_room": state.get("maneuver_room", {}),
            "control_phase": control_phase,
            "nearby_threats": nearby_threats,
            "battlefield_summary": battlefield_summary,
            "trajectory_prediction": state.get("trajectory_prediction", {}),
        },
        "constraints": [
            "Only reason about the nearby threats supplied in state; do not invent distant threats.",
            "JevSpark thinking never blocks local frame control; the browser continues small-step movement while this request is in flight.",
            "Use battlefield_summary.escape_lanes and nearest collision times to choose the least dangerous small steering bias.",
            "Prefer a safe lane with more maneuvering room over a short route toward an enemy or the exact center.",
            "Treat trajectory_prediction as the latest forecast context; prefer the small lane that avoids its predicted collision point.",
            "If no nearby collision is imminent, choose mode=recenter unless the player is already within the center corridor; then choose mode=hold_center.",
            "In recenter mode, choose the direction that reduces center_error_px, even when there is no threat.",
            "Use left/right/up/down as a small steering bias, not a teleport or frame-by-frame command.",
            "Choose urgency=evade only for a likely collision within the short decision horizon.",
            "Keep the player inside the arena and preserve room for the local autopilot to react.",
        ],
        "questions": {
            "enemy_type": {
                "type": "choice",
                "instructions": "Choose the next enemy profile for a simple space shooter wave.",
                "criteria": {
                    "scout": "fast, fragile enemy",
                    "tank": "slow, durable enemy",
                    "swarm": "many small enemies",
                },
            },
            "bullet_pattern": {
                "type": "choice",
                "instructions": "Choose the bullet pattern that makes the next wave challenging but playable.",
                "criteria": {
                    "aimed": "bullets aim at the player's current position",
                    "spray": "a wide spread of bullets",
                    "sweep": "a horizontal sweep across the arena",
                },
            },
            "difficulty": {
                "type": "score",
                "instructions": "Set the next wave difficulty based on the player's state.",
                "criteria": ["easy", "normal", "hard", "extreme"],
            },
            "mode": {
                "type": "choice",
                "instructions": "Choose the current branch of the conservative decision tree.",
                "criteria": {
                    "recenter": "no dangerous nearby threat; take a small step toward the preferred center corridor",
                    "hold_center": "no dangerous nearby threat and the player already has maneuvering room",
                    "adjust": "a nearby threat needs a small correction but not a full dodge",
                    "evade": "a collision is likely within 0.8 seconds; permit a stronger move",
                },
            },
            "movement": {
                "type": "choice",
                "instructions": "Choose the smallest useful movement bias for the next 1.2 seconds. In recenter mode, point toward preferred_center; in hold_center, choose hold.",
                "criteria": {
                    "hold": "keep the current lane and let the local autopilot make only tiny corrections",
                    "left": "drift left a short distance",
                    "right": "drift right a short distance",
                    "up": "move upward a short distance",
                    "down": "move downward a short distance",
                },
            },
            "urgency": {
                "type": "choice",
                "instructions": "Classify how urgently the local autopilot should change position.",
                "criteria": {
                    "observe": "no near collision; preserve position",
                    "adjust": "near threat exists; make a small correction",
                    "evade": "collision is likely soon; permit a stronger dodge",
                },
            },
        },
        "samples": 1,
    }


def build_trajectory_request(state: dict[str, Any]) -> dict[str, Any]:
    """Build a compact, separate request for short-horizon threat prediction."""
    decision_request = build_decision_request(state)
    decision_state = decision_request["state"]
    summary = decision_state.get("battlefield_summary", {}) or {}
    return {
        "model": "jev-latest",
        "controller": {
            "role": "short-horizon projectile trajectory predictor",
            "request_kind": "trajectory_prediction",
            "execution_model": "parallel_local_control",
            "horizon_ms": 900,
            "output_contract": "Return predicted collision point, time to crossing, and safest lane without issuing a long movement command.",
        },
        "state": {
            "player_position": decision_state["player_position"],
            "player_velocity": decision_state["player_velocity"],
            "nearby_threats": decision_state["nearby_threats"],
            "movement_context": {
                "escape_lanes": summary.get("escape_lanes", {}),
                "safe_target": summary.get("safe_target", {"x": 0, "y": 0}),
                "center_error_px": decision_state.get("center_error_px", 0),
                "maneuver_room": decision_state.get("maneuver_room", {}),
            },
        },
        "constraints": [
            "Predict only supplied nearby bullet and enemy trajectories.",
            "Use the player's small-step movement envelope when estimating a crossing point.",
            "Return a lane recommendation as context for a later movement decision; do not treat this request as the movement command.",
        ],
        "questions": {
            "predicted_collision_point": {
                "type": "structured",
                "instructions": "Estimate the earliest nearby bullet or enemy crossing point with the player's safe collision envelope.",
            },
            "recommended_lane": {
                "type": "choice",
                "instructions": "Choose the least dangerous small-step lane for the next short horizon.",
                "criteria": ["hold", "left", "right", "up", "down"],
            },
            "trajectory_confidence": {
                "type": "score",
                "instructions": "Estimate confidence in the predicted crossing point and lane.",
            },
        },
        "samples": 1,
    }


def _choice(answer: Any, field: str) -> str:
    if isinstance(answer, dict):
        value = answer.get("choice", answer.get("label", ""))
    else:
        value = answer
    value = str(value).strip().lower()
    return value if value in ALLOWED[field] else DEFAULT_DECISION[field]


def normalize_decision(response: dict[str, Any], state: dict[str, Any] | None = None) -> dict[str, Any]:
    answers = response.get("answers", {}) if isinstance(response, dict) else {}
    classification = answers.get("enemy_type", {}) or {}
    pattern = answers.get("bullet_pattern", {}) or {}
    difficulty = answers.get("difficulty", {}) or {}
    mode = answers.get("mode", {}) or {}
    movement = answers.get("movement", {}) or {}
    urgency = answers.get("urgency", {}) or {}

    def confidence_for(key: str) -> float:
        try:
            return max(0.0, min(1.0, float((answers.get(key, {}) or {}).get("confidence", 0.0) or 0.0)))
        except (TypeError, ValueError):
            return 0.0

    confidence_values = [
        confidence_for(key)
        for key in ("enemy_type", "bullet_pattern", "difficulty", "mode", "movement", "urgency")
    ]
    confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0.0
    mode_value = _choice(mode, "mode")
    movement_value = _choice(movement, "movement")
    if mode_value == "recenter" and movement_value == "hold" and state:
        position = state.get("player_position", {}) or {}
        center = state.get("preferred_center", {"x": 480, "y": 420}) or {"x": 480, "y": 420}
        try:
            delta_x = float(center.get("x", 480)) - float(position.get("x", 480))
            delta_y = float(center.get("y", 420)) - float(position.get("y", 420))
            if abs(delta_x) >= 24 or abs(delta_y) >= 24:
                movement_value = "left" if abs(delta_x) >= abs(delta_y) and delta_x < 0 else "right" if abs(delta_x) >= abs(delta_y) else "up" if delta_y < 0 else "down"
        except (AttributeError, TypeError, ValueError):
            pass
    urgency_value = _choice(urgency, "urgency")
    reason = movement.get("reason") if isinstance(movement, dict) else None
    reason = str(reason).strip() if reason else {
        "recenter": "No nearby collision is imminent; move a small step toward the center corridor.",
        "hold_center": "The player already has maneuvering room; preserve the center corridor.",
        "adjust": f"A small {movement_value} correction is preferred for the nearby threat.",
        "evade": f"A collision is likely soon; use {movement_value} to create escape room.",
    }[mode_value]
    difficulty_value = _choice(difficulty, "difficulty")
    return {
        "enemy_type": _choice(classification, "enemy_type"),
        "bullet_pattern": _choice(pattern, "bullet_pattern"),
        "difficulty": difficulty_value,
        "mode": mode_value,
        "movement": movement_value,
        "urgency": urgency_value,
        "confidence": round(confidence, 3),
        "reason": reason,
        "directive": f"{difficulty_value} {_choice(pattern, 'bullet_pattern')} wave · {mode_value} / {movement_value}",
    }


def _fallback_result(error: str = "djev unavailable") -> dict[str, Any]:
    return {**DEFAULT_DECISION, "api_ok": False, "error": error, "latency_ms": 0, "token_tps": 0, "result_speed": 0, "usage": {}}


def _fallback_trajectory(state: dict[str, Any], error: str = "djev unavailable") -> dict[str, Any]:
    summary = state.get("battlefield_summary", {}) or {}
    return {
        "prediction_source": "local-fallback",
        "recommended_lane": "hold",
        "predicted_collision_point": None,
        "trajectory_tracks": [],
        "confidence": 0.0,
        "api_ok": False,
        "error": error,
        "latency_ms": 0,
        "token_tps": 0,
        "result_speed": 0,
        "usage": {},
        "escape_lanes": summary.get("escape_lanes", {}),
    }


def normalize_trajectory(response: dict[str, Any], state: dict[str, Any] | None = None) -> dict[str, Any]:
    answers = response.get("answers", {}) if isinstance(response, dict) else {}
    point_answer = answers.get("predicted_collision_point", {}) or {}
    lane_answer = answers.get("recommended_lane", {}) or {}
    confidence_answer = answers.get("trajectory_confidence", {}) or {}
    if isinstance(point_answer, dict):
        point = point_answer.get("value", point_answer.get("point", point_answer.get("answer")))
    else:
        point = point_answer
    if isinstance(lane_answer, dict):
        lane = str(lane_answer.get("choice", lane_answer.get("label", "hold"))).strip().lower()
    else:
        lane = str(lane_answer or "hold").strip().lower()
    if lane not in {"hold", "left", "right", "up", "down"}:
        lane = "hold"
    if isinstance(confidence_answer, dict):
        confidence_value = confidence_answer.get("confidence", confidence_answer.get("score", confidence_answer.get("value", 0)))
    else:
        confidence_value = confidence_answer
    try:
        confidence = max(0.0, min(1.0, float(confidence_value or 0)))
    except (TypeError, ValueError):
        confidence = 0.0
    return {
        "prediction_source": "jev",
        "recommended_lane": lane,
        "predicted_collision_point": point,
        "trajectory_tracks": response.get("trajectory_tracks", response.get("tracks", [])) if isinstance(response, dict) else [],
        "confidence": round(confidence, 3),
        "escape_lanes": (state or {}).get("escape_lanes", {}),
    }


def _call_djev(payload: dict[str, Any]) -> tuple[dict[str, Any], float, dict[str, int]]:
    _load_env_file()
    base_url = os.environ.get("DJEV_URL", "http://127.0.0.1:8011").rstrip("/")
    request_body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("DJEV_API_KEY", os.environ.get("API_KEY", ""))
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(base_url + "/v1/systemone", data=request_body, headers=headers, method="POST")
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=45) as response:
        payload = json.loads(response.read().decode("utf-8"))
    elapsed_s = max(time.perf_counter() - started, 1e-9)
    usage = payload.get("usage", {}) or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    return payload, elapsed_s, {"input_tokens": input_tokens, "output_tokens": output_tokens}


def predict_trajectory(state: dict[str, Any]) -> dict[str, Any]:
    try:
        payload, elapsed_s, usage = _call_djev(build_trajectory_request(state))
        result = normalize_trajectory(payload, state)
        result.update({
            "api_ok": True,
            "latency_ms": round(elapsed_s * 1000, 1),
            "token_tps": round((usage["input_tokens"] + usage["output_tokens"]) / elapsed_s, 1),
            "result_speed": round(1 / elapsed_s, 2),
            "usage": usage,
        })
        return result
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return _fallback_trajectory(state, f"{type(exc).__name__}: {exc}")


def decide(state: dict[str, Any]) -> dict[str, Any]:
    try:
        payload, elapsed_s, usage = _call_djev(build_decision_request(state))
        result = normalize_decision(payload, state)
        result.update({
            "api_ok": True,
            "latency_ms": round(elapsed_s * 1000, 1),
            "token_tps": round((usage["input_tokens"] + usage["output_tokens"]) / elapsed_s, 1),
            "result_speed": round(1 / elapsed_s, 2),
            "usage": usage,
        })
        return result
    except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
        return _fallback_result(f"{type(exc).__name__}: {exc}")


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/space-shooter.html"}:
            self._send(200, HTML_PATH.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/health":
            self._send(200, b'{"ok":true}', "application/json")
        else:
            self._send(404, b'{"error":"not found"}', "application/json")

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/api/decision", "/api/trajectory"}:
            self._send(404, b'{"error":"not found"}', "application/json")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            state = json.loads(self.rfile.read(length).decode("utf-8"))
            state = state if isinstance(state, dict) else {}
            result = predict_trajectory(state) if self.path == "/api/trajectory" else decide(state)
            self._send(200, json.dumps(result).encode("utf-8"), "application/json")
        except (ValueError, json.JSONDecodeError) as exc:
            self._send(400, json.dumps({"error": str(exc)}).encode("utf-8"), "application/json")

    def log_message(self, format: str, *args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the one-file JevSpark space shooter demo")
    parser.add_argument("--host", default=os.environ.get("SHOOTER_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("SHOOTER_PORT", "7862")))
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Space shooter: http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
