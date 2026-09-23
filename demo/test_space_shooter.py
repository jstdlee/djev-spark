import json
import unittest
from pathlib import Path

from space_shooter_server import DEFAULT_DECISION, build_decision_request, build_trajectory_request, normalize_decision


class SpaceShooterTests(unittest.TestCase):
    def test_decision_request_contains_game_state_and_wave_questions(self):
        request = build_decision_request({"wave": 3, "score": 420, "lives": 2, "threat": "steady"})
        self.assertEqual(request["state"]["wave"], 3)
        self.assertEqual(request["state"]["score"], 420)
        self.assertIn("bullet_pattern", request["questions"])
        self.assertIn("enemy_type", request["questions"])

    def test_decision_request_is_near_threat_and_constraint_aware(self):
        request = build_decision_request({
            "wave": 3,
            "score": 420,
            "lives": 2,
            "player_position": {"x": 480, "y": 520},
            "bombs_remaining": 3,
            "nearby_bullets": [{"distance": 96, "time_to_closest": 0.42, "risk": 0.91}],
            "nearby_enemies": [{"distance": 150, "closing_speed": 120, "risk": 0.65}],
        })
        self.assertIn("controller", request)
        self.assertIn("nearby_threats", request["state"])
        self.assertIn("constraints", request)
        self.assertIn("movement", request["questions"])
        self.assertIn("urgency", request["questions"])
        self.assertIn("mode", request["questions"])
        self.assertIn("decision_tree", request["controller"])
        self.assertIn("preferred_center", request["state"])
        self.assertIn("center_delta", request["state"])
        self.assertIn("control_phase", request["state"])
        self.assertEqual(request["controller"]["decision_horizon_ms"], 1200)
        self.assertLessEqual(len(request["state"]["nearby_threats"]["bullets"]), 8)

    def test_decision_request_contains_structured_battlefield_summary(self):
        request = build_decision_request({
            "bombs_remaining": 2,
            "battlefield_summary": {
                "max_risk": 0.82,
                "nearest_bullet_tti": 0.41,
                "imminent_bullets": 2,
                "imminent_enemies": 1,
                "escape_lanes": {"left": 0.2, "right": 0.8, "up": 0.4, "down": 0.6},
            },
            "escape_lanes": {"left": 0.2, "right": 0.8, "up": 0.4, "down": 0.6},
        })
        summary = request["state"]["battlefield_summary"]
        self.assertIn("nearest_bullet_tti", summary)
        self.assertIn("escape_lanes", summary)
        self.assertIn("imminent_bullets", summary)
        self.assertEqual(request["controller"]["execution_model"], "parallel_local_control")

    def test_trajectory_request_is_separate_and_ready_for_movement_context(self):
        request = build_trajectory_request({
            "player_position": {"x": 480, "y": 420},
            "player_velocity": {"x": 4, "y": 0},
            "nearby_threats": {"radius_px": 360, "bullets": [{"x": 480, "y": 250, "vx": 0, "vy": 210, "risk": 0.8}], "enemies": []},
            "escape_lanes": {"left": 0.2, "right": 0.7, "up": 0.4, "down": 0.5},
        })
        self.assertEqual(request["controller"]["request_kind"], "trajectory_prediction")
        self.assertIn("predicted_collision_point", request["questions"])
        self.assertIn("recommended_lane", request["questions"])
        self.assertIn("movement_context", request["state"])

    def test_normalize_decision_returns_safe_fallback_for_bad_api_output(self):
        self.assertEqual(normalize_decision({}), DEFAULT_DECISION)
        self.assertEqual(normalize_decision({"answers": {"bullet_pattern": {"choice": "sweep"}}})["bullet_pattern"], "sweep")

    def test_normalize_decision_includes_conservative_movement_directive(self):
        result = normalize_decision({"answers": {
            "mode": {"choice": "recenter", "confidence": 0.8},
            "movement": {"choice": "left", "confidence": 0.8},
            "urgency": {"choice": "evade", "confidence": 0.9},
        }})
        self.assertEqual(result["mode"], "recenter")
        self.assertEqual(result["movement"], "left")
        self.assertEqual(result["urgency"], "evade")

    def test_recenter_fills_missing_direction_from_center_delta(self):
        result = normalize_decision(
            {"answers": {"mode": {"choice": "recenter"}}},
            {"player_position": {"x": 700, "y": 500}, "preferred_center": {"x": 480, "y": 420}},
        )
        self.assertEqual(result["mode"], "recenter")
        self.assertEqual(result["movement"], "left")

    def test_single_html_contains_game_panel_decision_panel_and_bridge_call(self):
        html = Path(__file__).with_name("space-shooter.html").read_text(encoding="utf-8")
        self.assertIn('id="game-panel"', html)
        self.assertIn('id="decision-panel"', html)
        self.assertIn('fetch("/api/decision"', html)

    def test_html_has_a_periodic_control_pulse_for_visible_loop(self):
        html = Path(__file__).with_name("space-shooter.html").read_text(encoding="utf-8")
        self.assertIn("DIRECTOR_PULSE_MS", html)
        self.assertIn('requestDecision("control pulse")', html)

    def test_html_has_local_autopilot_for_reactive_dodging(self):
        html = Path(__file__).with_name("space-shooter.html").read_text(encoding="utf-8")
        self.assertIn("AUTO PILOT ON", html)
        self.assertIn("findSafeTarget", html)
        self.assertIn("updateAutopilot", html)

    def test_html_has_small_step_motion_without_bomb_shortcuts(self):
        html = Path(__file__).with_name("space-shooter.html").read_text(encoding="utf-8")
        self.assertIn("MAX_AUTOPILOT_SPEED", html)
        self.assertNotIn('id="bomb-button"', html)
        self.assertNotIn("maybeUseBomb", html)

    def test_controller_declares_parallel_thinking_and_local_control(self):
        request = build_decision_request({"bombs_remaining": 3})
        self.assertEqual(request["controller"]["execution_model"], "parallel_local_control")
        self.assertTrue(any("JevSpark thinking never blocks local frame control" in item for item in request["constraints"]))

    def test_html_stops_decisions_after_game_over_and_uses_high_frequency_control(self):
        html = Path(__file__).with_name("space-shooter.html").read_text(encoding="utf-8")
        self.assertIn("LOCAL_CONTROL_TICK_MS", html)
        self.assertIn("stopDecisionLoop", html)
        self.assertIn("parallel: local control + Jev thinking", html)

    def test_html_sends_escape_lanes_and_runs_high_frequency_pulses(self):
        html = Path(__file__).with_name("space-shooter.html").read_text(encoding="utf-8")
        self.assertIn("escape_lanes", html)
        self.assertIn("battlefield_summary", html)
        self.assertIn("const DIRECTOR_PULSE_MS = 450", html)
        self.assertIn("const LOCAL_CONTROL_TICK_MS = 40", html)
        self.assertIn("trajectory_prediction", html)
        self.assertIn("requestTrajectoryPrediction", html)

    def test_enemy_escape_only_costs_hull_on_actual_player_collision(self):
        html = Path(__file__).with_name("space-shooter.html").read_text(encoding="utf-8")
        self.assertIn("if (overlaps(enemy, player)) hitPlayer();", html)

    def test_survival_model_has_a_center_corridor(self):
        html = Path(__file__).with_name("space-shooter.html").read_text(encoding="utf-8")
        self.assertIn("SAFE_CORRIDOR", html)
        self.assertIn("CENTER_TARGET", html)

    def test_prediction_is_used_as_a_small_directional_bias(self):
        html = Path(__file__).with_name("space-shooter.html").read_text(encoding="utf-8")
        self.assertIn("recommended_lane", html)
        self.assertIn("trajectoryPrediction.recommended_lane", html)

    def test_trajectory_forecast_scores_the_motion_path_and_all_lane_failure(self):
        html = Path(__file__).with_name("space-shooter.html").read_text(encoding="utf-8")
        self.assertIn("pathRiskAt", html)
        self.assertIn("trajectoryPrediction.predicted_collision_point", html)

    def test_survival_guard_keeps_bullet_density_playable_for_the_endurance_demo(self):
        html = Path(__file__).with_name("space-shooter.html").read_text(encoding="utf-8")
        self.assertIn("Math.max(.72, 1.9 / factor)", html)
        self.assertIn("[-.18, 0, .18]", html)


if __name__ == "__main__":
    unittest.main()
