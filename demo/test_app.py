import math
import unittest

from demo.app import (
    Metrics,
    action_for,
    dataset_size,
    domain_signals,
    extract_domain,
    filter_table_rows,
    normalize_label,
    render_status,
)


class DemoAppTests(unittest.TestCase):
    def test_normalize_labels_from_both_hugging_face_sources(self):
        self.assertEqual(normalize_label("spam"), "spam")
        self.assertEqual(normalize_label("ham"), "legitimate")
        self.assertEqual(normalize_label("phishing"), "scam")
        self.assertEqual(normalize_label("benign"), "legitimate")


    def test_action_mapping_uses_risk_and_confidence(self):
        self.assertEqual(action_for("scam", 0.92), "Quarantine")
        self.assertEqual(action_for("spam", 0.88), "Quarantine")
        self.assertEqual(action_for("legitimate", 0.91), "Deliver")
        self.assertEqual(action_for("spam", 0.54), "Review")
        self.assertEqual(action_for("legitimate", 0.91, sender_suspicious=True), "Review")

    def test_sender_domain_signals_are_visible_to_the_decision(self):
        self.assertEqual(extract_domain("alerts@Example.COM"), "example.com")
        self.assertIn("punycode", domain_signals("alerts@xn--secure-login-9za.example"))
        self.assertIn("ip-domain", domain_signals("alerts@192.0.2.10"))

    def test_dataset_sizes_and_wrong_only_filter(self):
        self.assertEqual(dataset_size("Curated phishing / benign"), 200)
        self.assertEqual(dataset_size("AURA rich phishing corpus"), 112013)
        rows = [[1, "a", "", "", "", "", "", "", "", "", "✅ match", "", "", ""]]
        rows.append([2, "b", "", "", "", "", "", "", "", "", "❌ miss", "", "", ""])
        self.assertEqual(len(filter_table_rows(rows, True)), 1)
        self.assertEqual(len(filter_table_rows(rows, False)), 2)


    def test_metrics_report_dynamic_token_speed_and_result_speed(self):
        metrics = Metrics()
        metrics.observe(input_tokens=100, output_tokens=20, elapsed_s=2.0, correct=True)
        metrics.observe(input_tokens=120, output_tokens=30, elapsed_s=1.0, correct=False)

        snapshot = metrics.snapshot()
        self.assertEqual(snapshot["processed"], 2)
        self.assertEqual(snapshot["correct"], 1)
        self.assertTrue(math.isclose(snapshot["result_speed"], 2 / 3))
        self.assertTrue(math.isclose(snapshot["token_tps"], 90.0))
        self.assertTrue(math.isclose(snapshot["output_tps"], 50 / 3))


    def test_dynamic_status_indicator_changes_with_run_state(self):
        self.assertIn("RUNNING", render_status(processed=2, total=5, latest_ms=800, token_tps=42))
        self.assertIn("COMPLETE", render_status(processed=5, total=5, latest_ms=800, token_tps=42))


if __name__ == "__main__":
    unittest.main()
