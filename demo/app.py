"""Gradio benchmark dashboard for the local djev-spark structured API.

The demo evaluates the running local API against labeled Hugging Face email
examples. It is intentionally a benchmark harness, not a mail gateway: no
email is sent, blocked, or modified.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import random
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


DATASETS = {
    "Both rich email corpora": (
        "kudzaiprichard/aura-phishing-email-corpus",
        "Teddyha/phishing_benign_email_dataset",
    ),
    "AURA rich phishing corpus": ("kudzaiprichard/aura-phishing-email-corpus",),
    "Curated phishing / benign": ("Teddyha/phishing_benign_email_dataset",),
}

# Current train-split sizes used to size the Examples control. The loader
# still uses the actual dataset length, so changed upstream data cannot cause
# out-of-range rows.
DATASET_SIZES = {
    "Teddyha/phishing_benign_email_dataset": 200,
    "kudzaiprichard/aura-phishing-email-corpus": 112013,
}

TABLE_HEADERS = [
    "#",
    "Source",
    "Sender",
    "Sender domain",
    "Subject",
    "True label",
    "Model label",
    "Risk",
    "Sender check",
    "Domain signal",
    "Judge",
    "Action",
    "Latency ms",
    "Live token/s",
]


@dataclass
class EmailRow:
    source: str
    sender: str
    subject: str
    body: str
    true_label: str
    raw_label: str = ""

    def state(self) -> dict[str, str]:
        return {
            "from": self.sender,
            "sender_domain": extract_domain(self.sender),
            "sender_domain_signals": ", ".join(domain_signals(self.sender)) or "none",
            "subject": self.subject,
            "body": self.body,
        }


@dataclass
class Metrics:
    processed: int = 0
    correct: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_elapsed_s: float = 0.0
    latest_elapsed_s: float = 0.0
    latest_token_tps: float = 0.0
    started_at: float = field(default_factory=time.perf_counter)

    def observe(self, input_tokens: int, output_tokens: int, elapsed_s: float, correct: bool) -> None:
        elapsed_s = max(float(elapsed_s), 1e-9)
        self.processed += 1
        self.correct += int(correct)
        self.input_tokens += max(0, int(input_tokens))
        self.output_tokens += max(0, int(output_tokens))
        self.total_elapsed_s += elapsed_s
        self.latest_elapsed_s = elapsed_s
        self.latest_token_tps = (input_tokens + output_tokens) / elapsed_s

    def snapshot(self) -> dict[str, float | int]:
        elapsed = max(self.total_elapsed_s, 1e-9)
        wall_elapsed = max(time.perf_counter() - self.started_at, 1e-9)
        total_tokens = self.input_tokens + self.output_tokens
        return {
            "processed": self.processed,
            "correct": self.correct,
            "accuracy": self.correct / self.processed if self.processed else 0.0,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "token_tps": total_tokens / elapsed if self.processed else 0.0,
            "output_tps": self.output_tokens / elapsed if self.processed else 0.0,
            "live_token_tps": self.latest_token_tps,
            "result_speed": self.processed / elapsed if self.processed else 0.0,
            "wall_result_speed": self.processed / wall_elapsed if self.processed else 0.0,
            "latest_ms": self.latest_elapsed_s * 1000,
            "average_ms": elapsed * 1000 / self.processed if self.processed else 0.0,
        }


def normalize_label(value: Any, positive_label: str = "spam") -> str:
    """Map common HF labels to the dashboard's three-class vocabulary."""

    text = str(value).strip().lower()
    if text in {"ham", "benign", "legitimate", "legit", "safe", "0", "false", "none"}:
        return "legitimate"
    if "phish" in text or "scam" in text or "fraud" in text:
        return "scam"
    if "spam" in text or "junk" in text or "unsolicited" in text:
        return "spam"
    if text in {"1", "true"}:
        return positive_label
    return "legitimate"


def normalize_prediction(value: Any) -> str:
    return normalize_label(value, positive_label="spam")


def action_for(label: str, confidence: float, sender_suspicious: bool = False) -> str:
    """Convert a model label into a demo-only mail-handling recommendation."""

    confidence = float(confidence or 0.0)
    if label in {"spam", "scam"}:
        return "Quarantine" if confidence >= 0.72 else "Review"
    if sender_suspicious:
        return "Review"
    return "Deliver" if confidence >= 0.60 else "Review"


def extract_domain(sender: str) -> str:
    """Extract a normalized sender domain for transparent demo heuristics."""

    value = str(sender or "").strip().lower()
    if "@" not in value:
        return ""
    return value.rsplit("@", 1)[1].strip().strip("<>[]")


def domain_signals(sender: str) -> list[str]:
    """Return explainable, local-only sender-domain warning signals.

    These are not a reputation service or production blocklist. They make it
    visible how sender evidence reaches the demo decision.
    """

    domain = extract_domain(sender)
    if not domain:
        return ["missing-domain"]
    signals: list[str] = []
    if domain.startswith("xn--") or ".xn--" in domain:
        signals.append("punycode")
    if re.fullmatch(r"(?:\d{1,3}\.){3}\d{1,3}", domain):
        signals.append("ip-domain")
    if len(domain) > 45:
        signals.append("long-domain")
    if domain.count("-") >= 3:
        signals.append("many-hyphens")
    if domain.endswith((".zip", ".mov", ".click", ".top", ".xyz", ".work", ".buzz")):
        signals.append("high-risk-tld")
    return signals


def dataset_size(selection: str) -> int:
    return sum(DATASET_SIZES.get(dataset_id, 0) for dataset_id in DATASETS.get(selection, DATASETS["Both rich email corpora"]))


def update_dataset_limit(selection: str, current: float | int) -> tuple[Any, str]:
    size = max(1, dataset_size(selection))
    value = min(max(2, int(current or 2)), size)
    try:
        import gradio as gr

        control = gr.Slider(minimum=2, maximum=size, value=value, step=1, label=f"Examples (max {size:,})")
    except ImportError:  # pragma: no cover
        control = value
    return control, f"Maximum available examples: **{size:,}**"


def render_status(processed: int, total: int, latest_ms: float, token_tps: float) -> str:
    if total and processed >= total:
        state, label = "complete", "COMPLETE"
    elif processed:
        state, label = "running", "RUNNING"
    else:
        state, label = "idle", "IDLE"
    return (
        f'<div class="run-status {state}">'
        f'<span class="status-dot"></span><strong>{label}</strong>'
        f'<span>{processed}/{total or "—"} results</span>'
        f'<span>latest {latest_ms:.0f} ms</span>'
        f'<span>live {token_tps:.1f} token/s</span>'
        "</div>"
    )


def render_progress(processed: int, total: int) -> str:
    ratio = min(1.0, processed / total) if total else 0.0
    return (
        '<div class="progress-wrap">'
        f'<div class="progress-track"><div class="progress-fill" style="width:{ratio * 100:.1f}%"></div></div>'
        f'<div class="progress-label">{processed} / {total or 0} handled ({ratio * 100:.0f}%)</div>'
        "</div>"
    )


def render_metrics(snapshot: dict[str, float | int]) -> str:
    cards = [
        ("Live token TPS", f"{snapshot['live_token_tps']:.1f}", "last result"),
        ("Average token TPS", f"{snapshot['token_tps']:.1f}", "input + output"),
        ("Result speed", f"{snapshot['result_speed']:.2f}/s", "API results"),
        ("Latest result", f"{snapshot['latest_ms']:.0f} ms", "round trip"),
        ("Accuracy", f"{snapshot['accuracy'] * 100:.1f}%", f"{snapshot['correct']}/{snapshot['processed']}"),
    ]
    return '<div class="metric-grid">' + "".join(
        f'<div class="metric-card"><div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div>'
        f'<div class="metric-note">{html.escape(note)}</div></div>'
        for label, value, note in cards
    ) + "</div>"


def filter_table_rows(rows: list[list[Any]], wrong_only: bool) -> list[list[Any]]:
    """Filter only the rendered table; aggregate metrics remain unchanged."""

    if not wrong_only or not rows:
        return rows
    # Before a benchmark runs, preview rows have no judge value yet.
    if len(rows[0]) <= 10 or all(str(row[10]) in {"", "—"} for row in rows):
        return rows
    return [row for row in rows if len(row) > 10 and not str(row[10]).startswith("✅")]


def _first(record: dict[str, Any], names: Iterable[str], default: str = "") -> Any:
    for name in names:
        value = record.get(name)
        if value not in (None, ""):
            return value
    return default


def _sender_from_text(text: str) -> str:
    match = re.search(r"(?im)^\s*(?:from|sender):\s*(.+)$", text or "")
    return match.group(1).strip() if match else "unknown@dataset"


def _subject_from_text(text: str) -> str:
    match = re.search(r"(?im)^\s*subject:\s*(.+)$", text or "")
    return match.group(1).strip() if match else "(no subject)"


def _row_from_record(record: dict[str, Any], dataset_id: str) -> EmailRow:
    text = str(_first(record, ("text", "email", "message", "content", "body", "description"), ""))
    subject = str(_first(record, ("subject", "title"), _subject_from_text(text)))
    body = str(_first(record, ("body", "text", "email", "message", "content", "description"), text))
    sender = str(_first(record, ("spoofed_sender", "sender", "from", "email_address"), _sender_from_text(text)))
    raw_label = str(_first(record, ("label", "class", "category", "target"), "0"))
    positive = "scam" if "phishing" in dataset_id.lower() else "spam"
    return EmailRow(dataset_id.split("/")[-1], sender, subject, body, normalize_label(raw_label, positive), raw_label)


def _usable_email(row: EmailRow) -> bool:
    """Keep examples with a real sender domain and meaningful body text."""

    return bool(extract_domain(row.sender)) and len(row.body.strip()) >= 20


def load_hf_rows(selection: str, limit: int, seed: int = 42) -> list[EmailRow]:
    """Load a deterministic, small benchmark slice from Hugging Face datasets."""

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - exercised only without optional UI deps
        raise RuntimeError("Install demo/requirements.txt before loading Hugging Face data.") from exc

    dataset_ids = DATASETS.get(selection, DATASETS["Both rich email corpora"])
    requested = max(1, int(limit))
    quotas = [requested // len(dataset_ids)] * len(dataset_ids)
    for index in range(requested % len(dataset_ids)):
        quotas[index] += 1
    rows: list[EmailRow] = []
    for dataset_id, quota in zip(dataset_ids, quotas):
        if quota <= 0:
            continue
        dataset = load_dataset(dataset_id, split="train", streaming=True)
        dataset = dataset.shuffle(seed=seed, buffer_size=min(10_000, max(1_000, quota * 4)))
        collected = 0
        for record in dataset:
            row = _row_from_record(dict(record), dataset_id)
            if not _usable_email(row):
                continue
            rows.append(row)
            collected += 1
            if collected >= quota:
                break
    random.Random(seed).shuffle(rows)
    return rows[: int(limit)]


def _load_env_file() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


class DjevClient:
    def __init__(self, base_url: str, api_key: str, samples: int = 1, timeout: float = 120.0):
        self.url = base_url.rstrip("/") + "/v1/systemone"
        self.api_key = api_key
        self.samples = max(1, int(samples))
        self.timeout = timeout

    def classify(self, row: EmailRow) -> dict[str, Any]:
        request_body = {
            "model": "jev-latest",
            "state": row.state(),
            "questions": {
                "classification": {
                    "type": "choice",
                    "instructions": "Classify this email as legitimate, spam, or scam/phishing.",
                    "criteria": {
                        "legitimate": "ordinary safe email with no malicious or unsolicited intent",
                        "spam": "unsolicited bulk or promotional email",
                        "scam": "phishing, fraud, impersonation, or a request intended to steal money or credentials",
                    },
                },
                "sender_suspicious": {
                    "type": "noul",
                    "instructions": "Does the sender identity or sender address look suspicious?",
                    "criteria": {"true": "suspicious or spoofed sender", "false": "credible sender"},
                },
                "risk": {
                    "type": "score",
                    "instructions": "How risky is it for the recipient to trust or act on this email?",
                    "criteria": ["low", "medium", "high", "critical"],
                },
            },
            "samples": self.samples,
        }
        payload = json.dumps(request_body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(self.url, data=payload, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode("utf-8"))


def parse_result(response: dict[str, Any]) -> dict[str, Any]:
    answers = response.get("answers", {})
    classification = answers.get("classification", {}) or {}
    sender = answers.get("sender_suspicious", {}) or {}
    risk = answers.get("risk", {}) or {}
    raw_label = classification.get("choice", classification.get("label", "legitimate"))
    label = normalize_prediction(raw_label)
    probabilities = classification.get("probabilities", {}) or {}
    confidence = float(classification.get("confidence", probabilities.get(raw_label, 0.0)) or 0.0)
    risk_probabilities = risk.get("probabilities", {}) or {}
    if risk_probabilities:
        risk_label = max(risk_probabilities, key=risk_probabilities.get)
    else:
        legend = risk.get("legend", {}) or {}
        risk_label = legend.get(str(round(float(risk.get("score", 0) or 0))), "low")
    sender_probability = float(sender.get("noul", 0.0) or 0.0)
    usage = response.get("usage", {}) or {}
    return {
        "label": label,
        "raw_label": str(raw_label),
        "confidence": confidence,
        "risk": str(risk_label),
        "sender_suspicious": sender_probability >= 0.5,
        "sender_probability": sender_probability,
        "input_tokens": int(usage.get("input_tokens", 0) or 0),
        "output_tokens": int(usage.get("output_tokens", 0) or 0),
    }


def preview_table(rows: list[EmailRow]) -> list[list[str]]:
    return [
        [
            idx,
            row.source,
            row.sender,
            extract_domain(row.sender) or "—",
            row.subject[:80],
            row.true_label,
            "—",
            "—",
            "—",
            ", ".join(domain_signals(row.sender)) or "—",
            "—",
            "—",
            "—",
            "—",
        ]
        for idx, row in enumerate(rows, 1)
    ]


def result_table(index: int, row: EmailRow, result: dict[str, Any], elapsed_s: float) -> list[str | int]:
    predicted = result["label"]
    correct = predicted == row.true_label
    verdict = "🟢 SAFE" if predicted == "legitimate" else "🔴 RISK"
    judge = "✅ match" if correct else "❌ miss"
    local_signals = domain_signals(row.sender)
    sender_alert = bool(local_signals) or bool(result["sender_suspicious"])
    return [
        index,
        row.source,
        row.sender,
        extract_domain(row.sender) or "—",
        row.subject[:80],
        row.true_label,
        f"{verdict} {predicted}",
        result["risk"],
        "⚠ suspicious" if sender_alert else "✅ clear",
        ", ".join(local_signals) or "—",
        judge,
        action_for(predicted, result["confidence"], sender_alert),
        f"{elapsed_s * 1000:.0f}",
        f"{(result['input_tokens'] + result['output_tokens']) / max(elapsed_s, 1e-9):.1f}",
    ]


def load_data(selection: str, limit: int) -> tuple[list[list[str]], list[dict[str, Any]], str, str, str, str]:
    rows = load_hf_rows(selection, int(limit))
    serialized = [row.__dict__ for row in rows]
    return (
        preview_table(rows),
        serialized,
        render_status(0, len(rows), 0, 0),
        render_metrics(Metrics().snapshot()),
        render_progress(0, len(rows)),
        f"Loaded {len(rows)} labeled email examples. Press **Run benchmark** to call the local API.",
    )


def run_benchmark(serialized_rows: list[dict[str, Any]], samples: int, wrong_only: bool = False):
    rows = [EmailRow(**row) for row in (serialized_rows or [])]
    if not rows:
        yield render_status(0, 0, 0, 0), render_metrics(Metrics().snapshot()), render_progress(0, 0), [], "Load a Hugging Face dataset first."
        return

    _load_env_file()
    client = DjevClient(os.environ.get("DJEV_URL", "http://127.0.0.1:8011"), os.environ.get("DJEV_API_KEY", os.environ.get("API_KEY", "")), samples)
    metrics = Metrics()
    table: list[list[str | int]] = []
    yield render_status(0, len(rows), 0, 0), render_metrics(metrics.snapshot()), render_progress(0, len(rows)), table, "Benchmark started. Waiting for the first local API result..."

    for index, row in enumerate(rows, 1):
        started = time.perf_counter()
        try:
            result = parse_result(client.classify(row))
            elapsed_s = time.perf_counter() - started
            correct = result["label"] == row.true_label
            metrics.observe(result["input_tokens"], result["output_tokens"], elapsed_s, correct)
            table.append(result_table(index, row, result, elapsed_s))
            snapshot = metrics.snapshot()
            yield (
                render_status(index, len(rows), snapshot["latest_ms"], snapshot["live_token_tps"]),
                render_metrics(snapshot),
                render_progress(index, len(rows)),
                filter_table_rows(table, wrong_only),
                f"Handled result {index}/{len(rows)} — {row.source} — {result['label']} — {action_for(result['label'], result['confidence'], result['sender_suspicious'])}",
            )
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as exc:
            elapsed_s = time.perf_counter() - started
            metrics.observe(0, 0, elapsed_s, False)
            local_signals = domain_signals(row.sender)
            table.append([
                index,
                row.source,
                row.sender,
                extract_domain(row.sender) or "—",
                row.subject[:80],
                row.true_label,
                "⚠ API error",
                "—",
                "⚠ suspicious" if local_signals else "—",
                ", ".join(local_signals) or "—",
                "❌ error",
                "Review",
                f"{elapsed_s * 1000:.0f}",
                "0.0",
            ])
            snapshot = metrics.snapshot()
            yield (
                render_status(index, len(rows), snapshot["latest_ms"], snapshot["live_token_tps"]),
                render_metrics(snapshot),
                render_progress(index, len(rows)),
                filter_table_rows(table, wrong_only),
                f"Result {index} failed: {type(exc).__name__}: {exc}",
            )


CSS = """
.run-status { display:flex; align-items:center; gap:12px; border-radius:12px; padding:12px 16px; background:#172033; color:#dbeafe; margin:8px 0 14px; }
.run-status.running { border:1px solid #38bdf8; box-shadow:0 0 18px #38bdf833; }
.run-status.complete { border:1px solid #34d399; background:#10261f; }
.run-status.idle { border:1px solid #64748b; }
.status-dot { width:11px; height:11px; display:inline-block; border-radius:50%; background:#64748b; }
.running .status-dot { background:#38bdf8; animation:pulse 1s infinite; }
.complete .status-dot { background:#34d399; }
.progress-track { height:12px; border-radius:8px; background:#1e293b; overflow:hidden; }
.progress-fill { height:100%; border-radius:8px; background:linear-gradient(90deg,#38bdf8,#34d399); transition:width .25s ease; }
.progress-label { text-align:right; color:#94a3b8; font-size:12px; margin-top:5px; }
.metric-grid { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:10px; margin:8px 0 14px; }
.metric-card { background:#111827; border:1px solid #263244; border-radius:12px; padding:12px; }
.metric-label,.metric-note { color:#94a3b8; font-size:12px; }
.metric-value { color:#f8fafc; font-size:22px; font-weight:700; margin:4px 0; }
@keyframes pulse { 50% { opacity:.35; transform:scale(.75); } }
"""


def build_demo():
    try:
        import gradio as gr
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install demo/requirements.txt before starting the Gradio app.") from exc

    empty_metrics = render_metrics(Metrics().snapshot())
    with gr.Blocks(title="djev-spark email judge") as demo:
        gr.Markdown("# djev-spark Email Judge\nBenchmark spam/phishing decisions and local inference speed.")
        with gr.Row():
            dataset = gr.Dropdown(list(DATASETS), value="Both rich email corpora", label="Hugging Face data")
            initial_size = dataset_size("Both rich email corpora")
            limit = gr.Slider(2, initial_size, value=8, step=1, label=f"Examples (max {initial_size:,})")
            samples = gr.Radio([1, 2, 4], value=1, label="Decision samples")
        dataset_info = gr.Markdown(f"Maximum available examples: **{initial_size:,}**")
        gr.Markdown(
            "**Decision samples:** `1` = fastest single stochastic read; `2` = two reads averaged; "
            "`4` = four reads averaged for a steadier decision, with higher latency and token cost."
        )
        with gr.Row():
            load_button = gr.Button("Load data", variant="secondary")
            run_button = gr.Button("Run benchmark", variant="primary")
            wrong_only = gr.Checkbox(False, label="Show only wrong detection")
        status = gr.HTML(render_status(0, 0, 0, 0), label="Run state")
        progress = gr.HTML(render_progress(0, 0), label="Progress")
        metrics = gr.HTML(empty_metrics, label="Live metrics")
        table = gr.Dataframe(headers=TABLE_HEADERS, datatype=["number"] + ["str"] * (len(TABLE_HEADERS) - 1), value=[], wrap=True, label="Email decisions")
        message = gr.Markdown("Load data to begin.")
        rows_state = gr.State([])

        dataset.change(update_dataset_limit, inputs=[dataset, limit], outputs=[limit, dataset_info])
        load_button.click(load_data, inputs=[dataset, limit], outputs=[table, rows_state, status, metrics, progress, message])
        wrong_only.change(filter_table_rows, inputs=[table, wrong_only], outputs=table)
        run_button.click(run_benchmark, inputs=[rows_state, samples, wrong_only], outputs=[status, metrics, progress, table, message])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", "7860")))
    args = parser.parse_args()
    build_demo().queue().launch(server_name=args.host, server_port=args.port, css=CSS)


if __name__ == "__main__":
    main()
