# djev-spark email judge demo

This Gradio app benchmarks the local structured djev-spark API against labeled
email examples from Hugging Face. It does not send email or modify a mailbox.

The benchmark uses field-complete email corpora: the recommended AURA corpus
has `sender`, `subject`, `body`, and `label` fields. Rows without a usable
sender domain or meaningful body are skipped. The previous text-only spam/ham
source is intentionally not used because it could not support sender-aware
decisions.

Default source: `kudzaiprichard/aura-phishing-email-corpus` (about 112k
training rows), with `Teddyha/phishing_benign_email_dataset` available as a
smaller comparison set.

## Start

From the repository root:

```bash
python3 -m venv .venv-demo
. .venv-demo/bin/activate
python -m pip install -r demo/requirements.txt
python demo/app.py --host 127.0.0.1 --port 7860
```

Open <http://127.0.0.1:7860>.

The app defaults to `http://127.0.0.1:8011` and reads `API_KEY` from the
repository `.env` file. Override these with `DJEV_URL` and `DJEV_API_KEY`.

## Dashboard signals

- `Live token TPS`: tokens per second for the most recent API result.
- `Average token TPS`: aggregate input plus output tokens divided by API time.
- `Result speed`: completed API results per second.
- `Latest result`: round-trip time for the most recent result.
- The blue pulsing `RUNNING` indicator changes to green `COMPLETE` when all
  examples have been handled.
- Each row includes a green safe/red risk badge, a ground-truth match/miss
  indicator, sender/domain signals, and a demo-only `Deliver`, `Review`, or
  `Quarantine` action.
- `Show only wrong detection` filters the table without changing aggregate
  metrics.
- `Decision samples` controls djev's stochastic reads per email: `1` is
  fastest, while `2` and `4` average more reads for a steadier result.

## Tests

```bash
python3 -m unittest -v demo.test_app
```

The first dataset load downloads and caches the selected Hugging Face data.

## Space shooter director demo

`space-shooter.html` is a single-page Canvas game with the playable game on the
left and the JevSpark decision/telemetry panel on the right. The browser keeps
the 60 FPS game loop local. At the beginning of a run and at each wave
boundary, it sends the compact game state to the local Python bridge, which
calls djev and reports the returned enemy profile, bullet pattern, difficulty,
confidence, API latency, token TPS, result speed, and token usage.
The director runs local control and trajectory tracking every 40ms, while
parallel JevSpark requests are issued every 450ms. Each request contains only
the nearest eight bullets and six enemies, a 1.2s decision horizon, the current
player velocity, and explicit conservative constraints. The trajectory request
is sent to `/api/trajectory`; its short forecast is fed into `/api/decision` as
movement context. The decision tree is
`evade → adjust → recenter → hold_center`:
JevSpark returns enemy/bullet strategy, a mode, a movement bias (`hold`,
`left`, `right`, `up`, or `down`), and urgency (`observe`, `adjust`, or
`evade`). With no nearby threat, the request includes `preferred_center`,
`center_delta`, and `center_error_px`, so `recenter` produces a small move
toward the center corridor instead of freezing in place.

The browser-side autopilot is enabled by default: it predicts short bullet
trajectories and uses the forecast only to choose the next small direction. It
keeps the ship inside a center corridor with a soft edge penalty, so repeated
JevSpark directions cannot pin the ship in a corner. The `Auto Pilot` button
switches to manual WASD/arrow/pointer control; this local safety loop never
waits for djev and there is no bomb shortcut.

Start it from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv-demo/bin/python -B demo/space_shooter_server.py \
  --host 127.0.0.1 --port 7862
```

Open <http://127.0.0.1:7862/>. The bridge keeps `API_KEY`/`DJEV_API_KEY` out
of the HTML and falls back to a playable deterministic strategy if djev is
unavailable.
