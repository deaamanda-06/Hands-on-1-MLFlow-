import os
from playwright.sync_api import sync_playwright

os.makedirs("fig", exist_ok=True)
B = "http://localhost:5000/#"
BASE = "c77ac8f535e9442691a7365d643275cd"
EPOCH = "28bfa9c31fde4099898fd5b31b7f7643"
AUTO = "a6a966f497c240438a7381c6c7d56c35"

pages = [
    ("ui1_experiment_runs", f"{B}/experiments/1/runs", 4500),
    ("ui2_run_baseline", f"{B}/experiments/1/runs/{BASE}", 3500),
    ("ui3_run_baseline_artifacts", f"{B}/experiments/1/runs/{BASE}/artifacts", 4000),
    ("ui4_epoch_metrics", f"{B}/experiments/1/runs/{EPOCH}/model-metrics", 5000),
    ("ui5_autolog_run", f"{B}/experiments/1/runs/{AUTO}", 3500),
    ("ui6_registry_model", f"{B}/models/breast-cancer-clf", 4500),
]


def dismiss(pg):
    for sel in ["button:has-text('Got it')", "button:has-text('Skip')", "button:has-text('Close')"]:
        try:
            el = pg.locator(sel).first
            if el.is_visible(timeout=500):
                el.click(); pg.wait_for_timeout(300)
        except Exception:
            pass
    # tutup panel MLflow Assistant (ikon X di kanan atas panel)
    try:
        pg.mouse.click(1420, 34); pg.wait_for_timeout(300)
    except Exception:
        pass


with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(locale="en-US", viewport={"width": 1500, "height": 850})
    pg = ctx.new_page()
    pg.goto(f"{B}/experiments/1"); pg.wait_for_timeout(3000)
    dismiss(pg)
    try:
        pg.get_by_text("Model training", exact=True).first.click(); pg.wait_for_timeout(1500)
    except Exception as e:
        print("no model-training toggle", str(e)[:80])
    for name, url, wait in pages:
        pg.goto(url); pg.wait_for_timeout(wait)
        dismiss(pg)
        pg.wait_for_timeout(800)
        pg.screenshot(path=f"fig/{name}.png")
        print("ok", name)
    b.close()
