#!/usr/bin/env python3
"""Exercise the official THUDM agent with two local GUI actions, not OSWorld scoring."""
import datetime
import importlib
import json
import os
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    cfg = json.loads((ROOT / "configs/paths.json").read_text())
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = cfg["browser_dir"]
    libs = Path(cfg["cache_dir"]) / "browser-system/root/usr/lib/x86_64-linux-gnu"
    os.environ["LD_LIBRARY_PATH"] = str(libs) + (":" + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else "")
    sys.path.insert(0, str(ROOT / "SCALE-CUA/osworld_eval"))
    agent_module = importlib.import_module("mm_agents.scalecua-osworld")
    from playwright.sync_api import sync_playwright
    base = f"http://127.0.0.1:{cfg['port']}"
    with urllib.request.urlopen(base + "/health", timeout=5) as response:
        assert response.status == 200
    output = ROOT / "runs" / ("thudm-sft-smoke-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output.mkdir(parents=True)
    calls = []

    def generate(messages):
        payload = {"model": cfg["served_model_name"], "messages": messages,
                   "tools": agent.get_tools(), "tool_choice": "auto", "temperature": 0,
                   "top_p": 1, "top_k": -1, "max_tokens": 1024}
        request = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=120) as response:
            result = json.load(response)
        message = result["choices"][0]["message"]
        calls.append({"usage": result.get("usage"), "message": message})
        return {key: value for key, value in message.items() if key in {"role", "content", "tool_calls"} and value is not None}

    agent = agent_module.ScaleCUAOSWorldAgent(screen_size=(1280, 720), image_size=(1280, 720),
                                            relative_coordinate=True, gen_func=generate,
                                            only_n_most_recent_images=5, min_removal_threshold=5)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
        page = browser.new_page(viewport={"width": 1280, "height": 720})
        page.set_content('''<html><body style="font:28px sans-serif;background:#eef2f8;text-align:center;padding-top:140px">
        <h1>GUI connection check</h1><p>Press Start, then press Finish.</p>
        <button id="start" style="font-size:32px;padding:24px 70px" onclick="this.hidden=true;document.querySelector('#finish').hidden=false">Start</button>
        <button id="finish" hidden style="font-size:32px;padding:24px 70px" onclick="this.hidden=true;document.querySelector('#status').textContent='Completed'">Finish</button>
        <p id="status">Waiting</p></body></html>''')
        parsed = []
        for step in range(5):
            screenshot = page.screenshot(path=str(output / f"step-{step}.png"))
            _, actions = agent.predict("Click Start, then click Finish to complete the GUI connection check.", {"screenshot": screenshot})
            parsed.append(actions)
            message = agent.messages[-1]
            for tool in message.get("tool_calls", []):
                assert tool["function"]["name"] == "computer"
                arguments = json.loads(tool["function"]["arguments"])
                action = arguments.get("action")
                if action in {"left_click", "mouse_move"}:
                    x, y = arguments["coordinate"]
                    assert 0 <= x <= 1000 and 0 <= y <= 1000
                    (page.mouse.click if action == "left_click" else page.mouse.move)(x * 1.28, y * 0.72)
                elif action not in {"wait", "screenshot"}:
                    raise AssertionError(f"Unsupported smoke action: {action}")
            if page.locator("#status").inner_text() == "Completed":
                break
        passed = page.locator("#status").inner_text() == "Completed"
        page.screenshot(path=str(output / "final.png"))
        browser.close()
    report = {"checked_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "passed": passed,
              "model_id": cfg["model_id"], "revision": cfg["model_revision"], "decisions": len(calls),
              "official_agent": "mm_agents.scalecua-osworld.ScaleCUAOSWorldAgent",
              "checks": ["image inference", "Hermes tool call parsing", "relative coordinates", "multi-turn screenshot history", "local DOM completion"],
              "scope": "Synthetic local browser smoke only; no OSWorld benchmark or RL update", "calls": calls,
              "official_parsed_actions": parsed, "artifacts": str(output)}
    (ROOT / "reports/smoke-thudm-sft.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": passed, "decisions": len(calls), "report": "reports/smoke-thudm-sft.json"}))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
