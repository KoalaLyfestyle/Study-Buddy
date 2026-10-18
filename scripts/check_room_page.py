"""Browser check for a built study room page (needs Playwright and Chromium).

    python scripts/check_room_page.py .studybuddy/rooms/<course>/week-01/index.html [--shots DIR]

Checks, on a phone-sized and a desktop-sized window:
- no horizontal scrolling;
- the first Tab reaches the "Skip to content" link, and Enter moves to the content;
- every table-of-contents link points at a section on the page;
- ticking "I have read this" and choosing a quiz answer survive a reload;
- no script errors.
Exits 1 and prints what failed. Only use it on the public sample course.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def check(page_url: str, shots: Path | None) -> list[str]:
    failures: list[str] = []
    with sync_playwright() as p:
        browser = p.chromium.launch()
        for name, size in (("phone", {"width": 390, "height": 844}), ("desktop", {"width": 1280, "height": 900})):
            context = browser.new_context(viewport=size)
            page = context.new_page()
            errors: list[str] = []
            page.on("pageerror", lambda exc: errors.append(str(exc)))
            page.goto(page_url)

            overflow = page.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
            if overflow > 0:
                failures.append(f"{name}: page scrolls sideways by {overflow}px")

            page.keyboard.press("Tab")
            if page.evaluate("document.activeElement.className") != "skip":
                failures.append(f"{name}: first Tab does not reach the skip link")
            page.keyboard.press("Enter")
            if page.evaluate("document.activeElement.id") != "main":
                failures.append(f"{name}: the skip link does not move focus to the content")

            missing = page.evaluate(
                "[...document.querySelectorAll('nav a[href^=\"#\"]')]"
                ".filter(a => !document.getElementById(a.getAttribute('href').slice(1))).length"
            )
            if missing:
                failures.append(f"{name}: {missing} contents links point nowhere")

            box = page.locator("input[data-read]").first
            box.focus()
            page.keyboard.press("Space")
            radio = page.locator("input[data-question]").last
            radio.check()
            if shots:
                shots.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(shots / f"room-{name}.png"), full_page=False)
            page.reload()
            if not page.locator("input[data-read]").first.is_checked():
                failures.append(f"{name}: reading progress is lost on reload")
            if not page.locator("input[data-question]").last.is_checked():
                failures.append(f"{name}: the quiz answer is lost on reload")
            failures.extend(f"{name}: script error: {e}" for e in errors)
            context.close()
        browser.close()
    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("html")
    parser.add_argument("--shots", type=Path)
    args = parser.parse_args()
    failures = check(Path(args.html).resolve().as_uri(), args.shots)
    for f in failures:
        print("FAIL", f)
    print("ok" if not failures else f"{len(failures)} problems")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
