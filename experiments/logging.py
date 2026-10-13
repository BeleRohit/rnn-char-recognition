"""Helpers for recording experiment output in Markdown."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


RESULTS_PATH = Path(__file__).with_name("RESULTS.md")


def append_experiment_results(
    title: str,
    metadata: dict[str, object],
    rows: list[dict[str, object]],
    results_path: Path = RESULTS_PATH,
) -> None:
    """Append one experiment run as a Markdown section."""
    results_path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%Y-%m-%d %H:%M:%S %Z")

    lines = [
        "",
        f"## {title}",
        "",
        f"Run timestamp: {timestamp}",
        "",
        "Metadata:",
    ]
    for key, value in metadata.items():
        lines.append(f"- `{key}`: `{value}`")

    if rows:
        headers = list(rows[0].keys())
        lines.extend(
            [
                "",
                "| " + " | ".join(headers) + " |",
                "| " + " | ".join("---" for _ in headers) + " |",
            ]
        )
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")

    lines.append("")
    with results_path.open("a", encoding="utf-8") as file:
        file.write("\n".join(lines))
