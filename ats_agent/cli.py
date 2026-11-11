"""Command-line entry point: python -m ats_agent resume.pdf --jd job.txt [--ai] [--json]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .parser import parse_resume
from .scorer import ATSReport, score_resume


def run(resume_path: str, jd_text: str | None = None, use_ai: bool = False) -> ATSReport:
    parsed = parse_resume(resume_path)
    report = score_resume(parsed, jd_text)
    if use_ai:
        from .llm import ai_review
        try:
            report.ai_review = ai_review(parsed.text, jd_text, report.to_dict())
        except Exception as exc:  # keep the offline score even if the AI step fails
            report.warnings.append(f"AI review skipped: {exc}")
    return report


def _bar(score: float, width: int = 20) -> str:
    filled = round(score / 100 * width)
    return "█" * filled + "░" * (width - filled)


def render(report: ATSReport) -> str:
    out = [f"\nATS SCORE: {report.overall_score}/100  ({report.grade})", "=" * 44]
    for c in report.categories:
        out.append(f"{c.name:<26} {_bar(c.score)} {c.score:5.1f}  (weight {c.weight:g}%)")
    if report.matched_keywords or report.missing_keywords:
        out.append("\nMatched keywords: " + (", ".join(report.matched_keywords[:25]) or "none"))
        out.append("Missing keywords: " + (", ".join(report.missing_keywords[:25]) or "none"))
    if report.suggestions:
        out.append("\nTop suggestions:")
        out += [f"  {i}. {s}" for i, s in enumerate(report.suggestions[:10], 1)]
    if report.ai_review:
        ai = report.ai_review
        out.append(f"\nAI REVIEW (Claude): {ai['ai_score']}/100")
        out.append(f"  {ai['verdict']}")
        for title, key in (("Strengths", "strengths"), ("Gaps", "gaps"),
                           ("Missing skills", "missing_skills"),
                           ("Recommendations", "top_recommendations")):
            if ai.get(key):
                out.append(f"\n  {title}:")
                out += [f"    - {x}" for x in ai[key]]
        if ai.get("rewritten_bullets"):
            out.append("\n  Bullet rewrites:")
            for b in ai["rewritten_bullets"]:
                out.append(f"    before: {b['original']}\n    after:  {b['improved']}")
    for w in report.warnings:
        out.append(f"\n⚠ {w}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score a resume for ATS compatibility.")
    p.add_argument("resume", help="Path to resume (.pdf, .docx, .txt)")
    p.add_argument("--jd", help="Path to a job description text file")
    p.add_argument("--jd-text", help="Job description passed inline")
    p.add_argument("--ai", action="store_true", help="Add a Claude-powered review (needs ANTHROPIC_API_KEY)")
    p.add_argument("--json", action="store_true", help="Print the full report as JSON")
    args = p.parse_args(argv)

    jd = args.jd_text or (Path(args.jd).read_text(encoding="utf-8") if args.jd else None)
    try:
        report = run(args.resume, jd, args.ai)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report.to_dict(), indent=2) if args.json else render(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
