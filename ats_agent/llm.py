"""Optional Claude-powered review that adds semantic judgment on top of the rule-based score."""

from __future__ import annotations

import json
import os

MODEL = os.environ.get("ATS_AGENT_MODEL", "claude-opus-5-5")

SYSTEM = """You are an expert technical recruiter and ATS (applicant tracking system) specialist.
You review resumes the way a modern ATS plus a human screener would: you check whether the
resume's skills and experience genuinely match the target role (including synonyms and
related skills a keyword matcher would miss), whether achievements are concrete, and whether
anything would hurt parsing. Be specific and actionable. Never suggest inventing experience;
only suggest surfacing or rewording what the candidate plausibly has."""

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ai_score", "verdict", "strengths", "gaps", "missing_skills",
                 "rewritten_bullets", "top_recommendations"],
    "properties": {
        "ai_score": {"type": "integer", "description": "0-100 holistic ATS + recruiter fit score"},
        "verdict": {"type": "string", "description": "One or two sentence summary"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "gaps": {"type": "array", "items": {"type": "string"}},
        "missing_skills": {"type": "array", "items": {"type": "string"}},
        "rewritten_bullets": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["original", "improved"],
                "properties": {"original": {"type": "string"}, "improved": {"type": "string"}},
            },
        },
        "top_recommendations": {"type": "array", "items": {"type": "string"}},
    },
}


def ai_review(resume_text: str, job_description: str | None, rule_report: dict) -> dict:
    """Ask Claude for a holistic review. Requires ANTHROPIC_API_KEY (or `ant auth login`)."""
    import anthropic

    client = anthropic.Anthropic()
    summary = {
        "rule_based_score": rule_report["overall_score"],
        "categories": {c["name"]: c["score"] for c in rule_report["categories"]},
        "missing_keywords": rule_report["missing_keywords"][:20],
    }
    jd = job_description.strip() if job_description else "(none provided: judge general ATS readiness)"
    prompt = (
        f"<job_description>\n{jd}\n</job_description>\n\n"
        f"<resume>\n{resume_text}\n</resume>\n\n"
        f"<rule_based_findings>\n{json.dumps(summary, indent=2)}\n</rule_based_findings>\n\n"
        "Review this resume for the role. Give up to 5 strengths, up to 5 gaps, missing skills, "
        "3-5 rewritten bullets taken from the resume (original text and an improved, quantified version), "
        "and the 5 highest-impact recommendations in priority order."
    )
    with client.beta.messages.stream(
        model=MODEL,
        max_tokens=16000,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        msg = stream.get_final_message()

    if msg.stop_reason == "refusal":
        raise RuntimeError("Claude declined to review this resume.")
    if msg.stop_reason == "max_tokens":
        raise RuntimeError("AI review was cut off (max_tokens reached).")
    text = "".join(b.text for b in msg.content if b.type == "text")
    return json.loads(text)
