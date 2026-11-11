from pathlib import Path

import docx

from ats_agent import parse_resume, score_resume
from ats_agent.scorer import extract_keywords

SAMPLES = Path(__file__).parent.parent / "samples"
JD = (SAMPLES / "job_description.txt").read_text()


def test_sample_scores_reasonably():
    report = score_resume(parse_resume(SAMPLES / "resume.txt"), JD)
    assert 60 <= report.overall_score <= 95
    assert "python" in report.matched_keywords
    assert "terraform" in report.missing_keywords


def test_no_jd_skips_keywords():
    report = score_resume(parse_resume(SAMPLES / "resume.txt"))
    assert all(c.name != "Keyword match" for c in report.categories)
    assert abs(sum(c.weight for c in report.categories) - 100) < 0.1


def test_poor_resume_scores_low(tmp_path):
    f = tmp_path / "bad.txt"
    f.write_text("John\nI am a hard worker and team player. I worked on many things.\n")
    report = score_resume(parse_resume(f), JD)
    assert report.overall_score < 40


def test_docx_with_table_is_penalised(tmp_path):
    d = docx.Document()
    for line in (SAMPLES / "resume.txt").read_text().splitlines():
        d.add_paragraph(line)
    d.add_table(rows=2, cols=2)
    path = tmp_path / "r.docx"
    d.save(path)
    parsed = parse_resume(path)
    assert parsed.table_count == 1
    fmt = next(c for c in score_resume(parsed, JD).categories if c.name.startswith("Formatting"))
    assert fmt.score < 95


def test_keywords_include_tools():
    kws = extract_keywords(JD)
    for k in ("python", "kubernetes", "postgresql", "ci/cd"):
        assert k in kws
