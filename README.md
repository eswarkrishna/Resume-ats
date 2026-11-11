# ATS Resume Scoring Agent

Scores a resume (PDF, DOCX, or TXT) for applicant-tracking-system compatibility, optionally
against a job description, and tells you exactly what to fix.

## How it scores (0-100)

| Category | Weight | What it checks |
|---|---|---|
| Keyword match | 35 | Top terms and repeated phrases from the job description found in the resume (light stemming, so "mentoring" matches "mentored") |
| Sections | 15 | Experience, Education, Skills headings; Summary; Projects/Certifications |
| Contact info | 10 | Email, phone, LinkedIn, GitHub/portfolio; flags contact details hidden in headers/footers |
| Formatting & parsability | 15 | Tables, images, multi-column layouts, header/footer content, odd symbols, bullet use, scanned PDFs |
| Content quality | 15 | Bullets starting with action verbs, quantified results, cliché phrases, first-person pronouns |
| Length | 10 | 400-900 words / 1-2 pages is ideal |

With no job description, keyword match is skipped and the other weights scale up to 100.
Suggestions are ordered by how many points each fix is worth.

Add `--ai` for a Claude review on top (default model `claude-opus-5-5`, override with
`ATS_AGENT_MODEL`): a holistic fit score, strengths, gaps, missing skills including synonyms
a keyword matcher misses, rewritten bullets, and prioritised recommendations. If the AI step
fails, you still get the offline score.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # only for --ai
```

## Use

```bash
python -m ats_agent samples/resume.txt --jd samples/job_description.txt
python -m ats_agent my_resume.pdf --jd-text "Senior Data Analyst ... SQL, Tableau ..." --ai
python -m ats_agent my_resume.docx --json        # machine-readable report
```

Web API:

```bash
uvicorn ats_agent.api:app --reload
curl -F resume=@my_resume.pdf -F job_description="$(cat jd.txt)" -F ai=true localhost:8000/score
```

From Python:

```python
from ats_agent.cli import run
report = run("resume.pdf", jd_text, use_ai=False)
print(report.overall_score, report.suggestions[:3])
```

## Tests

```bash
pytest -q
```

## Layout

- `ats_agent/parser.py` text + layout signals from PDF (pdfplumber) and DOCX (python-docx)
- `ats_agent/scorer.py` rule-based scoring and suggestions
- `ats_agent/llm.py` optional Claude review with structured JSON output
- `ats_agent/cli.py`, `ats_agent/api.py` CLI and FastAPI entry points
