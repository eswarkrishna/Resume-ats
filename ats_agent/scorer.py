"""Deterministic, rule-based ATS scoring. Works offline with no API key."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import asdict, dataclass, field

from .parser import ParsedResume

# Category weights (sum to 100). Without a job description, the keyword
# weight is redistributed across the other categories proportionally.
WEIGHTS = {
    "keywords": 35,
    "sections": 15,
    "contact": 10,
    "formatting": 15,
    "content": 15,
    "length": 10,
}

SECTION_PATTERNS = {
    "summary": r"(professional\s+)?summary|profile|objective|about\s+me",
    "experience": r"(work\s+|professional\s+)?experience|employment(\s+history)?|work\s+history",
    "education": r"education|academic\s+background|qualifications",
    "skills": r"(technical\s+|core\s+)?skills|technologies|competencies|tech\s+stack",
    "projects": r"projects|portfolio",
    "certifications": r"certifications?|licen[cs]es|courses",
}
REQUIRED_SECTIONS = ("experience", "education", "skills")

ACTION_VERBS = set("""
achieved accelerated administered analyzed architected automated boosted built championed
collaborated conceived configured consolidated coordinated created cut debugged decreased
delivered deployed designed developed directed drove eliminated enabled engineered enhanced
established evaluated executed expanded facilitated founded generated grew guided headed
implemented improved increased initiated innovated integrated introduced launched led
maintained managed mentored migrated modernized negotiated optimized orchestrated organized
oversaw owned partnered pioneered planned produced programmed published raised redesigned
reduced refactored resolved restructured revamped saved scaled secured shipped simplified
spearheaded streamlined strengthened supervised trained transformed tripled upgraded won wrote
""".split())

WEAK_PHRASES = [
    "responsible for", "duties included", "worked on", "helped with", "assisted with",
    "team player", "hard worker", "go-getter", "detail-oriented", "self-starter",
    "think outside the box", "synergy", "results-driven", "references available",
]

STOPWORDS = set("""
a about above after again against all also am an and any are as at be because been before
being below between both but by can could did do does doing down during each etc few for
from further had has have having he her here hers him his how i if in into is it its itself
just me more most my no nor not now of off on once only or other our ours out over own per
same she should so some such than that the their theirs them then there these they this
those through to too under until up us very via was we were what when where which while who
whom why will with would you your yours
ability able across apply applicant candidate candidates company degree environment equal
excellent experience familiarity good including job join knowledge looking must need
opportunity plus position preferred qualifications related required requirements responsibilities
role skills strong team understanding nice hands-on hands looking build building
develop development ideal seeking want help will work working year years well within new using use
""".split())

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(\+?\d{1,3}[\s.-]?)?(\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]?\d{3,4}")
LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w-]+", re.I)
URL_RE = re.compile(r"(github\.com/[\w-]+|https?://\S+|\b[\w-]+\.(dev|io|me|com)/?\S*)", re.I)
NUMBER_RE = re.compile(r"(\d+(\.\d+)?\s?%|\$\s?\d|\d+[kKmMbB]\b|\b\d{2,}\b|\d+x\b)")
BULLET_RE = re.compile(r"^\s*([•\-\*▪●◦‣–]|\d+\.)\s+", re.M)


@dataclass
class CategoryResult:
    name: str
    score: float  # 0-100 within the category
    weight: float
    details: dict = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)


@dataclass
class ATSReport:
    overall_score: int
    grade: str
    categories: list[CategoryResult]
    matched_keywords: list[str] = field(default_factory=list)
    missing_keywords: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ai_review: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------- helpers

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def _tokens(text: str) -> list[str]:
    # Keep tech tokens like c++, c#, node.js, ci/cd
    return re.findall(r"[a-z][a-z0-9+#./-]*[a-z0-9+#]|[a-z]", text.lower())


def extract_keywords(job_description: str, top_n: int = 40) -> list[str]:
    """Pull the most important unigrams and bigrams from a job description."""
    toks = [t.strip(".,/") for t in _tokens(job_description)]
    toks = [t for t in toks if t and t not in STOPWORDS and len(t) > 1 and not t.isdigit()]
    uni = Counter(toks)
    words = [t.strip(".,/") for t in _tokens(job_description)]
    bi = Counter(
        f"{a} {b}" for a, b in zip(words, words[1:])
        if a not in STOPWORDS and b not in STOPWORDS and len(a) > 1 and len(b) > 1
    )
    scored: dict[str, float] = {}
    for w, c in uni.items():
        scored[w] = c + (0.5 if any(ch in w for ch in "+#./") else 0)
    for b, c in bi.items():
        if c >= 2:  # repeated phrases are strong signals
            scored[b] = c * 1.5
    # Drop unigrams fully covered by a selected bigram with equal count
    ranked = sorted(scored.items(), key=lambda kv: (-kv[1], kv[0]))
    return [k for k, _ in ranked[:top_n]]


def _stem(word: str) -> str:
    """Very light stemmer so 'mentoring' matches 'mentored' and 'engineers' matches 'engineer'."""
    for suf in ("ization", "ations", "ation", "ings", "ing", "ers", "ed", "es", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 4:
            return word[: -len(suf)]
    return word


def _stem_text(text: str) -> str:
    return " ".join(_stem(t.strip(".,/")) for t in _tokens(text))


def _contains(haystack: str, kw: str, stemmed_haystack: str = "") -> bool:
    if re.search(r"(?<![a-z0-9])" + re.escape(kw) + r"(?![a-z0-9])", haystack):
        return True
    if stemmed_haystack:
        skw = _stem_text(kw)
        return re.search(r"(?<![a-z0-9])" + re.escape(skw) + r"(?![a-z0-9])", stemmed_haystack) is not None
    return False


def _find_sections(text: str) -> dict[str, bool]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    found = {k: False for k in SECTION_PATTERNS}
    for ln in lines:
        if len(ln) > 45:  # headings are short
            continue
        clean = re.sub(r"[^a-z &]", "", ln.lower()).strip()
        for name, pat in SECTION_PATTERNS.items():
            if re.fullmatch(rf"({pat})( & .*| and .*)?", clean):
                found[name] = True
    return found


# ---------------------------------------------------------------- categories

def _score_keywords(text: str, jd: str) -> tuple[CategoryResult, list[str], list[str]]:
    kws = extract_keywords(jd)
    hay = _norm(text)
    stemmed = _stem_text(text)
    matched = [k for k in kws if _contains(hay, k, stemmed)]
    missing = [k for k in kws if k not in matched]
    # Top-ranked keywords matter more.
    total_w = sum(1 / (1 + i * 0.05) for i in range(len(kws))) or 1
    got_w = sum(1 / (1 + i * 0.05) for i, k in enumerate(kws) if k in matched)
    score = 100 * got_w / total_w
    sugg = []
    if missing:
        sugg.append("Add these job-description terms where they truthfully apply: "
                    + ", ".join(missing[:12]) + ".")
    if score < 50:
        sugg.append("Mirror the exact wording of the job posting (e.g. its tool names and job title).")
    return (CategoryResult("Keyword match", round(score, 1), WEIGHTS["keywords"],
                           {"matched": len(matched), "total": len(kws)}, sugg), matched, missing)


def _score_sections(text: str) -> CategoryResult:
    found = _find_sections(text)
    score = 0.0
    sugg = []
    for s in REQUIRED_SECTIONS:
        if found[s]:
            score += 25
        else:
            sugg.append(f"Add a clearly labelled '{s.title()}' section heading.")
    if found["summary"]:
        score += 15
    else:
        sugg.append("Add a short 'Summary' at the top with your target title and core skills.")
    if found["projects"] or found["certifications"]:
        score += 10
    return CategoryResult("Sections", min(score, 100), WEIGHTS["sections"],
                          {k: v for k, v in found.items()}, sugg)


def _score_contact(parsed: ParsedResume) -> CategoryResult:
    text = parsed.text
    head = "\n".join(text.splitlines()[:15])
    checks = {
        "email": bool(EMAIL_RE.search(text)),
        "phone": bool(PHONE_RE.search(head)),
        "linkedin": bool(LINKEDIN_RE.search(text)),
        "portfolio_or_github": bool(URL_RE.search(text.replace("linkedin.com", ""))),
    }
    pts = {"email": 40, "phone": 30, "linkedin": 20, "portfolio_or_github": 10}
    score = sum(pts[k] for k, v in checks.items() if v)
    sugg = [f"Add your {k.replace('_', ' ')}." for k, v in checks.items() if not v and k != "portfolio_or_github"]
    if parsed.header_footer_text and EMAIL_RE.search(parsed.header_footer_text) and not EMAIL_RE.search(text):
        sugg.append("Your contact details are in the page header/footer; many ATS skip those. Move them into the body.")
        score = max(score - 30, 0)
    return CategoryResult("Contact info", score, WEIGHTS["contact"], checks, sugg)


def _score_formatting(parsed: ParsedResume) -> CategoryResult:
    score = 100.0
    sugg, details = [], {}
    if parsed.table_count:
        score -= min(25, 10 * parsed.table_count)
        sugg.append("Avoid tables for layout; ATS often scramble or drop table content.")
    if parsed.image_count:
        score -= min(15, 5 * parsed.image_count)
        sugg.append("Remove images, icons, and photos; ATS can't read them.")
    if parsed.multi_column_suspected:
        score -= 20
        sugg.append("Use a single-column layout; multi-column text can be read out of order.")
    if parsed.header_footer_text:
        score -= 5
        sugg.append("Keep important content out of headers and footers.")
    weird = len(re.findall(r"[^\x00-\x7F•–—’‘“”€£₹é]", parsed.text))
    details["unusual_chars"] = weird
    if weird > 15:
        score -= 10
        sugg.append("Replace decorative symbols/special characters with plain text.")
    bullets = len(BULLET_RE.findall(parsed.text))
    details["bullets"] = bullets
    if bullets < 5:
        score -= 10
        sugg.append("Use bullet points for achievements under each role.")
    if parsed.file_type not in {"pdf", "docx"}:
        score -= 5
        sugg.append("Submit as .docx or a text-based PDF for best ATS compatibility.")
    if len(parsed.text) < 200:
        score = min(score, 20)
    details.update(tables=parsed.table_count, images=parsed.image_count,
                   multi_column=parsed.multi_column_suspected)
    return CategoryResult("Formatting & parsability", max(score, 0), WEIGHTS["formatting"], details, sugg)


def _score_content(text: str) -> CategoryResult:
    lines = [ln.strip() for ln in text.splitlines() if len(ln.strip().split()) >= 4]
    bullet_lines = [BULLET_RE.sub("", ln) for ln in text.splitlines() if BULLET_RE.match(ln)] or lines
    n = max(len(bullet_lines), 1)
    first_words = [re.sub(r"[^a-z]", "", (ln.split() or [""])[0].lower()) for ln in bullet_lines]
    action = sum(1 for w in first_words if w in ACTION_VERBS)
    quantified = sum(1 for ln in bullet_lines if NUMBER_RE.search(ln))
    low = text.lower()
    weak = [p for p in WEAK_PHRASES if p in low]
    pronouns = len(re.findall(r"\b(i|me|my)\b", low))
    action_r, quant_r = action / n, quantified / n
    score = 45 * min(action_r / 0.6, 1) + 45 * min(quant_r / 0.4, 1) + 10
    score -= min(15, 4 * len(weak)) + min(10, pronouns)
    sugg = []
    if action_r < 0.6:
        sugg.append("Start more bullets with strong action verbs (Led, Built, Reduced, Shipped...).")
    if quant_r < 0.4:
        sugg.append("Quantify impact: add numbers, percentages, money, or scale to more bullets.")
    if weak:
        sugg.append("Replace weak or cliché phrases: " + ", ".join(f"'{w}'" for w in weak) + ".")
    if pronouns > 3:
        sugg.append("Drop first-person pronouns (I, me, my).")
    return CategoryResult("Content quality", round(max(min(score, 100), 0), 1), WEIGHTS["content"],
                          {"bullets_analyzed": n, "action_verb_ratio": round(action_r, 2),
                           "quantified_ratio": round(quant_r, 2), "weak_phrases": weak,
                           "first_person_pronouns": pronouns}, sugg)


def _score_length(parsed: ParsedResume) -> CategoryResult:
    words = len(parsed.text.split())
    if 400 <= words <= 900:
        score = 100
    elif 300 <= words < 400 or 900 < words <= 1200:
        score = 75
    elif 200 <= words < 300 or 1200 < words <= 1600:
        score = 50
    else:
        score = 25
    sugg = []
    if words < 400:
        sugg.append(f"Resume is short ({words} words). Add detail on impact, tools, and scope.")
    elif words > 900:
        sugg.append(f"Resume is long ({words} words). Trim older or less relevant content; aim for 1-2 pages.")
    if parsed.pages > 2:
        score = min(score, 50)
        sugg.append(f"Resume is {parsed.pages} pages; keep it to 1-2.")
    return CategoryResult("Length", score, WEIGHTS["length"], {"words": words, "pages": parsed.pages}, sugg)


def _grade(score: float) -> str:
    return ("Excellent" if score >= 85 else "Good" if score >= 70
            else "Fair" if score >= 55 else "Needs work")


def score_resume(parsed: ParsedResume, job_description: str | None = None) -> ATSReport:
    cats = [
        _score_sections(parsed.text),
        _score_contact(parsed),
        _score_formatting(parsed),
        _score_content(parsed.text),
        _score_length(parsed),
    ]
    matched = missing = []
    if job_description and job_description.strip():
        kw, matched, missing = _score_keywords(parsed.text, job_description)
        cats.insert(0, kw)
    else:
        scale = 100 / sum(c.weight for c in cats)
        for c in cats:
            c.weight = round(c.weight * scale, 2)
    overall = sum(c.score * c.weight for c in cats) / sum(c.weight for c in cats)
    # Biggest-impact suggestions first: weight x points lost.
    ordered = sorted(cats, key=lambda c: -(c.weight * (100 - c.score)))
    suggestions = [s for c in ordered for s in c.suggestions]
    warnings = list(parsed.warnings)
    if not job_description:
        warnings.append("No job description given; keyword match was skipped. Pass one for a role-specific score.")
    return ATSReport(round(overall), _grade(overall), cats, matched, missing, suggestions, warnings)
