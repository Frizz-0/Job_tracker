import asyncio
import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import httpx
import yaml
from groq import AsyncGroq
from pydantic import BaseModel

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

DB_PATH = "jobs.db"
GROQ_MODEL = "openai/gpt-oss-20b"
MAX_SCORES_PER_RUN = 20

class CompanyTarget(BaseModel):
    name: str
    ats: Literal["greenhouse", "lever", "ashby", "workable"]
    slug: str

class JobPosting(BaseModel):
    id: str
    company: str
    title: str
    location: str
    url: str
    description: str

class JobEvaluation(BaseModel):
    score: int
    region: Literal["UK", "EU", "US", "Remote", "Global/Other"]
    match_summary: str
    matched_skills: list[str]
    missing_gaps: list[str]
    red_flags: list[str]

FALLBACK_CV = """
Bharath Reddy Mohan — Belfast, UK (Open to UK hybrid/relocation & Global/Remote).
Education: MSc Artificial Intelligence, Ulster University (Jan 2026 – May 2027).
Experience: 6 years professional experience as 3D Generalist & Technical Artist (Unity, procedural 3D environments, shaders, synthetic datasets, pipeline automation).
Stack: Python 3.12, PyTorch, NumPy, Scikit-learn, OpenCV, ONNX/WASM edge inference, Grad-CAM XAI, LangGraph, LangChain, FastAPI, Docker, PostgreSQL, AWS (EC2, Lambda, DynamoDB, RDS).
Projects: Cutis Derm AI (multi-spectral skin lesion VLM/XAI diagnostic app), Multi-Robot Hospital Hazard Detection (ROS & Gazebo Bug2 state machine), LedgerVi (LangGraph multi-agent auditing), Parasite Egg Detection & OASIS Alzheimer's MRI research.
"""

def load_candidate_cv() -> str:
    if Path("cv.txt").exists():
        text = Path("cv.txt").read_text(encoding="utf-8").strip()
        if text:
            print("Loaded candidate CV from cv.txt")
            return text[:4500]
    if Path("cv.pdf").exists() and PdfReader is not None:
        reader = PdfReader("cv.pdf")
        pages = [p.extract_text() or "" for p in reader.pages]
        text = "\n".join(pages).strip()
        if text:
            print("Loaded candidate CV from cv.pdf")
            return text[:4500]
    print("Using built-in candidate CV profile (add cv.pdf or cv.txt to customize).")
    return FALLBACK_CV

def init_db(db_path: str = DB_PATH) -> tuple[sqlite3.Connection, bool]:
    is_first_run = not Path(db_path).exists()
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_jobs (
            job_id TEXT PRIMARY KEY,
            company TEXT,
            title TEXT,
            location TEXT DEFAULT 'Unknown',
            region TEXT DEFAULT 'UK',
            score INTEGER,
            uk_ok INTEGER DEFAULT 1,
            match_summary TEXT DEFAULT '',
            matched_skills TEXT DEFAULT '[]',
            missing_gaps TEXT DEFAULT '[]',
            red_flags TEXT DEFAULT '[]',
            is_active INTEGER DEFAULT 1,
            url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(seen_jobs)").fetchall()}
    migrations = {
        "location": "TEXT DEFAULT 'Unknown'",
        "region": "TEXT DEFAULT 'UK'",
        "match_summary": "TEXT DEFAULT ''",
        "matched_skills": "TEXT DEFAULT '[]'",
        "missing_gaps": "TEXT DEFAULT '[]'",
        "red_flags": "TEXT DEFAULT '[]'",
        "is_active": "INTEGER DEFAULT 1",
    }
    for col, col_def in migrations.items():
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE seen_jobs ADD COLUMN {col} {col_def}")
    conn.commit()
    return conn, is_first_run

def needs_scoring(conn: sqlite3.Connection, job_id: str) -> bool:
    row = conn.execute(
        "SELECT score, match_summary FROM seen_jobs WHERE job_id = ?", (job_id,)
    ).fetchone()
    if row is None:
        return True
    return row[0] > 0 and not row[1]

US_LOC_RE = re.compile(
    r"\b(united states|usa|u\.s\.|california|san francisco|mountain view|palo alto|sunnyvale|"
    r"menlo park|new york|nyc|seattle|austin|boston|chicago|los angeles|texas|massachusetts|"
    r"washington, dc|,\s*(ca|ny|wa|tx|ma|il|co|ga|va|nc|pa|nj))\b",
    re.IGNORECASE,
)
UK_LOC_RE = re.compile(
    r"\b(united kingdom|uk\b|u\.k\.|london|belfast|cambridge|oxford|edinburgh|manchester|"
    r"bristol|england|scotland|wales|northern ireland)\b",
    re.IGNORECASE,
)

def normalize_region(location: str, llm_region: str) -> str:
    loc = location or ""
    if US_LOC_RE.search(loc) and not UK_LOC_RE.search(loc):
        return "US"
    if UK_LOC_RE.search(loc) and not US_LOC_RE.search(loc):
        return "UK"
    if "remote" in loc.lower() and not US_LOC_RE.search(loc):
        return "Remote"
    return llm_region

def save_job_eval(conn: sqlite3.Connection, job: JobPosting, eval_res: JobEvaluation):
    true_region = normalize_region(job.location, eval_res.region)
    uk_ok = 1 if true_region in ("UK", "Remote") else 0
    conn.execute("""
        INSERT INTO seen_jobs (
            job_id, company, title, location, region, score, uk_ok,
            match_summary, matched_skills, missing_gaps, red_flags, is_active, url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
        ON CONFLICT(job_id) DO UPDATE SET
            location=excluded.location,
            region=excluded.region,
            score=excluded.score,
            uk_ok=excluded.uk_ok,
            match_summary=excluded.match_summary,
            matched_skills=excluded.matched_skills,
            missing_gaps=excluded.missing_gaps,
            red_flags=excluded.red_flags,
            is_active=1,
            url=excluded.url
    """, (
        job.id, job.company, job.title, job.location, true_region,
        eval_res.score, uk_ok, eval_res.match_summary,
        json.dumps(eval_res.matched_skills),
        json.dumps(eval_res.missing_gaps),
        json.dumps(eval_res.red_flags),
        job.url,
    ))
    conn.commit()

def mark_irrelevant(conn: sqlite3.Connection, job: JobPosting):
    conn.execute("""
        INSERT OR IGNORE INTO seen_jobs (job_id, company, title, location, score, is_active, url)
        VALUES (?, ?, ?, ?, 0, 1, ?)
    """, (job.id, job.company, job.title, job.location, job.url))
    conn.commit()

async def fetch_company_jobs(client: httpx.AsyncClient, target: CompanyTarget) -> list[JobPosting]:
    jobs: list[JobPosting] = []
    try:
        if target.ats == "greenhouse":
            r = await client.get(f"https://boards-api.greenhouse.io/v1/boards/{target.slug}/jobs?content=true")
            if r.status_code == 404:
                print(f"[SKIP] {target.name}: '{target.slug}' not on Greenhouse")
                return []
            r.raise_for_status()
            for item in r.json().get("jobs", []):
                jobs.append(JobPosting(
                    id=f"gh_{item['id']}",
                    company=target.name,
                    title=item["title"],
                    location=item.get("location", {}).get("name", "Unknown"),
                    url=item["absolute_url"],
                    description=item.get("content", "")[:2800],
                ))

        elif target.ats == "lever":
            r = await client.get(f"https://api.lever.co/v0/postings/{target.slug}?mode=json")
            if r.status_code == 404:
                print(f"[SKIP] {target.name}: '{target.slug}' not on Lever")
                return []
            r.raise_for_status()
            for item in r.json():
                jobs.append(JobPosting(
                    id=f"lv_{item['id']}",
                    company=target.name,
                    title=item["text"],
                    location=item.get("categories", {}).get("location", "Unknown"),
                    url=item["hostedUrl"],
                    description=item.get("descriptionPlain", "")[:2800],
                ))

        elif target.ats == "ashby":
            r = await client.get(f"https://api.ashbyhq.com/posting-api/job-board/{target.slug}")
            if r.status_code == 404:
                print(f"[SKIP] {target.name}: '{target.slug}' not on Ashby")
                return []
            r.raise_for_status()
            for item in r.json().get("jobs", []):
                jobs.append(JobPosting(
                    id=f"ab_{item['id']}",
                    company=target.name,
                    title=item["title"],
                    location=item.get("location", "Unknown"),
                    url=item["jobUrl"],
                    description=item.get("descriptionPlain", "")[:2800],
                ))

        elif target.ats == "workable":
            r = await client.get(f"https://apply.workable.com/api/v1/widget/accounts/{target.slug}")
            if r.status_code == 404:
                print(f"[SKIP] {target.name}: '{target.slug}' not on Workable")
                return []
            r.raise_for_status()
            for item in r.json().get("jobs", []):
                loc = ", ".join(filter(None, [item.get("city"), item.get("country")])) or "Unknown"
                jobs.append(JobPosting(
                    id=f"wk_{item['shortcode']}",
                    company=target.name,
                    title=item["title"],
                    location=loc,
                    url=item["url"],
                    description=f"{item.get('department', '')} | {loc} | {item.get('employment_type', '')}",
                ))
    except Exception as e:
        print(f"[WARN] Failed fetching {target.name} ({target.ats}): {e}")
    return jobs

TITLE_INCLUDE = re.compile(
    r"(computer vision|vision|machine learning|ml\b|ai\b|artificial intelligence|deep learning|"
    r"perception|imaging|robotics|multimodal|vlm|research engineer|applied scientist|3d|simulation)",
    re.IGNORECASE,
)
TITLE_EXCLUDE = re.compile(
    r"(director|vp\b|vice president|head of|intern\b|recruiter|sales|account|legal|clinical specialist|"
    r"people partner|designer|product manager|marketing|staff\b|principal\b|tech lead|manager)",
    re.IGNORECASE,
)

def passes_title_filter(title: str) -> bool:
    return bool(TITLE_INCLUDE.search(title)) and not bool(TITLE_EXCLUDE.search(title))

async def score_job(groq_client: AsyncGroq, cv_text: str, job: JobPosting) -> JobEvaluation:
    prompt = f"""
Audit how well this candidate's actual CV fits the job posting. Include both UK and global roles, giving a +7 priority boost for UK or UK-friendly Remote roles since the candidate is based in Belfast, UK.

Return ONLY valid JSON matching:
{{
  "score": int (0-100),
  "region": "UK" | "EU" | "US" | "Remote" | "Global/Other",
  "match_summary": "1-2 concise sentences citing specific CV projects/experience vs JD requirements",
  "matched_skills": ["3 to 5 specific matching skills/projects from the candidate's CV"],
  "missing_gaps": ["1 to 3 missing skills/requirements from JD, or empty list"],
  "red_flags": ["e.g. 'PhD required', '8+ YoE', 'US Security Clearance', or empty list"]
}}

Scoring rubric:
- 85-100: Strong match for Computer Vision, Medical Imaging, Robotics Perception, 3D/Simulation AI, or Applied VLM/Agentic Engineer matching the CV's PyTorch/OpenCV/3D/FastAPI stack (plus +7 boost if UK/Remote).
- 70-84: Solid Junior/Mid ML or AI Engineer role where MSc AI + 6 yrs technical 3D/engineering background transfers strongly.
- 50-69: Partial fit (adjacent role or asks for tools missing from CV like heavy C++/CUDA/Kubernetes).
- <50: Unrelated domain or hard senior/PhD barrier.

Candidate CV:
{cv_text}

Job Title: {job.title}
Company: {job.company}
Location: {job.location}
Description:
{job.description}
"""
    resp = await groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    data = json.loads(resp.choices[0].message.content)
    return JobEvaluation(**data)

async def send_discord_alert(client: httpx.AsyncClient, webhook_url: str, job: JobPosting, eval_res: JobEvaluation):
    if not webhook_url:
        return
    flags = ", ".join(eval_res.red_flags) if eval_res.red_flags else "None"
    skills = ", ".join(eval_res.matched_skills[:4])
    msg = (
        f"**[{eval_res.score}/100] [{eval_res.region}] {job.company} — {job.title}**\n"
        f"📍 `{job.location}` | 🔗 {job.url}\n"
        f"✅ **CV Fit:** {skills}\n"
        f"💡 {eval_res.match_summary}\n"
        f"⚠️ **Flags:** {flags}"
    )[:1900]
    resp = await client.post(webhook_url, json={"content": msg})
    if resp.status_code not in (200, 204):
        print(f"[DISCORD ERROR] HTTP {resp.status_code}: {resp.text}")
    await asyncio.sleep(1.0)

def export_markdown(conn: sqlite3.Connection, md_path: str = "JOBS.md"):
    rows = conn.execute("""
        SELECT score, region, company, title, location, url, created_at
        FROM seen_jobs
        WHERE score >= 65 AND is_active = 1
        ORDER BY score DESC, created_at DESC
    """).fetchall()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# CV & AI Engineering Job Radar",
        f"*Last updated: {now} | Showing {len(rows)} active matches (Score ≥ 65)*\n",
        "| Score | Region | Company | Role | Location | First Seen |",
        "| :---: | :---: | :--- | :--- | :--- | :--- |",
    ]
    for score, region, company, title, loc, url, created_at in rows:
        lines.append(f"| **{score}** | `{region}` | {company} | [{title}]({url}) | {loc} | {created_at[:10]} |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def export_dashboard_json(conn: sqlite3.Connection, json_path: str = "jobs.json"):
    # Only export active roles seen within the last 14 days so stale unapplied roles auto-prune
    rows = conn.execute("""
        SELECT job_id, score, region, company, title, location, match_summary,
               matched_skills, missing_gaps, red_flags, url, created_at
        FROM seen_jobs
        WHERE score >= 50
          AND is_active = 1
          AND datetime(created_at) >= datetime('now', '-14 days')
        ORDER BY score DESC, created_at DESC
    """).fetchall()

    payload = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "jobs": [
            {
                "id": r[0],
                "score": r[1],
                "region": normalize_region(r[5], r[2] or "Other"),
                "company": r[3],
                "title": r[4],
                "location": r[5] or "Unknown",
                "summary": r[6] or "",
                "matched_skills": json.loads(r[7] or "[]"),
                "missing_gaps": json.loads(r[8] or "[]"),
                "red_flags": json.loads(r[9] or "[]"),
                "url": r[10],
                "date": r[11][:10] if r[11] else "",
            }
            for r in rows
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

async def run_pipeline():
    conn, is_first_run = init_db()
    cv_text = load_candidate_cv()
    groq_client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")

    with open("companies.yaml", "r") as f:
        targets = [CompanyTarget(**c) for c in yaml.safe_load(f)]

    async with httpx.AsyncClient(timeout=15.0) as client:
        results = await asyncio.gather(*(fetch_company_jobs(client, t) for t in targets))
        all_jobs = [job for company_jobs in results for job in company_jobs]
        print(f"Fetched {len(all_jobs)} total jobs across {len(targets)} companies.")

        live_ids = {j.id for j in all_jobs}
        if live_ids:
            db_ids = {r[0] for r in conn.execute("SELECT job_id FROM seen_jobs WHERE is_active = 1").fetchall()}
            closed_ids = db_ids - live_ids
            if closed_ids:
                conn.executemany("UPDATE seen_jobs SET is_active = 0 WHERE job_id = ?", [(cid,) for cid in closed_ids])
                conn.commit()

        new_matches = 0
        scored_this_run = 0
        for job in all_jobs:
            if not passes_title_filter(job.title):
                mark_irrelevant(conn, job)
                continue
            if not needs_scoring(conn, job.id):
                conn.execute("UPDATE seen_jobs SET is_active = 1 WHERE job_id = ?", (job.id,))
                continue
            if scored_this_run >= MAX_SCORES_PER_RUN:
                continue

            try:
                eval_res = await score_job(groq_client, cv_text, job)
                save_job_eval(conn, job, eval_res)
                scored_this_run += 1

                if eval_res.score >= 78:
                    new_matches += 1
                    print(f"[MATCH {eval_res.score}] [{eval_res.region}] {job.company} — {job.title}")
                    await send_discord_alert(client, webhook_url, job, eval_res)

                await asyncio.sleep(0.5)
            except Exception as e:
                print(f"[WARN] Scoring failed for {job.company} - {job.title}: {e}")
                if "429" in str(e) or "rate_limit" in str(e).lower():
                    print("[STOP] Rate limit reached — halting scoring until next run.")
                    break

        conn.commit()
        export_markdown(conn)
        export_dashboard_json(conn)
        print(f"Run complete. Scored: {scored_this_run}/{MAX_SCORES_PER_RUN} | High-fit matches: {new_matches}.")

if __name__ == "__main__":
    asyncio.run(run_pipeline())