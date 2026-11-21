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

DB_PATH = "jobs.db"

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
    # Safely migrate existing DB columns if upgrading from v1
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
    # Re-evaluate previously filtered/unexplained jobs now that non-UK roles are included
    return row[0] > 0 and not row[1]

def save_job_eval(conn: sqlite3.Connection, job: JobPosting, eval_res: JobEvaluation):
    uk_ok = 1 if eval_res.region in ("UK", "Remote") else 0
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
        job.id, job.company, job.title, job.location, eval_res.region,
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
    r"(director|vp\b|vice president|head of|intern\b|recruiter|sales|account|legal|clinical specialist|people partner)",
    re.IGNORECASE,
)

def passes_title_filter(title: str) -> bool:
    return bool(TITLE_INCLUDE.search(title)) and not bool(TITLE_EXCLUDE.search(title))

CANDIDATE_CV = """
- Name & Location: Bharath Reddy Mohan (Belfast, UK — open to UK, EU, US, and Remote roles).
- Education: MSc in Artificial Intelligence, Ulster University (Jan 2026 – May 2027).
- Prior Experience: 6 years professional experience as a 3D Generalist & Technical Artist (3D environments, procedural generation, Unity, shaders, simulation assets, pipeline automation) prior to transitioning into AI/CV Engineering.
- Core AI/CV Stack: Python 3.12, PyTorch, NumPy, Scikit-learn, OpenCV, ONNX Runtime / WASM edge inference, Grad-CAM explainability, LangGraph, LangChain, FastAPI, Docker, PostgreSQL, AWS (EC2, Lambda, DynamoDB, RDS).
- Flagship Projects:
  1. Cutis Derm AI: Multi-spectral dermatology & skin lesion VLM/XAI diagnostic web app (ONNX/WASM, Grad-CAM heatmaps, FastAPI, Groq VLM).
  2. Multi-Robot Hospital Hazard Detection (ROS & Gazebo): Autonomous Bug2 navigation state machine, central GoalManager node, SDF simulation.
  3. LedgerVi & AI Investigator: Stateful multi-agent auditing & RAG workflows (LangGraph, PostgreSQL/pgvector, SearXNG).
  4. Alzheimer's Longitudinal MRI/Clinical Detection (OASIS dataset) & Custom CNN from scratch in NumPy.
- Target Roles: Computer Vision Engineer, Applied AI / ML Engineer, Robotics Perception / Simulation Engineer, Medical Imaging AI Engineer, Multimodal/VLM Engineer.
"""

SCORING_MODELS = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]

async def score_job(groq_client: AsyncGroq, job: JobPosting) -> JobEvaluation:
    prompt = f"""
Audit how well the candidate's CV fits this job posting across ALL locations globally (do NOT penalize non-UK locations).
Return ONLY valid JSON matching:
{{
  "score": int (0-100),
  "region": "UK" | "EU" | "US" | "Remote" | "Global/Other",
  "match_summary": "1-2 concise sentences on why the candidate fits or lacks fit",
  "matched_skills": ["3 to 5 specific matching skills/projects from candidate CV"],
  "missing_gaps": ["1 to 3 missing skills/requirements from JD, or empty list"],
  "red_flags": ["e.g. 'PhD required', '8+ YoE', 'US Security Clearance', or empty list"]
}}

Scoring rubric:
- 85-100: Strong match for Computer Vision, Medical Imaging, Robotics Perception, 3D/Simulation AI, or Applied VLM/Agentic Engineer matching PyTorch/OpenCV/3D/FastAPI stack.
- 70-84: Solid Junior/Mid ML or AI Engineer match where MSc AI + 6 yrs technical 3D/engineering background transfers strongly.
- 50-69: Partial fit (e.g., adjacent software/data role or asks for some tools candidate lacks like heavy C++/CUDA/Kubernetes).
- <50: Unrelated domain or hard senior/staff/PhD barrier.

Candidate CV:
{CANDIDATE_CV}

Job Title: {job.title}
Company: {job.company}
Location: {job.location}
Description:
{job.description}
"""
    last_err = None
    for model_name in SCORING_MODELS:
        try:
            resp = await groq_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,
            )
            data = json.loads(resp.choices[0].message.content)
            return JobEvaluation(**data)
        except Exception as e:
            last_err = e
            if "model_not_found" in str(e) or "429" in str(e):
                continue
            raise
    raise last_err

async def send_discord_alert(client: httpx.AsyncClient, webhook_url: str, job: JobPosting, eval_res: JobEvaluation):
    if not webhook_url:
        print("[WARN] DISCORD_WEBHOOK_URL is empty or not set in GitHub Secrets.")
        return

    flags = ", ".join(eval_res.red_flags) if eval_res.red_flags else "None"
    skills = ", ".join(eval_res.matched_skills[:4])
    msg = (
        f"**[{eval_res.score}/100] [{eval_res.region}] {job.company} — {job.title}**\n"
        f"📍 `{job.location}` | 🔗 {job.url}\n"
        f"✅ **Fit:** {skills}\n"
        f"💡 {eval_res.match_summary}\n"
        f"⚠️ **Flags:** {flags}"
    )[:1900]  # Enforce Discord's 2000-char limit

    resp = await client.post(webhook_url, json={"content": msg})
    if resp.status_code not in (200, 204):
        print(f"[DISCORD ERROR] HTTP {resp.status_code}: {resp.text}")
    else:
        print(f"[DISCORD SENT] {job.company} — {job.title}")
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
    rows = conn.execute("""
        SELECT job_id, score, region, company, title, location, match_summary,
               matched_skills, missing_gaps, red_flags, url, created_at
        FROM seen_jobs
        WHERE score >= 50 AND is_active = 1
        ORDER BY score DESC, created_at DESC
    """).fetchall()

    payload = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "jobs": [
            {
                "id": r[0],
                "score": r[1],
                "region": r[2] or "Other",
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
    groq_client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"])
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")

    with open("companies.yaml", "r") as f:
        targets = [CompanyTarget(**c) for c in yaml.safe_load(f)]

    async with httpx.AsyncClient(timeout=15.0) as client:
        results = await asyncio.gather(*(fetch_company_jobs(client, t) for t in targets))
        all_jobs = [job for company_jobs in results for job in company_jobs]
        print(f"Fetched {len(all_jobs)} total jobs across {len(targets)} companies.")

        # Mark closed jobs inactive so your dashboard stays fresh
        live_ids = {j.id for j in all_jobs}
        if live_ids:
            db_ids = {r[0] for r in conn.execute("SELECT job_id FROM seen_jobs WHERE is_active = 1").fetchall()}
            closed_ids = db_ids - live_ids
            if closed_ids:
                conn.executemany("UPDATE seen_jobs SET is_active = 0 WHERE job_id = ?", [(cid,) for cid in closed_ids])
                conn.commit()

        new_matches = 0
        for job in all_jobs:
            if not passes_title_filter(job.title):
                mark_irrelevant(conn, job)
                continue
            if not needs_scoring(conn, job.id):
                conn.execute("UPDATE seen_jobs SET is_active = 1 WHERE job_id = ?", (job.id,))
                continue

            try:
                eval_res = await score_job(groq_client, job)
                save_job_eval(conn, job, eval_res)

                if eval_res.score >= 78:
                    new_matches += 1
                    if not is_first_run:
                        await send_discord_alert(client, webhook_url, job, eval_res)
                    else:
                        print(f"[MATCH {eval_res.score}] [{eval_res.region}] {job.company} — {job.title}")

                await asyncio.sleep(0.35)
            except Exception as e:
                print(f"[WARN] Scoring failed for {job.company} - {job.title}: {e}")

        conn.commit()
        export_markdown(conn)
        export_dashboard_json(conn)
        print(f"Run complete. New high-fit matches: {new_matches} (first_run={is_first_run}).")

if __name__ == "__main__":
    asyncio.run(run_pipeline())