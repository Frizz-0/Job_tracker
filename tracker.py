import asyncio
import json
import os
import re
import sqlite3
from pathlib import Path
from typing import Literal
import httpx
import yaml
from groq import AsyncGroq
from pydantic import BaseModel
from datetime import datetime, timezone

DB_PATH = "jobs.db"

class CompanyTarget(BaseModel):
    name: str
    ats: Literal["greenhouse", "lever", "ashby"]
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
    match_summary: str
    uk_location_ok: bool
    red_flags: list[str]

def init_db(db_path: str = DB_PATH) -> tuple[sqlite3.Connection, bool]:
    is_first_run = not Path(db_path).exists()
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS seen_jobs (
            job_id TEXT PRIMARY KEY,
            company TEXT,
            title TEXT,
            score INTEGER,
            uk_ok INTEGER,
            url TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn, is_first_run

def is_seen(conn: sqlite3.Connection, job_id: str) -> bool:
    cur = conn.execute("SELECT 1 FROM seen_jobs WHERE job_id = ?", (job_id,))
    return cur.fetchone() is not None

def mark_seen(conn: sqlite3.Connection, job: JobPosting, score: int, uk_ok: bool = False):
    conn.execute(
        "INSERT OR IGNORE INTO seen_jobs (job_id, company, title, score, uk_ok, url) VALUES (?, ?, ?, ?, ?, ?)",
        (job.id, job.company, job.title, score, int(uk_ok), job.url),
    )
    conn.commit()

async def fetch_company_jobs(client: httpx.AsyncClient, target: CompanyTarget) -> list[JobPosting]:
    jobs: list[JobPosting] = []
    try:
        if target.ats == "greenhouse":
            r = await client.get(f"https://boards-api.greenhouse.io/v1/boards/{target.slug}/jobs?content=true")
            if r.status_code == 404:
                print(f"[SKIP] {target.name}: slug '{target.slug}' not found on Greenhouse")
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
                print(f"[SKIP] {target.name}: slug '{target.slug}' not found on Lever")
                return []
            r.raise_for_status()
            for item in r.json():
                jobs.append(JobPosting(
                    id=f"lv_{item['id']}",
                    company=target.name,
                    title=item["text"],
                    location=item.get("categories", {}).get("location", "Unknown"),
                    url=item["hostedUrl"],
                    description=item.get("descriptionPlain", "")[:6000],
                ))

        elif target.ats == "ashby":
            r = await client.get(f"https://api.ashbyhq.com/posting-api/job-board/{target.slug}")
            if r.status_code == 404:
                print(f"[SKIP] {target.name}: slug '{target.slug}' not found on Ashby")
                return []
            r.raise_for_status()
            for item in r.json().get("jobs", []):
                jobs.append(JobPosting(
                    id=f"ab_{item['id']}",
                    company=target.name,
                    title=item["title"],
                    location=item.get("location", "Unknown"),
                    url=item["jobUrl"],
                    description=item.get("descriptionPlain", "")[:6000],
                ))
    except Exception as e:
        print(f"[WARN] Failed fetching {target.name} ({target.ats}): {e}")
    return jobs

TITLE_INCLUDE = re.compile(
    r"(computer vision|vision|machine learning|ml\b|ai\b|artificial intelligence|deep learning|"
    r"perception|imaging|robotics|multimodal|vlm|research engineer|applied scientist)",
    re.IGNORECASE,
)
TITLE_EXCLUDE = re.compile(
    r"(director|vp\b|vice president|head of|intern\b|recruiter|sales|account|legal|clinical specialist|people partner)",
    re.IGNORECASE,
)

def passes_title_filter(title: str) -> bool:
    return bool(TITLE_INCLUDE.search(title)) and not bool(TITLE_EXCLUDE.search(title))

CANDIDATE_PROFILE = """
- Education: MSc in Artificial Intelligence (Ulster University, Belfast, UK).
- Background: 6 years technical experience (3D pipelines, simulation, technical automation) + hands-on AI/ML & Computer Vision engineering.
- Core Stack: Python, PyTorch, TensorFlow, OpenCV, ONNX Runtime, TensorRT, LangGraph, FastAPI, Docker, AWS.
- Key Projects: Multi-spectral skin lesion VLM/XAI diagnostic app (Grad-CAM, ONNX/WASM), ROS/Gazebo multi-robot autonomous hazard detection, multi-agent financial auditing (LangGraph).
- Target Roles: Computer Vision Engineer, Applied AI / ML Engineer, Perception Engineer (especially Healthcare AI, Medical Imaging, Robotics, or Multimodal/VLM systems).
- Location: Based in Belfast, UK (open to UK hybrid/relocation across UK cities or UK/EU remote).
"""

async def score_job(groq_client: AsyncGroq, job: JobPosting) -> JobEvaluation:
    prompt = f"""
Evaluate this job posting against the candidate profile. Return ONLY valid JSON matching:
{{"score": int (0-100), "match_summary": str (max 2 sentences), "uk_location_ok": bool, "red_flags": [str]}}

Scoring rubric:
- 85-100: Direct Computer Vision, Medical Imaging, Robotics Perception, or VLM/Applied AI Engineer role in UK/Remote matching PyTorch/OpenCV/edge/agentic stack.
- 70-84: Strong general ML/AI Engineer role where candidate's 6 yrs engineering + MSc AI fits well.
- <70: Wrong location (e.g., US-only on-site), hard PhD-only requirement, or unrelated tech stack.

Candidate Profile:
{CANDIDATE_PROFILE}

Job Title: {job.title}
Company: {job.company}
Location: {job.location}
Description:
{job.description}
"""
    resp = await groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    data = json.loads(resp.choices[0].message.content)
    return JobEvaluation(**data)

async def send_discord_alert(client: httpx.AsyncClient, webhook_url: str, job: JobPosting, eval_res: JobEvaluation):
    flags = ", ".join(eval_res.red_flags) if eval_res.red_flags else "None"
    print(f"[MATCH {eval_res.score}] {job.company} — {job.title} ({job.location}) | {job.url}")
    if not webhook_url:
        return
    payload = {
        "content": (
            f"**[{eval_res.score}/100] {job.company} — {job.title}**\n"
            f"📍 `{job.location}` | 🔗 {job.url}\n"
            f"💡 {eval_res.match_summary}\n"
            f"⚠️ Flags: {flags}"
        )
    }
    await client.post(webhook_url, json=payload)

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

        new_matches = 0
        for job in all_jobs:
            if is_seen(conn, job.id):
                continue
            if not passes_title_filter(job.title):
                mark_seen(conn, job, score=0, uk_ok=False)
                continue

            try:
                eval_res = await score_job(groq_client, job)
                mark_seen(conn, job, score=eval_res.score, uk_ok=eval_res.uk_location_ok)

                if eval_res.score >= 75 and eval_res.uk_location_ok:
                    new_matches += 1
                    # On initial DB seed, log top matches in Actions console without spamming Discord
                    if not is_first_run:
                        await send_discord_alert(client, webhook_url, job, eval_res)
                    else:
                        print(f"[SEED MATCH {eval_res.score}] {job.company} — {job.title} ({job.location}) -> {job.url}")

                await asyncio.sleep(0.4) # Gentle rate-limit buffer for Groq free tier
            except Exception as e:
                print(f"[WARN] Scoring failed for {job.company} - {job.title}: {e}")

        export_markdown(conn)
        export_dashboard_json(conn)
        print(f"Run complete. New high-fit matches: {new_matches} (first_run={is_first_run}).")
        

def export_markdown(conn: sqlite3.Connection, md_path: str = "JOBS.md"):
    rows = conn.execute("""
        SELECT score, company, title, url, created_at
        FROM seen_jobs
        WHERE score >= 60 AND uk_ok = 1
        ORDER BY score DESC, created_at DESC
    """).fetchall()

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# CV & AI Engineering Job Board",
        f"*Last updated: {now} | Showing {len(rows)} UK/Remote matches (Score ≥ 60)*\n",
        "| Score | Company | Role | First Seen |",
        "| :---: | :--- | :--- | :--- |",
    ]
    for score, company, title, url, created_at in rows:
        lines.append(f"| **{score}** | {company} | [{title}]({url}) | {created_at[:10]} |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

def export_dashboard_json(conn: sqlite3.Connection, json_path: str = "jobs.json"):
    rows = conn.execute("""
        SELECT job_id, score, company, title, url, created_at
        FROM seen_jobs
        WHERE score >= 50 AND uk_ok = 1
        ORDER BY score DESC, created_at DESC
    """).fetchall()

    payload = {
        "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "jobs": [
            {
                "id": r[0],
                "score": r[1],
                "company": r[2],
                "title": r[3],
                "url": r[4],
                "date": r[5][:10] if r[5] else "",
            }
            for r in rows
        ],
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    asyncio.run(run_pipeline())