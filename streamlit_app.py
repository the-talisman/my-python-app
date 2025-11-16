import tempfile
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import requests
from datetime import datetime

from analyzer import (
	detect_job_field,
	extract_skills,
	extract_text_from_pdf,
	skill_keywords,
)


# Try to import sentence-transformers for semantic matching; not fatal if missing
try:
	from sentence_transformers import SentenceTransformer
	import numpy as np

	S_EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
except Exception:
	S_EMBED_MODEL = None


def build_skill_chart(detected_skills: List[str]) -> Optional[plt.Figure]:
	"""Build a modern horizontal bar chart summarizing matched skills per field."""
	field_counts = {
		field: sum(1 for kw in keywords if kw in detected_skills)
		for field, keywords in skill_keywords.items()
	}
	field_counts = {k: v for k, v in field_counts.items() if v > 0}
	if not field_counts:
		return None

	# Sort by count descending for better visualization
	sorted_fields = sorted(field_counts.items(), key=lambda x: x[1], reverse=True)
	fields = [f[0] for f in sorted_fields]
	counts = [f[1] for f in sorted_fields]

	# Modern color gradient from deep blue to cyan
	colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(fields)))

	fig, ax = plt.subplots(figsize=(10, max(5, len(fields) * 0.5)))
	
	# Create horizontal bars with rounded edges
	bars = ax.barh(fields, counts, color=colors, edgecolor='white', linewidth=2, height=0.7)
	
	# Add value labels at the end of each bar
	for i, (bar, count) in enumerate(zip(bars, counts)):
		width = bar.get_width()
		ax.text(width + 0.1, bar.get_y() + bar.get_height()/2, 
				f'{int(count)}',
				ha='left', va='center', fontweight='bold', fontsize=10, color='#0a66c2')
	
	# Styling
	ax.set_xlabel("Number of Matched Skills", fontsize=12, fontweight='bold', color='#333')
	ax.set_ylabel("Job Fields", fontsize=12, fontweight='bold', color='#333')
	ax.set_title("📊 Resume Skill Distribution", fontsize=14, fontweight='bold', color='#0a66c2', pad=20)
	
	# Remove top and right spines for cleaner look
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	ax.spines['left'].set_color('#ddd')
	ax.spines['bottom'].set_color('#ddd')
	
	# Grid for better readability
	ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.7)
	ax.set_axisbelow(True)
	
	# Adjust layout
	plt.tight_layout()
	return fig


def _fs_str(field_obj: dict) -> str:
	if not field_obj:
		return ''
	return field_obj.get('stringValue', '')


def _fs_timestamp(field_obj: dict):
	if not field_obj:
		return None
	return field_obj.get('timestampValue')


def fetch_jobs_from_firestore() -> List[dict]:
	"""Fetch documents from the Firestore `jobs` collection using the REST API."""
	project_id = "skillsync20218118"
	url = f"https://firestore.googleapis.com/v1/projects/{project_id}/databases/(default)/documents/jobs"
	resp = requests.get(url, timeout=10)
	resp.raise_for_status()
	data = resp.json()
	docs = data.get('documents', [])
	jobs = []
	for d in docs:
		fields = d.get('fields', {})
		job = {
			'id': d.get('name', '').split('/')[-1],
			'title': _fs_str(fields.get('title')),
			'company': _fs_str(fields.get('company')),
			'location': _fs_str(fields.get('location')),
			'description': _fs_str(fields.get('description')),
			'createdAt': _fs_timestamp(fields.get('createdAt')),
		}
		jobs.append(job)
	return jobs


def score_jobs_against_skills(jobs: List[dict], skills: List[str]) -> List[dict]:
	"""Score each job by counting occurrences of detected skills in title/description/company."""
	skill_set = set(skills)
	scored = []
	
	# Determine which field the skills belong to
	field_counts = {}
	for skill in skills:
		for field, keywords in skill_keywords.items():
			if skill in keywords:
				field_counts[field] = field_counts.get(field, 0) + 1
	
	# Find dominant field
	dominant_field = max(field_counts.items(), key=lambda x: x[1])[0] if field_counts else None
	dominant_field_keywords = set(skill_keywords.get(dominant_field, [])) if dominant_field else set()
	
	for j in jobs:
		text = ' '.join([j.get('title',''), j.get('company',''), j.get('description',''), j.get('location','')]).lower()
		
		# Count direct skill matches
		direct_matches = sum(1 for sk in skill_set if sk in text)
		
		# Count field-relevant keywords in job posting
		field_matches = sum(1 for kw in dominant_field_keywords if kw in text) if dominant_field_keywords else 0
		
		# Calculate score: 60% from direct matches, 40% from field relevance
		if len(skill_set) > 0:
			direct_score = (direct_matches / len(skill_set)) * 60
		else:
			direct_score = 0
			
		if dominant_field_keywords:
			field_score = min(40, (field_matches / len(dominant_field_keywords)) * 100)
		else:
			field_score = direct_matches * 5
		
		total_score = round(direct_score + field_score, 1)
		
		if total_score > 0:
			j2 = dict(j)
			j2['score'] = total_score
			scored.append(j2)
	
	scored.sort(key=lambda x: x['score'], reverse=True)
	return scored


def semantic_score_jobs(jobs: List[dict], resume_text: str) -> List[dict]:
	"""Compute cosine similarity between resume text and each job posting."""
	if S_EMBED_MODEL is None:
		return []

	try:
		resume_emb = S_EMBED_MODEL.encode(resume_text, convert_to_numpy=True)
	except Exception:
		resume_emb = S_EMBED_MODEL.encode(resume_text)

	job_texts = [' '.join([j.get('title',''), j.get('company',''), j.get('description','')]) for j in jobs]
	try:
		job_embs = S_EMBED_MODEL.encode(job_texts, convert_to_numpy=True)
	except Exception:
		job_embs = S_EMBED_MODEL.encode(job_texts)

	def normalize(v):
		v = np.array(v, dtype=float)
		norm = np.linalg.norm(v)
		return v / (norm + 1e-9)

	r = normalize(resume_emb)
	scored = []
	for j, emb in zip(jobs, job_embs):
		v = normalize(emb)
		sim = float((r * v).sum())
		score = round(sim * 100, 2)
		if score > 0:
			j2 = dict(j)
			j2['score'] = score
			scored.append(j2)

	scored.sort(key=lambda x: x['score'], reverse=True)
	return scored


def main() -> None:
	primary_color = "#0a66c2"
	st.markdown(
		f"""
		<style>
		h1 {{ color: {primary_color} !important; }}
		h2 {{ color: {primary_color} !important; }}
		.stButton>button {{
			background: linear-gradient(90deg, {primary_color}, #155bd6) !important;
			color: white !important;
			border: none !important;
		}}
		</style>
		""",
		unsafe_allow_html=True,
	)

	st.title("AI Resume Analyzer by SkillSync")
	st.write("Upload your resume PDF and get detected skills + predicted career field.")

	uploaded_file = st.file_uploader("Choose your resume (PDF format)", type=["pdf"])
	if uploaded_file is None:
		return

	with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
		tmp_file.write(uploaded_file.read())
		uploaded_file_path = tmp_file.name

	text = extract_text_from_pdf(uploaded_file_path)
	skills = extract_skills(text)
	field = detect_job_field(text)

	st.subheader("Extracted Skills:")
	st.write(", ".join(skills) if skills else "(no skills detected)")

	st.subheader("Predicted Career Field:")
	st.success(field)

	fig = build_skill_chart(skills)
	if fig:
		st.subheader("📊 Skill Distribution")
		st.pyplot(fig)
	else:
		st.info("No specific field matches strong enough for charting.")

	st.subheader("Best Matching Jobs from Site")
	try:
		jobs = fetch_jobs_from_firestore()
	except Exception as e:
		st.warning(f"Could not fetch jobs from site: {e}")
		jobs = []

	if skills and jobs:
		matches = semantic_score_jobs(jobs, text) if S_EMBED_MODEL is not None else score_jobs_against_skills(jobs, skills)
		if matches:
			min_score = 30 if S_EMBED_MODEL is not None else 20
			relevant_matches = [m for m in matches if m['score'] >= min_score]
			if relevant_matches:
				st.write(f"Top {min(5, len(relevant_matches))} matches:")
				for job in relevant_matches[:5]:
					cols = st.columns([3, 1])
					with cols[0]:
						st.markdown(f"**{job['title']}** — {job.get('company','')} \n\n{job.get('description','')[:240]}{'...' if len(job.get('description',''))>240 else ''}")
						if job.get('location'):
							st.caption(job.get('location'))
					with cols[1]:
						st.metric("Score", job['score'])
				st.markdown("---")
			else:
				st.info("No closely matching jobs found. Try uploading a different resume.")
		else:
			st.info("No matching jobs found based on detected skills.")
	elif not skills:
		st.info("Upload a resume to see matching jobs from the site.")
	else:
		st.info("No jobs available on the site to match against.")


if __name__ == "__main__":
	main()
