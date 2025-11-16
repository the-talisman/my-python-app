from __future__ import annotations

from typing import List, Dict

# PDF extraction
import fitz  # PyMuPDF

# NLP
import nltk
from nltk.corpus import stopwords


def _ensure_nltk_resources() -> None:
	"""Ensure tokenizers and stopwords are available.

	Handles both legacy and new NLTK resource names (punkt vs punkt_tab).
	Falls back gracefully if downloads fail (tokenize by whitespace).
	"""
	# Stopwords
	try:
		stopwords.words("english")
	except LookupError:
		nltk.download("stopwords", quiet=True)

	# Tokenizers (punkt or punkt_tab depending on NLTK version)
	try:
		nltk.data.find("tokenizers/punkt")
	except LookupError:
		try:
			nltk.data.find("tokenizers/punkt_tab")
		except LookupError:
			# Try download new name first, then legacy
			try:
				nltk.download("punkt_tab", quiet=True)
			except Exception:
				nltk.download("punkt", quiet=True)


def extract_text_from_pdf(pdf_path: str) -> str:
	"""Extract text from a PDF file using PyMuPDF."""
	try:
		doc = fitz.open(pdf_path)
		text_parts: List[str] = []
		for page in doc:
			text_parts.append(page.get_text())
		return "\n".join(text_parts)
	except Exception:
		return ""


skill_keywords: Dict[str, List[str]] = {
	"cybersecurity": [
		"cybersecurity", "security", "penetration testing", "ethical hacking", "vulnerability",
		"firewall", "encryption", "network security", "malware", "threat analysis",
		"siem", "ids", "ips", "cissp", "ceh", "comptia security+", "soc",
		"incident response", "forensics", "risk assessment", "security audit",
	],
	"software engineering": [
		"software engineering", "software development", "programming", "coding",
		"algorithms", "data structures", "oop", "design patterns", "agile", "scrum",
		"git", "version control", "ci/cd", "testing", "debugging", "java", "c++",
		"c#", "python", "ruby", "go", "rust", "software architecture", "microservices",
	],
	"web development": [
		"html", "css", "javascript", "react", "vue", "angular", "node", "express",
		"django", "flask", "typescript", "php", "laravel", "wordpress", "api", "rest",
		"graphql", "mongodb", "mysql", "postgresql", "frontend", "backend", "fullstack",
		"responsive design", "web performance", "seo", "accessibility",
	],
	"mobile development": [
		"mobile", "ios", "android", "swift", "kotlin", "react native", "flutter",
		"xamarin", "mobile app", "app development", "objective-c", "java",
		"mobile ui", "app store", "google play", "push notifications",
	],
	"data science": [
		"data science", "data scientist", "pandas", "numpy", "scikit-learn",
		"statistics", "statistical analysis", "data visualization", "jupyter",
		"r", "tableau", "power bi", "data mining", "predictive modeling",
		"a/b testing", "hypothesis testing", "regression", "classification",
	],
	"artificial intelligence": [
		"ai", "artificial intelligence", "machine learning", "deep learning",
		"neural networks", "tensorflow", "keras", "pytorch", "nlp",
		"natural language processing", "computer vision", "opencv", "transformers",
		"bert", "gpt", "large language models", "llm", "reinforcement learning",
	],
	"cloud computing": [
		"cloud", "aws", "azure", "gcp", "google cloud", "cloud computing",
		"ec2", "s3", "lambda", "kubernetes", "docker", "containerization",
		"cloud architecture", "serverless", "iaas", "paas", "saas",
	],
	"devops": [
		"devops", "ci/cd", "jenkins", "gitlab", "github actions", "docker",
		"kubernetes", "terraform", "ansible", "chef", "puppet", "automation",
		"infrastructure as code", "monitoring", "prometheus", "grafana", "elk",
	],
	"database administration": [
		"database", "dba", "sql", "mysql", "postgresql", "oracle", "mongodb",
		"nosql", "redis", "cassandra", "database design", "query optimization",
		"backup", "recovery", "replication", "indexing", "stored procedures",
	],
	"qa testing": [
		"qa", "quality assurance", "testing", "test automation", "selenium",
		"junit", "pytest", "test cases", "manual testing", "regression testing",
		"performance testing", "load testing", "bug tracking", "jira", "qa engineer",
	],
	"blockchain": [
		"blockchain", "cryptocurrency", "bitcoin", "ethereum", "smart contracts",
		"solidity", "web3", "defi", "nft", "distributed ledger", "cryptography",
	],
	"game development": [
		"game development", "unity", "unreal engine", "game design", "3d modeling",
		"game programming", "c#", "c++", "gameplay", "game mechanics", "physics engine",
		"shader", "animation", "game art", "level design",
	],
	"marketing": [
		"marketing", "seo", "branding", "advertising", "content", "campaign",
		"social media", "digital marketing", "email marketing", "copywriting",
		"google analytics", "facebook ads", "instagram", "linkedin", "ppc",
	],
	"sales": [
		"sales", "crm", "negotiation", "lead generation", "b2b", "b2c",
		"salesforce", "cold calling", "account management", "business development",
		"customer relationship", "pipeline", "quota", "closing",
	],
	"graphic design": [
		"photoshop", "illustrator", "figma", "sketch", "ui", "ux", "adobe",
		"indesign", "logo design", "branding", "typography", "color theory",
		"canva", "wireframing", "prototyping", "visual design",
	],
	"healthcare": [
		"medical", "clinical", "patient care", "diagnosis", "surgery", "nurse",
		"nursing", "doctor", "physician", "healthcare", "hospital", "clinic",
		"pharmacist", "therapy", "rehabilitation", "medical records", "cpr",
		"first aid", "medication administration", "vital signs",
	],
	"finance": [
		"accounting", "finance", "budgeting", "excel", "audit", "tax",
		"financial analysis", "bookkeeping", "quickbooks", "sap", "gaap",
		"financial reporting", "investment", "banking", "treasury", "compliance",
	],
	"education": [
		"teacher", "teaching", "curriculum", "lesson planning", "training", "student",
		"classroom management", "education", "tutoring", "instructional design",
		"e-learning", "assessment", "grading", "mentoring", "pedagogy",
	],
	"engineering": [
		"cad", "mechanical", "electrical", "civil", "autocad", "solidworks",
		"engineering", "design", "simulation", "testing", "manufacturing",
		"quality control", "project management", "technical drawing", "matlab",
	],
	"law": [
		"law", "legal", "contract", "compliance", "litigation", "lawyer",
		"attorney", "paralegal", "legal research", "court", "trial",
		"negotiations", "legal writing", "regulations", "corporate law",
	],
	"hospitality": [
		"chef", "cook", "cooking", "culinary", "kitchen", "food preparation",
		"waiter", "waitress", "server", "bartender", "hospitality", "restaurant",
		"customer service", "food service", "catering", "menu planning",
		"food safety", "sanitation", "front desk", "hotel", "guest services",
		"housekeeping", "reservation", "banquet", "event planning",
	],
	"retail": [
		"retail", "cashier", "customer service", "sales associate", "merchandising",
		"inventory", "point of sale", "pos", "stock", "store", "shop",
		"customer relations", "product knowledge", "cash handling",
	],
	"construction": [
		"construction", "carpentry", "plumbing", "electrical work", "welding",
		"contractor", "building", "renovation", "hvac", "roofing", "masonry",
		"blueprint reading", "safety", "tools", "equipment operation",
	],
	"transportation": [
		"driver", "driving", "cdl", "truck", "delivery", "logistics",
		"transportation", "warehouse", "forklift", "shipping", "route planning",
		"vehicle maintenance", "safety compliance", "dispatcher",
	],
	"creative arts": [
		"artist", "photography", "videography", "editing", "video editing",
		"content creation", "creative", "drawing", "painting", "illustration",
		"animation", "3d modeling", "adobe premiere", "final cut", "after effects",
	],
	"administration": [
		"administrative", "office", "secretary", "receptionist", "scheduling",
		"data entry", "microsoft office", "word", "excel", "powerpoint",
		"filing", "organization", "correspondence", "phone", "calendar management",
	],
	"human resources": [
		"hr", "human resources", "recruiting", "recruitment", "hiring",
		"onboarding", "payroll", "employee relations", "benefits", "training",
		"performance management", "compensation", "talent acquisition",
	],
}


def _tokenize(text: str) -> List[str]:
	_ensure_nltk_resources()
	text = (text or "").lower()
	try:
		from nltk.tokenize import word_tokenize

		tokens = word_tokenize(text)
	except Exception:
		# Fallback to simple whitespace split if punkt is unavailable
		tokens = text.split()

	sw = set()
	try:
		sw = set(stopwords.words("english"))
	except Exception:
		pass

	return [t for t in tokens if t.isalpha() and t not in sw]


def _has_word_boundary_match(text: str, keyword: str) -> bool:
	"""Check if keyword exists with word boundaries to avoid false matches."""
	import re
	# Escape special regex characters in keyword
	escaped = re.escape(keyword)
	# Use word boundaries \b to match whole words/phrases
	# Allow for common separators like -, /, +
	pattern = r'\b' + escaped.replace(r'\ ', r'[\s\-/+]*') + r'\b'
	return bool(re.search(pattern, text, re.IGNORECASE))


def extract_skills(text: str) -> List[str]:
	"""Extract likely skills from resume text using precise keyword matching with word boundaries."""
	import re
	
	text_lower = (text or "").lower()
	# Normalize text: replace common separators with spaces for better matching
	normalized_text = re.sub(r'[/\-+]', ' ', text_lower)
	tokens = set(_tokenize(text))
	detected: List[str] = []
	
	for field, kws in skill_keywords.items():
		for kw in kws:
			kw_lower = kw.lower()
			
			# Use word boundary matching to avoid false positives
			# e.g., "rust" won't match "trust", "go" won't match "going"
			if _has_word_boundary_match(normalized_text, kw_lower):
				detected.append(kw)
			else:
				# Additional check for multi-word phrases where words appear separately
				parts = kw_lower.split()
				if len(parts) > 1:
					# Check if all parts exist as separate tokens
					if all(p in tokens for p in parts):
						detected.append(kw)
	
	# de-duplicate while preserving order
	seen = set()
	uniq: List[str] = []
	for k in detected:
		if k not in seen:
			seen.add(k)
			uniq.append(k)
	return uniq


def detect_job_field(text: str) -> str:
	"""Predict the most likely job field based on keyword counts with precise matching."""
	import re
	
	text_lower = (text or "").lower()
	normalized_text = re.sub(r'[/\-+]', ' ', text_lower)
	tokens = set(_tokenize(text))
	best_field = "General"
	best_score = 0
	
	for field, kws in skill_keywords.items():
		score = 0
		for kw in kws:
			kw_lower = kw.lower()
			
			# Use word boundary matching for accurate detection
			if _has_word_boundary_match(normalized_text, kw_lower):
				score += 1
			else:
				# Fallback for multi-word phrases
				parts = kw_lower.split()
				if len(parts) > 1 and all(p in tokens for p in parts):
					score += 1
		if score > best_score:
			best_score = score
			best_field = field
	return best_field