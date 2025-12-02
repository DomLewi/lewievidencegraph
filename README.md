LewiEvidenceGraph 🧬

An Autonomous Medical Research Agent for Evidence Synthesis.

EvidenceGraph Pro is a specialized Retrieval-Augmented Generation (RAG) system designed to automate the systematic review of medical literature. Unlike standard LLM interactions which rely on parametric memory (leading to hallucinations), this system uses a strict Supervisor-Worker Architecture anchored to live data from the National Library of Medicine (PubMed).

🚀 Key Features

Deterministic Retrieval: Uses Bio.Entrez to query the NCBI PubMed API directly.

Safety-First Architecture: Built on LangGraph, enforcing a multi-node workflow where "Auditor" agents must verify citations against source texts before approval.

Scored Curation: An LLM-based curator node grades papers on a 0-100 relevancy scale before they enter the synthesis context.

Audit Trail: Generates a downloadable JSON log of the entire reasoning process, including rejected drafts and valid citations.

🛠️ Tech Stack

Orchestration: Python, LangGraph

Bioinformatics: Biopython (Entrez, Medline)

Models: OpenAI GPT-4o via LangChain

Interface: Streamlit

🏗️ Architecture

The system operates as a Directed Cyclic Graph (DAG):

Investigator Node: Converts natural language into MeSH-optimized search vectors.

Harvester Node: Parallelized retrieval from PubMed (respecting NCBI rate limits).

Curator Node: Filters low-quality or irrelevant abstracts.

Analyst Node: Synthesizes consensus, contradictions, and limitations.

Auditor Node: The "Circuit Breaker" that rejects hallucinations and forces retries.

📦 Installation

Clone the repository:

git clone [https://github.com/YOUR_USERNAME/EvidenceGraph.git](https://github.com/YOUR_USERNAME/EvidenceGraph.git)


Install dependencies:

pip install -r requirements.txt


Run the application:

python -m streamlit run evidencegraphwebapp.py


🛡️ Disclaimer

This tool is for research acceleration purposes only. Always verify medical information with a qualified professional.
