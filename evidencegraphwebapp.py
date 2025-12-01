import streamlit as st
import os
import json
import re
import time
from typing import TypedDict, List, Dict, Any

# --- PAGE CONFIG ---
st.set_page_config(page_title="EvidenceGraph Pro", page_icon="🧬", layout="wide")

# --- CSS STYLING ---
st.markdown("""
<style>
    .report-box { border: 1px solid #ddd; padding: 20px; border-radius: 10px; background-color: #f9f9f9; }
    .status-pass { color: #28a745; font-weight: bold; }
    .status-fail { color: #dc3545; font-weight: bold; }
    .metric-card { background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin: 5px 0; }
</style>
""", unsafe_allow_html=True)

# --- DEPENDENCY CHECK ---
with st.spinner("Initializing AI Core & Entrez..."):
    try:
        from Bio import Entrez, Medline
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import SystemMessage, HumanMessage
        from langgraph.graph import StateGraph, END
    except ImportError as e:
        st.error(f"❌ Missing Dependency: {e}")
        st.info("Run this in terminal: `pip install langchain-openai langgraph biopython`")
        st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    
    user_api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    email_input = st.text_input("Email (Required for PubMed)", placeholder="your@email.com")
    
    # P0 FIX: Validate Email immediately
    if email_input and not re.match(r"[^@]+@[^@]+\.[^@]+", email_input):
        st.error("Invalid email format. NCBI requires a valid email.")
    
    st.divider()
    st.subheader("Research Parameters")
    
    strict_mode = st.checkbox("Strict Evidence Mode", value=True, help="Limits search to RCTs and Systematic Reviews.")
    search_width = st.slider("Search Breadth (Queries)", 3, 10, 5)
    fetch_limit = st.slider("Papers per Query", 5, 20, 10)
    keep_best_param = st.slider("Curator Selection", 3, 10, 5)
    
    st.info("Architecture: Bio.Entrez Retrieval -> LLM Curation -> RAG Synthesis")

# --- CORE LOGIC ---
if user_api_key:
    os.environ["OPENAI_API_KEY"] = user_api_key
    llm = ChatOpenAI(model="gpt-4o", temperature=0)

if email_input and re.match(r"[^@]+@[^@]+\.[^@]+", email_input):
    Entrez.email = email_input

# --- STATE DEFINITION ---
class ResearchState(TypedDict):
    topic: str
    config: Dict[str, Any]
    search_queries: List[str]
    raw_docs: List[Dict]
    selected_docs: List[Dict]
    synthesis_draft: str
    verified_report: Dict
    critique: str
    attempts: int

# --- NODE 1: INVESTIGATOR ---
def investigator_node(state):
    config = state['config']
    with st.chat_message("assistant"):
        st.write(f"**[INVESTIGATOR]:** Generating search strategy for: *{state['topic']}*...")
    
    filter_instruction = ""
    if config['strict_mode']:
        filter_instruction = " (Note: System will automatically append RCT/Review filters)."

    prompt = [
        SystemMessage(content="You are a Medical Librarian. Generate specific search queries. Use MeSH terms."),
        HumanMessage(content=f"Topic: {state['topic']}\nQuantity: {config['search_width']}\n{filter_instruction}\nOutput ONLY queries, one per line.")
    ]
    response = llm.invoke(prompt).content
    base_queries = [re.sub(r'^\d+\.\s*', '', q.strip()) for q in response.split('\n') if q.strip()]
    
    final_queries = []
    for q in base_queries:
        if config['strict_mode']:
            final_queries.append(f"({q}) AND (Randomized Controlled Trial[Publication Type] OR Review[Publication Type])")
        else:
            final_queries.append(q)
    
    with st.expander(f"View Search Strategy ({len(final_queries)} vectors)"):
        for q in final_queries: st.code(q, language="text")
        
    return {"search_queries": final_queries, "raw_docs": []}

# --- NODE 2: HARVESTER (Fixed Parsing & Rate Limiting) ---
def harvester_node(state):
    config = state['config']
    placeholder = st.empty()
    
    if not Entrez.email:
        st.error("Email required for PubMed access.")
        return {"status": "FAIL"}

    with placeholder.chat_message("assistant"):
        st.write(f"**[HARVESTER]:** Querying NCBI Entrez API...")
        
    all_docs = []
    seen_uids = set()
    
    progress_bar = st.progress(0)
    total_q = len(state['search_queries'])
    
    for idx, query in enumerate(state['search_queries']):
        try:
            # P0 FIX: Rate Limiting
            time.sleep(0.35) 
            
            # 1. Search
            handle = Entrez.esearch(db="pubmed", term=query, retmax=config['fetch_limit'])
            record = Entrez.read(handle)
            handle.close()
            id_list = record["IdList"]
            
            if not id_list:
                continue

            # P0 FIX: Rate Limiting before fetch
            time.sleep(0.35)

            # 2. Fetch with Bio.Medline (P1 FIX: Robust Parsing)
            handle = Entrez.efetch(db="pubmed", id=id_list, rettype="medline", retmode="text")
            records = Medline.parse(handle)
            
            for rec in records:
                uid = rec.get("PMID", "")
                if not uid or uid in seen_uids: continue
                
                title = rec.get("TI", "Unknown Title")
                abstract = rec.get("AB", "No Abstract Available.")
                
                # P1 FIX: Token Budgeting (Truncate Abstract)
                if len(abstract) > 4000:
                    abstract = abstract[:4000] + "... [TRUNCATED]"
                
                all_docs.append({
                    "uid": uid,
                    "title": title,
                    "content": abstract,
                    "source": "PubMed"
                })
                seen_uids.add(uid)
            
            handle.close()

        except Exception as e:
            pass # Keep moving on individual query failure
        
        progress_bar.progress((idx + 1) / total_q)
            
    progress_bar.empty()
    placeholder.empty()
    
    with st.chat_message("assistant"):
        st.write(f"**[HARVESTER]:** Retrieved {len(all_docs)} unique records via Entrez.")
        
    if not all_docs:
        st.error("Critical Failure: No documents retrieved. Try relaxing strict mode.")
        return {"raw_docs": [], "status": "FAIL"}

    return {"raw_docs": all_docs}

# --- NODE 3: CURATOR (Visibility Fix) ---
def curator_node(state):
    config = state['config']
    if not state['raw_docs']: return {"selected_docs": []}
    
    with st.chat_message("assistant"):
        st.write("**[CURATOR]:** Scoring evidence against rubric...")
        
    # P2 FIX: Dynamic Candidate Cap (Allow more context if model supports it)
    # We increase cap to 50 for gpt-4o, but keeping a limit prevents OOM.
    candidates = state['raw_docs'][:50]
    
    docs_text = "\n".join([f"ID:{d['uid']} | Title: {d['title']}" for d in candidates])
    
    prompt = [
        SystemMessage(content=f"""You are a Senior Editor. 
        Evaluate these papers for relevance to: '{state['topic']}'.
        Assign a score (0-100).
        Return a JSON list: [{{"uid": "123", "score": 90, "reason": "Reason"}}].
        Select Top {config['keep_best']}."""),
        HumanMessage(content=f"Candidates:\n{docs_text}")
    ]
    
    response = llm.invoke(prompt).content
    
    selected_docs = []
    try:
        json_str = re.search(r'\[.*\]', response, re.DOTALL).group()
        selections = json.loads(json_str)
        
        with st.expander("Evidence Scorecard"):
            for sel in selections:
                doc = next((d for d in candidates if d['uid'] == sel['uid']), None)
                if doc:
                    score = sel.get('score', 0)
                    color = "green" if score > 80 else "orange" if score > 50 else "red"
                    st.markdown(f":{color}[**{score}/100**] {doc['title']}")
                    st.caption(f"Reason: {sel.get('reason', 'N/A')}")
                    
                    doc['score'] = score
                    doc['reason'] = sel.get('reason')
                    selected_docs.append(doc)
    except Exception as e:
        # P1 FIX: Explicit Warning on Fallback
        st.warning(f"⚠️ Curator Logic Failed ({str(e)}). Falling back to simple ranking.")
        selected_docs = candidates[:config['keep_best']]
        
    return {"selected_docs": selected_docs, "attempts": 0}

# --- NODE 4: ANALYST ---
def analyst_node(state):
    attempt = state['attempts'] + 1
    with st.chat_message("assistant"):
        st.write(f"**[ANALYST]:** Synthesizing Evidence (Attempt {attempt})...")
    
    context = ""
    for d in state['selected_docs']:
        context += f"PAPER [PMID:{d['uid']}]: {d['title']}\nABSTRACT: {d['content']}\n\n"
    
    feedback = f"FIX ERROR: {state['critique']}" if state.get("critique") else ""

    prompt = [
        SystemMessage(content="""Synthesize a medical report. 
        Sections: Consensus, Contradictions, Limitations.
        RULES: Cite every claim with [PMID:XXXX]. No outside info."""),
        HumanMessage(content=f"Topic: {state['topic']}\n\nEVIDENCE:\n{context}\n\n{feedback}")
    ]
    
    response = llm.invoke(prompt).content
    return {"synthesis_draft": response, "attempts": attempt}

# --- NODE 5: AUDITOR ---
def auditor_node(state):
    draft = state['synthesis_draft']
    valid_pmids = [d['uid'] for d in state['selected_docs']]
    
    with st.chat_message("assistant"):
        st.write("**[AUDITOR]:** Verifying citations...")
    
    prompt = [
        SystemMessage(content=f"""You are a Citation Auditor. Allowed PMIDs: {valid_pmids}.
        Verify: 1. Are all PMIDs in the Allowed list? 2. Do claims match abstracts?
        Return 'APPROVED' or 'REJECTED: [Reason]'."""),
        HumanMessage(content=f"Draft:\n{draft}")
    ]
    
    review = llm.invoke(prompt).content
    
    if "APPROVED" in review.upper():
        st.toast("Audit Passed", icon="✅")
        report = {"topic": state['topic'], "synthesis": draft, "references": state['selected_docs']}
        return {"verified_report": report, "status": "COMPLETE"}
    else:
        st.toast(f"Audit Failed", icon="❌")
        return {"critique": review, "status": "RETRY"}

# --- GRAPH BUILDER ---
def build_graph():
    workflow = StateGraph(ResearchState)
    workflow.add_node("investigator", investigator_node)
    workflow.add_node("harvester", harvester_node)
    workflow.add_node("curator", curator_node)
    workflow.add_node("analyst", analyst_node)
    workflow.add_node("auditor", auditor_node)

    workflow.set_entry_point("investigator")
    workflow.add_edge("investigator", "harvester")
    workflow.add_edge("harvester", "curator")
    workflow.add_edge("curator", "analyst")
    workflow.add_edge("analyst", "auditor")
    
    def router(state):
        if state['status'] == "COMPLETE": return END
        if state['attempts'] > 2: return END
        return "analyst"

    workflow.add_conditional_edges("auditor", router, {"analyst": "analyst", END: END})
    return workflow.compile()

# --- MAIN UI LAYOUT ---
st.title("🧬 EvidenceGraph Pro")
st.caption("Bio.Entrez Edition | Medline Parsing | Compliance Hardened")

topic_input = st.text_input("Enter Medical Topic:", placeholder="e.g., Creatine supplementation efficacy in Traumatic Brain Injury")

if st.button("🚀 Start Deep Research"):
    if not user_api_key or not email_input:
        st.error("API Key and Email are required.")
        st.stop()
        
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email_input):
        st.error("Invalid Email Format.")
        st.stop()

    app = build_graph()
    
    # PACK CONFIG INTO STATE
    config_data = {
        "strict_mode": strict_mode,
        "search_width": search_width,
        "fetch_limit": fetch_limit,
        "keep_best": keep_best_param
    }
    
    initial_state = {
        "topic": topic_input, 
        "config": config_data,
        "search_queries": [], "raw_docs": [], 
        "selected_docs": [], "synthesis_draft": "", "verified_report": {}, 
        "critique": "", "attempts": 0
    }
    
    # Run the graph
    final_state = None
    with st.status("Running Evidence Architecture...", expanded=True):
        for output in app.stream(initial_state):
            for key, value in output.items():
                final_state = value
    
    # Display Result
    st.divider()
    if final_state and final_state.get('verified_report'):
        st.success("✅ Verification Complete")
        
        tab1, tab2 = st.tabs(["Synthesis", "Evidence Scorecard"])
        
        with tab1:
            st.markdown(final_state['verified_report']['synthesis'])
            
        with tab2:
            st.dataframe(final_state['verified_report']['references'])
        
        # Download Button
        json_str = json.dumps(final_state['verified_report'], indent=4)
        st.download_button(
            label="Download Audit Trail (JSON)",
            data=json_str,
            file_name="evidence_audit.json",
            mime="application/json"
        )
    else:
        st.error("❌ Research Failed: Unable to verify consensus.")