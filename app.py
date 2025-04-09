import streamlit as st
from Hybrid_Rag_Ui_Table import hybrid_search, call_llama, format_table
import re
import requests

# --- Optional: Pull secret from Streamlit Cloud ---
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")

# Override in case running locally
if not GROQ_API_KEY:
    from Hybrid_Rag_Ui_Table import GROQ_API_KEY as fallback
    GROQ_API_KEY = fallback

# Use updated Groq API key
def call_llama_updated(prompt):
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "llama3-8b-8192",
        "messages": [
            {"role": "system", "content": "You are an SHL test recommender. Suggest suitable assessments based on user input."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=data)
    return response.json()["choices"][0]["message"]["content"]

# Optional: Pull page content if URL is in query
def extract_url_content(query):
    url_match = re.search(r'https?://\S+', query)
    if url_match:
        try:
            page = requests.get(url_match.group(0), timeout=5)
            return page.text[:3000]  # Keep it concise
        except:
            return ""
    return ""

# --- Streamlit UI ---
st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")
st.title("🧠 SHL Assessment Recommender (Hybrid RAG + LLaMA)")

query = st.text_area("✍️ Enter a Job Description or Assessment Query (or a Link to a JD)", height=200)

if st.button("🔍 Recommend Assessments") and query:
    with st.spinner("Running Hybrid Search and LLaMA recommendation..."):
        url_content = extract_url_content(query)
        if url_content:
            query += f"\n\n[Extracted from link]:\n{url_content}"

        top_meta, top_docs = hybrid_search(query)
        table = format_table(top_meta)

        st.subheader("🔝 Top Matching Assessments")
        st.text(table)

        context = "\n\n".join(top_docs)
        prompt = f"Here is the context of available SHL tests:\n\n{context}\n\nBased on this, suggest the most relevant assessments for the following job description or query:\n{query}"
        response = call_llama_updated(prompt)

        st.subheader("💡 LLaMA Recommendation")
        st.write(response)
