import streamlit as st
import os
import requests

# Get API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("OPENROUTER_API_KEY not found in secrets!")
    st.stop()

# Page config
st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Knowledge Assistant")
st.subheader("Search organizational documents")

# Sample knowledge base
KNOWLEDGE_BASE = """
COMPANY CULTURE POLICY:
- Treat colleagues with respect
- Participate in team projects
- Attend monthly meetings
- Benefits: Health insurance, 401k, remote work

FINANCIAL POLICY:
- Budget <$5K: Manager approval
- Budget $5K-$50K: Director approval
- Budget >$50K: Board approval
- Reimbursement: Submit within 30 days with receipts

IT SECURITY:
- Password: 12+ chars, update every 90 days
- Device: Enable encryption, use VPN
- Data: Public, Internal, Confidential, Restricted

ONBOARDING:
Day 1: Welcome, office tour, intro to team
Week 1: IT setup, system training, orientation
Month 1: Compliance training, set goals

LEAVE POLICY:
- Vacation: 20 days/year (full-time), 10 days/year (part-time)
- Sick Leave: 10 days/year
- Holidays: 10 national + 2 company holidays
"""

def get_answer(question):
    """Get answer using OpenRouter"""
    try:
        prompt = f"""You are a company knowledge assistant. Answer questions based ONLY on this knowledge base:

KNOWLEDGE BASE:
{KNOWLEDGE_BASE}

Question: {question}

Answer concisely and accurately based only on the knowledge base provided."""
        
        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "HTTP-Referer": "https://streamlit.app",
            "X-Title": "Knowledge Assistant"
        }
        
        data = {
            "model": "z-ai/glm-5.3-flash",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0,
            "max_tokens": 500
        }
        
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code != 200:
            return f"Error: {response.status_code} - {response.json().get('error', {}).get('message', 'Unknown error')}"
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    except Exception as e:
        return f"Error: {str(e)}"

# UI
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Ask a Question")
    user_query = st.text_area(
        "What would you like to know?",
        placeholder="e.g., What's the budget approval process?",
        height=100,
        label_visibility="collapsed"
    )
    
    if st.button("🔍 Search", use_container_width=True, type="primary"):
        if user_query.strip():
            with st.spinner("Searching..."):
                answer = get_answer(user_query)
                st.markdown("### Answer")
                st.info(answer)
        else:
            st.warning("Please ask a question!")

with col2:
    st.markdown("### Available Topics")
    st.markdown("""
    📋 Culture Policy
    💰 Financial Guidelines
    🔒 IT Security
    👤 Onboarding Process
    🏖️ Leave Policy
    """)
    
    st.divider()
    st.markdown("### Example Questions")
    
    examples = [
        "Budget approval process?",
        "Vacation days?",
        "Password requirements?",
        "Onboarding steps?"
    ]
    
    for ex in examples:
        if st.button(ex, use_container_width=True, key=ex):
            st.session_state.query = ex

st.divider()
st.markdown("<p style='text-align: center; color: #888; font-size: 11px;'>📚 Knowledge Assistant | OpenRouter Powered</p>", unsafe_allow_html=True)
