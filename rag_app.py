import streamlit as st
import os
from dotenv import load_dotenv
import requests
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.documents import Document
import json

# Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found. Please set it in secrets.")
    st.stop()

# Page config
st.set_page_config(
    page_title="Knowledge Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    [data-testid="stChatMessageContainer"] { background-color: #f8f9fa; }
    </style>
""", unsafe_allow_html=True)

# Initialize embeddings once
@st.cache_resource
def get_embeddings():
    """Initialize HuggingFace embeddings"""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def get_llm():
    """Initialize OpenAI LLM"""
    return ChatOpenAI(
        model="gpt-3.5-turbo",
        api_key=OPENAI_API_KEY,
        temperature=0
    )

@st.cache_resource
def get_vector_store():
    """Initialize or load Chroma vector store with sample data"""
    embeddings = get_embeddings()
    
    # Sample organizational documents
    sample_docs = [
        Document(
            page_content="""Company Culture Policy
            
Our company is committed to fostering an inclusive and collaborative work environment. 
All employees are expected to:
- Treat colleagues with respect and dignity
- Participate actively in team projects
- Attend mandatory monthly culture meetings
- Report any issues through proper channels

Benefits: Health insurance, 401k matching, remote work flexibility""",
            metadata={"source": "culture_policy.txt", "topic": "HR"}
        ),
        Document(
            page_content="""Financial Policy Guidelines

Project Budget Approval:
- Under $5,000: Department Manager approval
- $5,000 - $50,000: Director approval  
- Over $50,000: Executive Board approval

Expense Reimbursement:
- Submit within 30 days with receipts
- Personal expenses not covered
- Travel requires pre-approval""",
            metadata={"source": "financial_policy.txt", "topic": "Finance"}
        ),
        Document(
            page_content="""IT Security Guidelines

Password Requirements:
- Minimum 12 characters
- Update every 90 days
- No sharing between employees

Device Security:
- Enable disk encryption
- Use VPN for remote access
- Lock device when away
- Report lost devices immediately

Data Classification:
- Public, Internal, Confidential, Restricted
- Follow access control policies
- Secure deletion required""",
            metadata={"source": "it_security.txt", "topic": "IT"}
        ),
        Document(
            page_content="""Employee Onboarding Checklist

Day 1:
- Welcome meeting with HR
- Office tour and access setup
- Introduction to team members
- Review company policies

Week 1:
- IT setup (laptop, email, accounts)
- System access training
- Department orientation
- Assign mentor/buddy

Month 1:
- Complete compliance training
- Meet department head
- Set 30-day goals
- Attend culture meeting""",
            metadata={"source": "onboarding.txt", "topic": "HR"}
        ),
    ]
    
    # Create or load vector store
    vector_store = Chroma.from_documents(
        documents=sample_docs,
        embedding=embeddings,
        persist_directory=".chroma_db"
    )
    
    return vector_store

def search_knowledge_base(query: str) -> str:
    """Search knowledge base and get answer"""
    try:
        vector_store = get_vector_store()
        llm = get_llm()
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        
        # Get answer
        result = qa_chain({"query": query})
        return result["result"]
        
    except Exception as e:
        return f"Error searching knowledge base: {str(e)}"

# UI
st.title("📚 Knowledge Assistant")
st.subheader("Search organizational documents and policies")

col1, col2 = st.columns([2, 1], gap="large")

with col1:
    st.markdown("### Search Knowledge Base")
    user_query = st.text_area(
        "What would you like to know?",
        placeholder="e.g., What's the budget approval process? How do I get reimbursed?",
        height=100,
        label_visibility="collapsed"
    )
    
    if st.button("🔍 Search", use_container_width=True, type="primary"):
        if user_query.strip():
            with st.spinner("🔎 Searching documents..."):
                try:
                    answer = search_knowledge_base(user_query)
                    st.markdown("### Answer")
                    st.info(answer)
                except Exception as e:
                    st.error(f"Error: {str(e)}")
        else:
            st.warning("Please enter a question!")

with col2:
    st.markdown("### 📖 Quick Help")
    st.markdown("""
    **Available Topics:**
    - HR Policies
    - Financial Guidelines
    - IT Security
    - Onboarding
    
    **Example Queries:**
    - Budget approval limits?
    - Password requirements?
    - Reimbursement process?
    - Onboarding steps?
    """)
    
    st.divider()
    
    st.markdown("### Popular Questions")
    questions = [
        "What are the budget approval limits?",
        "How do I get reimbursed for expenses?",
        "What's the password policy?",
        "What's the onboarding process?"
    ]
    
    for q in questions:
        if st.button(q, use_container_width=True, key=q):
            st.session_state.user_query = q

st.divider()
st.markdown("""
<div style='text-align: center; color: #888; font-size: 12px;'>
    <p>📚 Knowledge Assistant | Powered by RAG + LLMs</p>
</div>
""", unsafe_allow_html=True)
