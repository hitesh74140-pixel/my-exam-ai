import streamlit as st
import os
import shutil
import requests
from engine import process_book, process_text_input, generate_exam

st.set_page_config(page_title="Global AI Exam Generator", layout="wide", page_icon="📝")

st.title("📝 Global AI Exam Generator")
st.write("Upload a document OR paste text directly, configure settings, choose your AI model, and generate complete tests instantly.")

st.warning(
    "⚠️ **Important Notice:** Please upload or paste **specific topics or chapters** instead of full textbooks. "
    "Uploading large documents at once can exceed memory limits."
)

UPLOAD_DIR = "./uploaded_materials"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Helper function to dynamically fetch active models from Groq API
@st.cache_data(ttl=3600)  # Caches the active model list for 1 hour
def get_active_groq_models():
    api_key = os.environ.get("GROQ_API_KEY")
    default_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    
    if not api_key:
        return default_models
        
    try:
        url = "https://api.groq.com/openai/v1/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            # Extract active text model IDs, filtering out whisper/audio or guard models
            fetched_models = [
                model["id"] for model in data.get("data", [])
                if "whisper" not in model["id"].lower() and "guard" not in model["id"].lower()
            ]
            if fetched_models:
                # Prioritize top production models at the top of the dropdown
                priority_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
                sorted_models = [m for m in priority_models if m in fetched_models] + \
                                [m for m in fetched_models if m not in priority_models]
                return sorted_models
    except Exception:
        pass
        
    return default_models

# Sidebar configuration panel
with st.sidebar:
    st.header("📂 Data Ingestion")
    
    tab1, tab2 = st.tabs(["📄 Upload File", "✏️ Paste Text"])
    
    with tab1:
        uploaded_file = st.file_uploader("Upload Material (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
        if uploaded_file is not None:
            file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
            if not os.path.exists(file_path):
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                with st.spinner("Processing document into cloud database..."):
                    try:
                        process_book(file_path)
                        st.success("Document uploaded and indexed successfully!")
                    except Exception as e:
                        st.error(f"Processing error: {str(e)}")
            else:
                st.info("Document loaded and ready.")
                
    with tab2:
        raw_text = st.text_area("Paste your material/text here:", height=250)
        if st.button("Process Pasted Text", use_container_width=True):
            if raw_text.strip():
                with st.spinner("Processing text into cloud database..."):
                    try:
                        process_text_input(raw_text)
                        st.success("Text processed and indexed successfully!")
                    except Exception as e:
                        st.error(f"Processing error: {str(e)}")
            else:
                st.warning("Please paste some text before clicking process.")
            
    st.header("🗑️ Reset Application")
    if st.button("Wipe Current Database", use_container_width=True):
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
        if os.path.exists(UPLOAD_DIR):
            shutil.rmtree(UPLOAD_DIR)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
        st.success("App data cleared!")
        st.rerun()

st.subheader("🛠️ Step-by-Step Test Specification Configuration")
col1, col2 = st.columns(2)

# Dynamically fetch active Groq models
active_models = get_active_groq_models()

with col1:
    exam_topic = st.text_input("Target Topic / Chapter Name", placeholder="e.g., Photosynthesis, Chapter 2")
    difficulty_level = st.selectbox("Select Academic Rigor Level", ["Easy", "Medium", "Hard"])
    
    # Select from live fetched Groq models
    selected_model = st.selectbox(
        "🤖 Select AI Model (Live Active Models)",
        options=active_models
    )

with col2:
    num_mcqs = st.slider("Multiple Choice Questions (MCQs)", min_value=0, max_value=25, value=5)
    num_subj = st.slider("Subjective / Essay Questions", min_value=0, max_value=15, value=3)
    num_fib = st.slider("Fill in the Blanks (FIB)", min_value=0, max_value=20, value=5)

st.markdown("---")

if st.button("Generate Paper", type="primary", use_container_width=True):
    if not os.path.exists("./chroma_db"):
        st.error("Operation Denied: Please upload a file or paste text in the sidebar panel first.")
    elif not exam_topic.strip():
        st.error("Operation Denied: Please enter a target topic to query.")
    elif num_mcqs == 0 and num_subj == 0 and num_fib == 0:
        st.error("Operation Denied: Choose at least 1 question type variation to build.")
    else:
        with st.spinner("AI cloud is analyzing content and writing questions..."):
            try:
                generated_paper = generate_exam(
                    topic=exam_topic,
                    difficulty=difficulty_level,
                    mcq_cnt=num_mcqs,
                    subj_cnt=num_subj,
                    fib_cnt=num_fib,
                    model_name=selected_model
                )
                
                st.subheader("📄 Generated Assessment")
                st.markdown(generated_paper)
                
                st.download_button(
                    label="📥 Save Examination Document (.txt)",
                    data=generated_paper,
                    file_name=f"Assessment_{exam_topic.replace(' ', '_')}.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Generation failure: {str(e)}")
