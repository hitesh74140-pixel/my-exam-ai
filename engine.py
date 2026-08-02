import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

DB_DIR = "./chroma_db"

def process_book(file_path: str):
    """Detects file format, cuts text into chunks, and saves to a vector database."""
    _, file_extension = os.path.splitext(file_path.lower())
    
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif file_extension == ".docx":
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(file_path)
    elif file_extension == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported format: {file_extension}")
        
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        persist_directory=DB_DIR
    )
    return vector_store

def generate_exam(topic: str, difficulty: str, mcq_cnt: int, subj_cnt: int, fib_cnt: int) -> str:
    """Retrieves context and asks Groq Cloud AI to build the exam."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
    
    # Retrieve top 5 matching document chunks
    docs = vector_store.similarity_search(topic, k=5)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    api_key = os.environ.get("GROQ_API_KEY")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, groq_api_key=api_key)
    
    system_prompt = (
        "You are an expert academic curriculum developer and examination officer.\n"
        "Generate a structured examination paper based ONLY on the provided book context below.\n"
        "Do not use outside knowledge or make up facts.\n\n"
        f"TEST SETTINGS:\n"
        f"- Target Academic Difficulty: {difficulty}\n"
        f"- Required Multiple Choice Questions (MCQ): {mcq_cnt}\n"
        f"- Required Subjective Questions: {subj_cnt}\n"
        f"- Required Fill in the Blanks (FIB): {fib_cnt}\n\n"
        "FORMATTING:\n"
        "1. Organize the paper into separate labeled sections for each question type.\n"
        "2. For MCQs, provide exactly 4 clear choices labeled A, B, C, and D.\n"
        "3. For Fill in the Blanks, use underlines (e.g., '_____') where the missing word belongs.\n\n"
        "ANSWER KEY:\n"
        "At the absolute end of your output, add a clear divider line called '--- SOLUTIONS AND ANSWER KEY ---'.\n"
        "Provide correct answers and explanations for every question generated.\n\n"
        f"Book Context:\n{context_text}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Generate the custom assessment paper covering the topic: {input}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"input": topic})
    return response.content
