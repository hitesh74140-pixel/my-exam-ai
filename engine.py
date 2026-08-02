import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

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
    return index_documents(docs)

def process_text_input(text_content: str):
    """Processes pasted raw text into chunks and saves to vector database."""
    docs = [Document(page_content=text_content, metadata={"source": "user_pasted_text"})]
    return index_documents(docs)

def index_documents(docs):
    """Splits documents and indexes into Chroma vector store."""
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
    
    # Retrieve top matching document chunks
    docs = vector_store.similarity_search(topic, k=5)
    context_text = "\n\n".join([doc.page_content for doc in docs])
    
    api_key = os.environ.get("GROQ_API_KEY")
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3, groq_api_key=api_key)
    
    system_prompt = (
        "You are an expert academic curriculum developer and examination officer.\n"
        "Generate a structured examination paper based ONLY on the provided context below.\n"
        "Do not use outside knowledge or make up facts.\n\n"
        f"TEST SETTINGS:\n"
        f"- Target Academic Difficulty: {difficulty}\n"
        f"- Required Multiple Choice Questions (MCQ): {mcq_cnt}\n"
        f"- Required Subjective Questions: {subj_cnt}\n"
        f"- Required Fill in the Blanks (FIB): {fib_cnt}\n\n"
        "FORMATTING & ANSWER KEY PLACEMENT:\n"
        "1. Organize the paper into separate sections for each question type.\n"
        "2. CRITICAL RULE: For EVERY single question generated, provide the correct answer and brief explanation DIRECTLY underneath that question.\n"
        "   Example Format:\n"
        "   Q1: What is ...?\n"
        "   A) Choice 1  B) Choice 2  C) Choice 3  D) Choice 4\n"
        "   --> Answer: B) Choice 2\n"
        "   --> Explanation: ...\n\n"
        "3. For Fill in the Blanks, use underlines (e.g., '_____') and output the answer directly underneath.\n\n"
        f"Context Material:\n{context_text}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "Generate the custom assessment paper covering the topic: {input}")
    ])
    
    chain = prompt | llm
    response = chain.invoke({"input": topic})
    return response.content
