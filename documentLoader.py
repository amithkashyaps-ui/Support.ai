from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os

app = FastAPI()

# Allow CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Utility Functions ===
def load_documents(directory_path: str):
    loader = DirectoryLoader(directory_path, glob="**/*.md")
    return loader.load()

def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_documents(docs)

def create_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

def create_vectorstore(splits, embedding_model, persist_directory):
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    # vectorstore.persist()
    return vectorstore

def create_prompt_template():
    template = """You are an expert assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Keep the answer concise.

Context:
{context}

Question: {question}

Answer:"""
    return PromptTemplate(template=template, input_variables=["context", "question"])

def initialize_qa_chain(vectorstore, prompt):
    llm = ChatGroq(
        model="llama3-70b-8192",
        temperature=0.2,
        groq_api_key=os.environ["GROQ_API_KEY"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )

# === Initialize Components ===
directory_path = "../talkToAI/data"
persist_directory = "chroma_db"
docs = load_documents(directory_path)
splits = split_documents(docs)
embedding_model = create_embedding_model()
vectorstore = create_vectorstore(splits, embedding_model, persist_directory)
prompt = create_prompt_template()
qa_chain = initialize_qa_chain(vectorstore, prompt)

# === API Endpoint ===
class QuestionRequest(BaseModel):
    question: str

@app.post("/api/ask")
async def ask_question(req: QuestionRequest):
    try:
        result = qa_chain({"query": req.question})
        return {
            "result": result["result"],
            # "sources": [
            #     {
            #         "metadata": doc.metadata,
            #         "page_content": doc.page_content
            #     }
            #     for doc in result["source_documents"]
            # ]
        }
    except Exception as e:
        return {"error": str(e)}
