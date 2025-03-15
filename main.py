import os
import shutil
import uuid
from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory import ConversationBufferMemory
# from openai.error import AuthenticationError
from openai import OpenAIError

app = FastAPI()

# Store uploaded PDF path
pdf_path = "uploaded_document.pdf"
vector_store = None  # Global variable for ChromaDB
chat_sessions = {}  # Dictionary to store chat history per UUID


def process_pdf(api_key: str):
    """
    Loads the uploaded PDF, extracts text, generates embeddings, and stores in ChromaDB.
    """
    global vector_store

    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=400, detail="No PDF uploaded.")

    try:
        # Set OpenAI API key dynamically
        os.environ["OPENAI_API_KEY"] = api_key

        # Load and split PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # Generate embeddings and store in ChromaDB
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(texts, embeddings, persist_directory="./chroma_db")

    except OpenAIError:
        raise HTTPException(status_code=401, detail="Invalid or expired OpenAI API key.")


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...), api_key: str = Query(...)):
    """
    Endpoint to upload a PDF document and process it.
    """
    try:
        # Save the uploaded file
        with open(pdf_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process the PDF
        process_pdf(api_key)

        # Generate unique UUID for chat session
        session_id = str(uuid.uuid4())

        # Create buffer memory for this session
        chat_sessions[session_id] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        return {"message": "PDF uploaded and processed successfully.", "session_id": session_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/ask/")
async def ask_question(query: str, session_id: str, api_key: str = Query(...)):
    """
    Endpoint to ask a question based on the uploaded PDF with memory.
    """
    global vector_store

    if vector_store is None:
        raise HTTPException(status_code=400, detail="No PDF has been uploaded or processed.")

    if session_id not in chat_sessions:
        raise HTTPException(status_code=404, detail="Invalid session_id. Please upload a new PDF to start a session.")

    try:
        # Set OpenAI API key dynamically
        os.environ["OPENAI_API_KEY"] = api_key

        # Get memory for this session
        memory = chat_sessions[session_id]

        # Create Conversational Retrieval Chain with memory
        retriever = vector_store.as_retriever()
        llm = ChatOpenAI(model_name="gpt-4-turbo")  # Change to "gpt-3.5-turbo" if needed
        qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

        # Query the model
        response = qa_chain.run(query)

        return {"session_id": session_id, "query": query, "response": response}

    except OpenAIError:
        raise HTTPException(status_code=401, detail="Invalid or expired OpenAI API key.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
