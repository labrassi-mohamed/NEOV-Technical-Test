import os
import getpass
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = "your_api_key_here"
    
# Function to ingest documents (PDF or text files)
def ingest_documents(docs_path):
    file_ext = os.path.splitext(docs_path)[1].lower()
    
    # Select the appropriate loader based on file extension
    if file_ext == '.pdf':
        loader = PyPDFLoader(docs_path)
    elif file_ext == '.txt':
        loader = TextLoader(docs_path)
    else:
        raise ValueError(f"Unsupported file format: {file_ext}")
    
    # Load the document
    documents = loader.load()
    
    # Split the document into smaller chunks for better processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    documents = text_splitter.split_documents(documents)
    
    return documents

# Function to embed the documents using Google Generative AI and store them in ChromaDB
def embed_documents(documents): 
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    return Chroma.from_documents(documents=documents, embedding=embeddings)

# Ingest and embed the documents from the specified file
vectorstore = embed_documents(ingest_documents("your_document_path"))

# Create a retriever to perform similarity-based document search
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

# Initialize the language model (LLM) using Google's Gemini-1.5-Pro
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)

# Define the system prompt to guide the LLM's behavior
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

# Create a chat prompt template that includes the system prompt and user input
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Create a question-answering chain using the LLM and prompt
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# Create a full RAG (Retrieval-Augmented Generation) pipeline
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Interactive loop for user queries
while True:
    input_message = input("Ask a question: ")
    if input_message.lower() == "exit":
        break
    
    response = rag_chain.invoke({"input": input_message})
    print(response["answer"])
