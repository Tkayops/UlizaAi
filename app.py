from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
import os
from dotenv import load_dotenv

from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex

# Load env variables
load_dotenv()

# Environment Variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

# Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Mistral Setup
llm = MistralAI(model="mistral-medium-latest", temperature=0.1)
embed_model = MistralAIEmbedding(model_name="mistral-embed")

Settings.llm = llm
Settings.embed_model = embed_model

# Load & Index Constitution
Constitution_docs = SimpleDirectoryReader(
    input_files=["./documents/Constitution.pdf"]
).load_data()

Constitution_index = VectorStoreIndex.from_documents(Constitution_docs)
Constitution_query_engine = Constitution_index.as_query_engine(similarity_top_k=5)

# FastAPI App
app = FastAPI(title="Kenyan Constitution Query API")


# Models


class QueryRequest(BaseModel):
    chat_id: str
    question: str

class CreateChatRequest(BaseModel):
    title: str


# Root


@app.get("/")
def read_root():
    return {"message": "Welcome to the Kenyan Constitution Query API"}

# 
# Create New Chat


@app.post("/chats")
async def create_chat(request: CreateChatRequest):
    response = supabase.table("chats").insert({
        "title": request.title
    }).execute()

    return response.data[0]

#
# Get All Chats


@app.get("/chats")
async def get_chats():
    response = supabase.table("chats").select("*").order("created_at", desc=True).execute()
    return response.data


# Get Chat History (Messages)


@app.get("/chats/{chat_id}")
async def get_chat_messages(chat_id: str):
    response = supabase.table("messages") \
        .select("*") \
        .eq("chat_id", chat_id) \
        .order("created_at") \
        .execute()

    return response.data


# Delete Chat (Cascade deletes messages)


@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    supabase.table("chats").delete().eq("id", chat_id).execute()
    return {"message": "Chat deleted successfully"}


# Query + Save Chat


@app.post("/query")
async def query_constitution(request: QueryRequest):
    question = request.question

    # Save user question
    supabase.table("messages").insert({
        "chat_id": request.chat_id,
        "role": "user",
        "content": question
    }).execute()

    # Get AI response
    response = Constitution_query_engine.query(question)
    answer = str(response)

    # Save AI response
    supabase.table("messages").insert({
        "chat_id": request.chat_id,
        "role": "assistant",
        "content": answer
    }).execute()

    return {
        "chat_id": request.chat_id,
        "question": question,
        "answer": answer
    }
