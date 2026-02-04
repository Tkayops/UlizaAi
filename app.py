from mistralai import Mistral
from fastapi import FastAPI
from pydantic import BaseModel


import os

from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent




os.environ["MISTRAL_API_KEY"] = "0uLqHjUaxWHIBObcCbwSac4J3l0CLuas"

llm = MistralAI(model="mistral-medium-latest", temperature=0.1)
embed_model = MistralAIEmbedding(model_name="mistral-embed")

Settings.llm = llm
Settings.embed_model = embed_model

#Loading Data
Constitution_docs = SimpleDirectoryReader(input_files=["./documents/Constitution.pdf"]).load_data()

#Indexing Data
Constitution_index = VectorStoreIndex.from_documents(Constitution_docs)
Constitution_query_engine = Constitution_index.as_query_engine(similarity_top_k=5)

#Querying Data
#print('What would you love to know about the Kenyan Constitution :')
#x = input()


query_engine_tools = [

    QueryEngineTool(
        query_engine=Constitution_query_engine,
        metadata=ToolMetadata(
            name="Constitution_2010",
            description="Provides information about Kenyan Constitution for year 2010",
            
        ),
        

    ),
]

agent = ReActAgent(
    tools=query_engine_tools,
    llm=llm,
    verbose=True,
)


# FastAPI app
app = FastAPI(title="Kenyan Constitution Query API")

class QueryRequest(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Kenyan Constitution Query API"}

@app.post("/query")
async def query_constitution(request: QueryRequest):
    """
    Endpoint to query the Kenyan Constitution.
    """
    question = request.question
    # Use the query engine to get a response
    response = Constitution_query_engine.query(question)
    return {"question": question, "answer": str(response)}









