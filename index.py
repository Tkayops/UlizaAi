import nest_asyncio

nest_asyncio.apply()

import os



from llama_index.llms.mistralai import MistralAI
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.core import Settings
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex



os.environ["MISTRAL_API_KEY"] = "0uLqHjUaxWHIBObcCbwSac4J3l0CLuas"

llm = MistralAI(model="open-mixtral-8x22b", temperature=0.1)
embed_model = MistralAIEmbedding(model_name="mistral-embed")

Settings.llm = llm
Settings.embed_model = embed_model

#Loading Data
uber_docs = SimpleDirectoryReader(input_files=["./documents/Constitution.pdf"]).load_data()

#Indexing Data
uber_index = VectorStoreIndex.from_documents(uber_docs)
uber_query_engine = uber_index.as_query_engine(similarity_top_k=5)

#Querying Data

from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import FunctionCallingAgent

query_engine_tools = [

    QueryEngineTool(
        query_engine=uber_query_engine,
        metadata=ToolMetadata(
            name="Constitution_2010",
            description="Provides information about Kenyan Constitution for year 2010",
        ),
    ),
]

agent = FunctionCallingAgent.from_tools(
    query_engine_tools,
    llm=llm,
    verbose=True,
    allow_parallel_tool_calls=False,
)

response = agent.chat("What are the circumstances in which a passport could be denied?Give all the details expalaining it to the full")






