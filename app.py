import uvicorn
from fastapi import FastAPI, Request, HTTPException
from src.graphs.graph_builder import GraphBuilder
from src.llms.groqllm import GroqLLM

import os
from dotenv import load_dotenv
load_dotenv()

app=FastAPI()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

## API's
@app.post("/blogs")
async def create_blogs(request: Request):
    data=await request.json()
    topic=data.get("topic", "")
    if not topic:
        raise HTTPException(status_code=400, detail="Missing 'topic' in request body")

    ## get the llm object
    groqllm=GroqLLM()
    llm=groqllm.get_llm()

    ## get the graph
    graph_builder=GraphBuilder(llm).setup_graph(usecase="topic")
    state=await graph_builder.ainvoke({"topic":topic})

    return {"data": state}

if __name__=="__main__":
    uvicorn.run("app:app",host="0.0.0.0", port=8000,reload=True)