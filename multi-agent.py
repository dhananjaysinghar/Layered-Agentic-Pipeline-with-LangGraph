import chainlit as cl
from typing import TypedDict, Optional
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_core.runnables import RunnableLambda

# === LLMs ===
llm_base = OllamaLLM(model="mistral")
llm_streaming = OllamaLLM(model="mistral", streaming=True)

# === Agentic State ===
class AgentState(TypedDict):
    question: str
    rephrased: Optional[str]
    retrieved: Optional[str]
    answer: Optional[str]
    summary: Optional[str]

# === Tooling Logic ===

@tool
def search_confluence(query: str) -> str:
    # Simulated integration
    return f"[Confluence] Found internal doc for: {query}"

@tool
def search_bitbucket(query: str) -> str:
    return f"[Bitbucket] Found relevant repo and file snippet for: {query}"

@tool
def query_postgres(query: str) -> str:
    return f"[PostgreSQL] Found matching records for: {query}"

@tool
def query_graphql(query: str) -> str:
    return f"[GraphQL] Received schema-matching results for: {query}"

@tool
def summarize_text(text: str) -> str:
    return f"📝 Summary: {text[:150]}..."

# All tools
tools = [
    search_confluence,
    search_bitbucket,
    query_postgres,
    query_graphql
]

# === LangGraph Nodes ===

def rephrase(state: AgentState) -> AgentState:
    prompt = f"Rephrase this question to be more specific and clear:\n\n{state['question']}"
    result = llm_base.invoke(prompt)
    return {**state, "rephrased": result}

def retrieve(state: AgentState) -> AgentState:
    agent = initialize_agent(tools, llm_base, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
    result = agent.run(state["rephrased"])
    return {**state, "retrieved": result}

def generate_answer(state: AgentState) -> AgentState:
    prompt = f"""Using this info:\n{state['retrieved']}\n\nAnswer the question: {state['rephrased']}"""
    result = llm_base.invoke(prompt)
    return {**state, "answer": result}

def summarize(state: AgentState) -> AgentState:
    summary = summarize_text.invoke(state["answer"])
    return {**state, "summary": summary}

# === LangGraph Workflow ===

builder = StateGraph(AgentState)
builder.add_node("Rephrase", RunnableLambda(rephrase))
builder.add_node("Retrieve", RunnableLambda(retrieve))
builder.add_node("Answer", RunnableLambda(generate_answer))
builder.add_node("Summarize", RunnableLambda(summarize))

builder.set_entry_point("Rephrase")
builder.add_edge("Rephrase", "Retrieve")
builder.add_edge("Retrieve", "Answer")
builder.add_edge("Answer", "Summarize")
builder.add_edge("Summarize", END)

graph = builder.compile()

# === Chainlit Streaming ===

async def stream_response(title, content):
    msg = cl.Message(content="", author=title)
    await msg.send()

    final = ""
    async for chunk in llm_streaming.astream(content):
        final += chunk
        await msg.stream_token(chunk)
    msg.content = final.strip()
    await msg.update()

# === Chainlit Trigger ===

@cl.on_message
async def handle_user_input(msg: cl.Message):
    q = msg.content.strip()
    await cl.Message(content=f"❓ {q}").send()

    result = graph.invoke({"question": q})

    await stream_response("🔄 Rephrased", result["rephrased"])
    await stream_response("🔍 Retrieved Info", result["retrieved"])
    await stream_response("📘 Answer", result["answer"])
    await stream_response("📝 Summary", result["summary"])
