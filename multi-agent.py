import chainlit as cl
from typing import TypedDict, Optional
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_core.runnables import RunnableLambda
import re
import logging

# === Setup Logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === LLMs ===
llm_base = OllamaLLM(model="mistral")
llm_streaming = OllamaLLM(model="mistral", streaming=True)

# === Agent State Definition ===
class AgentState(TypedDict):
    question: str
    rephrased: Optional[str]
    retrieved: Optional[dict]
    answer: Optional[str]
    summary: Optional[str]

# === Simulated Tool Integrations ===
def safe_tool_call(tool_func, query: str) -> str:
    try:
        return tool_func(query)
    except Exception as e:
        logger.error(f"Error in tool {tool_func.__name__} with query {query}: {e}")
        return f"❌ Error while fetching from {tool_func.__name__}."

@tool
def search_confluence(query: str) -> str:
    if "pipeline" in query.lower():
        return "📄 Confluence: Found CI/CD pipeline docs: https://confluence.mycorp.local/pages/viewpage.action?pageId=123456"
    return f"📄 Confluence: No docs found for '{query}'"

@tool
def search_bitbucket(query: str) -> str:
    return f"📁 Bitbucket: Found matching repo `devops-scripts` with file `deploy.yml` for query '{query}'."

@tool
def query_postgres(query: str) -> str:
    if "orders" in query.lower():
        return "🗃️ PostgreSQL: SELECT * FROM orders WHERE status = 'pending'; → 13 rows"
    return f"🗃️ PostgreSQL: No matching rows for '{query}'"

@tool
def query_graphql(query: str) -> str:
    if "user" in query.lower():
        return "🔗 GraphQL: Queried `getUser(id)` → returned user profile object"
    return f"🔗 GraphQL: No matching query for '{query}'"

@tool
def search_field_mapping(query: str) -> str:
    if "user_id" in query.lower():
        return "🗺️ Field Mapping: Found field mapping for `user_id` to `user_profile.id`"
    elif "order_status" in query.lower():
        return "🗺️ Field Mapping: Found field mapping for `order_status` to `order_status_code`"
    return f"🗺️ Field Mapping: No mappings found for '{query}'"

@tool
def summarize_text(text: str) -> str:
    return f"📝 Summary: {text[:150]}..."

# List of tools available to the Retrieval Agent
tools = {
    "confluence": search_confluence,
    "bitbucket": search_bitbucket,
    "postgresql": query_postgres,
    "graphql": query_graphql,
    "field mapping": search_field_mapping,
}

# === LangGraph Node Implementations ===

def rephrase(state: AgentState) -> AgentState:
    try:
        prompt = f"Rephrase this question to be clearer:\n\n{state['question']}"
        result = llm_base.invoke(prompt)
        return {**state, "rephrased": result}
    except Exception as e:
        logger.error(f"Error rephrasing question: {e}")
        return {**state, "rephrased": "❌ Unable to rephrase the question."}

def retrieve(state: AgentState) -> AgentState:
    try:
        # Regex pattern to match tool names from the question
        tool_names = "|".join(tools.keys())
        pattern = re.compile(f"({tool_names})", re.IGNORECASE)

        # Search for a tool name in the rephrased question
        found_tools = pattern.findall(state["rephrased"])

        # If no tool is mentioned, search all tools
        if not found_tools:
            found_tools = list(tools.keys())

        # Initialize agent with selected tools
        tool_results = {}
        for tool_name in found_tools:
            logger.info(f"Searching with tool: {tool_name}")
            tool_results[tool_name] = safe_tool_call(tools[tool_name], state["rephrased"])

        return {**state, "retrieved": tool_results}

    except Exception as e:
        logger.error(f"Error in retrieving data: {e}")
        return {**state, "retrieved": {"error": "❌ Error in retrieving information."}}

def generate_answer(state: AgentState) -> AgentState:
    try:
        prompt = f"Based on this info:\n{state['retrieved']}\n\nAnswer the question: {state['rephrased']}"
        result = llm_base.invoke(prompt)
        return {**state, "answer": result}
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return {**state, "answer": "❌ Unable to generate an answer."}

def summarize(state: AgentState) -> AgentState:
    try:
        summary = summarize_text.invoke(state["answer"])
        return {**state, "summary": summary}
    except Exception as e:
        logger.error(f"Error summarizing answer: {e}")
        return {**state, "summary": "❌ Unable to summarize the answer."}

# === LangGraph Flow ===

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

# === Streaming Output in Chainlit ===

async def stream_response(title, content):
    try:
        msg = cl.Message(content="", author=title)
        await msg.send()

        full_text = ""
        async for chunk in llm_streaming.astream(content):
            full_text += chunk
            await msg.stream_token(chunk)
        msg.content = full_text.strip()
        await msg.update()
    except Exception as e:
        logger.error(f"Error during streaming response: {e}")
        await cl.Message(content="❌ Error during streaming response.").send()

# === Chainlit Entry Point ===

@cl.on_message
async def handle_user_input(msg: cl.Message):
    try:
        q = msg.content.strip()
        await cl.Message(content=f"❓ {q}").send()

        result = graph.invoke({"question": q})

        await stream_response("🔄 Rephrased", result["rephrased"])

        # Display tool-wise retrieved results
        retrieved_info = ""
        for tool_name, result in result["retrieved"].items():
            retrieved_info += f"**{tool_name}:** {result}\n\n"
        await stream_response("🔍 Retrieved Info", retrieved_info)

        await stream_response("📘 Answer", result["answer"])
        await stream_response("📝 Summary", result["summary"])
    except Exception as e:
        logger.error(f"Error handling user input: {e}")
        await cl.Message(content="❌ An error occurred while processing your request.").send()
