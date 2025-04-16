import chainlit as cl
from typing import TypedDict, Optional
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, END
from langchain.tools import tool
from langchain_core.runnables import RunnableLambda
from difflib import SequenceMatcher
import re
import logging
import asyncio

# === Logging Setup ===
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

# === Tools ===
@tool
def search_confluence(query: str) -> str:
    if "pipeline" in query.lower():
        return "Confluence: Found CI/CD pipeline docs: https://confluence.mycorp.local/pages/viewpage.action?pageId=123456"
    return f"Confluence: No docs found for '{query}'"

@tool
def search_bitbucket(query: str) -> str:
    return f"Bitbucket: Found matching repo `devops-scripts` with file `deploy.yml` for query '{query}'."

@tool
def query_postgres(query: str) -> str:
    if "orders" in query.lower():
        return "PostgreSQL: SELECT * FROM orders WHERE status = 'pending'; → 13 rows"
    return f"PostgreSQL: No matching rows for '{query}'"

@tool
def query_graphql(query: str) -> str:
    if "user" in query.lower():
        return "GraphQL: Queried `getUser(id)` → returned user profile object"
    return f"GraphQL: No matching query for '{query}'"

@tool
def search_field_mapping(query: str) -> str:
    if "user_id" in query.lower():
        return "Field Mapping: Found field mapping for `user_id` to `user_profile.id`"
    elif "order_status" in query.lower():
        return "Field Mapping: Found field mapping for `order_status` to `order_status_code`"
    return f"Field Mapping: No mappings found for '{query}'"

@tool
def summarize_text(text: str) -> str:
    return f"Summary: {text[:150]}..."

tools = {
    "confluence": search_confluence,
    "bitbucket": search_bitbucket,
    "postgresql": query_postgres,
    "graphql": query_graphql,
    "field mapping": search_field_mapping,
}

# === Utilities ===
def safe_tool_call(tool_func, query: str) -> str:
    try:
        return tool_func(query)
    except Exception as e:
        logger.error(f"[{tool_func.__name__}] Tool error: {e}")
        return f"Error while calling {tool_func.__name__}."

async def async_safe_tool_call(tool_name, query):
    return (tool_name, safe_tool_call(tools[tool_name], query))

# def rank_results(query, tool_results):
#     rankings = []
#     for tool, result in tool_results.items():
#         score = 1
#         if query.lower() in result.lower():
#             score += 2
#         if "Error" in result:
#             score -= 1
#         rankings.append({"tool": tool, "score": score})
#     return rankings

def rank_results(query, tool_results, error_penalty=1, match_boost=2):
    def similarity(a, b):
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    rankings = []
    for tool, result in tool_results.items():
        score = 1  # base score

        # Boost for match
        if query.lower() in result.lower():
            score += match_boost
        else:
            score += int(similarity(query, result) * match_boost)

        # Penalty for errors
        if "error" in result.lower():
            score -= error_penalty

        rankings.append({"tool": tool, "score": score})

    # Sort by descending score
    return sorted(rankings, key=lambda x: x["score"], reverse=True)

# === Nodes ===
def rephrase(state: AgentState) -> AgentState:
    try:
        prompt = f"Rephrase this question to be clearer:\n\n{state['question']}"
        rephrased = llm_base.invoke(prompt)
        return {**state, "rephrased": rephrased}
    except Exception as e:
        logger.error(f"Rephrase error: {e}")
        return {**state, "rephrased": "Unable to rephrase the question."}

async def retrieve(state: AgentState) -> AgentState:
    query = state.get("rephrased", "")
    toolset = []

    try:
        lowered_query = query.lower()

        if "search in all" in lowered_query:
            toolset = list(tools.keys())
        elif match := re.search(r'search\s+(only\s+)?in\s+((?:\w+(?:\s+and\s+)?)+)', lowered_query):
            mentioned = re.split(r'\s+and\s+|\s*,\s*', match.group(2))
            toolset = [tool.strip() for tool in mentioned if tool.strip() in tools]
        else:
            matches = re.findall(f"({'|'.join(tools.keys())})", lowered_query, re.IGNORECASE)
            toolset = list(set(match.lower() for match in matches)) if matches else []

            if not toolset:
                suggestion_prompt = f"Available tools: {list(tools.keys())}.\nQuery: \"{query}\". Suggest most relevant tools (Python list only)."
                try:
                    suggestions = eval(llm_base.invoke(suggestion_prompt))
                    toolset = [tool for tool in suggestions if tool in tools]
                except Exception:
                    toolset = list(tools.keys())

        results = await asyncio.gather(*[async_safe_tool_call(tool, query) for tool in toolset])
        tool_results = dict(results)

        ranked = rank_results(query, tool_results)
        sorted_results = {entry["tool"]: tool_results[entry["tool"]] for entry in sorted(ranked, key=lambda x: -x["score"])}

        return {**state, "retrieved": sorted_results}

    except Exception as e:
        logger.error(f"Retrieve error: {e}")
        return {**state, "retrieved": {"error": "Retrieval failed."}}

def generate_answer(state: AgentState) -> AgentState:
    try:
        prompt = f"Based on this information:\n{state['retrieved']}\n\nAnswer the question: {state['rephrased']}"
        result = llm_base.invoke(prompt)
        return {**state, "answer": result}
    except Exception as e:
        logger.error(f"Generate answer error: {e}")
        return {**state, "answer": "Unable to generate an answer."}

def summarize(state: AgentState) -> AgentState:
    try:
        summary = summarize_text.invoke(state["answer"])
        return {**state, "summary": summary}
    except Exception as e:
        logger.error(f"Summarize error: {e}")
        return {**state, "summary": "Unable to summarize the answer."}

# === LangGraph ===
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

# === Streaming Utility ===
async def stream_response(title: str, content: str):
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
        logger.error(f"Streaming error: {e}")
        await cl.Message(content="Error during streaming response.").send()

# === Chainlit Message Handler ===
@cl.on_message
async def handle_user_input(msg: cl.Message):
    try:
        question = msg.content.strip()
        await cl.Message(content=f"{question}").send()

        result = await graph.ainvoke({"question": question})

        await stream_response("Rephrased", result["rephrased"])

        retrieved_text = "\n\n".join(f"{k}: {v}" for k, v in result["retrieved"].items())
        await stream_response("Retrieved Info", retrieved_text)

        await stream_response("Answer", result["answer"])
        await stream_response("Summary", result["summary"])

    except Exception as e:
        logger.error(f"Handler error: {e}")
        await cl.Message(content="An error occurred while processing your request.").send()
