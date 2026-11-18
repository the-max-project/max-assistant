# Copyright (c) 2025, Robert Begg
# Licensed under the MIT License. See LICENSE for more details.
"""
This module defines a framework for managing conversation history and invoking a large language model (LLM)
to generate responses. It includes mechanisms for pruning conversation history to ensure efficient interaction
with the LLM, as well as a configurable reasoning engine implemented as a state graph.

The module initializes an LLM using the ChatOllama model, constructs conversation nodes for pruning messages
and generating AI responses, and builds an execution graph workflow for multi-node communication.
The reasoning engine incorporates both stateful and asynchronous operations to handle conversational data.

Classes and functions are structured to allow seamless integration of the reasoning engine into
chat applications or AI-powered assistants.
"""

import logging
import json
from typing import Any, Literal
import uuid

import tiktoken

# Added SystemMessage to the LangChain imports
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, ToolCall, RemoveMessage, trim_messages
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_neo4j.checkpoint.aio import AsyncNeo4jSaver

from max_assistant.agent.prompts import ChatPromptTemplate, MessagesPlaceholder, senior_assistant_prompt
from max_assistant.agent.state import GraphState
from max_assistant.tools.registry import ToolRegistry
from max_assistant.tools.time_tools import get_current_datetime
from max_assistant.utils.datetime_utils import current_datetime


logger = logging.getLogger(__name__)


def process_and_summarize(raw_input_data):
    try:
        # Attempt to parse the string as JSON
        data = json.loads(raw_input_data)

        # If it succeeds, run your JSON summary logic
        if isinstance(data, list):
            return f"JSON List: {json.dumps(data[:2])} ... Total records: {len(data)}"
        return f"JSON Object: {json.dumps(list(data.keys())[:3])} keys..."

    except (json.JSONDecodeError, TypeError):
        # Fallback handling if it is plain text, html, or malformed
        # Truncate safely using the slice conditional we used earlier
        clean_text = str(raw_input_data).strip()
        truncated = clean_text[:47] + "..." if len(clean_text) > 50 else clean_text
        return f"Non-JSON Data (Raw Text Summary): {truncated}"


# Initialize a fast, local tokenizer
tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(messages: list) -> int:
    total_tokens = 0
    for m in messages:
        content = str(m.content) if m.content else ""
        # Heuristic: ~4 characters per token is highly efficient for pruning calculations
        total_tokens += len(content) // 4
    return total_tokens


# --- Build the Graph ---
# noinspection PyTypeChecker
async def create_reasoning_engine(
        llm: ChatOllama,
        tool_registry: ToolRegistry, ):
    """Builds the graph with pruning, model calls, and tool execution."""

    # 1. Initialize Tools from the registry
    logger.info("Collecting tools from registry...")
    tools = tool_registry.get_all_tools()
    tools.append(get_current_datetime)  # Add standalone tools
    logger.debug(f"Tools List: {tools}")
    llm_with_tools = llm.bind_tools(tools)
    llm_without_tools = llm
    logger.info(f"Reasoning engine configured with {len(tools)} tools.")

    # 2. Define Nodes that will be part of the graph

    def prepare_input(state: GraphState)-> dict[str, Any]:
        """
        Prepares the message history for the model invocation turn.
        Intercepts push notifications from background system tasks if present.
        """
        logger.info("Node: prepare_input")

        #  Intercept background reminder push events ---
        external_event = state.get("external_event")
        if external_event:
            event_type = external_event.get("type", "REMINDER")
            reminder_text = external_event.get("text", "")
            item_id = external_event.get("task_id") or external_event.get("item_id") or "unknown"
            logger.info(f"prepare_input: Intercepted background event [{event_type}] id=[{item_id}]")

            event_message = HumanMessage(
                content=f"[BACKGROUND SYSTEM DIRECTIVE | TYPE: {event_type}]\n{reminder_text}"
            )

            return {
                "messages": [event_message],
                "external_event": {},
                "is_background": True,
            }

        # --- Fallback: Normal conversational turn processing ---
        text = state.get("transcribed_text", "").strip()
        if text:
            return {
                "messages": [HumanMessage(content=text)],
                "transcribed_text": "",
                "is_background": False,
            }
        return {"is_background": False}

    async def call_model(state: GraphState)-> dict[str, Any]:
        """
        Node to invoke the LLM with the current state. The user's input is already
        in the message history.
        """
        logger.debug("Calling model with current history.")

        current_messages = state.get("messages", [])
        trimmed_messages = trim_messages(
            current_messages,
            max_tokens=128000,
            strategy="last",
            token_counter=count_tokens,
            include_system=True,
            allow_partial=False,
        )

        if state.get("is_background"):
            bg_prompt = ChatPromptTemplate.from_messages([
                ("system",
                 "You are Max, a warm, patient companion speaking proactively to the user.\n"
                 "User profiles: {user_info}\n"
                 "Current time context: {current_datetime}\n\n"
                 "CRITICAL EXECUTION DIRECTIVES:\n"
                 "1. The latest message is an internal SYSTEM DIRECTIVE instructing you what to say to the user. "
                 "It is NOT the user speaking to you. Never reply as the user.\n"
                 "2. Speak directly to the user in 1-2 warm, simple sentences appropriate for dementia care:\n"
                 "   - REMINDER: Gently announce that their reminder time has arrived.\n"
                 "   - WAKE_UP_BRIEFING: Deliver a calm morning greeting using the provided summary.\n"
                 "   - PLAYBOOK_PROMPT: Gently ask the optional activity question or share the family update.\n"
                 "3. NEVER call tools during a background trigger. If asking a question, ask it and stop—wait for the user's reply on the next turn.\n"
                 "4. Output ONLY the spoken words for the user. Never read aloud internal IDs, UUIDs, tool names, markdown, or JSON."),
                MessagesPlaceholder(variable_name="messages")
            ])
            chain = bg_prompt | llm_without_tools
            msg_tag = "TASK"
        else:
            # Fall back to standard interactive conversational persona
            chain = senior_assistant_prompt | llm_with_tools
            msg_tag = "HUMAN"

        # Get the last message from trimmed_messages
        last_message = trimmed_messages[-1] if trimmed_messages else None

        # Check if it's a HumanMessage (user input)
        logger.info("==================== Calling Model: ===================")
        if isinstance(last_message, HumanMessage):
            last_user_prompt = last_message.content
            logger.info(f"▶▶▶ [{msg_tag}] [{last_user_prompt}]")
        elif isinstance(last_message, ToolMessage):
            last_tool_message = last_message.content
            logger.info(f"▶▶▶ [Tool] [{process_and_summarize(last_tool_message)}]")
        else:
            logger.info(f"Calling Model: with prompt [{trimmed_messages[-1]}] ")

        # The 'messages' in the state now contains the user's latest input.
        response = await chain.ainvoke({
            "user_info": state["userinfo"],
            "current_datetime": current_datetime(),
            "messages": trimmed_messages,
        })

        if response.tool_calls:
            logger.info(f"◀◀◀ [Tool Call] {response.tool_calls}")
        else:
            logger.info(f"◀◀◀ [AI]  {repr(response.content)}")

        if response.tool_calls:
            # It's a standard tool call, just return it
            return {"messages": [response]}

        try:
            # Ensure content is natively a string before feeding it to json.loads
            if isinstance(response.content, str):
                try:
                    # Check if the *content* is a JSON tool call
                    content_json = json.loads(response.content)
                    if isinstance(content_json, dict) and "name" in content_json:
                        logger.warning("Raw JSON tool call detected. Re-formatting message.")

                        # Create a proper AIMessage with a tool_calls attribute
                        tool_call_obj = ToolCall(
                            name=content_json["name"],
                            args=content_json.get("parameters", {}),
                            id=f"call_{uuid.uuid4().hex[:16]}"  # Create a new ID
                        )

                        # Create a new message that has the 'tool_calls'
                        # attribute that should_continue is looking for.
                        new_response = AIMessage(
                            content="",  # Content is now empty
                            tool_calls=[tool_call_obj],
                            id=response.id
                        )
                        return {"messages": [new_response]}
                except (json.JSONDecodeError, TypeError):
                    # It's just a regular text response, not JSON
                    pass

        except (json.JSONDecodeError, TypeError):
            # It's just a regular text response, not JSON
            pass

            # It's a regular text response, return it as-is
        return {"messages": [response]}

    def should_continue(state: GraphState) -> Literal["execute_tools", "__end__"]:
        """Conditional node to decide whether to execute tools or end."""
        if state.get("is_background"):
            logger.info("Node: should_continue (background turn) - Return= __end__")
            return "__end__"

        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            logger.info("Node: should_continue - Return= execute_tools")
            return "execute_tools"

        logger.info("Node: should_continue - Return= __end__")
        return "__end__"

    # 3. Build the workflow
    workflow = StateGraph(GraphState)

    workflow.add_node("prepare_input", prepare_input)
    workflow.add_node("agent", call_model)
    workflow.add_node("execute_tools", ToolNode(tools))

    # 4. Add edges
    workflow.set_entry_point("prepare_input")
    workflow.add_edge("prepare_input", "agent")
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        ["execute_tools", "__end__"]
    )
    workflow.add_edge("execute_tools", "agent")

    # Instantiate Neo4j Checkpointer
    checkpointer = AsyncNeo4jSaver(tool_registry.db_client.driver)
    await checkpointer.setup()

    # 5. Compile and return
    compiled_graph = workflow.compile(checkpointer=checkpointer)

    return compiled_graph