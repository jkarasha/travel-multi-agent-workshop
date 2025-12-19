import asyncio
import json
import logging
import os
import uuid
from typing import Literal
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver
from datetime import datetime, UTC

from src.app.services.azure_open_ai import model
from src.app.services.azure_cosmos_db import patch_active_agent, sessions_container, update_session_container

from langgraph_checkpoint_cosmosdb import CosmosDBSaver
from src.app.services.azure_cosmos_db import DATABASE_NAME, checkpoint_container

local_interactive_mode = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce noise from verbose libraries
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.identity").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("azure.cosmos").setLevel(logging.WARNING)

PROMPT_DIR = os.path.join(os.path.dirname(__file__), 'prompts')

# load prompts
def load_prompt(agent_name: str) -> str:
    """Load prompt from .prompty file."""
    file_path = os.path.join(PROMPT_DIR, f"{agent_name}.prompty")
    logger.info(f"Loading promot for agent '{agent_name}' from {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except FileNotFoundError:
        logger.error(f"Prompt file for agent '{agent_name}' not found.")
        return f"You are a {agent_name} agent in a travel planning application."
        
def filter_tools_by_prefix(tools, prefixes):
    """Filter tools by given prefixes."""
    return [tool for tool in tools if any(tool.name.startswith(prefix) for prefix in prefixes)]

# global variables for MCP session management
_mcp_client = None
_session_context = None
_persistent_session = None

# Global agent variables
orchestrator_agent = None
hotel_agent = None
activity_agent = None
dining_agent = None
itinerary_generator_agent = None

# define agents & tools
async def setup_agents():
    global orchestrator_agent, hotel_agent, activity_agent, dining_agent, itinerary_generator_agent
    global _mcp_client, _session_context, _persistent_session

    logger.info("Starting Travel Assistant MCP client...")

    #Load authentication configuration
    try:
        simple_token = os.getenv("MCP_AUTH_TOKEN")

        logger.info("Client Authentication Configuration:")
        logger.info(f" Simple Token: {'SET' if simple_token else 'NOT SET'}")
        
        # Determine authentication method
        if simple_token:
            auth_mode = 'simple_token'
            logger.info(f"  Mode: Simple Token (Development)")
        else:
            auth_mode = "none"
            logger.info("   Mode: No Authentication")
    except ImportError:
        simple_token = None
        logger.info("Client Authentication: dependencies unavailable - no auth")

    logger.info("    - Transport: streamable_http")
    logger.info(f"   - Server URL: {os.getenv('MCP_SERVER_URL', 'http://localhost:8000')}/mcp/")
    logger.info(f"   - Authentication Mode: {auth_mode.upper()}")
    logger.info("    - Status: Ready to connect\n")

    # MCP Client configuration
    client_config = {
        "travel_tools": {
            "transport": "streamable_http",
            "url": os.getenv("MCP_SERVER_BASE_URL", "http://localhost:8080") + "/mcp/",
        }
    }
    # Add authentication if configured
    client_config["travel_tools"]["headers"] = {
        "Authorization": f"Bearer {simple_token}"
    }
    logger.info("🔐 Added Bearer token authentication to client")

    _mcp_client = MultiServerMCPClient(client_config)
    logger.info("✅ MCP Client initialized successfully")

    # Create persistent session
    _session_context = _mcp_client.session("travel_tools")
    _persistent_session = await _session_context.__aenter__()

    # Load all MCP tools
    all_tools = await load_mcp_tools(_persistent_session)

    logger.info("[DEBUG] All tools registered from Travel Assistant MCP server:")
    for tool in all_tools:
        logger.info(f"  - {tool.name}")

    # ========================================================================
    # Tool Distribution for Agents
    # ========================================================================

    # Orchestrator: Session management + all transfer tools
    orchestrator_tools = filter_tools_by_prefix(all_tools, [
        "create_session", "get_session_context", "append_turn",
        "transfer_to_"  # All transfer tools
    ])
    itinerary_generator_tools = filter_tools_by_prefix(all_tools, [
        "create_new_trip", "update_trip", "get_trip_details",
        "transfer_to_orchestrator"
    ])
    hotel_tools = filter_tools_by_prefix(all_tools, [
        "discover_places",  # Search hotels
        "recall_memories",
        "transfer_to_orchestrator", "transfer_to_itinerary_generator"
    ])
    activity_tools = filter_tools_by_prefix(all_tools, [
        "discover_places",  # Search attractions
        "recall_memories",
        "transfer_to_orchestrator", "transfer_to_itinerary_generator"
    ])
    dining_tools = filter_tools_by_prefix(all_tools, [
        "discover_places",  # Search restaurants
        "recall_memories",
        "transfer_to_orchestrator", "transfer_to_itinerary_generator"
    ])

    # Create agents with their tools
    orchestrator_agent = create_react_agent(
        model,
        orchestrator_tools,
        state_modifier=load_prompt("orchestrator")
    )

    itinerary_generator_agent = create_react_agent(
        model,
        itinerary_generator_tools,
        state_modifier=load_prompt("itinerary_generator")
    )

    hotel_agent = create_react_agent(
        model,
        hotel_tools,
        state_modifier=load_prompt("hotel_agent")
    )

    activity_agent = create_react_agent(
        model,
        activity_tools,
        state_modifier=load_prompt("activity_agent")
    )

    dining_agent = create_react_agent(
        model,
        dining_tools,
        state_modifier=load_prompt("dining_agent")
    )

# define functions
async def call_orchestrator_agent(state: MessagesState, config) -> Command[Literal["orchestrator", "human"]]:
    """
    Orchestrator agent: Routes requests using transfer_to_ tools.
    Checks for active agent and routes directly if found.
    Stores every message in database.
    """
    thread_id = config["configurable"].get("thread_id", "UNKNOWN_THREAD_ID")
    user_id = config["configurable"].get("userId", "UNKNOWN_USER_ID")
    tenant_id = config["configurable"].get("tenantId", "UNKNOWN_TENANT_ID")

    # Add context about available parameters
    state["messages"].append(SystemMessage(
        content=f"If tool to be called requires tenantId='{tenant_id}', userId='{user_id}', session_id='{thread_id}', include these in the JSON parameters when invoking the tool. Do not ask the user for them."
    ))

    # Check for active agent in database
    try:
        logging.info(f"Looking up active agent for thread {thread_id}")
        session_doc = sessions_container.read_item(
            item=thread_id,
            partition_key=[tenant_id, user_id, thread_id]
        )
        activeAgent = session_doc.get('activeAgent', 'unknown')
    except Exception as e:
        logger.debug(f"No active agent found: {e}")
        activeAgent = None

    # Initialize session if needed (for local testing)
    if activeAgent is None:
        update_session_container({
            "id": thread_id,
            "sessionId": thread_id,
            "tenantId": tenant_id,
            "userId": user_id,
            "title": "New Conversation",
            "createdAt": datetime.now(UTC).isoformat(),
            "lastActivityAt": datetime.now(UTC).isoformat(),
            "status": "active",
            "messageCount": 0
        })

    logger.info(f"Active agent from DB: {activeAgent}")

    # Always call orchestrator to analyze the message and decide routing
    # Don't blindly route to the last active agent - user's request may have changed
    response = await orchestrator_agent.ainvoke(state, config)
    return Command(update=response, goto="human")

async def call_itinerary_generator_agent(state: MessagesState, config) -> Command[
    Literal["itinerary_generator", "orchestrator", "human"]]:
    """
    Itinerary Generator: Synthesizes all gathered info into day-by-day plan.
    """
    thread_id = config["configurable"].get("thread_id", "UNKNOWN_THREAD_ID")
    user_id = config["configurable"].get("userId", "UNKNOWN_USER_ID")
    tenant_id = config["configurable"].get("tenantId", "UNKNOWN_TENANT_ID")

    logger.info("📋 Itinerary Generator synthesizing plan...")

    # Patch active agent in database
    if local_interactive_mode:
        patch_active_agent(tenant_id or "cli-test", user_id or "cli-test", thread_id, "itinerary_generator_agent")

    # Add context about available parameters
    state["messages"].append(SystemMessage(
        content=f"If tool to be called requires tenantId='{tenant_id}', userId='{user_id}', session_id='{thread_id}', include these in the JSON parameters when invoking the tool. Do not ask the user for them."
    ))

    response = await itinerary_generator_agent.ainvoke(state, config)
    return Command(update=response, goto="human")

async def call_hotel_agent(state: MessagesState, config) -> Command[
    Literal["hotel", "itinerary_generator", "orchestrator", "human"]]:
    """
    Hotel Agent: Searches accommodations and stores hotel preferences.
    """
    thread_id = config["configurable"].get("thread_id", "UNKNOWN_THREAD_ID")
    user_id = config["configurable"].get("userId", "UNKNOWN_USER_ID")
    tenant_id = config["configurable"].get("tenantId", "UNKNOWN_TENANT_ID")

    # Patch active agent in database
    if local_interactive_mode:
        patch_active_agent(tenant_id or "cli-test", user_id or "cli-test", thread_id, "hotel_agent")

    # Add context about available parameters
    state["messages"].append(SystemMessage(
        content=f"If tool to be called requires tenantId='{tenant_id}', userId='{user_id}', session_id='{thread_id}', include these in the JSON parameters when invoking the tool. Do not ask the user for them."
    ))

    response = await hotel_agent.ainvoke(state, config)
    return Command(update=response, goto="human")

async def call_activity_agent(state: MessagesState, config) -> Command[
    Literal["activity", "itinerary_generator", "orchestrator", "human"]]:
    """
    Activity Agent: Searches attractions and stores activity preferences.
    """
    thread_id = config["configurable"].get("thread_id", "UNKNOWN_THREAD_ID")
    user_id = config["configurable"].get("userId", "UNKNOWN_USER_ID")
    tenant_id = config["configurable"].get("tenantId", "UNKNOWN_TENANT_ID")

    # Patch active agent in database
    if local_interactive_mode:
        patch_active_agent(tenant_id or "cli-test", user_id or "cli-test", thread_id, "activity_agent")

    # Add context about available parameters
    state["messages"].append(SystemMessage(
        content=f"If tool to be called requires tenantId='{tenant_id}', userId='{user_id}', session_id='{thread_id}', include these in the JSON parameters when invoking the tool. Do not ask the user for them."
    ))

    response = await activity_agent.ainvoke(state, config)
    return Command(update=response, goto="human")

async def call_dining_agent(state: MessagesState, config) -> Command[
    Literal["dining", "itinerary_generator", "orchestrator", "human"]]:
    """
    Dining Agent: Searches restaurants and stores dining preferences.
    """
    thread_id = config["configurable"].get("thread_id", "UNKNOWN_THREAD_ID")
    user_id = config["configurable"].get("userId", "UNKNOWN_USER_ID")
    tenant_id = config["configurable"].get("tenantId", "UNKNOWN_TENANT_ID")

    # Patch active agent in database
    if local_interactive_mode:
        patch_active_agent(tenant_id or "cli-test", user_id or "cli-test", thread_id, "dining_agent")

    # Add context about available parameters
    state["messages"].append(SystemMessage(
        content=f"If tool to be called requires tenantId='{tenant_id}', userId='{user_id}', session_id='{thread_id}', include these in the JSON parameters when invoking the tool. Do not ask the user for them."
    ))

    response = await dining_agent.ainvoke(state, config)
    return Command(update=response, goto="human")

def human_node(state: MessagesState, config) -> None:
    """
    Human node: Interrupts for user input in interactive mode.
    """
    interrupt(value="Ready for user input.")
    return None

# define workflow
async def cleanup_persistent_session():
    """Clean up the persistent MCP session when the application shuts down"""
    global _session_context, _persistent_session

    if _session_context is not None and _persistent_session is not None:
        try:
            await _session_context.__aexit__(None, None, None)
            logger.info("✅ MCP persistent session cleaned up successfully")
        except Exception as e:
            logger.error(f"Error cleaning up MCP session: {e}")

def get_active_agent(state: MessagesState, config) -> str:
    """
    Extract active agent from ToolMessage or fallback to Cosmos DB.
    This is used by the router to determine which specialized agent to call.
    Also checks if auto-summarization should be triggered.
    """
    thread_id = config["configurable"].get("thread_id", "UNKNOWN_THREAD_ID")
    user_id = config["configurable"].get("userId", "UNKNOWN_USER_ID")
    tenant_id = config["configurable"].get("tenantId", "UNKNOWN_TENANT_ID")

    activeAgent = None

    # Search for last ToolMessage and try to extract `goto`
    for message in reversed(state['messages']):
        if isinstance(message, ToolMessage):
            try:
                content_json = json.loads(message.content)
                activeAgent = content_json.get("goto")
                if activeAgent:
                    logger.info(f"🎯 Extracted activeAgent from ToolMessage: {activeAgent}")
                    break
            except Exception as e:
                logger.debug(f"Failed to parse ToolMessage content: {e}")

    # Fallback: Cosmos DB lookup if needed
    if not activeAgent:
        try:
            session_doc = sessions_container.read_item(
                item=thread_id,
                partition_key=[tenant_id, user_id, thread_id]
            )
            activeAgent = session_doc.get('activeAgent', 'unknown')
            logger.info(f"Active agent from DB: {activeAgent}")
        except Exception as e:
            logger.error(f"Error retrieving active agent from DB: {e}")
            activeAgent = "unknown"

    # If activeAgent is unknown or None, default to orchestrator
    if activeAgent in [None, "unknown"]:
        logger.info(f"🔀 activeAgent is '{activeAgent}', defaulting to Orchestrator")
        activeAgent = "orchestrator"

    return activeAgent

def build_agent_graph():
    logger.info("🏗️  Building multi-agent graph...")

    builder = StateGraph(MessagesState)
    builder.add_node("orchestrator", call_orchestrator_agent)
    builder.add_node("hotel", call_hotel_agent)
    builder.add_node("activity", call_activity_agent)
    builder.add_node("dining", call_dining_agent)
    builder.add_node("itinerary_generator", call_itinerary_generator_agent)
    builder.add_node("human", human_node)

    builder.add_edge(START, "orchestrator")

    # Orchestrator routing - can route to any specialized agent
    builder.add_conditional_edges(
        "orchestrator",
        get_active_agent,
        {
            "hotel": "hotel",
            "activity": "activity",
            "dining": "dining",
            "itinerary_generator": "itinerary_generator",
            "human": "human",  # Wait for user input
            "orchestrator": "orchestrator",  # fallback
        }
    )

    # Hotel routing - can call itinerary_generator or orchestrator
    builder.add_conditional_edges(
        "hotel",
        get_active_agent,
        {
            "itinerary_generator": "itinerary_generator",
            "orchestrator": "orchestrator",
            "hotel": "hotel",  # Can stay in hotel
        }
    )

    # Activity routing - can call itinerary_generator or orchestrator
    builder.add_conditional_edges(
        "activity",
        get_active_agent,
        {
            "itinerary_generator": "itinerary_generator",
            "orchestrator": "orchestrator",
            "activity": "activity",  # Can stay in activity
        }
    )

    # Dining routing - can call itinerary_generator or orchestrator
    builder.add_conditional_edges(
        "dining",
        get_active_agent,
        {
            "itinerary_generator": "itinerary_generator",
            "orchestrator": "orchestrator",
            "dining": "dining",  # Can stay in dining
        }
    )

    # Itinerary Generator routing - can return to orchestrator or stay
    builder.add_conditional_edges(
        "itinerary_generator",
        get_active_agent,
        {
            "orchestrator": "orchestrator",
            "itinerary_generator": "itinerary_generator",  # Can stay to handle follow-ups
        }
    )

    checkpointer = CosmosDBSaver(
        database_name=DATABASE_NAME,
        container_name=checkpoint_container
    )
    graph = builder.compile(checkpointer=checkpointer)

    return graph

async def interactive_chat():
    """
    Interactive CLI for testing the travel assistant.
    Similar to banking app's interactive mode.
    """
    global local_interactive_mode
    local_interactive_mode = True

    thread_id = str(uuid.uuid4())
    thread_config = {
        "configurable": {
            "thread_id": thread_id,
            "userId": "Tony",
            "tenantId": "Marvel"
        }
    }

    print("\n" + "=" * 70)
    print("🌍 Travel Assistant - Interactive Test Mode")
    print("=" * 70)
    print("Type 'exit' to end the conversation")
    print("=" * 70 + "\n")

    # Build graph
    graph = build_agent_graph()

    user_input = input("You: ")

    while user_input.lower() != "exit":
        input_message = {"messages": [{"role": "user", "content": user_input}]}
        response_found = False

        async for update in graph.astream(input_message, config=thread_config, stream_mode="updates"):
            for node_id, value in update.items():
                if isinstance(value, dict) and value.get("messages"):
                    last_message = value["messages"][-1]
                    if isinstance(last_message, AIMessage):
                        print(f"{node_id}: {last_message.content}\n")
                        response_found = True

        if not response_found:
            logger.debug("No AI response received.")

        user_input = input("You: ")

    print("\n👋 Goodbye!")

if __name__ == "__main__":
    # Setup agents and run interactive chat
    async def main():
        await setup_agents()
        await interactive_chat()
    asyncio.run(main())