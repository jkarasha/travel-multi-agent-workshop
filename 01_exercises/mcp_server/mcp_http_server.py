import sys
import os
import logging
import json
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# Add python directory to path so we can import src modules
current_dir = os.path.dirname(os.path.abspath(__file__))
python_dir = os.path.join(current_dir, '..', 'python')
sys.path.insert(0, python_dir)

from src.app.services.azure_cosmos_db import (  # noqa: E402
    create_session_record,
    create_summary,
    get_all_user_memories,
    get_message_by_id,
    get_session_by_id,
    get_session_messages,
    get_session_summaries,
    get_user_summaries,
    query_memories,
    query_places_hybrid,
    create_trip,
    get_trip,
    store_memory,
    supersede_memory,
    trips_container,
    update_memory_last_used
)

from src.app.services.azure_open_ai import get_openai_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce noise from verbose libraries
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("azure.identity").setLevel(logging.WARNING)
logging.getLogger("azure.identity._credentials.environment").setLevel(logging.WARNING)
logging.getLogger("azure.identity._credentials.managed_identity").setLevel(logging.WARNING)
logging.getLogger("azure.identity._credentials.chained").setLevel(logging.WARNING)
logging.getLogger("azure.cosmos").setLevel(logging.WARNING)
logging.getLogger("azure.cosmos._cosmos_http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("mcp").setLevel(logging.WARNING)
logging.getLogger("mcp.client.streamable_http").setLevel(logging.WARNING)

# Suppress SSE, OpenAI, urllib3, and LangSmith debug logs
logging.getLogger("sse_starlette.sse").setLevel(logging.WARNING)
logging.getLogger("openai._base_client").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.WARNING)
logging.getLogger("langsmith.client").setLevel(logging.WARNING)

# Suppress service initialization logs
logging.getLogger("src.app.services.azure_open_ai").setLevel(logging.WARNING)
logging.getLogger("src.app.services.azure_cosmos_db").setLevel(logging.WARNING)

# Prompt directory
PROMPT_DIR = os.path.join(os.path.dirname(__file__), '..', 'python', 'src', 'app', 'prompts')

# Load environment variables
try:
    load_dotenv('.env', override=False)

    # Load authentication configuration
    simple_token = os.getenv("MCP_AUTH_TOKEN")
    base_url = os.getenv("MCP_SERVER_BASE_URL", "http://localhost:8080")

    print("🔐 Authentication Configuration:")
    print(f"   Simple Token: {'SET' if simple_token else 'NOT SET'}")
    print(f"   Base URL: {base_url}")

    # Determine authentication mode
    if simple_token:
        auth_mode = "simple_token"
        print("✅ SIMPLE TOKEN MODE ENABLED (Development)")
        print(f"   Token: {simple_token[:8]}...")
    else:
        auth_mode = "none"
        print("⚠️  NO AUTHENTICATION - All requests accepted")

except ImportError as e:
    auth_mode = "none"
    simple_token = None
    print(f"❌ OAuth dependencies not available: {e}")

# Initialize MCP server
print("\n🚀 Initializing Travel Assistant MCP Server...")
port = int(os.getenv("PORT", 8080))
mcp = FastMCP("TravelAssistantTools", host="0.0.0.0", port=port)

print(f"✅ Travel Assistant MCP server initialized")
print(f"🌐 Server will be available at: http://0.0.0.0:{port}")
print(f"📋 Authentication mode: {auth_mode.upper()}\n")


# ============================================================================
# 1. Agent Transfer Tools (for Orchestrator Routing)
# ============================================================================

@mcp.tool()
def transfer_to_orchestrator(
    reason: str
) -> str:
    """
    Transfer conversation back to the Orchestrator agent.

    Use this when:
    - Task is complete and user needs general assistance
    - User has a new question that doesn't fit specialized agents
    - General conversation, greetings, clarifications needed

    Examples:
    - After completing a specific task
    - User says "Thanks" or changes topic
    - User asks general questions about the system

    Args:
        reason: Why you're transferring to this agent

    Returns:
        JSON with goto field for routing
    """

    logger.info(f"🔄 Transfer to Orchestrator: {reason}")

    return json.dumps({
        "goto": "orchestrator",
        "reason": reason,
        "message": "Transferring back to Orchestrator for general assistance."
    })

@mcp.tool()
def transfer_to_itinerary_generator(
    reason: str
) -> str:
    """
    Transfer conversation to the Itinerary Generator agent.

    Use this when:
    - User explicitly requests an itinerary or day-by-day plan
    - User says "create itinerary", "plan my days", "generate schedule"
    - User wants a complete trip plan synthesized

    Examples:
    - "Create an itinerary for my trip"
    - "Plan my 4 days in Paris"
    - "Generate a schedule with everything we discussed"

    Args:
        reason: Why you're transferring to this agent

    Returns:
        JSON with goto field for routing
    """

    logger.info(f"🔄 Transfer to Itinerary Generator: {reason}")

    return json.dumps({
        "goto": "itinerary_generator",
        "reason": reason,
        "message": "Transferring to Itinerary Generator to create your day-by-day plan."
    })

@mcp.tool()
def transfer_to_hotel(
        reason: str
) -> str:
    """
    Transfer conversation to the Hotel Agent.

    Use this when:
    - User wants to search for hotels or accommodations
    - User is sharing hotel/lodging preferences (boutique, quiet, central, etc.)
    - User asks about places to stay

    Examples:
    - "Find hotels in Paris"
    - "I prefer quiet hotels away from tourist areas"
    - "Where should I stay?"

    Args:
        reason: Why you're transferring to this agent

    Returns:
        JSON with goto field for routing
    """

    logger.info(f"🔄 Transfer to Hotel Agent: {reason}")

    return json.dumps({
        "goto": "hotel",
        "reason": reason,
        "message": "Transferring to Hotel Agent to find accommodations for you."
    })

@mcp.tool()
def transfer_to_activity(
        reason: str
) -> str:
    """
    Transfer conversation to the Activity Agent.

    Use this when:
    - User wants to discover attractions, museums, landmarks
    - User is sharing activity preferences (art, history, nature, etc.)
    - User asks about things to do or see

    Examples:
    - "What should I do in Barcelona?"
    - "Find art museums"
    - "I love history and architecture"

    Args:
        reason: Why you're transferring to this agent

    Returns:
        JSON with goto field for routing
    """

    logger.info(f"🔄 Transfer to Activity Agent: {reason}")

    return json.dumps({
        "goto": "activity",
        "reason": reason,
        "message": "Transferring to Activity Agent to discover attractions for you."
    })

@mcp.tool()
def transfer_to_dining(
        reason: str
) -> str:
    """
    Transfer conversation to the Dining Agent.

    Use this when:
    - User wants restaurant or cafe recommendations
    - User is sharing dietary preferences or cuisine interests
    - User asks where to eat

    Examples:
    - "Find vegetarian restaurants"
    - "I'm pescatarian and like local bistros"
    - "Where should I have dinner?"

    Args:
        reason: Why you're transferring to this agent

    Returns:
        JSON with goto field for routing
    """

    logger.info(f"🔄 Transfer to Dining Agent: {reason}")

    return json.dumps({
        "goto": "dining",
        "reason": reason,
        "message": "Transferring to Dining Agent to find restaurants for you."
    })

@mcp.tool()
def transfer_to_summarizer(
        reason: str
) -> str:
    """
    Transfer conversation to the Summarizer agent.

    Use this when:
    - User asks for a recap or summary of the conversation
    - Conversation has become long (12+ turns)
    - User wants to review what's been discussed or planned

    Examples:
    - "Summarize our conversation"
    - "What have we planned so far?"
    - "Give me a recap"

    Args:
        reason: Why you're transferring to this agent

    Returns:
        JSON with goto field for routing
    """

    logger.info(f"🔄 Transfer to Summarizer: {reason}")

    return json.dumps({
        "goto": "summarizer",
        "reason": reason,
        "message": "Transferring to Summarizer to compress and recap our conversation."
    })

# ============================================================================
# 2. Place Discovery Tools
# ============================================================================

@mcp.tool()
def discover_places(
        geo_scope: str,
        query: str,
        user_id: str,
        tenant_id: str = "",
        filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Memory-aware place search with hybrid RRF retrieval (for chat assistant).

    Args:
        geo_scope: Geographic scope (e.g., "barcelona")
        query: Natural language search query
        user_id: User identifier (for memory alignment)
        tenant_id: Tenant identifier
        filters: Optional filters dict with:
            - type: "hotel" | "restaurant" | "attraction" (optional)
            - dietary: ["vegan", "seafood"] (optional)
            - accessibility: ["wheelchair-friendly"] (optional)
            - priceTier: "budget" | "moderate" | "luxury" (optional)

    Returns:
        List of places with match reasons and memory alignment scores
    """
    # Parse filters
    filters = filters or {}
    place_type = filters.get("type")
    dietary = filters.get("dietary", [])
    accessibility = filters.get("accessibility", [])
    price_tier = filters.get("priceTier")

    # Convert single values to lists if needed
    if dietary and not isinstance(dietary, list):
        dietary = [dietary]
    if accessibility and not isinstance(accessibility, list):
        accessibility = [accessibility]

    # Query places using hybrid RRF search
    try:
        places = query_places_hybrid(
            query=query,
            geo_scope_id=geo_scope,
            place_type=place_type,
            dietary=dietary,
            accessibility=accessibility,
            price_tier=price_tier
        )
        logger.info(f"✅ Hybrid RRF returned {len(places)} results")
    except Exception as e:
        logger.error(f"❌ Error in hybrid search: {e}")
        import traceback
        logger.error(f"{traceback.format_exc()}")
        return []

        # Get user memories for alignment
    logger.info(f"🧠 Recalling user memories...")
    memories = recall_memories(
        user_id=user_id,
        tenant_id=tenant_id,
        query=query
    )
    logger.info(f"🧠 Found {len(memories)} memories")
    
    # Memory alignment scoring and match reason generation
    used_memory_ids = set()  # Track which memories were actually used
    
    for place in places:
        alignment_score = 0.0
        match_reasons = ["Hybrid search match (text + semantic)"]
    
        # Check alignment with user memories
        if memories:
            for memory in memories:
                memory_facets = memory.get("facets", {})
                memory_id = memory.get("id")
                memory_used = False
    
                # Dietary alignment
                if "dietary" in memory_facets:
                    memory_dietary = memory_facets["dietary"]
                    place_dietary = place.get("dietary", [])
                    if memory_dietary in place_dietary:
                        alignment_score += 0.3
                        match_reasons.append(f"Matches your {memory_dietary} preference")
                        memory_used = True
    
                # Price tier alignment
                if "priceTier" in memory_facets:
                    memory_price = memory_facets["priceTier"]
                    place_price = place.get("priceTier")
                    if memory_price == place_price:
                        alignment_score += 0.2
                        match_reasons.append(f"Matches your {place_price} preference")
                        memory_used = True
    
                # Accessibility alignment
                if "accessibility" in memory_facets:
                    memory_access = memory_facets["accessibility"]
                    place_access = place.get("accessibility", [])
                    if memory_access in place_access:
                        alignment_score += 0.3
                        match_reasons.append(f"Accessible: {memory_access}")
                        memory_used = True
    
                # Track this memory as used if it influenced the recommendation
                if memory_used and memory_id:
                    used_memory_ids.add(memory_id)
    
        # Add memory alignment to place
        place["memoryAlignment"] = min(alignment_score, 1.0)
        place["matchReasons"] = match_reasons
    
    # Update lastUsedAt only for memories that were actually used
    if used_memory_ids:
        logger.info(f"🔄 Updating lastUsedAt for {len(used_memory_ids)} memories that influenced recommendations")
        for memory_id in used_memory_ids:
            update_memory_last_used(
                memory_id=memory_id,
                user_id=user_id,
                tenant_id=tenant_id
            )
    #
    logger.info(f"✅ Returning {len(places)} places with memory alignment")
    return places

# ============================================================================
# 3. Trip Management Tools
# ============================================================================

@mcp.tool()
def create_new_trip(
        user_id: str,
        tenant_id: str,
        scope: Dict[str, str],
        dates: Dict[str, str],
        travelers: List[str],
        constraints: Optional[Dict[str, Any]] = None,
        days: Optional[List[Dict[str, Any]]] = None,
        trip_duration: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create a new trip itinerary.

    Args:
        user_id: User identifier
        tenant_id: Tenant identifier
        scope: Trip scope (type: "city", id: "barcelona")
        dates: Trip dates (start, end in ISO format)
        travelers: List of traveler user IDs
        constraints: Optional constraints (budgetTier, etc.)
        days: Optional list of day-by-day itinerary (dayNumber, date, activities, etc.)
        trip_duration: Optional total number of days (calculated from days array if not provided)

    Returns:
        Dictionary with tripId and details
    """
    logger.info(f"🎒 Creating trip for user: {user_id} with {len(days or [])} days")

    trip_id = create_trip(
        user_id=user_id,
        tenant_id=tenant_id,
        scope=scope,
        dates=dates,
        travelers=travelers,
        constraints=constraints or {},
        days=days or [],
        trip_duration=trip_duration
    )

    return {
        "tripId": trip_id,
        "scope": scope,
        "dates": dates,
        "tripDuration": trip_duration or len(days or []),
        "daysCount": len(days or [])
    }

@mcp.tool()
def get_trip_details(
        trip_id: str,
        user_id: str,
        tenant_id: str = ""
) -> Optional[Dict[str, Any]]:
    """
    Get trip details by ID.

    Args:
        trip_id: Trip identifier
        user_id: User identifier
        tenant_id: Tenant identifier

    Returns:
        Trip dictionary or None if not found
    """
    logger.info(f"📋 Getting trip: {trip_id}")
    return get_trip(trip_id, user_id, tenant_id)

@mcp.tool()
def update_trip(
        trip_id: str,
        user_id: str,
        tenant_id: str,
        updates: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Update trip details (add days, modify constraints, etc.).

    Args:
        trip_id: Trip identifier
        user_id: User identifier
        tenant_id: Tenant identifier
        updates: Dictionary of fields to update

    Returns:
        Updated trip dictionary
    """
    logger.info(f"📝 Updating trip: {trip_id}")

    # Get existing trip
    trip = get_trip(trip_id, user_id, tenant_id)
    if not trip:
        raise ValueError(f"Trip {trip_id} not found")

    # Apply updates
    trip.update(updates)

    # Save to Cosmos DB
    if trips_container:
        trips_container.upsert_item(trip)

    return trip

# ============================================================================
# 4. Session Management Tools
# ============================================================================

@mcp.tool()
def create_session(
        user_id: str,
        tenant_id: str = "",
        title: str = None,
        activeAgent: str = "orchestrator"
) -> Dict[str, Any]:
    """
    Create a new conversation session with proper initialization.

    Args:
        user_id: User identifier
        tenant_id: Tenant identifier (default: empty string)
        title: Optional session title
        activeAgent: Active agent (default: empty string)

    Returns:
        Dictionary with session details including sessionId
    """
    logger.info(f"🆕 Creating session for user: {user_id}")
    session = create_session_record(user_id, tenant_id, activeAgent, title)
    return {
        "sessionId": session["sessionId"],
        "userId": user_id,
        "title": session["title"],
        "createdAt": session["createdAt"]
    }

@mcp.tool()
def get_session_context(
        session_id: str,
        tenant_id: str,
        user_id: str,
        include_summaries: bool = True
) -> Dict[str, Any]:
    """
    Retrieve conversation context (recent messages + summaries).

    Args:
        session_id: Session identifier
        tenant_id: Tenant identifier
        user_id: User identifier
        include_summaries: Whether to include summaries (default: True)

    Returns:
        Dictionary with messages, summaries, and metadata
    """
    logger.info(f"📖 Getting context for session: {session_id}")

    messages = get_session_messages(session_id, tenant_id, user_id)
    session_info = get_session_by_id(session_id, tenant_id, user_id)

    result = {
        "messages": messages,
        "sessionInfo": session_info,
        "messageCount": len(messages)
    }

    if include_summaries:
        summaries = get_session_summaries(session_id, tenant_id, user_id)
        result["summaries"] = summaries
        result["summaryCount"] = len(summaries)

    return result

# ============================================================================
# 5. Memory Lifecycle Tools
# ============================================================================

@mcp.tool()
def recall_memories(
    user_id: str,
    tenant_id: str,
    query: str,
    min_salience: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Smart hybrid retrieval of relevant memories.

    Args:
        user_id: User identifier
        tenant_id: Tenant identifier
        query: Search query for semantic search
        min_salience: Minimum salience threshold (default: 0.0)

    Returns:
        List of memory dictionaries with scores and match reasons
    """
    logger.info(f"🔍 Recalling memories for user: {user_id}")
    # For now, return top memories by salience
    memories = query_memories(
        user_id=user_id,
        tenant_id=tenant_id,
        query=query,
        min_salience=min_salience
    )

    return memories

@mcp.tool()
def extract_preferences_from_message(
        message: str,
        role: str,
        user_id: str,
        tenant_id: str
) -> Dict[str, Any]:
    """
    Extract travel preferences from a user or assistant message using LLM.
    Smart enough to skip greetings, simple yes/no, and other non-preference messages.

    Args:
        message: The message text to analyze
        role: Message role (user/assistant)
        user_id: User identifier (for logging)
        tenant_id: Tenant identifier (for logging)

    Returns:
        Dictionary with:
        - shouldExtract: bool (whether to extract)
        - skipReason: str (reason if skipped)
        - preferences: list of extracted preferences with category, value, text, salience, type
    """
    logger.info(f"🔍 Extracting preferences from {role} message for user {user_id}")

    try:
        # Load prompty template
        template = load_prompty_template("preference_extraction.prompty")

        # Call LLM
        response_text = call_llm_with_prompt(
            template=template,
            variables={"message": message, "role": role},
            temperature=0.3
        )

        # Parse JSON response
        response_json = json.loads(response_text)

        logger.info(f"✅ Extraction complete: shouldExtract={response_json.get('shouldExtract', False)}")
        return response_json

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        return {
            "shouldExtract": False,
            "skipReason": "LLM response parsing error",
            "preferences": []
        }
    except Exception as e:
        logger.error(f"Error extracting preferences: {e}")
        return {
            "shouldExtract": False,
            "skipReason": f"Error: {str(e)}",
            "preferences": []
        }

def load_prompty_template(filename: str) -> str:
    """Load prompty file content (strips frontmatter, returns system+user sections)"""
    file_path = os.path.join(PROMPT_DIR, filename)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            # Remove frontmatter (--- ... ---) if present
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    content = parts[2].strip()
            return content
    except FileNotFoundError:
        logger.error(f"Prompty file not found: {file_path}")
        raise

def call_llm_with_prompt(template: str, variables: Dict[str, Any], temperature: float = 0.3) -> str:
    """
    Call Azure OpenAI with a prompt template and variables.

    Args:
        template: Prompt template with {{variable}} placeholders
        variables: Dictionary of variable values to substitute
        temperature: LLM temperature (default 0.3 for structured output)

    Returns:
        LLM response content as string (with markdown code blocks stripped if present)
    """
    client = get_openai_client()

    # Substitute variables in template
    prompt = template
    for key, value in variables.items():
        prompt = prompt.replace(f"{{{{{key}}}}}", str(value))

    response = client.chat.completions.create(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2000
    )

    content = response.choices[0].message.content

    # Strip markdown code blocks if present (```json ... ``` or ``` ... ```)
    if content.startswith("```"):
        # Remove opening ```json or ```
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        # Remove closing ```
        if content.endswith("```"):
            content = content[:-3]
        content = content.strip()

    return content

@mcp.tool()
def resolve_memory_conflicts(
        new_preferences: List[Dict[str, Any]],
        user_id: str,
        tenant_id: str
) -> Dict[str, Any]:
    """
    Resolve conflicts between new preferences and existing memories using LLM.

    Args:
        new_preferences: List of new preferences to check
        user_id: User identifier
        tenant_id: Tenant identifier

    Returns:
        Dictionary with:
        - resolutions: list of resolution decisions for each preference
          - conflict: bool
          - conflictsWith: str (existing memory text)
          - conflictingMemoryId: str
          - severity: none/low/high
          - decision: auto-resolve/require-confirmation
          - strategy: explanation
          - action: store-new/update-existing/store-both/ask-user
    """
    logger.info(f"⚖️  Resolving conflicts for {len(new_preferences)} preferences")

    try:
        # Query existing memories
        existing_memories = get_all_user_memories(
            user_id=user_id,
            tenant_id=tenant_id
        )

        # Format existing memories for LLM
        existing_prefs_text = "\n".join([
            f"- [{mem.get('type')}] {mem.get('text')} (salience: {mem.get('salience')}, id: {mem.get('memoryId')})"
            for mem in existing_memories
        ])

        # Format new preferences for LLM
        new_prefs_text = json.dumps(new_preferences, indent=2)

        # Load prompty template
        template = load_prompty_template("memory_conflict_resolution.prompty")

        # Call LLM
        response_text = call_llm_with_prompt(
            template=template,
            variables={
                "existing_preferences": existing_prefs_text,
                "new_preferences": new_prefs_text
            },
            temperature=0.3
        )

        # Parse JSON response
        response_json = json.loads(response_text)

        # Count severity levels
        high_severity_count = sum(1 for r in response_json.get("resolutions", []) if r.get("severity") == "high")
        low_severity_count = sum(1 for r in response_json.get("resolutions", []) if r.get("severity") == "low")

        logger.info(f"✅ Conflict resolution complete: {high_severity_count} high, {low_severity_count} low severity")

        return response_json

    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        return {"resolutions": []}
    except Exception as e:
        logger.error(f"Error resolving conflicts: {e}")
        return {"resolutions": []}

@mcp.tool()
def store_resolved_preferences(
        resolutions: List[Dict[str, Any]],
        user_id: str,
        tenant_id: str,
        justification: str
) -> Dict[str, Any]:
    """
    Store preferences that have been auto-resolved (no user confirmation needed).
    Skip preferences that require user confirmation or are duplicates.

    Args:
        resolutions: List of resolution decisions from resolve_memory_conflicts
        user_id: User identifier
        tenant_id: Tenant identifier
        justification: Source message ID or reasoning

    Returns:
        Dictionary with:
        - stored: list of stored memory IDs
        - skipped: list of preferences that were skipped (duplicates)
        - needsConfirmation: list of preferences requiring user confirmation
        - superseded: list of old memory IDs that were marked as superseded
    """
    logger.info(f"💾 Storing resolved preferences for user {user_id}")

    stored = []
    skipped = []
    needs_confirmation = []
    superseded = []

    try:
        for resolution in resolutions:
            decision = resolution.get("decision")
            action = resolution.get("action")
            new_pref = resolution.get("newPreference", {})
            strategy = resolution.get("strategy")

            if action == "skip" or decision == "skip":
                skipped.append({
                    "preference": new_pref,
                    "reason": resolution.get("strategy", "Duplicate or covered by existing memory")
                })
                logger.info(f"⏭️  Skipping duplicate preference: {new_pref.get('text')}")
                continue

            if decision == "require-confirmation":
                # Skip and add to confirmation list
                needs_confirmation.append({
                    "preference": new_pref,
                    "conflict": resolution.get("conflictsWith"),
                    "strategy": strategy
                })
                logger.info(f"⏸️  Skipping preference (needs confirmation): {new_pref.get('text')}")
                continue

            # Auto-resolve actions
            if action == "store-new":
                # Before storing, build detailed justification
                category = new_pref.get("category", "preference")
                value = new_pref.get("value", "")
                pref_text = new_pref.get("text", "")

                detailed_justification = f"User stated {category} preference: {value} - {pref_text}"
                if strategy:
                    detailed_justification += f" ({strategy})"

                # Store new preference
                memory_id = store_memory(
                    user_id=user_id,
                    tenant_id=tenant_id,
                    memory_type=new_pref.get("type", "declarative"),
                    text=new_pref.get("text"),
                    facets={new_pref.get("category"): {"value": new_pref.get("value")}},
                    salience=new_pref.get("salience", 0.7),
                    justification=detailed_justification
                )
                stored.append(memory_id)
                logger.info(f"✅ Stored new preference: {memory_id}")

            elif action == "update-existing":
                # Build detailed justification before storing
                category = new_pref.get("category", "preference")
                value = new_pref.get("value", "")
                pref_text = new_pref.get("text", "")

                detailed_justification = f"User updated {category} preference: {value} - {pref_text}"
                if strategy:
                    detailed_justification += f" ({strategy})"

                # Mark old as superseded and store new
                old_memory_id = resolution.get("conflictingMemoryId")

                # Store new preference first
                memory_id = store_memory(
                    user_id=user_id,
                    tenant_id=tenant_id,
                    memory_type=new_pref.get("type", "declarative"),
                    text=new_pref.get("text"),
                    facets={new_pref.get("category"): {"value": new_pref.get("value")}},
                    salience=new_pref.get("salience", 0.7),
                    justification=detailed_justification
                )
                stored.append(memory_id)
                logger.info(f"✅ Stored updated preference: {memory_id}")

                # Now supersede the old memory
                if old_memory_id:
                    success = supersede_memory(
                        memory_id=old_memory_id,
                        user_id=user_id,
                        tenant_id=tenant_id,
                        superseded_by=memory_id
                    )
                    if success:
                        superseded.append(old_memory_id)
                        logger.info(f"🔄 Superseded old memory: {old_memory_id} with {memory_id}")
                    else:
                        logger.warning(f"⚠️ Failed to supersede old memory: {old_memory_id}")

            elif action == "store-both":
                # Build detailed justification
                category = new_pref.get("category", "preference")
                value = new_pref.get("value", "")
                pref_text = new_pref.get("text", "")

                detailed_justification = f"User added complementary {category} preference: {value} - {pref_text}"
                if strategy:
                    detailed_justification += f" ({strategy})"

                # Store new preference (old one remains)
                memory_id = store_memory(
                    user_id=user_id,
                    tenant_id=tenant_id,
                    memory_type=new_pref.get("type", "declarative"),
                    text=new_pref.get("text"),
                    facets={new_pref.get("category"): {"value": new_pref.get("value")}},
                    salience=new_pref.get("salience", 0.7),
                    justification=detailed_justification
                )
                stored.append(memory_id)
                logger.info(f"✅ Stored complementary preference: {memory_id}")

        return {
            "stored": stored,
            "skipped": skipped,
            "needsConfirmation": needs_confirmation,
            "superseded": superseded,
            "storedCount": len(stored),
            "skippedCount": len(skipped),
            "confirmationCount": len(needs_confirmation)
        }

    except Exception as e:
        logger.error(f"Error storing preferences: {e}")
        return {
            "stored": stored,
            "skipped": skipped,
            "needsConfirmation": needs_confirmation,
            "superseded": superseded,
            "error": str(e)
        }

# ============================================================================
# 6. Summarization Tools
# ============================================================================

@mcp.tool()
def mark_span_summarized(
        session_id: str,
        tenant_id: str,
        user_id: str,
        summary_text: str,
        span: Dict[str, str],
        supersedes: List[str],
        generate_embedding_flag: bool = True
) -> Dict[str, Any]:
    """
    Atomically create summary and set TTL on source messages.

    Args:
        session_id: Session identifier
        tenant_id: Tenant identifier
        user_id: User identifier
        summary_text: Summary content
        span: Dictionary with fromMessageId and toMessageId
        supersedes: List of message IDs being superseded
        generate_embedding_flag: Whether to generate embedding (default: True)

    Returns:
        Dictionary with summaryId and metadata
    """
    logger.info(f"📝 Creating summary for session: {session_id}")

    # Get the last message being summarized to extract its timestamp
    to_message_id = span.get("toMessageId")
    last_message = get_message_by_id(
        message_id=to_message_id,
        session_id=session_id,
        tenant_id=tenant_id,
        user_id=user_id
    )

    # Extract timestamp or fallback to current time
    if last_message and last_message.get("ts"):
        last_message_ts = last_message.get("ts")
    else:
        from datetime import datetime
        last_message_ts = datetime.utcnow().isoformat() + "Z"
        logger.warning(f"Could not find timestamp for message {to_message_id}, using current time")

    summary_id = create_summary(
        session_id=session_id,
        tenant_id=tenant_id,
        user_id=user_id,
        summary_text=summary_text,
        span=span,
        summary_timestamp=last_message_ts,
        supersedes=supersedes
    )

    return {
        "summaryId": summary_id,
        "supersededCount": len(supersedes),
        "summaryTimestamp": last_message_ts
    }


@mcp.tool()
def get_summarizable_span(
        session_id: str,
        tenant_id: str,
        user_id: str,
        min_messages: int = 20,
        retention_window: int = 10
) -> Dict[str, Any]:
    """
    Return message range suitable for summarization.

    Args:
        session_id: Session identifier
        tenant_id: Tenant identifier
        user_id: User identifier
        min_messages: Minimum messages needed for summarization (default: 20)
        retention_window: Number of recent messages to keep (default: 10)

    Returns:
        Dictionary with span info and messages
    """
    logger.info(f"📊 Finding summarizable span for session: {session_id}")

    # Get all messages (excluding superseded ones)
    messages = get_session_messages(
        session_id=session_id,
        tenant_id=tenant_id,
        user_id=user_id,
        include_superseded=False
    )

    if len(messages) < min_messages:
        return {
            "canSummarize": False,
            "reason": f"Not enough messages (have {len(messages)}, need {min_messages})",
            "messageCount": len(messages)
        }

    # Keep recent messages, summarize older ones
    # Messages are returned in DESC order, so reverse for chronological
    messages_chronological = list(reversed(messages))
    messages_to_summarize = messages_chronological[:-retention_window]

    if not messages_to_summarize:
        return {
            "canSummarize": False,
            "reason": "All messages within retention window",
            "messageCount": len(messages)
        }

    return {
        "canSummarize": True,
        "span": {
            "fromMessageId": messages_to_summarize[0]["messageId"],
            "toMessageId": messages_to_summarize[-1]["messageId"]
        },
        "messageCount": len(messages_to_summarize),
        "totalMessages": len(messages),
        "retentionWindow": retention_window
    }


@mcp.tool()
def get_all_user_summaries(
        user_id: str,
        tenant_id: str
) -> List[Dict[str, Any]]:
    """
    Retrieve all conversation summaries for a user across all sessions.
    Useful when user asks "Show me my past trips" or "What have we discussed before?".

    Args:
        user_id: User identifier
        tenant_id: Tenant identifier

    Returns:
        List of summary objects containing sessionId, text, and createdAt

    """
    logger.info(f"📚 Retrieving all summaries for user: {user_id}")

    summaries = get_user_summaries(
        user_id=user_id,
        tenant_id=tenant_id
    )

    # Return simplified format for agent consumption
    return [
        {
            "summaryId": s.get("summaryId"),
            "sessionId": s.get("sessionId"),
            "text": s.get("text"),
            "createdAt": s.get("createdAt"),
            "span": s.get("span")
        }
        for s in summaries
    ]

# ============================================================================
# Server Startup
# ============================================================================

if __name__ == "__main__":
    print("Starting Travel Assistant MCP server...")

    # Configure server options
    server_options = {
        "transport": "streamable-http"
    }

    print("🔓 Starting server without built-in authentication...")
    print("💡 For OAuth, use a reverse proxy like nginx or API gateway")

    try:
        mcp.run(**server_options)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)    