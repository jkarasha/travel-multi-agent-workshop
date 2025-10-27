import sys
import os
import logging
from typing import Any, Dict, List, Optional
from langsmith import traceable
from mcp.server.fastmcp import FastMCP

from src.app.services.azure_open_ai import generate_embedding
from src.app.services.azure_cosmos_db import (
    create_session_record,
    get_session_by_id,
    append_message,
    get_session_messages,
    get_session_summaries,
    create_summary,
    store_memory,
    query_memories,
    query_places,
    create_trip,
    get_trip,
    record_api_event,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv('.env.oauth', override=False)
    
    # Load authentication configuration
    simple_token = os.getenv("MCP_AUTH_TOKEN")
    github_client_id = os.getenv("GITHUB_CLIENT_ID")
    github_client_secret = os.getenv("GITHUB_CLIENT_SECRET")
    base_url = os.getenv("MCP_SERVER_BASE_URL", "http://localhost:8080")
    
    print("🔐 Authentication Configuration:")
    print(f"   Simple Token: {'SET' if simple_token else 'NOT SET'}")
    print(f"   GitHub Client ID: {'SET' if github_client_id else 'NOT SET'}")
    print(f"   Base URL: {base_url}")
    
    # Determine authentication mode
    if github_client_id and github_client_secret:
        auth_mode = "github_oauth"
        print("✅ GITHUB OAUTH MODE ENABLED")
    elif simple_token:
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
# 1. Session Management Tools
# ============================================================================

@mcp.tool()
@traceable
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
@traceable
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


@mcp.tool()
@traceable
def append_turn(
    session_id: str,
    tenant_id: str,
    user_id: str,
    role: str,
    content: str,
    tool_call: Optional[Dict] = None,
    keywords: Optional[List[str]] = None,
    generate_embedding_flag: bool = True
) -> Dict[str, Any]:
    """
    Atomically store a message and update session metadata.
    
    Args:
        session_id: Session identifier
        tenant_id: Tenant identifier
        user_id: User identifier
        role: Message role (user/assistant/system)
        content: Message content
        tool_call: Optional tool call information
        keywords: Optional list of keywords
        generate_embedding_flag: Whether to generate embedding (default: True)
        
    Returns:
        Dictionary with messageId and metadata
    """
    logger.info(f"💬 Appending {role} message to session: {session_id}")
    
    # Generate embedding if requested
    embedding = None
    if generate_embedding_flag and content:
        try:
            embedding = generate_embedding(content)
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
    
    message_id = append_message(
        session_id=session_id,
        tenant_id=tenant_id,
        user_id=user_id,
        role=role,
        content=content,
        tool_call=tool_call,
        embedding=embedding,
        keywords=keywords
    )
    
    return {
        "messageId": message_id,
        "sessionId": session_id,
        "role": role,
        "embeddingGenerated": embedding is not None
    }
# ============================================================================
# 2. Memory Lifecycle Tools
# ============================================================================

@mcp.tool()
@traceable
def store_user_memory(
    user_id: str,
    tenant_id: str,
    memory_type: str,
    text: str,
    facets: Dict[str, Any],
    salience: float,
    justification: str,
    generate_embedding_flag: bool = True
) -> Dict[str, Any]:
    """
    Store a user memory with appropriate TTL and indexing.
    
    Args:
        user_id: User identifier
        tenant_id: Tenant identifier
        memory_type: Type of memory (declarative/episodic/procedural)
        text: Memory text content
        facets: Structured facets (dietary, mobility, timeOfDay, etc.)
        salience: Importance score (0.0-1.0)
        justification: Source message ID or reasoning
        generate_embedding_flag: Whether to generate embedding (default: True)
        
    Returns:
        Dictionary with memoryId and metadata
    """
    logger.info(f"🧠 Storing {memory_type} memory for user: {user_id}")
    
    # Validate memory type
    if memory_type not in ["declarative", "episodic", "procedural"]:
        raise ValueError(f"Invalid memory type: {memory_type}")
    
    # Generate embedding if requested
    embedding = None
    if generate_embedding_flag and text:
        try:
            embedding = generate_embedding(text)
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
    
    memory_id = store_memory(
        user_id=user_id,
        tenant_id=tenant_id,
        memory_type=memory_type,
        text=text,
        facets=facets,
        salience=salience,
        justification=justification,
        embedding=embedding
    )
    
    return {
        "memoryId": memory_id,
        "type": memory_type,
        "salience": salience,
        "embeddingGenerated": embedding is not None
    }


@mcp.tool()
@traceable
def recall_memories(
    user_id: str,
    tenant_id: str,
    query: Optional[str] = None,
    memory_types: Optional[List[str]] = None,
    min_salience: float = 0.0
) -> List[Dict[str, Any]]:
    """
    Smart hybrid retrieval of relevant memories.
    
    Args:
        user_id: User identifier
        tenant_id: Tenant identifier
        query: Optional search query
        memory_types: Optional filter by memory types
        min_salience: Minimum salience threshold (default: 0.0)
        
    Returns:
        List of memory dictionaries with scores and match reasons
    """
    logger.info(f"🔍 Recalling memories for user: {user_id}")
    
    # TODO: Implement hybrid search with vector similarity
    # For now, return top memories by salience
    memories = query_memories(
        user_id=user_id,
        tenant_id=tenant_id,
        memory_types=memory_types,
        min_salience=min_salience
    )
    
    return memories


# ============================================================================
# 3. Summarization Tools
# ============================================================================

@mcp.tool()
@traceable
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
    
    # Generate embedding if requested
    embedding = None
    if generate_embedding_flag and summary_text:
        try:
            embedding = generate_embedding(summary_text)
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
    
    summary_id = create_summary(
        session_id=session_id,
        tenant_id=tenant_id,
        user_id=user_id,
        summary_text=summary_text,
        span=span,
        embedding=embedding,
        supersedes=supersedes
    )
    
    return {
        "summaryId": summary_id,
        "supersededCount": len(supersedes),
        "embeddingGenerated": embedding is not None
    }


@mcp.tool()
@traceable
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


# ============================================================================
# 4. Place Discovery Tools
# ============================================================================

@mcp.tool()
@traceable
def discover_places(
    geo_scope: str,
    query: str,
    user_id: str,
    tenant_id: str = "",
    filters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Memory-aware place search with hybrid retrieval.
    
    Args:
        geo_scope: Geographic scope (e.g., "barcelona")
        query: Search query
        user_id: User identifier (for memory alignment)
        tenant_id: Tenant identifier
        filters: Optional filters (type, dietary, priceTier, etc.)
        
    Returns:
        List of places with match reasons and memory alignment scores
    """
    logger.info(f"🗺️  ========== DISCOVER_PLACES TOOL CALLED ==========")
    logger.info(f"🗺️  Parameters:")
    logger.info(f"     - geo_scope: {geo_scope}")
    logger.info(f"     - query: {query}")
    logger.info(f"     - user_id: {user_id}")
    logger.info(f"     - tenant_id: {tenant_id}")
    logger.info(f"     - filters: {filters}")
    
    # Get user memories for alignment
    logger.info(f"🧠 Recalling user memories...")
    memories = recall_memories(
        user_id=user_id,
        tenant_id=tenant_id,
        query=query,
        memory_types=["declarative", "episodic"]
    )
    logger.info(f"🧠 Found {len(memories)} memories")
    
    # Parse filters - convert list to string if needed (defensive programming)
    place_type = filters.get("type") if filters else None
    if isinstance(place_type, list):
        if place_type:
            logger.warning(f"⚠️  place_type passed as list {place_type}, using first element")
            place_type = place_type[0]
        else:
            place_type = None
    # Handle pipe-separated types (e.g. "restaurant|cafe") - take first one
    if place_type and "|" in place_type:
        types = place_type.split("|")
        logger.warning(f"⚠️  place_type contains pipe-separated values {types}, using first: '{types[0]}'")
        place_type = types[0]
    
    dietary = filters.get("dietary") if filters else None
    if isinstance(dietary, list):
        if dietary:
            logger.warning(f"⚠️  dietary passed as list {dietary}, using first element")
            dietary = dietary[0]
        else:
            dietary = None
    
    price_tier = filters.get("priceTier") if filters else None
    
    logger.info(f"🔍 Parsed filters:")
    logger.info(f"     - place_type: {place_type}")
    logger.info(f"     - dietary: {dietary}")
    logger.info(f"     - price_tier: {price_tier}")

    logger.info(f"🔢 Generating embedding for query...")
    try:
        vectors = generate_embedding(query)
        logger.info(f"✅ Embedding generated successfully (dimension: {len(vectors)})")
    except Exception as e:
        logger.error(f"❌ Failed to generate embedding: {e}")
        raise
    
    # Query places
    logger.info(f"🔍 Calling query_places from Cosmos DB...")
    try:
        places = query_places(
            vectors=vectors,
            geo_scope_id=geo_scope,
            place_type=place_type,
            dietary=dietary,
            price_tier=price_tier,
        )
        logger.info(f"✅ query_places returned {len(places)} results")
        if places:
            logger.info(f"📍 Sample place: {places[0].get('name', 'N/A')} (type: {places[0].get('type', 'N/A')})")
        else:
            logger.warning(f"⚠️  No places returned from query_places!")
    except Exception as e:
        logger.error(f"❌ Error in query_places: {e}")
        logger.error(f"❌ Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        raise
    
    # Memory alignment scoring and match reason generation
    logger.info(f"🎯 Starting memory alignment scoring...")
    for idx, place in enumerate(places):
        alignment_score = 0.0
        match_reasons = []
        
        # Base reason: Vector similarity to query
        if place.get("similarityScore"):
            match_reasons.append(f"Semantic match (score: {place['similarityScore']:.2f})")
        
        # Check alignment with user memories
        if memories:
            for memory in memories:
                memory_facets = memory.get("facets", {})
                
                # Dietary alignment
                if "dietary" in memory_facets:
                    memory_dietary = memory_facets["dietary"]
                    place_dietary = place.get("restaurantSpecific", {}).get("dietaryOptions", [])
                    if memory_dietary in place_dietary:
                        alignment_score += 0.3
                        match_reasons.append(f"Matches your {memory_dietary} preference")
                
                # Price tier alignment
                if "priceTier" in memory_facets:
                    memory_price = memory_facets["priceTier"]
                    place_price = place.get("priceTier")
                    if memory_price == place_price:
                        alignment_score += 0.2
                        match_reasons.append(f"Matches your {place_price} preference")
                
                # Ambiance/style alignment
                if "style" in memory_facets:
                    memory_style = memory_facets["style"]
                    place_amenities = place.get("hotelSpecific", {}).get("amenities", [])
                    place_categories = place.get("activitySpecific", {}).get("categories", [])
                    if memory_style in place_amenities or memory_style in place_categories:
                        alignment_score += 0.2
                        match_reasons.append(f"Matches your preference for {memory_style}")
                
                # Accessibility alignment
                if "accessibility" in memory_facets:
                    memory_access = memory_facets["accessibility"]
                    place_access = place.get("accessibility", [])
                    if memory_access in place_access:
                        alignment_score += 0.3
                        match_reasons.append(f"Accessible: {memory_access}")
        
        # Location/geo match
        if geo_scope.lower() in place.get("geoScope", "").lower():
            match_reasons.append(f"Located in {geo_scope}")
        
        # If no specific reasons found, add generic reason
        if not match_reasons:
            match_reasons.append("Location match")
        
        # Cap alignment score at 1.0
        place["memoryAlignment"] = min(alignment_score, 1.0)
        place["matchReasons"] = match_reasons
        
        if idx == 0:  # Log first place details
            logger.info(f"📍 Place {idx+1}: {place.get('name')} - alignment: {place['memoryAlignment']:.2f}, reasons: {match_reasons}")
    
    # Sort by combined score (similarity + memory alignment)
    logger.info(f"📊 Sorting places by combined score...")
    places.sort(
        key=lambda p: (p.get("similarityScore", 0.0) * 0.6) + (p.get("memoryAlignment", 0.0) * 0.4),
        reverse=True
    )
    
    logger.info(f"✅ discover_places returning {len(places)} places")
    logger.info(f"🗺️  ========== DISCOVER_PLACES TOOL COMPLETED ==========")
    return places


# ============================================================================
# 5. Trip Management Tools
# ============================================================================

@mcp.tool()
@traceable
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
@traceable
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
@traceable
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
    from src.app.services.azure_cosmos_db import trips_container
    if trips_container:
        trips_container.upsert_item(trip)
    
    return trip


# ============================================================================
# 6. Cross-Thread Search Tools
# ============================================================================

@mcp.tool()
@traceable
def search_user_threads(
    user_id: str,
    tenant_id: str,
    query: str,
    mode: str = "hybrid",
    since: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Hybrid search across user's conversation history.
    
    Args:
        user_id: User identifier
        tenant_id: Tenant identifier
        query: Search query
        mode: Search mode (hybrid/semantic/fulltext)
        since: Optional ISO date to filter recent conversations
        
    Returns:
        List of matches grouped by thread with scores
    """
    logger.info(f"🔍 Searching user threads for: {query}")
    
    from src.app.services.azure_cosmos_db import messages_container
    
    if not messages_container:
        return []
    
    # Generate query embedding for semantic search
    query_embedding = None
    if mode in ["hybrid", "semantic"]:
        try:
            query_embedding = generate_embedding(query)
        except Exception as e:
            logger.warning(f"Failed to generate query embedding: {e}")
    
    # Search messages (simplified - full implementation would use vector search)
    query_filter = """
    SELECT TOP 10 c.threadId, c.messageId, c.content, c.ts, c.role
    FROM c 
    WHERE c.userId = @userId 
    AND c.tenantId = @tenantId
    AND CONTAINS(LOWER(c.content), LOWER(@query))
    ORDER BY c.ts DESC
    """
    
    params = [
        {"name": "@userId", "value": user_id},
        {"name": "@tenantId", "value": tenant_id},
        {"name": "@query", "value": query},
    ]
    
    if since:
        query_filter = query_filter.replace(
            "ORDER BY",
            "AND c.ts >= @since ORDER BY"
        )
        params.append({"name": "@since", "value": since})
    
    results = list(messages_container.query_items(
        query=query_filter,
        parameters=params,
        enable_cross_partition_query=True
    ))
    
    # Group by thread
    threads_map = {}
    for msg in results:
        thread_id = msg["threadId"]
        if thread_id not in threads_map:
            threads_map[thread_id] = {
                "threadId": thread_id,
                "matches": [],
                "totalScore": 0.0
            }
        
        threads_map[thread_id]["matches"].append({
            "messageId": msg["messageId"],
            "content": msg["content"],
            "timestamp": msg["ts"],
            "role": msg["role"],
            "score": 0.8  # Placeholder
        })
        threads_map[thread_id]["totalScore"] += 0.8
    
    return list(threads_map.values())


# ============================================================================
# 7. API Event Tools
# ============================================================================

@mcp.tool()
@traceable
def record_api_call(
    session_id: str,
    tenant_id: str,
    provider: str,
    operation: str,
    request: Dict[str, Any],
    response: Dict[str, Any],
    keywords: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Store API event with auto-extracted keywords.
    
    Args:
        session_id: Session identifier
        tenant_id: Tenant identifier
        provider: API provider name (e.g., "FlightsAPI")
        operation: Operation name (e.g., "search")
        request: Request parameters
        response: Response data
        keywords: Optional list of keywords
        
    Returns:
        Dictionary with eventId and metadata
    """
    logger.info(f"📡 Recording API call: {provider}.{operation}")
    
    event_id = record_api_event(
        session_id=session_id,
        tenant_id=tenant_id,
        provider=provider,
        operation=operation,
        request=request,
        response=response,
        keywords=keywords
    )
    
    return {
        "eventId": event_id,
        "provider": provider,
        "operation": operation
    }


# ============================================================================
# 9. Agent Transfer Tools (for Orchestrator Routing)
# ============================================================================

@mcp.tool()
@traceable
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
    import json
    logger.info(f"🔄 Transfer to Hotel Agent: {reason}")
    
    return json.dumps({
        "goto": "hotel",
        "reason": reason,
        "message": "Transferring to Hotel Agent to find accommodations for you."
    })


@mcp.tool()
@traceable
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
    import json
    logger.info(f"🔄 Transfer to Activity Agent: {reason}")
    
    return json.dumps({
        "goto": "activity",
        "reason": reason,
        "message": "Transferring to Activity Agent to discover attractions for you."
    })


@mcp.tool()
@traceable
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
    import json
    logger.info(f"🔄 Transfer to Dining Agent: {reason}")
    
    return json.dumps({
        "goto": "dining",
        "reason": reason,
        "message": "Transferring to Dining Agent to find restaurants for you."
    })


@mcp.tool()
@traceable
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
    import json
    logger.info(f"🔄 Transfer to Itinerary Generator: {reason}")
    
    return json.dumps({
        "goto": "itinerary_generator",
        "reason": reason,
        "message": "Transferring to Itinerary Generator to create your day-by-day plan."
    })


@mcp.tool()
@traceable
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
    import json
    logger.info(f"🔄 Transfer to Summarizer: {reason}")
    
    return json.dumps({
        "goto": "summarizer",
        "reason": reason,
        "message": "Transferring to Summarizer to compress and recap our conversation."
    })


@mcp.tool()
@traceable
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
    import json
    logger.info(f"🔄 Transfer to Orchestrator: {reason}")
    
    return json.dumps({
        "goto": "orchestrator",
        "reason": reason,
        "message": "Transferring back to Orchestrator for general assistance."
    })


# ============================================================================
# Server Startup
# ============================================================================

if __name__ == "__main__":
    print("Starting Banking Tools MCP server...")

    # Configure server options
    server_options = {
        "transport": "streamable-http"
    }

    print("� Starting server without built-in authentication...")
    print("💡 For OAuth, use a reverse proxy like nginx or API gateway")

    try:
        mcp.run(**server_options)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)
