# Hotel Agent Flow: "Recommend hotels in Paris"

This document describes the end-to-end flow when a user asks for hotel recommendations.

---

## Simple Flow (Step-by-Step)

1. **You type** "Recommend hotels in Paris" in the Angular UI.
2. **Frontend → Backend (FastAPI :8000)**: it POSTs to  
   `POST /tenant/{tenantId}/user/{userId}/sessions/{sessionId}/completion` with your message text.
3. **Backend loads the LangGraph workflow** (or returns `503` if agents/MCP aren't ready), then starts/resumes the conversation state (checkpoint).
4. **Orchestrator agent runs first**:
   - Decides this is a hotel task.
   - Calls the MCP tool **`transfer_to_hotel`** (returns a small JSON like `{"goto":"hotel", ...}`).
5. **LangGraph router sees `goto=hotel`** (from the tool output) and routes to the **Hotel agent** node.
6. **Hotel agent calls MCP tools** (over HTTP to MCP server :8080), typically:
   - **`recall_memories(user_id, tenant_id, query=...)`** to pull your stored preferences (budget, neighborhood, etc.)
   - **`discover_places(geo_scope="paris", query="hotel recommendations ...", filters={"type":"hotel", ...})`**
7. **MCP server executes those tools**:
   - Queries **Cosmos DB** for hotels (hybrid/vector search) and for relevant memories.
   - Returns a ranked list of hotel candidates + match reasons / memory alignment.
8. **Hotel agent writes the response** (recommendations + reasoning, maybe follow-up questions).
9. **Backend stores messages/debug info in Cosmos**, then returns the updated message list to the frontend.
10. **Frontend renders** the assistant response in the chat.

---

## Sequence Diagram

```
┌──────────┐     ┌───────────┐     ┌──────────────┐     ┌────────────┐     ┌───────────┐     ┌──────────┐
│  Browser │     │  Angular  │     │   FastAPI    │     │  LangGraph │     │    MCP    │     │ Cosmos / │
│   (You)  │     │  :4200    │     │    :8000     │     │  Workflow  │     │   :8080   │     │  OpenAI  │
└────┬─────┘     └─────┬─────┘     └──────┬───────┘     └─────┬──────┘     └─────┬─────┘     └────┬─────┘
     │                 │                  │                   │                  │                │
     │ "Recommend      │                  │                   │                  │                │
     │  hotels in      │                  │                   │                  │                │
     │  Paris"         │                  │                   │                  │                │
     │────────────────>│                  │                   │                  │                │
     │                 │                  │                   │                  │                │
     │                 │ POST /sessions/  │                   │                  │                │
     │                 │ {id}/completion  │                   │                  │                │
     │                 │─────────────────>│                   │                  │                │
     │                 │                  │                   │                  │                │
     │                 │                  │ workflow.ainvoke()│                  │                │
     │                 │                  │──────────────────>│                  │                │
     │                 │                  │                   │                  │                │
     │                 │                  │         ┌─────────┴─────────┐        │                │
     │                 │                  │         │   ORCHESTRATOR    │        │                │
     │                 │                  │         │ "This is a hotel  │        │                │
     │                 │                  │         │  request..."      │        │                │
     │                 │                  │         └─────────┬─────────┘        │                │
     │                 │                  │                   │                  │                │
     │                 │                  │                   │ transfer_to_     │                │
     │                 │                  │                   │ hotel(reason)    │                │
     │                 │                  │                   │─────────────────>│                │
     │                 │                  │                   │                  │                │
     │                 │                  │                   │<─ {"goto":"hotel"}                │
     │                 │                  │                   │                  │                │
     │                 │                  │         ┌─────────┴─────────┐        │                │
     │                 │                  │         │   HOTEL AGENT     │        │                │
     │                 │                  │         │ (specialized)     │        │                │
     │                 │                  │         └─────────┬─────────┘        │                │
     │                 │                  │                   │                  │                │
     │                 │                  │                   │ recall_memories  │                │
     │                 │                  │                   │ (user prefs)     │                │
     │                 │                  │                   │─────────────────>│                │
     │                 │                  │                   │                  │ query_memories │
     │                 │                  │                   │                  │───────────────>│
     │                 │                  │                   │                  │<── memories ───│
     │                 │                  │                   │<─ user prefs ────│                │
     │                 │                  │                   │                  │                │
     │                 │                  │                   │ discover_places  │                │
     │                 │                  │                   │ (paris, hotel)   │                │
     │                 │                  │                   │─────────────────>│                │
     │                 │                  │                   │                  │ hybrid_search  │
     │                 │                  │                   │                  │───────────────>│
     │                 │                  │                   │                  │<── hotels ─────│
     │                 │                  │                   │<─ ranked hotels ─│                │
     │                 │                  │                   │                  │                │
     │                 │                  │         ┌─────────┴─────────┐        │                │
     │                 │                  │         │  Hotel agent      │        │                │
     │                 │                  │         │  generates reply  │        │                │
     │                 │                  │         │  with Azure OpenAI│        │                │
     │                 │                  │         └─────────┬─────────┘        │                │
     │                 │                  │                   │                  │                │
     │                 │                  │<── response ──────│                  │                │
     │                 │                  │                   │                  │                │
     │                 │                  │ store messages    │                  │                │
     │                 │                  │ in Cosmos DB      │                  │                │
     │                 │                  │───────────────────────────────────────────────────────>│
     │                 │                  │                   │                  │                │
     │                 │<─ [messages] ────│                   │                  │                │
     │                 │                  │                   │                  │                │
     │<── render ──────│                  │                   │                  │                │
     │   response      │                  │                   │                  │                │
     │                 │                  │                   │                  │                │
```

---

## Simplified Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              USER SENDS MESSAGE                                  │
│                        "Recommend hotels in Paris"                               │
└────────────────────────────────────┬────────────────────────────────────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │       Angular Frontend         │
                    │   POST /api/.../completion     │
                    └────────────────┬───────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │      FastAPI Backend :8000     │
                    │  • Load checkpoint (resume)    │
                    │  • Call workflow.ainvoke()     │
                    └────────────────┬───────────────┘
                                     │
           ┌─────────────────────────┼─────────────────────────┐
           │                  LangGraph Workflow               │
           │                                                   │
           │   ┌─────────────────────────────────────────┐     │
           │   │            ORCHESTRATOR                 │     │
           │   │  • Analyzes: "hotel" intent             │     │
           │   │  • Calls: transfer_to_hotel() via MCP   │     │
           │   └──────────────────┬──────────────────────┘     │
           │                      │ goto: "hotel"              │
           │                      ▼                            │
           │   ┌─────────────────────────────────────────┐     │
           │   │            HOTEL AGENT                  │     │
           │   │  • Calls: recall_memories() via MCP     │     │
           │   │  • Calls: discover_places() via MCP     │     │
           │   │  • Generates response (Azure OpenAI)    │     │
           │   └──────────────────┬──────────────────────┘     │
           └──────────────────────┼────────────────────────────┘
                                  │
                                  ▼
                    ┌────────────────────────────────┐
                    │       MCP Server :8080         │
                    │  • recall_memories → Cosmos    │
                    │  • discover_places → Cosmos    │
                    │    (hybrid vector search)      │
                    └────────────────┬───────────────┘
                                     │
                                     ▼
           ┌─────────────────────────────────────────────────┐
           │              Azure Services                     │
           │  ┌──────────────────┐  ┌──────────────────┐     │
           │  │   Cosmos DB      │  │   Azure OpenAI   │     │
           │  │ • places         │  │ • embeddings     │     │
           │  │ • memories       │  │ • chat model     │     │
           │  │ • sessions       │  │   (gpt-4o, etc.) │     │
           │  └──────────────────┘  └──────────────────┘     │
           └─────────────────────────────────────────────────┘
                                     │
                                     ▼
                    ┌────────────────────────────────┐
                    │     RESPONSE TO FRONTEND       │
                    │ "Here are 5 hotels in Paris:   │
                    │  1. Hotel Le Marais ...        │
                    │  2. Boutique Montmartre ..."   │
                    └────────────────────────────────┘
```

---

## Key Takeaways

| Step | Component | What happens |
|------|-----------|--------------|
| 1 | **Angular** | Sends user message to FastAPI |
| 2 | **FastAPI** | Resumes LangGraph state, invokes workflow |
| 3 | **Orchestrator** | Classifies intent → calls `transfer_to_hotel` (MCP tool) |
| 4 | **Router** | Sees `goto: hotel`, routes to Hotel agent |
| 5 | **Hotel agent** | Calls `recall_memories` + `discover_places` (MCP tools) |
| 6 | **MCP server** | Executes Cosmos queries (vector/hybrid search) |
| 7 | **Hotel agent** | Composes final answer using Azure OpenAI |
| 8 | **FastAPI** | Stores messages, returns response to frontend |
| 9 | **Angular** | Renders the assistant reply in chat UI |

---

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────────────────┐
│                           TRAVEL ASSISTANT                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│   Browser ──► Angular :4200 ──► FastAPI :8000 ──► LangGraph Workflow       │
│                                       │                  │                 │
│                                       │           ┌──────┴──────┐          │
│                                       │           │ Agents:     │          │
│                                       │           │ Orchestrator│          │
│                                       │           │ Hotel       │          │
│                                       │           │ Activity    │          │
│                                       │           │ Dining      │          │
│                                       │           │ Itinerary   │          │
│                                       │           │ Summarizer  │          │
│                                       │           └──────┬──────┘          │
│                                       │                  │                 │
│                                       │                  ▼                 │
│                                       │           MCP Server :8080         │
│                                       │           (Tools Gateway)          │
│                                       │                  │                 │
│                                       ▼                  ▼                 │
│                              ┌────────────────────────────────┐            │
│                              │        Azure Services          │            │
│                              │  • Cosmos DB (data storage)    │            │
│                              │  • Azure OpenAI (LLM + embed)  │            │
│                              └────────────────────────────────┘            │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

