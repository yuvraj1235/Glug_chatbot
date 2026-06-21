# Graph Report - Glug_chatbot  (2026-06-21)

## Corpus Check
- 11 files · ~1,921 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 52 nodes · 54 edges · 10 communities (8 shown, 2 thin omitted)
- Extraction: 87% EXTRACTED · 13% INFERRED · 0% AMBIGUOUS · INFERRED: 7 edges (avg confidence: 0.63)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `95684892`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]

## God Nodes (most connected - your core abstractions)
1. `Glug Chatbot Backend` - 7 edges
2. `LLMService` - 6 edges
3. `chat()` - 5 edges
4. `ChatRequest` - 5 edges
5. `clean_html()` - 4 edges
6. `extract_text_deduplicated()` - 4 edges
7. `scrape_all_endpoints()` - 4 edges
8. `get_pyq_response()` - 4 edges
9. `Setup & Installation` - 4 edges
10. `1. `POST /api/chat`` - 4 edges

## Surprising Connections (you probably didn't know these)
- `chat()` --calls--> `ChatResponse`  [INFERRED]
  chatbot-backend/app/routes/chat.py → chatbot-backend/app/schemas/chat.py
- `chat()` --calls--> `get_pyq_response()`  [INFERRED]
  chatbot-backend/app/routes/chat.py → chatbot-backend/app/services/pyq.py
- `ChatRequest` --uses--> `LLMService`  [INFERRED]
  chatbot-backend/app/routes/chat.py → chatbot-backend/app/services/llm.py
- `LLMService` --uses--> `LLMService`  [INFERRED]
  chatbot-backend/app/routes/chat.py → chatbot-backend/app/services/llm.py
- `ingest_data()` --calls--> `scrape_all_endpoints()`  [INFERRED]
  chatbot-backend/app/routes/ingest.py → chatbot-backend/app/services/api_scraper.py

## Import Cycles
- None detected.

## Communities (10 total, 2 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.24
Nodes (8): str, ingest_data(), clean_html(), extract_text_deduplicated(), Removes HTML tags and normalizes whitespace., Recursively extracts all meaningful text values from nested JSON data strings,, Scrapes all endpoints, processes data into vector documents, and stores them in, scrape_all_endpoints()

### Community 1 - "Community 1"
Cohesion: 0.29
Nodes (6): str, ChatRequest, LLMService, chat(), get_llm_service(), LLMService

### Community 2 - "Community 2"
Cohesion: 0.20
Nodes (9): 1. Prerequisites, 2. Install Dependencies, 3. Configure Environment Variables, Directory Structure, Features, Glug Chatbot Backend, License, Running the Server (+1 more)

### Community 3 - "Community 3"
Cohesion: 0.33
Nodes (6): 1. `POST /api/chat`, 2. `GET /api/chat/models`, API Endpoints, Request Body, Standard Response, Streaming Response (SSE Format)

### Community 4 - "Community 4"
Cohesion: 0.67
Nodes (3): BaseModel, ChatRequest, ChatResponse

### Community 5 - "Community 5"
Cohesion: 0.50
Nodes (3): str, get_pyq_response(), Checks if the user's message is asking for PYQs, Previous Year Papers,     Study

## Knowledge Gaps
- **15 isolated node(s):** `str`, `str`, `str`, `Glug_chatbot`, `Features` (+10 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `chat()` connect `Community 1` to `Community 4`, `Community 5`?**
  _High betweenness centrality (0.167) - this node is a cross-community bridge._
- **Are the 2 inferred relationships involving `LLMService` (e.g. with `ChatRequest` and `LLMService`) actually correct?**
  _`LLMService` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 2 inferred relationships involving `chat()` (e.g. with `ChatResponse` and `get_pyq_response()`) actually correct?**
  _`chat()` has 2 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `ChatRequest` (e.g. with `ChatRequest` and `LLMService`) actually correct?**
  _`ChatRequest` has 3 INFERRED edges - model-reasoned connections that need verification._
- **What connects `str`, `Removes HTML tags and normalizes whitespace.`, `Recursively extracts all meaningful text values from nested JSON data strings,` to the rest of the system?**
  _19 weakly-connected nodes found - possible documentation gaps or missing edges._