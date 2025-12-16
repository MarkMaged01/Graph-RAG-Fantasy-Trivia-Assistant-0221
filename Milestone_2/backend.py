import streamlit as st
import os
from input_processing import FPLPreprocessor
# This import must match your actual filename (graph_retrieval.py)
from graph_retrieval import GraphRetriever 
from layer2_embeddings import EmbeddingRetriever, model1, model2, driver
from layer3LLM import HFModels, merge_context, build_prompt, run_model

HF_TOKEN = os.getenv("HF_TOKEN")

@st.cache_resource
def get_backend_components():
    """
    Initializes the heavy AI models only ONCE.
    Streamlit will remember this result for future interactions.
    """
    print("⏳ Initializing backend (this should happen only once)...")
    try:
        # 1. Load NLP Processor (Layer 1)
        proc = FPLPreprocessor()
        
        # 2. Load Graph Connection (Layer 2.a – Baseline KG)
        retr = GraphRetriever()

        # 3. Load Embedding Retriever (Layer 2.b – Vector / Hybrid)
        emb_retr = EmbeddingRetriever(driver, model1, model2)
        
        # 4. Load LLM (Layer 3 – slow part we want to cache)
        llm_instance = HFModels(token=HF_TOKEN)
        llm_instance.load_models(only=["mistral"]) 
        
        print("✅ Backend ready!")
        return proc, retr, emb_retr, llm_instance
        
    except Exception as e:
        print(f"❌ Backend initialization failed: {e}")
        return None, None, None, None

# Load the global instances
processor, retriever, embedding_retriever, llm = get_backend_components()

# ==========================================
# QUERY PROCESSING LOGIC
# ==========================================
def process_query(query_text, model_name="mistral", retrieval_mode="baseline", embedding_model="minilm"):
    """
    Orchestrates RAG with selectable Model and Retrieval Mode.
    """
    response = {
        "intent": "Unknown",
        "entities": {},
        "kg_context": [],
        "llm_answer": "Error generating answer.",
        "cypher_query": "N/A",
        "layer_details": {
            "layer1": {},
            "layer2": {},
            "layer3": {}
        }
    }

    # Safety check: If backend failed to load
    if not processor or not retriever or not llm:
        response["llm_answer"] = "Backend failed to initialize. Check terminal logs."
        return response

    try:
        # --- LAYER 1: PREPROCESSING ---
        intent = processor.classify_intent(query_text)
        entities = processor.extract_entities(query_text)
        
        response["intent"] = intent
        response["entities"] = entities
        
        # Track Layer 1 details
        response["layer_details"]["layer1"] = {
            "input": query_text,
            "output": {
                "intent": intent,
                "entities": entities
            },
            "description": "Processed user query to detect intent and extract entities (players, teams, positions, etc.)"
        }

        # --- LAYER 2: RETRIEVAL ---
        graph_data = []
        embeddings_data = []
        retrieval_method_used = None
        cypher_query = "N/A"
        query_params = {}
        result_shape = {}

        # Mode A: Baseline (Graph Only) or Hybrid
        if retrieval_mode in ["Baseline (Graph)", "Hybrid", "baseline"]:
            retrieval_result = retriever.route_query(intent, entities)
            retrieval_method_used = "Graph (Neo4j Cypher Query)"
            
            # Handle new format (dict with data, query, params, result_shape)
            if isinstance(retrieval_result, dict):
                graph_data = retrieval_result.get("data", [])
                cypher_query = retrieval_result.get("query", "N/A")
                query_params = retrieval_result.get("params", {})
                result_shape = retrieval_result.get("result_shape", {})
            else:
                # Fallback for old format (just list of data)
                graph_data = retrieval_result if isinstance(retrieval_result, list) else []
            
            # Normalize keys for UI visualization
            normalized_data = []
            if graph_data:
                for item in graph_data:
                    # Normalized data is used for the Graph Viz in app.py
                    normalized_data.append(item)
            response["kg_context"] = normalized_data
            response["cypher_query"] = cypher_query

        # Mode B: Embeddings (Vector Only) or Hybrid
        if retrieval_mode in ["Embeddings (Vector)", "Hybrid"] and embedding_retriever:
            # If we are purely in embedding mode, override description.
            if retrieval_mode == "Embeddings (Vector)":
                retrieval_method_used = "Vector Embeddings"
            elif retrieval_mode == "Hybrid":
                retrieval_method_used = "Hybrid (Graph + Vector)"

            # Use Layer 2.b to get embedding-based contextual snippets
            # ... inside Mode B block ...
            try:
                # Map UI names to model keys if necessary, or pass directly
                # UI might send "MiniLM" or "MPNet", map them to "minilm" or "mpnet"
                model_key_map = {
                    "MiniLM": "minilm",
                    "MPNet": "mpnet",
                    "minilm": "minilm",
                    "mpnet": "mpnet"
                }

                selected_key = model_key_map.get(embedding_model, "minilm")

                embeddings_data = embedding_retriever.get_embedding_context_from_query(
                    query_text=query_text,
                    model_key=selected_key, # <--- USE DYNAMIC PARAMETER
                    top_k=10,
                )
            except Exception as e:
                # Fail soft – keep going with whatever graph data we have
                print(f"Embedding retrieval error: {e}")

        # Track Layer 2 details
        response["layer_details"]["layer2"] = {
            "input": {
                "intent": intent,
                "entities": entities,
                "retrieval_mode": retrieval_mode,
            },
            "output": {
                "retrieval_method": retrieval_method_used or "None",
                # Separate counts so UI can clearly compare baseline vs embeddings
                "graph_records": len(graph_data),
                "embedding_records": len(embeddings_data),
                # Backwards-compatible aggregate count
                "records_retrieved": len(graph_data) if graph_data else len(embeddings_data),
                # Samples for display
                "graph_sample": graph_data[:3] if len(graph_data) > 3 else graph_data,
                "embedding_sample": embeddings_data[:5] if len(embeddings_data) > 5 else embeddings_data,
                # Graph-specific metadata
                "cypher_query": cypher_query,
                "query_params": query_params,
                "result_shape": result_shape,
            },
            "description": (
                "Layer 2 combined baseline Neo4j graph retrieval (Layer 2.a) "
                "and optional vector-embedding retrieval (Layer 2.b), depending on the selected mode."
            ),
        }

        # --- LAYER 3: LLM GENERATION ---
        
        # Ensure the requested model is actually loaded
        if not llm.is_ready(model_name):
            try:
                # We don't cache this dynamic switch to avoid memory bloat
                llm.load_models(only=[model_name])
            except Exception:
                response["llm_answer"] = f"Failed to load model: {model_name}"
                return response

        # Merge context
        context = merge_context(baseline=graph_data, embeddings=embeddings_data)
        
        # If no data found, be honest
        if not context["context_text"].strip():
            response["llm_answer"] = "I couldn't find relevant data in the Knowledge Graph for this query."
            response["layer_details"]["layer3"] = {
                "input": {
                    "context": "No context available",
                    "query": query_text
                },
                "output": {
                    "answer": response["llm_answer"]
                },
                "description": "No context was found, so LLM returned a message indicating insufficient data"
            }
            return response

        # Build Prompt & Run
        prompt = build_prompt(query_text, context)
        llm_result = run_model(prompt, model_name, llm, context["context_text"])
        
        response["llm_answer"] = llm_result["answer"]
        
        # Track Layer 3 details
        response["layer_details"]["layer3"] = {
            "input": {
                "context_length": len(context["context_text"]),
                "context_preview": context["context_text"][:200] + "..." if len(context["context_text"]) > 200 else context["context_text"],
                "query": query_text,
                "model": model_name
            },
            "output": {
                "answer": llm_result["answer"],
                "latency": llm_result.get("latency", 0),
                "hallucination_check": llm_result.get("hallucination", {})
            },
            "description": f"Generated answer using {model_name} model based on retrieved context"
        }

    except Exception as e:
        response["llm_answer"] = f"An error occurred: {str(e)}"
        response["layer_details"]["error"] = str(e)
    
    return response