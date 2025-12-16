import streamlit as st
import networkx as nx
import plotly.graph_objects as go
from backend import process_query 

# Page config
st.set_page_config(page_title="FPL Graph-RAG Assistant", page_icon="soccer", layout="wide")

st.title("⚽ FPL Graph-RAG Assistant")
st.markdown("Your AI helper for Fantasy Premier League — powered by Neo4j & LLMs.")

# ==========================================
# SIDEBAR CONTROLS
# ==========================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. Model Selection
    model_choice = st.selectbox(
        "Choose LLM Model:",
        ["mistral", "gemma", "zephyr"],
        index=0
    )
    
    # 2. Retrieval Method Selection
    retrieval_choice = st.radio(
        "Retrieval Method:",
        ["Baseline (Graph)", "Hybrid"],
        index=0
    )

    # 3. Embedding Model Selection (Dynamic)
    # Only show this if using Vector/Hybrid mode
    if retrieval_choice in ["Embeddings (Vector)", "Hybrid"]:
        embedding_model_choice = st.radio(
            "Choose Embedding Model:",
            ("MiniLM", "MPNet"),
            index=0,
            help="MiniLM is faster. MPNet is more accurate but slower."
        )

    else:
        embedding_model_choice = "MiniLM" # Default fallback (unused in Baseline)

    st.divider()
    
    # 4. System Controls
    if st.button("Rebuild Index", key="btn_rebuild"):
        st.cache_resource.clear()
        st.rerun()

# ==========================================
# MAIN CHAT INTERFACE
# ==========================================

if prompt := st.chat_input("Ask your FPL question..."):
    
    # Show user message
    with st.chat_message("user"):
        st.write(prompt)

    # Process Query
    with st.spinner(f"Reasoning with {model_choice} ({retrieval_choice})..."):
        # Pass the selected embedding model to the backend
        result = process_query(
            prompt, 
            model_name=model_choice, 
            retrieval_mode=retrieval_choice,
            embedding_model=embedding_model_choice 
        )

    # --- SECTION A: FINAL ANSWER ---
    with st.chat_message("assistant"):
        st.subheader("💡 Answer")
        st.write(result["llm_answer"])

    # --- SECTION B: LAYER-BY-LAYER PROCESSING ---
    st.subheader("🔧 Processing Pipeline")
    
    layer_details = result.get("layer_details", {})
    
    # Layer 1
    if "layer1" in layer_details:
        with st.expander("📥 Layer 1: Input Processing", expanded=True):
            layer1 = layer_details["layer1"]
            st.markdown(f"**What Layer 1 did:** {layer1.get('description', 'N/A')}")
            st.markdown("**Input:**")
            st.code(layer1.get("input", "N/A"), language=None)
            st.markdown("**Output (sent to Layer 2):**")
            st.json(layer1.get("output", {}))
    
    # Layer 2
    if "layer2" in layer_details:
        with st.expander("🔍 Layer 2: Retrieval (Baseline vs Embeddings)", expanded=True):
            layer2 = layer_details["layer2"]
            st.markdown(f"**What Layer 2 did:** {layer2.get('description', 'N/A')}")
            st.markdown("**Input (received from Layer 1):**")
            st.json(layer2.get("input", {}))
            st.markdown("**Output (sent to Layer 3):**")
            output = layer2.get("output", {})

            # High-level summary
            st.write(f"**Retrieval Method:** {output.get('retrieval_method', 'N/A')}")
            st.write(
                f"**Baseline Graph Rows:** {output.get('graph_records', 0)} | "
                f"**Embedding Hits:** {output.get('embedding_records', 0)}"
            )
            
            # --- Baseline (Graph) section ---
            if retrieval_choice in ["Baseline (Graph)", "Hybrid"]:
                st.markdown("### 🧩 Baseline Graph Retrieval (Layer 2.a)")

                # Display Cypher Query
                cypher_query = output.get("cypher_query", "N/A")
                if cypher_query and cypher_query != "N/A":
                    st.markdown("**🔷 Cypher Query Used:**")
                    st.code(cypher_query, language="cypher")
                    
                    query_params = output.get("query_params", {})
                    if query_params:
                        st.markdown("**Query Parameters:**")
                        st.json(query_params)
                
                # Display Result Shape
                result_shape = output.get("result_shape", {})
                if result_shape and "error" not in result_shape:
                    st.markdown("**📊 Result Shape:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Rows:** {result_shape.get('row_count', 0)}")
                        st.write(f"**Columns:** {len(result_shape.get('columns', []))}")
                    with col2:
                        if result_shape.get("columns"):
                            st.write("**Column Names:**")
                            st.code(", ".join(result_shape["columns"]))
                    
                    if result_shape.get("column_types"):
                        st.markdown("**Column Types:**")
                        st.json(result_shape["column_types"])
                elif result_shape.get("error"):
                    st.error(f"Query Error: {result_shape['error']}")
                
                graph_sample = output.get("graph_sample") or []
                if graph_sample:
                    st.markdown("**Sample Baseline Data (first few records):**")
                    st.dataframe(graph_sample)
                else:
                    st.info("No baseline graph data retrieved")

            # --- Embedding section ---
            if retrieval_choice in ["Embeddings (Vector)", "Hybrid"]:
                st.markdown(f"### 🧠 Embedding Retrieval (Layer 2.b) - Model: {embedding_model_choice}")
                embedding_sample = output.get("embedding_sample") or []
                if embedding_sample:
                    st.markdown("**Top Embedding Hits (sorted by similarity score):**")
                    st.dataframe(embedding_sample)
                else:
                    st.info("No embedding-based results for this query / mode")
    
    # Layer 3
    if "layer3" in layer_details:
        with st.expander("🤖 Layer 3: LLM Generation", expanded=True):
            layer3 = layer_details["layer3"]
            st.markdown(f"**What Layer 3 did:** {layer3.get('description', 'N/A')}")
            st.markdown("**Input (received from Layer 2):**")
            input_data = layer3.get("input", {})
            st.write(f"**Context Length:** {input_data.get('context_length', 0)} characters")
            st.write(f"**Model Used:** {input_data.get('model', 'N/A')}")
            st.write(f"**Context Preview:**")
            st.code(input_data.get("context_preview", "N/A"), language=None)
            st.markdown("**Output (Final Answer):**")
            output_data = layer3.get("output", {})
            st.success(output_data.get("answer", "N/A"))
            if output_data.get("latency"):
                st.caption(f"⏱️ Generation Time: {output_data['latency']:.2f}s")
            if output_data.get("hallucination_check"):
                hall_check = output_data["hallucination_check"]
                if hall_check.get("hallucinated"):
                    st.warning(f"⚠️ Potential hallucination detected. Missing terms: {hall_check.get('missing_terms', [])}")
                else:
                    st.success("✅ No hallucination detected")

    # --- SECTION C: TRANSPARENCY (Context & Logic) ---
    with st.expander("🔍 See Retrieved Context & Logic", expanded=False):
        
        # 1. Intent & Entities
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Intent Detected:**")
            st.info(result["intent"])
        with c2:
            st.markdown("**Entities Extracted:**")
            st.json(result["entities"])

        # 2. Raw Context
        st.markdown("**🗂️ Raw Knowledge Graph Context:**")
        if result["kg_context"]:
            st.dataframe(result["kg_context"])
        else:
            st.warning("No direct graph matches found.")

    # --- SECTION C: GRAPH VISUALIZATION ---
    if result["kg_context"]:
        st.subheader("🕸️ Context Graph Visualization")
        
        # Build NetworkX Graph from result context
        G = nx.Graph()
        
        # Add nodes/edges based on what keys are present
        for item in result["kg_context"]:
            # Try to find relevant keys for nodes
            p = item.get("Player") or item.get("player") or item.get("Name")
            t = item.get("Team") or item.get("team")
            
            if p:
                G.add_node(p, type="Player", color="lightblue")
            if t:
                G.add_node(t, type="Team", color="orange")
            if p and t:
                G.add_edge(p, t)

        if G.number_of_nodes() > 0:
            # Layout
            pos = nx.spring_layout(G, k=0.5)
            
            # Edges Trace
            edge_x, edge_y = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])

            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1, color='#888'),
                hoverinfo='none',
                mode='lines'
            )

            # Nodes Trace
            node_x, node_y, node_text, node_color = [], [], [], []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                # Color logic
                node_color.append('orange' if G.nodes[node].get('type') == 'Team' else '#00CC96')

            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="bottom center",
                marker=dict(
                    showscale=False,
                    color=node_color,
                    size=25,
                    line_width=2
                )
            )

            fig = go.Figure(data=[edge_trace, node_trace],
                        layout=go.Layout(
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=0,l=0,r=0,t=0),
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Graph data found, but could not automatically determine connections for visualization.")

    st.markdown("---")
    st.caption("MS3 FPL Graph-RAG | Powered by Streamlit")

else:
    # Landing state (No list, just instructions)
    st.info("👋 Type your question in the box below to start.")