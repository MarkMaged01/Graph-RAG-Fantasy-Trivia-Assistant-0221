import pandas as pd
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')

# Load Neo4j credentials
def read_config(path="C:/Users/mokam/Downloads/Milestone_2 (last last)/Milestone_2/config.txt"):
    cfg = {}
    with open(path, "r") as f:
        for line in f:
            key, val = line.strip().split("=")
            cfg[key] = val
    return cfg

config = read_config()
URI = config["URI"]
USERNAME = config["USERNAME"]
PASSWORD = config["PASSWORD"]

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))

# Load embedding models
print("Loading embedding models...")
model1 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
model2 = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
print("Models loaded!")

class EmbeddingRetriever:
    """
    Semantic search functionality for FPL players
    """
    
    def __init__(self, driver, model1, model2):
        self.driver = driver
        self.model1 = model1
        self.model2 = model2
    
    def search_similar_players_by_name(self, player_name, top_k=5, model='model1'):
        """
        Find players similar to a given player by name
        """
        with self.driver.session() as session:
            # Get the target player's embedding
            result = session.run("""
                MATCH (p:Player {player_name: $name})
                RETURN p.embedding_model1 as emb1, 
                       p.embedding_model2 as emb2
                LIMIT 1
            """, {'name': player_name})
            
            record = result.single()
            if not record:
                return f"Player '{player_name}' not found in database"
            
            # Use the appropriate embedding
            query_embedding = record['emb1'] if model == 'model1' else record['emb2']
            index_name = f"player_embedding_{model}"
            
            # Perform vector similarity search
            similar_result = session.run(f"""
                CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
                YIELD node, score
                MATCH (node)-[:PLAYS_AS]->(pos:Position)
                RETURN node.player_name as name,
                       pos.name as position,
                       node.total_career_points as total_points,
                       node.total_goals as goals,
                       node.total_assists as assists,
                       node.avg_points_per_game as avg_ppg,
                       node.appearances as appearances,
                       score
                ORDER BY score DESC
            """, {
                'index_name': index_name,
                'top_k': top_k,
                'query_vector': query_embedding
            })
            
            results = []
            for rec in similar_result:
                results.append({
                    'name': rec['name'],
                    'position': rec['position'],
                    'total_points': rec['total_points'],
                    'goals': rec['goals'],
                    'assists': rec['assists'],
                    'avg_ppg': round(rec['avg_ppg'], 2),
                    'appearances': rec['appearances'],
                    'similarity': round(rec['score'], 4)
                })
            
            return results
    
    def search_by_description(self, description, top_k=5, model='model1'):
        """
        Find players matching a natural language description
        Example: "High scoring forward with many goals"
        """
        # Generate embedding for the description
        if model == 'model1':
            query_embedding = self.model1.encode(description).tolist()
            index_name = "player_embedding_model1"
        else:
            query_embedding = self.model2.encode(description).tolist()
            index_name = "player_embedding_model2"
        
        with self.driver.session() as session:
            result = session.run(f"""
                CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
                YIELD node, score
                MATCH (node)-[:PLAYS_AS]->(pos:Position)
                RETURN node.player_name as name,
                       pos.name as position,
                       node.total_career_points as total_points,
                       node.total_goals as goals,
                       node.total_assists as assists,
                       node.avg_points_per_game as avg_ppg,
                       node.appearances as appearances,
                       score
                ORDER BY score DESC
            """, {
                'index_name': index_name,
                'top_k': top_k,
                'query_vector': query_embedding
            })
            
            results = []
            for rec in result:
                results.append({
                    'name': rec['name'],
                    'position': rec['position'],
                    'total_points': rec['total_points'],
                    'goals': rec['goals'],
                    'assists': rec['assists'],
                    'avg_ppg': round(rec['avg_ppg'], 2),
                    'appearances': rec['appearances'],
                    'similarity': round(rec['score'], 4)
                })
            
            return results
    
    def search_similar_by_position(self, player_name, position=None, top_k=5, model='model1'):
        """
        Find similar players, optionally filtering by position
        """
        with self.driver.session() as session:
            # Get the target player's embedding and position
            result = session.run("""
                MATCH (p:Player {player_name: $name})-[:PLAYS_AS]->(pos:Position)
                RETURN p.embedding_model1 as emb1, 
                       p.embedding_model2 as emb2,
                       pos.name as player_position
                LIMIT 1
            """, {'name': player_name})
            
            record = result.single()
            if not record:
                return f"Player '{player_name}' not found in database"
            
            query_embedding = record['emb1'] if model == 'model1' else record['emb2']
            target_position = position or record['player_position']
            index_name = f"player_embedding_{model}"
            
            # Search with position filter
            similar_result = session.run(f"""
                CALL db.index.vector.queryNodes($index_name, $top_k * 2, $query_vector)
                YIELD node, score
                MATCH (node)-[:PLAYS_AS]->(pos:Position {{name: $position}})
                RETURN node.player_name as name,
                       pos.name as position,
                       node.total_career_points as total_points,
                       node.total_goals as goals,
                       node.total_assists as assists,
                       node.avg_points_per_game as avg_ppg,
                       node.appearances as appearances,
                       score
                ORDER BY score DESC
                LIMIT $top_k
            """, {
                'index_name': index_name,
                'top_k': top_k,
                'query_vector': query_embedding,
                'position': target_position
            })
            
            results = []
            for rec in similar_result:
                results.append({
                    'name': rec['name'],
                    'position': rec['position'],
                    'total_points': rec['total_points'],
                    'goals': rec['goals'],
                    'assists': rec['assists'],
                    'avg_ppg': round(rec['avg_ppg'], 2),
                    'appearances': rec['appearances'],
                    'similarity': round(rec['score'], 4)
                })
            
            return results
        
    def get_embedding_context_from_query(self, query_text, model_key="minilm", top_k=10):
        """
        Adapter method used by backend.py
        Returns list of dicts suitable for RAG context
        """

        # Map UI/backend model keys → embedding indexes
        model_key_to_embedding = {
            "minilm": "model1",   # all-MiniLM-L6-v2 (384d)
            "mpnet": "model2"       # mpnet / stronger model (768d)
        }

        model = model_key_to_embedding.get(model_key, "model1")

        results = self.search_by_description(
            description=query_text,
            top_k=top_k,
            model=model
        )

        # Normalize for backend + app.py
        context = []
        for r in results:
            context.append({
                "name": r["name"],
                "position": r["position"],
                "total_points": r["total_points"],
                "goals": r["goals"],
                "assists": r["assists"],
                "avg_ppg": r["avg_ppg"],
                "appearances": r["appearances"],
                "similarity": r["similarity"],
                "source": "embedding"
            })

        return context

def compare_embedding_models(searcher, query, top_k=5):
    """
    Compare results from both embedding models
    """
    print(f"\n{'='*80}")
    print(f"Query: {query}")
    print(f"{'='*80}")
    
    # Test with both models
    print("\nModel 1 Results (all-MiniLM-L6-v2):")
    print("-" * 80)
    results1 = searcher.search_by_description(query, top_k=top_k, model='model1')
    for i, player in enumerate(results1, 1):
        print(f"{i}. {player['name']} ({player['position']}) - "
              f"Points: {player['total_points']}, Goals: {player['goals']}, "
              f"Assists: {player['assists']}, Similarity: {player['similarity']}")
    
    print("\nModel 2 Results (all-mpnet-base-v2):")
    print("-" * 80)
    results2 = searcher.search_by_description(query, top_k=top_k, model='model2')
    for i, player in enumerate(results2, 1):
        print(f"{i}. {player['name']} ({player['position']}) - "
              f"Points: {player['total_points']}, Goals: {player['goals']}, "
              f"Assists: {player['assists']}, Similarity: {player['similarity']}")

def demo_searches():
    """
    Demonstrate various semantic search capabilities
    """
    searcher = SemanticSearchFPL(driver, model1, model2)
    
    print("\n" + "="*80)
    print("FPL SEMANTIC SEARCH DEMO")
    print("="*80)
    
    # Example 1: Find similar players by name
    print("\n\n### Example 1: Find players similar to Mohamed Salah ###")
    results = searcher.search_similar_players_by_name("Mohamed Salah", top_k=5, model='model1')
    if isinstance(results, str):
        print(results)
    else:
        for i, player in enumerate(results, 1):
            print(f"{i}. {player['name']} ({player['position']}) - "
                  f"Points: {player['total_points']}, Goals: {player['goals']}, "
                  f"Assists: {player['assists']}, Similarity: {player['similarity']}")
    
    # Example 2: Search by description
    compare_embedding_models(
        searcher,
        "who scored more points in gw 17 between mohamed salah and erling haaland",
        top_k=5
    )
    
    # Example 3: Another description
    compare_embedding_models(
        searcher,
        "Salah goals in gw 17 season 2022-23",
        top_k=5
    )
    
    # Example 4: Creative midfielder
    compare_embedding_models(
        searcher,
        "Highest total points for defenders in gw 20",
        top_k=5
    )

    # Example 5: Creative midfielder
    compare_embedding_models(
        searcher,
        "best total points in gw 26",
        top_k=5
    )
    
    # Example 5: Find similar defenders
    print("\n\n### Example 5: Find defenders similar to specific player ###")
    player_to_search = "Virgil van Dijk"
    results = searcher.search_similar_by_position(player_to_search, position='DEF', top_k=5, model='model1')
    if isinstance(results, str):
        # Try alternative defender
        player_to_search = "Aaron Cann"
        results = searcher.search_similar_by_position(player_to_search, position='DEF', top_k=5, model='model1')
    
    if not isinstance(results, str):
        print(f"Defenders similar to {player_to_search}:")
        for i, player in enumerate(results, 1):
            print(f"{i}. {player['name']} - "
                  f"Points: {player['total_points']}, Clean Sheets: Consider match data, "
                  f"Similarity: {player['similarity']}")

# Interactive search function
def interactive_search():
    """
    Interactive search interface
    """
    searcher = SemanticSearchFPL(driver, model1, model2)
    
    print("\n" + "="*80)
    print("INTERACTIVE FPL SEMANTIC SEARCH")
    print("="*80)
    print("\nOptions:")
    print("1. Search by player name (find similar players)")
    print("2. Search by description")
    print("3. Compare both embedding models")
    print("4. Exit")
    
    while True:
        print("\n" + "-"*80)
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            player_name = input("Enter player name: ").strip()
            model = input("Choose model (model1/model2, default=model1): ").strip() or 'model1'
            top_k = int(input("How many results? (default=5): ").strip() or '5')
            
            results = searcher.search_similar_players_by_name(player_name, top_k=top_k, model=model)
            
            if isinstance(results, str):
                print(f"\n{results}")
            else:
                print(f"\nPlayers similar to {player_name}:")
                for i, player in enumerate(results, 1):
                    print(f"{i}. {player['name']} ({player['position']}) - "
                          f"Points: {player['total_points']}, Similarity: {player['similarity']}")
        
        elif choice == '2':
            description = input("Enter description: ").strip()
            model = input("Choose model (model1/model2, default=model1): ").strip() or 'model1'
            top_k = int(input("How many results? (default=5): ").strip() or '5')
            
            results = searcher.search_by_description(description, top_k=top_k, model=model)
            
            print(f"\nPlayers matching '{description}':")
            for i, player in enumerate(results, 1):
                print(f"{i}. {player['name']} ({player['position']}) - "
                      f"Points: {player['total_points']}, Similarity: {player['similarity']}")
        
        elif choice == '3':
            description = input("Enter description: ").strip()
            top_k = int(input("How many results? (default=5): ").strip() or '5')
            compare_embedding_models(searcher, description, top_k=top_k)
        
        elif choice == '4':
            print("\nExiting...")
            break
        
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    try:
        # Run demo first
        demo_searches()
        
        # Then open interactive mode
        print("\n\nWould you like to try interactive search? (y/n): ", end='')
        if input().strip().lower() == 'y':
            interactive_search()
    
    finally:
        driver.close()
        print("\nConnection closed.")