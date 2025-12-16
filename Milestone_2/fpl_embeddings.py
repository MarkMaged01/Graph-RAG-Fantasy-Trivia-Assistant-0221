import pandas as pd
import numpy as np
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

# Load the dataset
df = pd.read_csv("C:/Users/mokam/Downloads/Milestone_2 (last last)/Milestone_2/fpl_two_seasons.csv")

# Initialize embedding models
print("Loading embedding models...")
model1 = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')  # 384 dimensions
model2 = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')  # 768 dimensions
print("Models loaded successfully!")

def aggregate_player_stats(df):
    """
    Aggregate player statistics across all their appearances
    to create a comprehensive player profile
    """
    # Group by player name and element
    player_stats = df.groupby(['name', 'element', 'position']).agg({
        'total_points': 'sum',
        'goals_scored': 'sum',
        'assists': 'sum',
        'minutes': 'sum',
        'bonus': 'sum',
        'clean_sheets': 'sum',
        'goals_conceded': 'sum',
        'own_goals': 'sum',
        'penalties_saved': 'sum',
        'penalties_missed': 'sum',
        'yellow_cards': 'sum',
        'red_cards': 'sum',
        'saves': 'sum',
        'bps': 'mean',
        'influence': 'mean',
        'creativity': 'mean',
        'threat': 'mean',
        'ict_index': 'mean',
        'form': 'mean'
    }).reset_index()
    
    # Calculate appearances
    appearances = df.groupby(['name', 'element']).size().reset_index(name='appearances')
    player_stats = player_stats.merge(appearances, on=['name', 'element'])
    
    # Calculate average points per game
    player_stats['avg_points_per_game'] = player_stats['total_points'] / player_stats['appearances']
    
    return player_stats

def construct_text_description(row):
    """
    Construct a natural language description from player statistics
    """
    position_full = {
        'GKP': 'Goalkeeper',
        'DEF': 'Defender',
        'MID': 'Midfielder',
        'FWD': 'Forward'
    }
    
    pos = position_full.get(row['position'], row['position'])
    
    # Build description based on position
    if row['position'] == 'GKP':
        description = (
            f"Player {row['name']} is a {pos} who played {int(row['appearances'])} matches "
            f"with {int(row['minutes'])} total minutes. "
            f"Performance: {int(row['total_points'])} total points, "
            f"{int(row['clean_sheets'])} clean sheets, {int(row['saves'])} saves, "
            f"{int(row['goals_conceded'])} goals conceded, {int(row['penalties_saved'])} penalties saved. "
            f"Discipline: {int(row['yellow_cards'])} yellow cards, {int(row['red_cards'])} red cards. "
            f"Average points per game: {row['avg_points_per_game']:.2f}. "
            f"Form rating: {row['form']:.2f}."
        )
    
    elif row['position'] == 'DEF':
        description = (
            f"Player {row['name']} is a {pos} who played {int(row['appearances'])} matches "
            f"with {int(row['minutes'])} total minutes. "
            f"Performance: {int(row['total_points'])} total points, "
            f"{int(row['goals_scored'])} goals, {int(row['assists'])} assists, "
            f"{int(row['clean_sheets'])} clean sheets, {int(row['goals_conceded'])} goals conceded. "
            f"Bonus points: {int(row['bonus'])}. "
            f"Discipline: {int(row['yellow_cards'])} yellow cards, {int(row['red_cards'])} red cards. "
            f"Average points per game: {row['avg_points_per_game']:.2f}. "
            f"Advanced metrics - BPS: {row['bps']:.2f}, Influence: {row['influence']:.2f}, "
            f"Creativity: {row['creativity']:.2f}, Threat: {row['threat']:.2f}, ICT Index: {row['ict_index']:.2f}. "
            f"Form rating: {row['form']:.2f}."
        )
    
    else:  # MID or FWD
        description = (
            f"Player {row['name']} is a {pos} who played {int(row['appearances'])} matches "
            f"with {int(row['minutes'])} total minutes. "
            f"Performance: {int(row['total_points'])} total points, "
            f"{int(row['goals_scored'])} goals, {int(row['assists'])} assists. "
            f"Bonus points: {int(row['bonus'])}. Clean sheets: {int(row['clean_sheets'])}. "
            f"Discipline: {int(row['yellow_cards'])} yellow cards, {int(row['red_cards'])} red cards. "
            f"Average points per game: {row['avg_points_per_game']:.2f}. "
            f"Advanced metrics - BPS: {row['bps']:.2f}, Influence: {row['influence']:.2f}, "
            f"Creativity: {row['creativity']:.2f}, Threat: {row['threat']:.2f}, ICT Index: {row['ict_index']:.2f}. "
            f"Form rating: {row['form']:.2f}."
        )
    
    return description

def generate_embeddings(player_stats):
    """
    Generate embeddings using both models
    """
    print("\nConstructing text descriptions...")
    player_stats['text_description'] = player_stats.apply(construct_text_description, axis=1)
    
    print("Generating embeddings with Model 1 (all-MiniLM-L6-v2)...")
    embeddings_model1 = model1.encode(player_stats['text_description'].tolist(), show_progress_bar=True)
    
    print("Generating embeddings with Model 2 (all-mpnet-base-v2)...")
    embeddings_model2 = model2.encode(player_stats['text_description'].tolist(), show_progress_bar=True)
    
    # Add embeddings to dataframe
    player_stats['embedding_model1'] = embeddings_model1.tolist()
    player_stats['embedding_model2'] = embeddings_model2.tolist()
    
    return player_stats

def create_vector_indexes():
    """
    Create vector indexes in Neo4j for both embedding models
    """
    print("\nCreating vector indexes in Neo4j...")
    
    with driver.session() as session:
        # Drop existing indexes if they exist
        try:
            session.run("DROP INDEX player_embedding_model1 IF EXISTS")
            session.run("DROP INDEX player_embedding_model2 IF EXISTS")
        except:
            pass
        
        # Create vector index for model 1 (384 dimensions)
        session.run("""
            CREATE VECTOR INDEX player_embedding_model1 IF NOT EXISTS
            FOR (p:Player)
            ON p.embedding_model1
            OPTIONS {indexConfig: {
                `vector.dimensions`: 384,
                `vector.similarity_function`: 'cosine'
            }}
        """)
        
        # Create vector index for model 2 (768 dimensions)
        session.run("""
            CREATE VECTOR INDEX player_embedding_model2 IF NOT EXISTS
            FOR (p:Player)
            ON p.embedding_model2
            OPTIONS {indexConfig: {
                `vector.dimensions`: 768,
                `vector.similarity_function`: 'cosine'
            }}
        """)
        
    print("Vector indexes created successfully!")

def store_embeddings_in_neo4j(player_stats):
    """
    Store embeddings in Neo4j Player nodes
    """
    print("\nStoring embeddings in Neo4j...")
    
    with driver.session() as session:
        for idx, row in player_stats.iterrows():
            session.run("""
                MATCH (p:Player {player_name: $name, player_element: $element})
                SET p.embedding_model1 = $emb1,
                    p.embedding_model2 = $emb2,
                    p.text_description = $desc,
                    p.total_career_points = $total_points,
                    p.total_goals = $goals,
                    p.total_assists = $assists,
                    p.avg_points_per_game = $avg_ppg,
                    p.appearances = $appearances
            """, {
                'name': row['name'],
                'element': int(row['element']),
                'emb1': row['embedding_model1'],
                'emb2': row['embedding_model2'],
                'desc': row['text_description'],
                'total_points': int(row['total_points']),
                'goals': int(row['goals_scored']),
                'assists': int(row['assists']),
                'avg_ppg': float(row['avg_points_per_game']),
                'appearances': int(row['appearances'])
            })
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(player_stats)} players...")
    
    print(f"All {len(player_stats)} players processed successfully!")

def verify_embeddings():
    """
    Verify that embeddings were stored correctly
    """
    print("\nVerifying embeddings...")
    
    with driver.session() as session:
        result = session.run("""
            MATCH (p:Player)
            WHERE p.embedding_model1 IS NOT NULL AND p.embedding_model2 IS NOT NULL
            RETURN count(p) as count
        """)
        count = result.single()['count']
        print(f"Total players with embeddings: {count}")
        
        # Show example
        result = session.run("""
            MATCH (p:Player)
            WHERE p.embedding_model1 IS NOT NULL
            RETURN p.player_name as name, 
                   p.total_career_points as points,
                   size(p.embedding_model1) as emb1_size,
                   size(p.embedding_model2) as emb2_size
            LIMIT 3
        """)
        
        print("\nSample players with embeddings:")
        for record in result:
            print(f"  - {record['name']}: {record['points']} points, "
                  f"Embedding sizes: {record['emb1_size']}, {record['emb2_size']}")

def main():
    print("="*60)
    print("FPL Player Embeddings Generation")
    print("="*60)
    
    # Step 1: Aggregate player statistics
    print("\nStep 1: Aggregating player statistics...")
    player_stats = aggregate_player_stats(df)
    print(f"Total unique players: {len(player_stats)}")
    
    # Step 2: Generate embeddings
    print("\nStep 2: Generating embeddings...")
    player_stats = generate_embeddings(player_stats)
    
    # Step 3: Create vector indexes
    print("\nStep 3: Creating vector indexes...")
    create_vector_indexes()
    
    # Step 4: Store embeddings in Neo4j
    print("\nStep 4: Storing embeddings in Neo4j...")
    store_embeddings_in_neo4j(player_stats)
    
    # Step 5: Verify
    verify_embeddings()
    
    print("\n" + "="*60)
    print("Embeddings generation completed successfully!")
    print("="*60)
    print("\nYou can now use semantic similarity search on players!")
    print("Example: Find players similar to 'Mohamed Salah'")

if __name__ == "__main__":
    try:
        main()
    finally:
        driver.close()
