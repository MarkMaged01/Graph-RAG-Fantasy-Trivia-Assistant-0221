import json
from neo4j import GraphDatabase

# 1. Load Credentials from config.txt
config = {}
base_path = "config.txt" # Ensure this is in the same folder

try:
    with open(base_path, "r") as f:
        for line in f:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                config[key] = value
except FileNotFoundError:
    print("⚠️ Config file not found. Using defaults.")

uri = config.get("URI", "neo4j://localhost:7687")
user = config.get("USERNAME", "neo4j")
password = config.get("PASSWORD", "12345678")

# 2. Connect and Fetch
def fetch_data():
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    # default structure
    metadata = {
        "players": [],
        "teams": [],
        "seasons": [], 
        "metrics": ["points", "goals", "assists", "clean sheets", "minutes", 
                    "yellow cards", "red cards", "bonus", "ict_index", 
                    "top", "best", "price", "cost", "value"] 
    }

    try:
        with driver.session() as session:
            print("⏳ Connecting to Neo4j...")
            
            # Fetch Players
            result = session.run("MATCH (p:Player) RETURN DISTINCT p.player_name AS name")
            metadata["players"] = [r["name"] for r in result]
            print(f"   Found {len(metadata['players'])} players.")

            # Fetch Teams
            result = session.run("MATCH (t:Team) RETURN DISTINCT t.name AS name")
            metadata["teams"] = [r["name"] for r in result]
            print(f"   Found {len(metadata['teams'])} teams.")

            # Fetch Seasons
            result = session.run("MATCH (s:Season) RETURN DISTINCT s.season_name AS name")
            metadata["seasons"] = [r["name"] for r in result]
            print(f"   Found {len(metadata['seasons'])} seasons.")

    except Exception as e:
        print(f"❌ Error connecting to Neo4j: {e}")
    finally:
        driver.close()

    return metadata

if __name__ == "__main__":
    data = fetch_data()
    
    # Save to JSON for the other script to use
    with open("fpl_metadata.json", "w") as f:
        json.dump(data, f, indent=4)
    print("✅ 'fpl_metadata.json' has been created successfully.")