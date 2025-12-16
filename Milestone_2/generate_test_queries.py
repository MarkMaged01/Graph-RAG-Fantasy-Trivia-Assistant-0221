import json
import random

# 1. Load the Metadata
try:
    with open("fpl_metadata.json", "r") as f:
        kb = json.load(f)
except FileNotFoundError:
    print("Error: Run fetch_metadata.py first!")
    exit()

# 2. Define Query Templates
templates = [
    # Intent: PLAYER_STATS
    "How many {metric} did {player} get in {season}?",
    "What were {player}'s {metric} in GW {gw}?",
    "Show me the {metric} for {player}.",
    "Did {player} score any goals in {season}?",
    
    # Intent: RECOMMENDATION / RANKING
    "Who are the top {position}s in {season}?",
    "Recommend a {position} from {team}.",
    "Who is the best player in {team}?",
    "Get top players by {metric} in {season}.",
    
    # Intent: COMPARE_PLAYERS
    "Compare {player} vs {player2}.",
    "Who is better, {player} or {player2}?",
    "Is {player} better than {player2}?",
    
    # Intent: GENERAL_INFO
    "Which team does {player} play for?",
    "What position is {player}?"
]

# 3. Helper to get random items
def get_random(key):
    return random.choice(kb[key])

def generate_queries(n=100):
    queries = []
    positions = ["Forward", "Midfielder", "Defender", "Goalkeeper", "FWD", "MID", "DEF", "GK"]
    
    for _ in range(n):
        template = random.choice(templates)
        
        # Fill placeholders dynamically
        q = template.format(
            player=get_random("players"),
            player2=get_random("players"), # For comparisons
            team=get_random("teams"),
            season=get_random("seasons"),
            metric=random.choice(kb["metrics"]),
            gw=random.randint(1, 38),
            position=random.choice(positions)
        )
        queries.append(q)
    return queries

# 4. Generate and Save
if __name__ == "__main__":
    test_set = generate_queries(100)
    
    with open("test_queries.txt", "w") as f:
        for q in test_set:
            f.write(q + "\n")
            
    print(f"✅ Generated 100 test queries in 'test_queries.txt'")
    # Preview first 5
    print("Preview:")
    for i in range(5):
        print(f"- {test_set[i]}")