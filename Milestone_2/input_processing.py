import spacy
import re
import json
import os
from rapidfuzz import process, fuzz
from neo4j import GraphDatabase

# Initialize Spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class FPLPreprocessor:
    def __init__(self):
        self.metadata_file = "fpl_metadata.json"
        self.config_file = "config.txt"
        
        # 1. AUTO-FETCH METADATA
        if not os.path.exists(self.metadata_file):
            self._fetch_and_save_metadata()
        
        try:
            with open(self.metadata_file, "r") as f:
                self.kb = json.load(f)
        except:
            self.kb = {"players": [], "teams": [], "seasons": [], "metrics": []}

        self.kb.setdefault("positions", ["Goalkeeper", "Defender", "Midfielder", "Forward", "GK", "DEF", "MID", "FWD"])
        
        # 2. DEFINING INTENTS
        self.intents = {
            "PLAYER_STATS": [
                "score", "scored", "scoring", "points", "goals", "assist", "stats", "statistics",
                "clean sheet", "minutes", "red card", "yellow card", "bonus", "xg", "threat", 
                "creativity", "form", "influence"
            ],
            "RECOMMENDATION": [
                "top", "best", "rank", "better", "recommend", "captain", "option", 
                "replacement", "budget", "value", "highest", "most", "essential",
                "transfer", "market", "bought", "sold", "rise", "drop", 
                "trending", "popular", "transfers in", "transfers out", "buy", "sell"
            ],
            "COMPARE_PLAYERS": [
                "compare", "comparison", "better than", "better", "vs", "or", "higher", "difference"
            ],
            "FIXTURE_QUERY": [
                "fixture", "schedule", "difficulty", "fdr", "upcoming", "opponent", 
                "run of games", "next match", "play", "playing", "match", "against"
            ],
            "GENERAL_INFO": [
                "who is", "team", "position", "ownership", "price", "cost",
                "injury", "injured", "news", "fit", "available", "status", "suspended", "ban", "return"
            ]
        }

    def _fetch_and_save_metadata(self):
        config = {}
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r") as f:
                    for line in f:
                        if "=" in line:
                            k, v = line.strip().split("=", 1)
                            config[k] = v.strip()
            except: pass
        
        uri = config.get("URI", "neo4j://localhost:7687")
        user = config.get("USERNAME", "neo4j")
        password = config.get("PASSWORD", "12345678")

        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            data = {"players": [], "teams": [], "seasons": [], "metrics": ["points", "goals", "assists", "clean sheets", "minutes", "yellow cards", "red cards", "bonus", "ict_index", "top", "best", "price", "cost", "value"]}
            with driver.session() as session:
                data["players"] = [r["name"] for r in session.run("MATCH (p:Player) RETURN DISTINCT p.player_name AS name")]
                data["teams"] = [r["name"] for r in session.run("MATCH (t:Team) RETURN DISTINCT t.name AS name")]
                data["seasons"] = [r["name"] for r in session.run("MATCH (s:Season) RETURN DISTINCT s.season_name AS name")]
            
            with open(self.metadata_file, "w") as f:
                json.dump(data, f, indent=4)
            print(f"✅ Success! Saved {len(data['players'])} players to {self.metadata_file}")
            driver.close()
        except Exception as e:
            print(f"❌ Neo4j Error: {e}")

    def classify_intent(self, text):
        text_lower = text.lower()
        if " vs " in text_lower or " compare " in text_lower:
            return "COMPARE_PLAYERS"
            
        scores = {k: 0 for k in self.intents}
        for intent, keywords in self.intents.items():
            for k in keywords:
                if k in text_lower: scores[intent] += 1
        
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "GENERAL_INFO"

    def extract_entities(self, text):
        entities = {
            "player": [], "team": [], "position": [],
            "season": None, "gameweek": None, "metric": []
        }
        
        # 1. Season & Gameweek
        s_match = re.search(r"20\d{2}(-\d{2})?", text)
        if s_match:
            f = s_match.group(0)
            entities["season"] = f"{int(f)-1}-{str(f)[-2:]}" if len(f) == 4 else f
        gw_match = re.search(r"(?:gw|gameweek)\s*(\d+)", text, re.IGNORECASE)
        if gw_match: entities["gameweek"] = int(gw_match.group(1))

        # 2. Position & Metrics
        # Store found metrics to exclude them from player search later
        found_keywords = set() 
        
        for pos in self.kb["positions"]:
            if re.search(r'\b' + re.escape(pos) + r'\b', text, re.IGNORECASE):
                if pos not in entities["position"]: 
                    entities["position"].append(pos)
                    found_keywords.add(pos.lower())
                
        for m in self.kb["metrics"]:
            # Basic match
            if m in text.lower(): 
                entities["metric"].append(m)
                # Split multi-word metrics like "clean sheets" to avoid partial matching players
                for word in m.split():
                    found_keywords.add(word.lower())

        # 3. ENTITY EXTRACTION
        candidates = set()
        stop_words = {
            "is", "in", "at", "on", "the", "a", "an", "of", "or", "vs", "gw", 
            "did", "does", "get", "show", "me", "tell", "about", "check", "list",
            "give", "find", "best", "top", "recommend", "how", "many", "who", "what",
            "compare", "difference", "between", "stats", "price", "cost", "value",
            "play", "playing", "injury", "news", "team", "season", "gameweek",
            "player", "week", "this", "that", "for", "with", "form", "score","which", "had", "have", "has", "was", "were", "are"
        }
        
        # Tokenization
        doc = nlp(text)
        for t in doc:
            if t.pos_ == "PROPN": candidates.add(t.text)
            
        for word in text.split():
            clean_word = re.sub(r'[^\w\s]', '', word)
            # Skip short words AND words already identified as metrics (e.g., "points")
            if len(clean_word) > 2 and clean_word.lower() not in stop_words and clean_word.lower() not in found_keywords:
                candidates.add(clean_word)

        # --- A. TEAM MATCHING ---
        found_teams = []
        teams_to_remove_from_candidates = set()
        
        for cand in candidates:
            # Score 90 keeps teams strict (Newcastle vs Newcastle United)
            matches = process.extract(cand, self.kb["teams"], scorer=fuzz.WRatio, limit=2, score_cutoff=90)
            for name, score, _ in matches:
                found_teams.append(name)
                teams_to_remove_from_candidates.add(cand)

        entities["team"] = list(set(found_teams))
        candidates = candidates - teams_to_remove_from_candidates

        # --- B. PLAYER MATCHING (FIXED for Single Names like "Salah") ---
        found_players = [] 
        for cand in candidates:
            # FIX: Switched back to WRatio, which handles "Salah" vs "Mohamed Salah" (Score ~90).
            # We use score_cutoff=85 to block "best" (noise) but allow "Salah" (partial match).
            matches = process.extract(cand, self.kb["players"], scorer=fuzz.WRatio, limit=5, score_cutoff=85)
            
            for name, score, _ in matches:
                found_players.append((name, score))

        found_players.sort(key=lambda x: x[1], reverse=True)
        
        final_players = []
        for name, score in found_players:
            is_conflict = False
            for existing_name, existing_score in final_players:
                if name in existing_name or existing_name in name:
                    is_conflict = True
                    break
            if not is_conflict:
                final_players.append((name, score))

        
        
        clean_final_players = []

        text_lower = text.lower()

        for name, score in final_players:
            name_parts = name.lower().split()

            if not any(re.search(rf"\b{re.escape(part)}\b", text_lower) for part in name_parts):
                continue
            clean_final_players.append((name, score))

        final_players = clean_final_players


        entities["player"] = [p[0] for p in final_players]

        return entities

if __name__ == "__main__":
    processor = FPLPreprocessor()
    q = "Salah points"
    print(f"Query: {q}")
    print(f"Entities: {processor.extract_entities(q)}")