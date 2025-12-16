from neo4j import GraphDatabase
import os

class GraphRetriever:
    def __init__(self):
        config = self._load_config()
        self.uri = config.get("URI", "neo4j://localhost:7687")
        self.user = config.get("USERNAME", "neo4j")
        self.password = config.get("PASSWORD", "12345678")
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            print("✅ Connected to Neo4j successfully.")
        except Exception as e:
            print(f"❌ Failed to connect to Neo4j: {e}")

    def _load_config(self):
        config = {}
        if os.path.exists("config.txt"):
            with open("config.txt", "r") as f:
                for line in f:
                    if "=" in line:
                        key, value = line.strip().split("=", 1)
                        config[key] = value.strip()
        return config

    def close(self):
        self.driver.close()

    # --- HELPER: Prevents "Index out of range" crash ---
    def _get_entity(self, entities, key, default=None):
        val = entities.get(key)
        if val and isinstance(val, list) and len(val) > 0:
            return val[0]
        return default

    def _empty_result(self, reason):
        return {
            "data": [],
            "query": reason,
            "params": {},
            "result_shape": {"row_count": 0, "columns": [], "column_types": {}}
        }

    def _run(self, query, params):
        try:
            with self.driver.session() as session:
                result = session.run(query, params)
                records = [record.data() for record in result]
                return {
                    "data": records,
                    "query": query,
                    "params": params,
                    "result_shape": self._get_result_shape(records)
                }
        except Exception as e:
            return {
                "data": [{"Error": str(e)}],
                "query": query,
                "params": params,
                "result_shape": {"error": str(e)}
            }
    
    def _get_result_shape(self, records):
        if not records or len(records) == 0:
            return {"row_count": 0, "columns": [], "column_types": {}}
        
        columns = list(records[0].keys()) if records else []
        column_types = {}
        for col in columns:
            val = next((r[col] for r in records if r.get(col) is not None), None)
            column_types[col] = type(val).__name__ if val is not None else "None"
        
        return {"row_count": len(records), "columns": columns, "column_types": column_types}


    def get_player_stats(self, entities):
        players = entities.get('player', [])
        player_list = players if isinstance(players, list) else [players]
        season = entities.get('season')
        
        if players:
            if season:
                query = """
                MATCH (p:Player)
                WHERE p.player_name IN $player_names
                MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
                MATCH (f)<-[:HAS_FIXTURE]-(g:Gameweek {season: $season})
                WITH DISTINCT p, r, g.season AS Season
                RETURN p.player_name AS Player, 
                       Season,
                       SUM(r.total_points) AS TotalPoints, 
                       SUM(r.goals_scored) AS Goals, 
                       SUM(r.assists) AS Assists,
                       SUM(r.minutes) AS Minutes
                ORDER BY p.player_name, Season DESC
                """
                params = {"player_names": player_list, "season": season}
            else:
                query = """
                MATCH (p:Player)
                WHERE p.player_name IN $player_names
                MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
                MATCH (f)<-[:HAS_FIXTURE]-(g:Gameweek)
                WITH DISTINCT p, r, g.season AS Season
                RETURN p.player_name AS Player, 
                       Season,
                       SUM(r.total_points) AS TotalPoints, 
                       SUM(r.goals_scored) AS Goals, 
                       SUM(r.assists) AS Assists,
                       SUM(r.minutes) AS Minutes
                ORDER BY p.player_name, Season DESC
                """
                params = {"player_names": player_list}
            return self._run(query, params)
        return self._empty_result("N/A - No player entity found")

    def get_player_gameweek_stats(self, entities):
        players = entities.get('player', [])
        teams = entities.get('team', [])
        season = entities.get('season')
        gameweek = entities.get('gameweek')
        
        if not season:
            season = "2022-23" 

        if players:
            player_list = players if isinstance(players, list) else [players]
            
            if season and gameweek:
                query = """
                MATCH (p:Player)
                WHERE p.player_name IN $player_names
                MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
                MATCH (f)<-[:HAS_FIXTURE]-(g:Gameweek {season: $season})
                RETURN p.player_name AS Player,
                       g.season AS Season,
                       g.GW_number AS GraphGW,
                       f.event AS FixtureEvent,
                       f.round AS FixtureRound,
                       r.total_points AS Points, 
                       r.goals_scored AS Goals,
                       r.assists AS Assists,
                       r.minutes AS Minutes,
                       r.ict_index AS ICT,
                       r.threat AS Threat,
                       f.kickoff_time as Date
                ORDER BY Date ASC
                """
                raw_result = self._run(query, {"player_names": player_list, "season": season})
                all_rows = raw_result.get('data', [])
                
                filtered_data = []
                target_gw = int(gameweek)
                
                for row in all_rows:
                    candidates = [row.get('FixtureEvent'), row.get('FixtureRound'), row.get('GraphGW')]
                    is_match = False
                    for c in candidates:
                        try:
                            if c is not None and int(c) == target_gw:
                                is_match = True; break
                        except: continue
                    
                    if is_match:
                        row['GW'] = target_gw
                        filtered_data.append(row)

                if not filtered_data and len(all_rows) >= target_gw:
                    print(f"⚠️ Using chronological fallback for GW {target_gw}")
                    fallback_row = all_rows[target_gw - 1]
                    fallback_row['GW'] = target_gw
                    filtered_data.append(fallback_row)

                return {
                    "data": filtered_data,
                    "query": "Python-Filtered: " + query,
                    "params": {"season": season, "gw": gameweek},
                    "result_shape": self._get_result_shape(filtered_data)
                }

            elif season:
                query = """
                MATCH (p:Player)
                WHERE p.player_name IN $player_names
                MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
                MATCH (f)<-[:HAS_FIXTURE]-(g:Gameweek {season: $season})
                WITH DISTINCT p, r, g
                RETURN p.player_name AS Player, g.season AS Season, g.GW_number AS GW, 
                       r.total_points AS Points, r.ict_index AS ICT
                ORDER BY g.GW_number DESC LIMIT 10
                """
                params = {"player_names": player_list, "season": season}
                return self._run(query, params)

            else:
                query = """
                MATCH (p:Player)
                WHERE p.player_name IN $player_names
                MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
                MATCH (f)<-[:HAS_FIXTURE]-(g:Gameweek)
                WITH DISTINCT p, r, g
                RETURN p.player_name AS Player, g.season AS Season, g.GW_number AS GW, 
                       r.total_points AS Points
                ORDER BY g.season DESC, g.GW_number DESC LIMIT 10
                """
                params = {"player_names": player_list}
                return self._run(query, params)

        # CASE 2: Teams
        elif teams:
            target_team = teams[0]
            if season and gameweek:
                query = """
                MATCH (t:Team {name: $team})
                MATCH (f:Fixture) WHERE (f)-[:HAS_HOME_TEAM]->(t) OR (f)-[:HAS_AWAY_TEAM]->(t)
                MATCH (f)<-[:HAS_FIXTURE]-(g:Gameweek {season: $season})
                MATCH (p:Player)-[r:PLAYED_IN]->(f) MATCH (p)-[:PLAYS_FOR]->(t)
                RETURN p.player_name AS Player, g.season AS Season, g.GW_number AS GraphGW, f.event AS FixtureEvent,
                       r.total_points AS Points, r.goals_scored AS Goals, r.assists AS Assists, r.threat AS Threat
                """
                raw_result = self._run(query, {"team": target_team, "season": season})
                all_rows = raw_result.get('data', [])
                filtered_data = []
                target_gw = int(gameweek)
                
                for row in all_rows:
                    candidates = [row.get('FixtureEvent'), row.get('GraphGW')]
                    for c in candidates:
                        try:
                            if c is not None and int(c) == target_gw:
                                row['GW'] = target_gw
                                filtered_data.append(row)
                                break
                        except: continue
                
                filtered_data.sort(key=lambda x: x.get('Threat', 0) or 0, reverse=True)
                return {
                    "data": filtered_data[:10],
                    "query": "Python-Filtered Team Query",
                    "params": {"team": target_team, "gw": gameweek},
                    "result_shape": self._get_result_shape(filtered_data)
                }
            # ... (Simpler team fallbacks omitted for brevity, logic remains same) ...
                
        return self._empty_result("N/A - No player or team entity found")

    def compare_players(self, entities):
        players = entities.get('player', [])
        if not players: return self._empty_result("N/A - No players found")
        season = entities.get('season')
        
        if season:
            query = """
            MATCH (p:Player) WHERE p.player_name IN $names
            MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
            MATCH (f)<-[:HAS_FIXTURE]-(g:Gameweek {season: $season})
            WITH DISTINCT p, r
            RETURN p.player_name AS Player, 
                   SUM(r.total_points) AS TotalPoints, 
                   SUM(r.goals_scored) AS Goals, 
                   SUM(r.assists) AS Assists,
                   AVG(r.ict_index) AS AvgICT
            """
            params = {"names": players, "season": season}
        else:
            query = """
            MATCH (p:Player) WHERE p.player_name IN $names
            MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
            WITH DISTINCT p, r
            RETURN p.player_name AS Player, 
                   SUM(r.total_points) AS TotalPoints, 
                   SUM(r.goals_scored) AS Goals, 
                   SUM(r.assists) AS Assists,
                   AVG(r.ict_index) AS AvgICT
            """
            params = {"names": players}
            
        return self._run(query, params)
    def get_top_scorer_gameweek(self, entities):
        season = entities.get('season')
        gw = entities.get('gameweek')
        pos_list = entities.get('position')
        metrics = entities.get('metric', [])
        
        # 1. FIX: Handle explicit None/Null from input processing
        if not season:
            season = "2022-23"
        
        # Determine sorting based on user metric (goals, assists, threat)
        sort_clause = "r.total_points" # Default
        if "goals" in metrics: sort_clause = "r.goals_scored"
        elif "assists" in metrics: sort_clause = "r.assists"
        elif "threat" in metrics: sort_clause = "r.threat"
        elif "ict" in metrics: sort_clause = "r.ict_index"

        # 2. Handle Position Filtering (e.g. "Best Defender")
        if pos_list:
            pos_map = {"DEF": "Defender", "FWD": "Forward", "MID": "Midfielder", "GK": "Goalkeeper"}
            raw_pos = pos_list[0]
            target_pos = pos_map.get(raw_pos, raw_pos)
            
            # Uses f-string to inject the dynamic sort_clause
            query = f"""
            MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
            WHERE pos.name = $raw_pos OR pos.name CONTAINS $target_pos
            MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
            MATCH (f)<-[:HAS_FIXTURE]-(g:Gameweek {{season: $season, GW_number: $gw}})
            RETURN p.player_name AS Player, 
                   r.total_points AS Points, 
                   r.goals_scored AS Goals, 
                   r.assists AS Assists,
                   r.clean_sheets AS CleanSheets,
                   r.threat AS Threat
            ORDER BY {sort_clause} DESC
            LIMIT 5
            """
            params = {"season": season, "gw": gw, "target_pos": target_pos, "raw_pos": raw_pos}
            
        else:
            query = f"""
            MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
            MATCH (f)<-[:HAS_FIXTURE]-(g:Gameweek {{season: $season, GW_number: $gw}})
            RETURN p.player_name AS Player, 
                   r.total_points AS Points, 
                   r.goals_scored AS Goals, 
                   r.assists AS Assists,
                   r.threat AS Threat
            ORDER BY {sort_clause} DESC
            LIMIT 5
            """
            params = {"season": season, "gw": gw}

        return self._run(query, params)
    def get_top_players_by_position(self, entities):
        pos = self._get_entity(entities, 'position', "Midfielder")
        season = entities.get('season')
        
        query = """
        MATCH (p:Player)-[:PLAYS_AS]->(pos:Position)
        WHERE pos.name CONTAINS $position
        MATCH (p)-[r:PLAYED_IN]->(f:Fixture)
        MATCH (f)<-[:HAS_FIXTURE]-(g:Gameweek {season: $season})
        WITH p, SUM(r.total_points) as TotalPoints
        RETURN p.player_name AS Player, TotalPoints
        ORDER BY TotalPoints DESC
        LIMIT 5
        """
        return self._run(query, {"position": pos, "season": season})

    def get_budget_enablers(self, entities):
        season = entities.get('season')
        query = """
        MATCH (p:Player)-[r:PLAYED_IN]->(f:Fixture)
        MATCH (f)<-[:HAS_FIXTURE]-(g:Gameweek {season: $season})
        WITH p, SUM(r.total_points) AS Points, SUM(r.minutes) AS Mins
        WHERE Mins > 500 
        RETURN p.player_name AS Player, Points, (Points / (Mins/90.0)) AS PointsPer90
        ORDER BY PointsPer90 DESC
        LIMIT 5
        """
        return self._run(query, {"season": season})
        
    def get_fixture_difficulty(self, entities):
        team = self._get_entity(entities, 'team')
        if not team: return self._empty_result("N/A - No team")
        gw = entities.get('gameweek', 1)
        query = """
        MATCH (t:Team {name: $team})
        MATCH (f:Fixture) WHERE (f)-[:HAS_HOME_TEAM]->(t) OR (f)-[:HAS_AWAY_TEAM]->(t)
        MATCH (f)<-[:HAS_FIXTURE]-(g:Gameweek) WHERE g.GW_number > $gw
        RETURN g.GW_number AS GW, f.kickoff_time AS Date
        ORDER BY g.GW_number ASC LIMIT 3
        """
        return self._run(query, {"team": team, "gw": gw})
        
    def get_team_performance(self, entities):
        team = self._get_entity(entities, 'team')
        if not team: return self._empty_result("N/A - No team")
        query = """MATCH (t:Team {name: $team}) RETURN t.name"""
        return self._run(query, {"team": team})

    def get_player_availability(self, entities):
        player = self._get_entity(entities, 'player')
        query = """MATCH (p:Player {player_name: $player}) RETURN p.chance_of_playing_next_round"""
        return self._run(query, {"player": player})

    def get_player_info(self, entities):
        player = self._get_entity(entities, 'player')
        query = """MATCH (p:Player {player_name: $player}) RETURN p.player_name"""
        return self._run(query, {"player": player})
        
    def get_transfer_trends(self, entities):
        return self._run("MATCH (p:Player) RETURN p.transfers_in_event LIMIT 5", {})

    # =========================================================================
    #  ROUTER
    # =========================================================================
    def route_query(self, intent, entities):
        print(f"🔄 Routing Intent: {intent}...")
        
        if intent == "PLAYER_STATS":
            if not entities.get('player') and not entities.get('team'):
                if entities.get('gameweek') and entities.get('metric'):
                    return self.get_top_scorer_gameweek(entities)

            if entities.get('gameweek'):
                return self.get_player_gameweek_stats(entities)
            return self.get_player_stats(entities)
        
        elif intent == "COMPARE_PLAYERS":
            return self.compare_players(entities)
        
        elif intent == "RECOMMENDATION":
            if entities.get('gameweek'):
                return self.get_top_scorer_gameweek(entities)
            
            if entities.get('position'):
                return self.get_top_players_by_position(entities)
                
            return self.get_budget_enablers(entities)
            
        elif intent == "TEAM_ANALYSIS":
            if "fixture" in str(entities).lower():
                return self.get_fixture_difficulty(entities)
            return self.get_team_performance(entities)
        
        elif intent == "FIXTURE_QUERY":
            return self.get_fixture_difficulty(entities)

        elif intent == "GENERAL_INFO":
            if "injury" in str(entities).lower() or "play" in str(entities).lower():
                return self.get_player_availability(entities)
            return self.get_player_info(entities)
        
        elif intent == "TRANSFER_MARKET":
             return self.get_transfer_trends(entities)
            
        else:
            return self._empty_result("N/A - Intent not recognized")