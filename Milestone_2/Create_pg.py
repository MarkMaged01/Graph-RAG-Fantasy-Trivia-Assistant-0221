import pandas as pd
from neo4j import GraphDatabase



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


def run_query(query, params=None):
    with driver.session() as session:
        session.run(query, params or {})



# Load CSV (your corrected columns)
df = pd.read_csv("C:/Users/mokam/Downloads/Milestone_2 (last last)/Milestone_2/fpl_two_seasons.csv")

df["season"] = df["season"].astype(str)
df["GW"] = df["GW"].astype(int)
df["fixture"] = df["fixture"].astype(int)



# Build Knowledge Graph
def build_kg():


    run_query("MATCH (n) DETACH DELETE n")


    season_rows = [{"season_name": s} for s in df["season"].unique()]
    run_query("""
        UNWIND $rows AS r
        MERGE (:Season {season_name:r.season_name})
    """, {"rows": season_rows})


    gws = df.groupby(["season", "GW"]).size().reset_index()[["season", "GW"]]
    gw_rows = gws.to_dict("records")
    run_query("""
        UNWIND $rows AS r
        MERGE (:Gameweek {season:r.season, GW_number:r.GW})
    """, {"rows": gw_rows})


    fixes = df.groupby(["season","fixture","kickoff_time"]).size().reset_index()[["season","fixture","kickoff_time"]]
    fix_rows = fixes.to_dict("records")

    run_query("""
        UNWIND $rows AS r
        MERGE (:Fixture {
            season:r.season,
            fixture_number:r.fixture,
            kickoff_time:r.kickoff_time
        })
    """, {"rows": fix_rows})


    teams = set(df["home_team"].unique()).union(df["away_team"].unique())
    team_rows = [{"name": t} for t in teams]

    run_query("""
        UNWIND $rows AS r
        MERGE (:Team {name:r.name})
    """, {"rows": team_rows})


    players = df.groupby(["name","element"]).size().reset_index()[["name","element"]]
    pl_rows = players.to_dict("records")

    run_query("""
        UNWIND $rows AS r
        MERGE (:Player {player_name:r.name, player_element:r.element})
    """, {"rows": pl_rows})


    pos_rows = [{"name": p} for p in df["position"].unique()]
    run_query("""
        UNWIND $rows AS r
        MERGE (:Position {name:r.name})
    """, {"rows": pos_rows})


    run_query("""
        MATCH (s:Season)
        MATCH (g:Gameweek {season:s.season_name})
        MERGE (s)-[:HAS_GW]->(g)
    """)


    run_query("""
        MATCH (g:Gameweek)
        MATCH (f:Fixture {season:g.season})
        MERGE (g)-[:HAS_FIXTURE]->(f)
    """)


    home_rows = df.groupby(["season","fixture","home_team"]).size().reset_index()[["season","fixture","home_team"]]
    run_query("""
        UNWIND $rows AS r
        MATCH (f:Fixture {season:r.season, fixture_number:r.fixture})
        MATCH (t:Team {name:r.home_team})
        MERGE (f)-[:HAS_HOME_TEAM]->(t)
    """, {"rows": home_rows.to_dict("records")})


    away_rows = df.groupby(["season","fixture","away_team"]).size().reset_index()[["season","fixture","away_team"]]
    run_query("""
        UNWIND $rows AS r
        MATCH (f:Fixture {season:r.season, fixture_number:r.fixture})
        MATCH (t:Team {name:r.away_team})
        MERGE (f)-[:HAS_AWAY_TEAM]->(t)
    """, {"rows": away_rows.to_dict("records")})


    plays_rows = df.groupby(["name","element","position"]).size().reset_index()[["name","element","position"]]
    run_query("""
        UNWIND $rows AS r
        MATCH (p:Player {player_name:r.name, player_element:r.element})
        MATCH (pos:Position {name:r.position})
        MERGE (p)-[:PLAYS_AS]->(pos)
    """, {"rows": plays_rows.to_dict("records")})


        # 13. PLAYED_IN with all properties (add position)
    rel_rows = df.to_dict("records")

    run_query("""
        UNWIND $rows AS r
        MATCH (p:Player {player_name:r.name, player_element:r.element})
        MATCH (f:Fixture {season:r.season, fixture_number:r.fixture})
        MERGE (p)-[pl:PLAYED_IN]->(f)
        SET pl.position = r.position,
            pl.minutes = r.minutes,
            pl.goals_scored = r.goals_scored,
            pl.assists = r.assists,
            pl.total_points = r.total_points,
            pl.bonus = r.bonus,
            pl.clean_sheets = r.clean_sheets,
            pl.goals_conceded = r.goals_conceded,
            pl.own_goals = r.own_goals,
            pl.penalties_saved = r.penalties_saved,
            pl.penalties_missed = r.penalties_missed,
            pl.yellow_cards = r.yellow_cards,
            pl.red_cards = r.red_cards,
            pl.saves = r.saves,
            pl.bps = r.bps,
            pl.influence = r.influence,
            pl.creativity = r.creativity,
            pl.threat = r.threat,
            pl.ict_index = r.ict_index,
            pl.form = r.form
    """, {"rows": rel_rows})



    print("Knowledge Graph created successfully!")


if __name__ == "__main__":
    build_kg()
    driver.close()
