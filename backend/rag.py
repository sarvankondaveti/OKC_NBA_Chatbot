import os
import json
import sqlalchemy as sa
import pandas as pd
from sqlalchemy import text
from backend.config import DB_DSN, EMBED_MODEL, LLM_MODEL
from backend.utils import ollama_embed, ollama_generate

BASE_DIR = os.path.dirname(__file__)
QUESTIONS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "part1", "questions.json"))
ANSWERS_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "part1", "answers.json"))
TEMPLATE_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "part1", "answers_template.json"))

def get_team_name(team_id, teams_df):
    """Get team name from team_id"""
    team = teams_df[teams_df['team_id'] == team_id]
    if not team.empty:
        return f"{team.iloc[0]['city']} {team.iloc[0]['name']}"
    return f"Team_{team_id}"

def get_player_name(player_id, players_df):
    """Get player name from player_id"""
    player = players_df[players_df['player_id'] == player_id]
    if not player.empty:
        return f"{player.iloc[0]['first_name']} {player.iloc[0]['last_name']}"
    return f"Player_{player_id}"

def is_player_question(question):
    """Determine if question is asking about player stats vs game results"""
    player_keywords = ['player', 'scored', 'points', 'assists', 'rebounds', 'leading scorer', 'triple-double']
    return any(keyword.lower() in question.lower() for keyword in player_keywords)

def retrieve_games(cx, qvec, k=5):
    """Retrieve similar games from game_details"""
    sql = (
        "SELECT game_id, game_timestamp, home_team_id, away_team_id, home_points, away_points, "
        "1 - (embedding <=> (:q)::vector) AS score FROM game_details "
        "WHERE embedding IS NOT NULL ORDER BY embedding <-> (:q)::vector LIMIT :k"
    )
    return cx.execute(text(sql), {"q": qvec, "k": k}).mappings().all()

def retrieve_players(cx, qvec, k=10):
    """Retrieve similar player performances from player_box_scores"""
    sql = (
        "SELECT game_id, person_id, team_id, points, assists, offensive_reb, defensive_reb, "
        "steals, blocks, starter, seconds, "
        "1 - (embedding <=> (:q)::vector) AS score FROM player_box_scores "
        "WHERE embedding IS NOT NULL ORDER BY embedding <-> (:q)::vector LIMIT :k"
    )
    return cx.execute(text(sql), {"q": qvec, "k": k}).mappings().all()

def build_game_context(rows, teams_df):
    """Build context from game results with team names"""
    context_lines = []
    for r in rows:
        home_team = get_team_name(r['home_team_id'], teams_df)
        away_team = get_team_name(r['away_team_id'], teams_df)
        date = pd.to_datetime(r['game_timestamp']).strftime('%Y-%m-%d')
        context_lines.append(
            f"Game {r['game_id']} on {date}: {away_team} vs {home_team}, "
            f"Final: {home_team} {r['home_points']}, {away_team} {r['away_points']}"
        )
    return "\n".join(context_lines)

def build_player_context(rows, teams_df, players_df, cx):
    """Build context from player performances with names"""
    context_lines = []
    for r in rows:
        player_name = get_player_name(r['person_id'], players_df)
        team_name = get_team_name(r['team_id'], teams_df)
        total_reb = r['offensive_reb'] + r['defensive_reb']
        
        # Get game info for this performance
        game_info = cx.execute(text(
            "SELECT game_timestamp, home_team_id, away_team_id, home_points, away_points "
            "FROM game_details WHERE game_id = :gid"
        ), {"gid": r['game_id']}).mappings().first()
        
        if game_info:
            date = pd.to_datetime(game_info['game_timestamp']).strftime('%Y-%m-%d')
            home_team = get_team_name(game_info['home_team_id'], teams_df)
            away_team = get_team_name(game_info['away_team_id'], teams_df)
            
            context_lines.append(
                f"Game {r['game_id']} on {date} ({away_team} vs {home_team}): "
                f"{player_name} ({team_name}) - {r['points']} pts, {r['assists']} ast, "
                f"{total_reb} reb, {r['steals']} stl, {r['blocks']} blk"
            )
    return "\n".join(context_lines)


def answer(question, context, question_id, evidence):
    """Generate answer using enhanced context and proper formatting"""
    
    # Enhanced prompt for better JSON formatting
    prompt = f"""You are an NBA statistics expert. Answer the question using ONLY the provided context.

IMPORTANT: Your response must be valid JSON in this exact format based on question ID {question_id}:

For game questions (IDs 1,2,3): {{"points": number}} or {{"winner": "Team Name", "score": "XXX-XXX"}}
For player questions (IDs 4,5,7,8,9,10): {{"player_name": "First Last", "points": number}}
For triple-double questions (ID 6): {{"player_name": "First Last", "points": number, "rebounds": number, "assists": number}}

Context:
{context}

Question: {question}

Answer (JSON only):"""
    
    return ollama_generate(LLM_MODEL, prompt)


if __name__ == "__main__":
    print("Starting Enhanced RAG Pipeline...")
    eng = sa.create_engine(DB_DSN)
    
    # Load reference data
    teams_df = pd.read_sql("SELECT team_id, city, name, abbreviation FROM teams", eng)
    players_df = pd.read_sql("SELECT player_id, first_name, last_name FROM players", eng)
    
    with open(QUESTIONS_PATH, encoding="utf-8") as f:
        qs = json.load(f)
    
    results = []
    
    with eng.begin() as cx:
        for i, q in enumerate(qs):
            print(f"Processing question {q['id']}: {q['question'][:50]}...")
            
            # Create embedding for question
            qvec = ollama_embed(EMBED_MODEL, q["question"])
            
            # Determine if it's a player or game question
            if is_player_question(q["question"]):
                # Retrieve player performances
                player_rows = retrieve_players(cx, qvec, 10)
                context = build_player_context(player_rows, teams_df, players_df, cx)
                evidence = [{"table": "player_box_score", "id": int(r["game_id"])} for r in player_rows[:3]]
            else:
                # Retrieve games
                game_rows = retrieve_games(cx, qvec, 5)
                context = build_game_context(game_rows, teams_df)
                evidence = [{"table": "game_details", "id": int(r["game_id"])} for r in game_rows[:3]]
            
            # Generate answer
            raw_answer = answer(q["question"], context, q["id"], evidence)
            
            # Try to parse JSON response, fallback if needed
            try:
                # Clean up the response to extract JSON
                clean_answer = raw_answer.strip()
                if clean_answer.startswith('```json'):
                    clean_answer = clean_answer.replace('```json', '').replace('```', '')
                elif clean_answer.startswith('```'):
                    clean_answer = clean_answer.replace('```', '')
                
                parsed_answer = json.loads(clean_answer.strip())
                
                results.append({
                    "id": q["id"],
                    "result": {**parsed_answer, "evidence": evidence}
                })
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON for question {q['id']}, using raw answer")
                results.append({
                    "id": q["id"],
                    "result": {"raw_answer": raw_answer, "evidence": evidence}
                })
    
    # Save results
    with open(ANSWERS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Completed! Generated answers for {len(results)} questions.")
    print(f"Results saved to: {ANSWERS_PATH}")
