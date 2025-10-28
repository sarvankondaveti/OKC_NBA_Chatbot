from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlalchemy as sa
from backend.config import DB_DSN, EMBED_MODEL, LLM_MODEL
from backend.utils import ollama_embed, ollama_generate
from sqlalchemy import text

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

eng = sa.create_engine(DB_DSN)

class Q(BaseModel):
    question: str

@app.post("/api/chat")
def answer(q: Q):
    print(f'Received question: {q.question}')
    
    try:
        # Step 1: Generate embedding for the question using Ollama
        print("Generating question embedding...")
        qvec = ollama_embed(EMBED_MODEL, q.question)
        print(f"Generated embedding with {len(qvec)} dimensions")
        
        with eng.begin() as cx:
            # Step 2: Semantic retrieval using pgvector
            print("Performing semantic retrieval...")
            
            # Smart query routing - check if question needs player data
            question_lower = q.question.lower()
            
            # Default to game data (safer approach)
            needs_player_data = False
            
            # Check for player-specific questions
            player_indicators = [
                'leading scorer', 'triple-double', 'recorded a triple',
                'points did', 'scored', 'rebounds did', 'assists did',
                'performance', 'stats', 'statistics', 'how many points',
                'player had', 'player scored', 'player recorded'
            ]
            
            # Also check if question mentions specific player names or individual stats
            if any(indicator in question_lower for indicator in player_indicators):
                needs_player_data = True
            
            print(f"Question routing: {'PLAYER' if needs_player_data else 'GAME'} search for: {q.question[:50]}...")
            
            if needs_player_data:
                # Use semantic search for player data
                print("Using semantic search for player data...")
                player_rows = cx.execute(
                    text(
                        "SELECT p.game_id, p.person_id, p.points, p.offensive_reb + p.defensive_reb as rebounds, "
                        "p.assists, pl.first_name, pl.last_name, g.game_timestamp, "
                        "ht.name as home_team, at.name as away_team "
                        "FROM player_box_scores p "
                        "JOIN players pl ON p.person_id = pl.player_id "
                        "JOIN game_details g ON p.game_id = g.game_id "
                        "JOIN teams ht ON g.home_team_id = ht.team_id "
                        "JOIN teams at ON g.away_team_id = at.team_id "
                        "ORDER BY p.embedding <-> CAST(:q AS vector) LIMIT :k"
                    ),
                    {"q": qvec, "k": 5},  
                ).fetchall()
                combined_rows = []  # Use player data instead
            else:
                # Get game data for team/score questions
                combined_rows = cx.execute(
                    text(
                        "SELECT g.game_id, g.game_timestamp, g.home_team_id, g.away_team_id, "
                        "g.home_points, g.away_points, ht.name as home_team, at.name as away_team "
                        "FROM game_details g "
                        "JOIN teams ht ON g.home_team_id = ht.team_id "
                        "JOIN teams at ON g.away_team_id = at.team_id "
                        "ORDER BY g.embedding <-> CAST(:q AS vector) LIMIT :k"
                    ),
                    {"q": qvec, "k": 1},  
                ).fetchall()
                player_rows = []  
            
            print(f"Found {len(combined_rows)} games and {len(player_rows)} players")
            
            # Step 3: Build context from retrieved data
            context = ""
            evidence = []
            
            # Add game data if available (simplified for speed)
            if combined_rows:
                for row in combined_rows:
                    game_id = int(row.game_id)
                    home_points = int(row.home_points) if row.home_points else 0
                    away_points = int(row.away_points) if row.away_points else 0
                    home_team = str(row.home_team) if row.home_team else ""
                    away_team = str(row.away_team) if row.away_team else ""
                    timestamp = str(row.game_timestamp) if row.game_timestamp else ""
                    
                    # Clean date format - only show YYYY-MM-DD
                    date = timestamp[:10] if timestamp else ""
                    context += f"{away_team} {away_points}-{home_points} {home_team} {date}\n"
                    evidence.append({"table": "game_details", "id": game_id})
            
            # Add player data if available (simplified for speed)
            if player_rows:
                for row in player_rows:
                    game_id = int(row.game_id) if row.game_id else 0
                    points = int(row.points) if row.points else 0
                    rebounds = int(row.rebounds) if row.rebounds else 0
                    assists = int(row.assists) if row.assists else 0
                    first_name = str(row.first_name) if row.first_name else ""
                    last_name = str(row.last_name) if row.last_name else ""
                    timestamp = str(row.game_timestamp) if row.game_timestamp else ""
                    
                    # Clean date format - only show YYYY-MM-DD
                    date = timestamp[:10] if timestamp else ""
                    context += f"{first_name} {last_name}: {points} points, {rebounds} rebounds, {assists} assists {date}\n"
                    evidence.append({"table": "player_box_scores", "id": game_id})
        
        # Step 4: Generate answer using Llama with optimized prompt
        print("Generating answer with Llama...")
        
        if context.strip():
            # Optimized prompt for accuracy and speed
            prompt = f"""NBA Data:
{context.strip()}

Q: {q.question}
A: Based on this data,"""
            
            response = ollama_generate(LLM_MODEL, prompt)
        else:
            response = f"No specific data found for: {q.question}"
        
        print("Generated response successfully")
        
        return {
            "answer": response,
            "evidence": evidence[:5]  # Limit evidence to 5 items
        }
        
    except Exception as e:
        print(f"Error in RAG pipeline: {e}")
        return {
            "answer": f"Sorry, I encountered an error processing your question: {str(e)}",
            "evidence": []
        }