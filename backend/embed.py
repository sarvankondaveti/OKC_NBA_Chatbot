import pandas as pd
import sqlalchemy as sa
from sqlalchemy import text
from backend.config import DB_DSN, EMBED_MODEL
from backend.utils import ollama_embed


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

def game_row_text(r, teams_df):
    """Enhanced game embedding with team names and readable format"""
    ts = pd.to_datetime(r.game_timestamp, utc=True)
    date = ts.strftime('%B %d, %Y')  
    
    home_team = get_team_name(r.home_team_id, teams_df)
    away_team = get_team_name(r.away_team_id, teams_df)
    
    # Determine winner
    if r.home_points > r.away_points:
        winner = home_team
        final_score = f"{r.home_points}-{r.away_points}"
    else:
        winner = away_team
        final_score = f"{r.away_points}-{r.home_points}"
    
    return (
        f"NBA Game on {date}: {away_team} vs {home_team}. "
        f"Final score: {home_team} {r.home_points}, {away_team} {r.away_points}. "
        f"Winner: {winner} ({final_score}). "
        f"Season: {r.season}-{r.season+1}. "
        f"Game ID: {r.game_id}"
    )

def player_row_text(r, teams_df, players_df):
    """Create embedding text for player box scores"""
    player_name = get_player_name(r.person_id, players_df)
    team_name = get_team_name(r.team_id, teams_df)
    
    # Create readable stats summary
    stats_parts = []
    if r.points > 0:
        stats_parts.append(f"{r.points} points")
    if r.assists > 0:
        stats_parts.append(f"{r.assists} assists")
    if r.defensive_reb + r.offensive_reb > 0:
        total_reb = r.defensive_reb + r.offensive_reb
        stats_parts.append(f"{total_reb} rebounds")
    if r.steals > 0:
        stats_parts.append(f"{r.steals} steals")
    if r.blocks > 0:
        stats_parts.append(f"{r.blocks} blocks")
    
    stats_text = ", ".join(stats_parts) if stats_parts else "0 points"
    
    # Check for notable performances
    performance_notes = []
    if r.points >= 30:
        performance_notes.append("high scoring game")
    if r.assists >= 10:
        performance_notes.append("double-digit assists")
    if (r.defensive_reb + r.offensive_reb) >= 10:
        performance_notes.append("double-digit rebounds")
    if r.points >= 10 and r.assists >= 10 and (r.defensive_reb + r.offensive_reb) >= 10:
        performance_notes.append("triple-double")
    
    performance_text = " (" + ", ".join(performance_notes) + ")" if performance_notes else ""
    
    return (
        f"Player performance: {player_name} from {team_name} scored {stats_text} "
        f"in game {r.game_id}{performance_text}. "
        f"Starter: {'Yes' if r.starter else 'No'}. "
        f"Minutes played: {r.seconds/60:.1f}"
    )


def main():
    print("Starting Enhanced Embedding Process")
    eng = sa.create_engine(DB_DSN)
    
    # Load reference data for team and player names
    print("Loading reference data...")
    teams_df = pd.read_sql("SELECT team_id, city, name, abbreviation FROM teams", eng)
    players_df = pd.read_sql("SELECT player_id, first_name, last_name FROM players", eng)
    
    with eng.begin() as cx:
        cx.execute(text('ALTER DATABASE nba REFRESH COLLATION VERSION'))
        
        # Setup game_details embeddings
        print("Setting up game_details embeddings...")
        cx.execute(text("ALTER TABLE IF EXISTS game_details ADD COLUMN IF NOT EXISTS embedding vector(768);"))
        cx.execute(text("CREATE INDEX IF NOT EXISTS idx_game_details_embedding ON game_details USING hnsw (embedding vector_cosine_ops);"))
        
        # Setup player_box_scores embeddings
        print("Setting up player_box_scores embeddings...")
        cx.execute(text("ALTER TABLE IF EXISTS player_box_scores ADD COLUMN IF NOT EXISTS embedding vector(768);"))
        cx.execute(text("CREATE INDEX IF NOT EXISTS idx_player_box_scores_embedding ON player_box_scores USING hnsw (embedding vector_cosine_ops);"))
        
        # Process game_details embeddings
        print("Processing game_details embeddings...")
        games_df = pd.read_sql(
            "SELECT game_id, season, game_timestamp, home_team_id, away_team_id, home_points, away_points FROM game_details ORDER BY game_timestamp DESC, game_id DESC",
            cx,
        )
        
        for i, (_, r) in enumerate(games_df.iterrows()):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(games_df)} games...")
            text_content = game_row_text(r, teams_df)
            vec = ollama_embed(EMBED_MODEL, text_content)
            cx.execute(text("UPDATE game_details SET embedding = :v WHERE game_id = :gid"), 
                      {"v": vec, "gid": int(r.game_id)})
        
        # Process player_box_scores embeddings (sample for performance)
        print("Processing player_box_scores embeddings...")
        players_df_filtered = pd.read_sql(
            """SELECT game_id, person_id, team_id, starter, seconds, points, fg2_made, fg2_attempted, 
                      fg3_made, fg3_attempted, ft_attempted, ft_made, offensive_reb, defensive_reb, 
                      assists, steals, blocks, turnovers, defensive_fouls, offensive_fouls 
               FROM player_box_scores 
               WHERE points >= 20 OR assists >= 8 OR (defensive_reb + offensive_reb) >= 10 
                     OR (points >= 10 AND assists >= 10 AND (defensive_reb + offensive_reb) >= 10)
               ORDER BY points DESC, assists DESC
               LIMIT 5000""",
            cx,
        )
        
        for i, (_, r) in enumerate(players_df_filtered.iterrows()):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(players_df_filtered)} player performances...")
            text_content = player_row_text(r, teams_df, players_df)
            vec = ollama_embed(EMBED_MODEL, text_content)
            cx.execute(text("UPDATE player_box_scores SET embedding = :v WHERE game_id = :gid AND person_id = :pid"), 
                      {"v": vec, "gid": int(r.game_id), "pid": int(r.person_id)})
    
    print(f"Finished Enhanced Embeddings:")
    print(f"  - {len(games_df)} game_details rows updated")
    print(f"  - {len(players_df_filtered)} player_box_scores rows updated")


if __name__ == "__main__":
    main()
