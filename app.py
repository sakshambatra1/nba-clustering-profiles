import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from rapidfuzz import process
import requests
import io 


# Load Data from Google Drive
url = "https://drive.google.com/uc?id=1cfxqTo-BGmgsfbPq46c1fccVatrpbmki"
response = requests.get(url, verify=False)  # Bypassing SSL verification
df = pd.read_csv(io.StringIO(response.text))

# Define Position-Based Metrics
position_metrics = {
    "PG": ["AST", "FG%", "3P%", "PPS", "STL+BLK", "TOV", "FT%"],
    "SG": ["AST", "FG%", "3P%", "PPS", "STL+BLK", "FGA", "TS%", "FT%"],
    "SF": ["AST", "FG%", "3P%", "PPS", "STL+BLK", "TRB", "TS%", "FT%"],
    "PF": ["AST", "FG%", "2P%", "TRB", "STL+BLK", "PPS", "BLK", "REB%"],
    "C": ["AST", "FG%", "2P%", "TRB", "STL+BLK", "ORB_Rate", "BLK", "REB%"]
}

# Function to Determine Position
def get_position(player_name):
    player_row = df[df["Player"] == player_name]
    if not player_row.empty:
        pos_values = player_row[["Pos_PG", "Pos_SG", "Pos_SF", "Pos_PF", "Pos_C"]].values.flatten()
        positions = ["PG", "SG", "SF", "PF", "C"]
        return positions[np.argmax(pos_values)]
    return "SF"

# Function to Find Closest Matching Player
def find_closest_player(player_name):
    players_list = df["Player"].tolist()
    match, score, _ = process.extractOne(player_name, players_list)
    return match if score > 70 else None  # Only return if confidence score is high

# Function to Create Radar Chart
def radar_chart(player1, player2):
    pos1 = get_position(player1)
    pos2 = get_position(player2)
    stats_to_use = position_metrics.get(pos1, position_metrics["SF"])
    
    stats1 = df[df["Player"] == player1][stats_to_use].values.flatten()
    stats2 = df[df["Player"] == player2][stats_to_use].values.flatten()
    
    labels = stats_to_use
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    stats1 = np.concatenate((stats1, [stats1[0]]))
    stats2 = np.concatenate((stats2, [stats2[0]]))
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
    ax.fill(angles, stats1, color='blue', alpha=0.25, label=player1)
    ax.fill(angles, stats2, color='red', alpha=0.25, label=player2)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.legend()
    st.pyplot(fig)

# Streamlit App
st.title("NBA Player Clustering Explorer")
st.write("Find players with similar profiles based on per 100 possessions data.")

# Player Search Feature
player_name = st.text_input("Search for a Player (format: 2024-2025 Luka Doncic):")
if player_name:
    closest_match = find_closest_player(player_name)
    if closest_match:
        player_info = df[df["Player"] == closest_match]
        st.write(f"### {closest_match} belongs to: {player_info.iloc[0]['Cluster Name']}")
        
        # Show similar players
        similar_players = df[df["Cluster Name"] == player_info.iloc[0]["Cluster Name"]]
        selected_similar = st.selectbox("Compare with:", similar_players["Player"].unique())
        
        # Generate Radar Chart
        if selected_similar:
            radar_chart(closest_match, selected_similar)
    else:
        st.write("Player not found. Try refining your search.")
