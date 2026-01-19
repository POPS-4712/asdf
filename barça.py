import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson
import plotly.graph_objects as go

# Configuración técnica
API_KEY = "9cf1175deaaf48c5b596552f69658d6e"
BARCA_ID = 81
HEADERS = {'X-Auth-Token': API_KEY}

@st.cache_data(ttl=3600)
def cargar_datos_totales():
    # Clasificación y últimos resultados
    standings = requests.get("https://api.football-data.org/v4/competitions/PD/standings", headers=HEADERS).json()['standings'][0]['table']
    history = requests.get(f"https://api.football-data.org/v4/teams/{BARCA_ID}/matches?status=FINISHED&limit=20", headers=HEADERS).json()['matches']
    next_m = requests.get(f"https://api.football-data.org/v4/teams/{BARCA_ID}/matches?status=SCHEDULED&limit=1", headers=HEADERS).json()['matches'][0]
    return standings, history, next_m

standings, history, next_match = cargar_datos_totales()

st.title("🏟️ Predictor: Spotify Camp Nou")

# --- INPUTS DE REALISMO ---
st.sidebar.header("Variables del Partido")
bajas = st.sidebar.multiselect("Bajas clave del Barça", ["Lewandowski", "Lamine Yamal", "Pedri", "Gavi", "Araujo", "Ter Stegen"])
cansancio = st.sidebar.checkbox("¿Viene de jugar Champions hace < 4 días?")

# --- CÁLCULO DE FUERZA ---
goles_ultimos = [m['score']['fullTime']['home'] if m['homeTeam']['id'] == BARCA_ID else m['score']['fullTime']['away'] for m in history]
fuerza_base = np.mean(goles_ultimos)

# Ajuste por Bajas (-10% por cada jugador clave)
penalizacion_bajas = 1 - (len(bajas) * 0.10)
# Ajuste por Cansancio (-5%)
penalizacion_fisica = 0.95 if cansancio else 1.0

# --- LÓGICA RIVAL ---
rival = next_match['homeTeam'] if next_match['homeTeam']['id'] != BARCA_ID else next_match['awayTeam']
es_local = next_match['homeTeam']['id'] == BARCA_ID
stats_rival = next((t for t in standings if t['team']['id'] == rival['id']), None)

if stats_rival:
    # Efecto Camp Nou: +25% si es local / -15% si es visitante (desplazamiento)
    factor_campo = 1.25 if es_local else 0.85
    defensa_rival_index = (stats_rival['goalsAgainst'] / stats_rival['playedGames']) / 1.3
    
    # Lambda Final Barça
    lambda_barca = fuerza_base * factor_campo * penalizacion_bajas * penalizacion_fisica * defensa_rival_index
    # Lambda Rival (Basado en goles encajados por el Barça: media 0.9)
    lambda_rival = (stats_rival['goalsFor'] / stats_rival['playedGames']) * 0.9 * (0.8 if es_local else 1.1)

    # --- PREDICCIÓN ---
    prob_m = np.outer(poisson.pmf(range(7), lambda_barca), poisson.pmf(range(7), lambda_rival))
    vic = np.sum(np.tril(prob_m, -1))
    emp = np.sum(np.diag(prob_m))
    der = np.sum(np.triu(prob_m, 1))

    # Interfaz
    st.header(f"Barça vs {rival['name']}")
    if es_local: st.success("🏟️ Partido en el Spotify Camp Nou")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Victoria Barça", f"{vic*100:.1f}%")
    col2.metric("Empate", f"{emp*100:.1f}%")
    col3.metric("Victoria Rival", f"{der*100:.1f}%")

    marcador = np.unravel_index(np.argmax(prob_m), prob_m.shape)
    st.info(f"📍 Resultado más probable: Barça {marcador[0]} - {marcador[1]} {rival['name']}")
