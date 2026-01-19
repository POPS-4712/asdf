import streamlit as st
import requests
import numpy as np
from scipy.stats import poisson

# Configuración de acceso
API_KEY = "9cf1175deaaf48c5b596552f69658d6e"
BARCA_ID = 81
HEADERS = {'X-Auth-Token': API_KEY}

# Configuración de estilo directa
st.set_page_config(page_title="Barça Predictor", page_icon="🔵🔴")
st.markdown("""
    <style>
    .main { background-color: #004d98; color: white; }
    .stMetric { background-color: #A50044; padding: 15px; border-radius: 10px; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def obtener_todo_automatico():
    # 1. Traer tabla para nivel del rival
    standings = requests.get("https://api.football-data.org/v4/competitions/PD/standings", headers=HEADERS).json()['standings'][0]['table']
    # 2. Traer racha del Barça
    history = requests.get(f"https://api.football-data.org/v4/teams/{BARCA_ID}/matches?status=FINISHED&limit=15", headers=HEADERS).json()['matches']
    # 3. Traer el próximo partido
    next_m = requests.get(f"https://api.football-data.org/v4/teams/{BARCA_ID}/matches?status=SCHEDULED&limit=1", headers=HEADERS).json()['matches'][0]
    return standings, history, next_m

try:
    standings, history, next_match = obtener_todo_automatico()

    # --- PROCESAMIENTO AUTOMÁTICO ---
    # Media de goles racha actual
    goles_barca = [m['score']['fullTime']['home'] if m['homeTeam']['id'] == BARCA_ID else m['score']['fullTime']['away'] for m in history]
    fuerza_barca = np.mean(goles_barca)

    # Datos del Rival
    es_local = next_match['homeTeam']['id'] == BARCA_ID
    rival = next_match['homeTeam'] if not es_local else next_match['awayTeam']
    stats_rival = next((t for t in standings if t['team']['id'] == rival['id']), None)

    # Cálculo Realista Automático
    # Factor Camp Nou: 1.25 si es en casa
    factor_casa = 1.25 if es_local else 0.85
    
    # Dificultad basada en la defensa del rival (si no hay datos de liga, usamos 1.1 por defecto)
    defensa_rival = (stats_rival['goalsAgainst'] / stats_rival['playedGames']) / 1.3 if stats_rival else 1.1
    
    # Goles esperados (Lambda)
    lambda_barca = fuerza_barca * factor_casa * defensa_rival
    lambda_rival = 1.0 # Promedio de goles recibidos por el Barça actual

    # Probabilidades
    prob_m = np.outer(poisson.pmf(range(6), lambda_barca), poisson.pmf(range(6), lambda_rival))
    vic = np.sum(np.tril(prob_m, -1))
    marcador = np.unravel_index(np.argmax(prob_m), prob_m.shape)

    # --- INTERFAZ DIRECTA ---
    st.title("🔵🔴 Próximo Partido")
    st.header(f"{next_match['homeTeam']['name']} vs {next_match['awayTeam']['name']}")
    
    if es_local:
        st.subheader("🏟️ Spotify Camp Nou")
    
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probabilidad de Victoria", f"{vic*100:.1f}%")
    with col2:
        st.metric("Goles Esperados Barça", f"{lambda_barca:.2f}")

    st.divider()
    st.markdown(f"<h1 style='text-align: center;'>Predicción: {marcador[0]} - {marcador[1]}</h1>", unsafe_allow_html=True)

except Exception as e:
    st.error("Esperando datos de la API... Asegúrate de que el Barça tenga partidos programados.")
