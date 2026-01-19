import streamlit as st
import requests
import pandas as pd
import numpy as np
from scipy.stats import poisson
import plotly.express as px

# Configuración de la página
st.set_page_config(page_title="Barça Predictor", page_icon="🔵🔴")
st.title("🔵🔴 FC Barcelona: Predictor Estadístico")

API_KEY = "9cf1175deaaf48c5b596552f69658d6e"
HEADERS = {'X-Auth-Token': API_KEY}
BARCA_ID = 81

@st.cache_data(ttl=3600)
def get_data():
    # Histórico de goles
    url_hist = f"https://api.football-data.org/v4/teams/{BARCA_ID}/matches?status=FINISHED&limit=20"
    res_hist = requests.get(url_hist, headers=HEADERS).json()
    
    # Próximo partido
    url_next = f"https://api.football-data.org/v4/teams/{BARCA_ID}/matches?status=SCHEDULED&limit=1"
    res_next = requests.get(url_next, headers=HEADERS).json()
    
    goles_f, goles_c = [], []
    for m in res_hist['matches']:
        if m['homeTeam']['id'] == BARCA_ID:
            goles_f.append(m['score']['fullTime']['home'])
            goles_c.append(m['score']['fullTime']['away'])
        else:
            goles_f.append(m['score']['fullTime']['away'])
            goles_c.append(m['score']['fullTime']['home'])
            
    proximo = res_next['matches'][0] if res_next['matches'] else None
    return np.mean(goles_f), np.mean(goles_c), proximo

m_favor, m_contra, proximo = get_data()

# --- INTERFAZ DE USUARIO ---
col1, col2 = st.columns(2)
with col1:
    st.metric("Goles Favor (Media)", f"{m_favor:.2f}")
with col2:
    st.metric("Goles Contra (Media)", f"{m_contra:.2f}")

if proximo:
    rival = proximo['homeTeam']['name'] if proximo['homeTeam']['id'] != BARCA_ID else proximo['awayTeam']['name']
    st.subheader(f"Próximo Encuentro: vs {rival}")
    
    # Slider para ajustar la dificultad del rival
    nivel_rival = st.slider("Nivel defensivo del rival (Goles que suele recibir)", 0.5, 3.0, 1.2)
    
    # Cálculo
    exp_b = (m_favor + nivel_rival) / 2
    exp_r = m_contra
    
    prob_m = np.outer(poisson.pmf(range(6), exp_b), poisson.pmf(range(6), exp_r))
    vic = np.sum(np.tril(prob_m, -1))
    emp = np.sum(np.diag(prob_m))
    der = np.sum(np.triu(prob_m, 1))

    # Gráfico de tarta
    fig = px.pie(values=[vic, emp, der], names=['Victoria Barça', 'Empate', 'Derrota'], 
                 color_discrete_sequence=['#004d98', '#A5A5A5', '#DB0030'])
    st.plotly_chart(fig)

    st.write(f"**Marcador más probable:** {np.unravel_index(np.argmax(prob_m), prob_m.shape)[0]} - {np.unravel_index(np.argmax(prob_m), prob_m.shape)[1]}")