import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import poisson

st.set_page_config(page_title="Pronostici Serie A", layout="wide")

st.title("⚽ Predictor Serie A - Modello di Poisson")
st.markdown("Applicazione statistica per elaborare i migliori pronostici basati sulla distribuzione dei gol.")

data = {
    'Squadra': [
        'Inter', 'Milan', 'Juventus', 'Atalanta', 'Roma', 
        'Lazio', 'Napoli', 'Fiorentina', 'Bologna', 'Torino',
        'Monza', 'Udinese', 'Genoa', 'Lecce', 'Verona',
        'Cagliari', 'Empoli', 'Parma', 'Como', 'Venezia'
    ],
    'GF_Casa': [2.1, 1.9, 1.8, 2.2, 1.6, 1.5, 1.7, 1.4, 1.3, 1.1, 1.0, 1.2, 1.1, 0.9, 1.0, 1.1, 0.9, 1.0, 0.9, 0.8],
    'GS_Casa': [0.7, 0.9, 0.6, 1.1, 1.0, 1.1, 0.8, 1.0, 0.9, 1.0, 1.2, 1.3, 1.1, 1.2, 1.3, 1.2, 1.2, 1.4, 1.3, 1.4],
    'GF_Trasferta': [1.8, 1.6, 1.4, 1.7, 1.3, 1.2, 1.4, 1.1, 1.0, 0.9, 0.8, 0.9, 0.9, 0.7, 0.8, 0.8, 0.7, 0.8, 0.8, 0.6],
    'GS_Trasferta': [0.8, 1.1, 0.7, 1.3, 1.2, 1.3, 0.9, 1.2, 1.1, 1.2, 1.4, 1.5, 1.3, 1.4, 1.5, 1.4, 1.3, 1.6, 1.5, 1.6]
}
# 1. Dati dimostrativi della stagione (Medie Gol Fatti/Subiti)
# In un'evoluzione futura questi dati potranno essere scaricati automaticamente via API

df = pd.DataFrame(data)

# Calcolo medie di campionato
media_gf_casa = df['GF_Casa'].mean()
media_gs_casa = df['GS_Casa'].mean()
media_gf_trasferta = df['GF_Trasferta'].mean()
media_gs_trasferta = df['GS_Trasferta'].mean()

# Funzione per calcolare le probabilità con la Distribuzione di Poisson
def calcola_probabilita(squadra_casa, squadra_trasferta):
    riga_casa = df[df['Squadra'] == squadra_casa].iloc[0]
    riga_trasferta = df[df['Squadra'] == squadra_trasferta].iloc[0]
    
    # Forza attacco e difesa
    attacco_casa = riga_casa['GF_Casa'] / media_gf_casa
    difesa_casa = riga_casa['GS_Casa'] / media_gs_casa
    
    attacco_trasferta = riga_trasferta['GF_Trasferta'] / media_gf_trasferta
    difesa_trasferta = riga_trasferta['GS_Trasferta'] / media_gs_trasferta
    
    # Gol attesi (Lambda)
    lambda_casa = attacco_casa * difesa_trasferta * media_gf_casa
    lambda_trasferta = attacco_trasferta * difesa_casa * media_gf_trasferta
    
    # Matrice di probabilità per i risultati da 0x0 a 5x5
    max_gol = 6
    matrice = np.zeros((max_gol, max_gol))
    
    for i in range(max_gol):
        for j in range(max_gol):
            matrice[i, j] = poisson.pmf(i, lambda_casa) * poisson.pmf(j, lambda_trasferta)
            
    prob_1 = np.sum(np.tril(matrice, -1))
    prob_X = np.sum(np.diag(matrice))
    prob_2 = np.sum(np.triu(matrice, 1))
    
    # Calcolo Under / Over 2.5
    gol_totali = np.add.outer(np.arange(max_gol), np.arange(max_gol))
    prob_under25 = np.sum(matrice[gol_totali < 2.5])
    prob_over25 = np.sum(matrice[gol_totali > 2.5])
    
    # Calcolo Goal / NoGoal
    prob_nogoal = np.sum(matrice[0, :]) + np.sum(matrice[:, 0]) - matrice[0, 0]
    prob_goal = 1 - prob_nogoal
    
    return {
        'lambda_casa': lambda_casa,
        'lambda_trasferta': lambda_trasferta,
        '1': prob_1,
        'X': prob_X,
        '2': prob_2,
        'Under 2.5': prob_under25,
        'Over 2.5': prob_over25,
        'Goal': prob_goal,
        'NoGoal': prob_nogoal
    }

# Interfaccia Utente
col1, col2 = st.columns(2)

squadre = df['Squadra'].tolist()

with col1:
    sq_casa = st.selectbox("Squadra in Casa", squadre, index=0)

with col2:
    sq_trasferta_options = [s for s in squadre if s != sq_casa]
    sq_trasferta = st.selectbox("Squadra in Trasferta", sq_trasferta_options, index=0)

if st.button("Calcola Pronostico"):
    res = calcola_probabilita(sq_casa, sq_trasferta)
    
    st.subheader(f"📊 Pronostico: {sq_casa} vs {sq_trasferta}")
    
    col_a, col_b = st.columns(2)
    col_a.metric("Gol Attesi Casa", f"{res['lambda_casa']:.2f}")
    col_b.metric("Gol Attesi Trasferta", f"{res['lambda_trasferta']:.2f}")
    
    st.markdown("---")
    st.write("### Probabilità Esiti 1X2")
    c1, c2, c3 = st.columns(3)
    c1.metric("Vittoria Casa (1)", f"{res['1']*100:.1f}%")
    c2.metric("Pareggio (X)", f"{res['X']*100:.1f}%")
    c3.metric("Vittoria Trasferta (2)", f"{res['2']*100:.1f}%")
    
    st.write("### Altri Mercati")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Under 2.5", f"{res['Under 2.5']*100:.1f}%")
    m2.metric("Over 2.5", f"{res['Over 2.5']*100:.1f}%")
    m3.metric("Goal", f"{res['Goal']*100:.1f}%")
    m4.metric("NoGoal", f"{res['NoGoal']*100:.1f}%")
    
    # Suggerimento del pronostico migliore
    st.markdown("---")
    st.write("### 💡 Suggerimento Miglior Pronostico")
    
    opzioni = {
        '1': res['1'],
        'X': res['X'],
        '2': res['2'],
        'Under 2.5': res['Under 2.5'],
        'Over 2.5': res['Over 2.5'],
        'Goal': res['Goal'],
        'NoGoal': res['NoGoal']
    }
    
    miglior_esito = max(opzioni, key=opzioni.get)
    percentuale = opzioni[miglior_esito] * 100
    
    st.success(f"L'esito statisticamente più probabile è **{miglior_esito}** con la probabilità del **{percentuale:.1f}%**")

st.markdown("---")
st.write("### Tabella Dati Squadre")
st.dataframe(df)
7
