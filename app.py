import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson

st.set_page_config(page_title="Predictor Serie A Live & H2H", layout="wide", initial_sidebar_state="expanded")

st.title("⚽ Predictor Serie A - Live & Confronto H2H")
st.markdown("Applicazione statistica interattiva per l'elaborazione di pronostici e l'analisi Head-to-Head (H2H).")

# Recupero API Key dai Secrets di Streamlit o da input manuale
api_key = st.secrets.get("FOOTBALL_API_KEY", "")

if not api_key:
    api_key = st.sidebar.text_input("Inserisci API Key Football-Data.org", type="password")

@st.cache_data(ttl=3600)  # Mantiene i dati in cache per 1 ora
def carica_dati_serie_a(key):
    url = "https://api.football-data.org/v4/competitions/SA/standings"
    headers = {"X-Auth-Token": key}
    
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None, f"Errore API: {response.status_code}"
        
        data = response.json()
        standings = data['standings'][0]['table']
        
        squadre = []
        for item in standings:
            giocate = item['playedGames'] if item['playedGames'] > 0 else 1
            
            squadre.append({
                'ID': item['team']['id'],
                'Squadra': item['team']['name'],
                'Posizione': item['position'],
                'Giocate': item['playedGames'],
                'Vinte': item['won'],
                'Pareggiate': item['draw'],
                'Perse': item['lost'],
                'Punti': item['points'],
                'GF_Totali': item['goalsFor'],
                'GS_Totali': item['goalsAgainst'],
                'DR': item['goalDifference'],
                'GF_Casa': item['home']['goalsFor'] / max(1, item['home']['played']),
                'GS_Casa': item['home']['goalsAgainst'] / max(1, item['home']['played']),
                'GF_Trasferta': item['away']['goalsFor'] / max(1, item['away']['played']),
                'GS_Trasferta': item['away']['goalsAgainst'] / max(1, item['away']['played'])
            })
            
        return pd.DataFrame(squadre), None
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=3600)
def carica_h2h_dati(key, team1_id, team2_id):
    url = f"https://api.football-data.org/v4/teams/{team1_id}/matches?limit=50"
    headers = {"X-Auth-Token": key}
    try:
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return []
        matches = response.json().get('matches', [])
        h2h_matches = []
        for m in matches:
            home_id = m['homeTeam']['id']
            away_id = m['awayTeam']['id']
            if (home_id == team1_id and away_id == team2_id) or (home_id == team2_id and away_id == team1_id):
                if m['status'] == 'FINISHED':
                    h2h_matches.append({
                        'Data': m['utcDate'][:10],
                        'Casa': m['homeTeam']['name'],
                        'Risultato': f"{m['score']['fullTime']['home']} - {m['score']['fullTime']['away']}",
                        'Trasferta': m['awayTeam']['name'],
                        'Competizione': m['competition']['name']
                    })
        return h2h_matches
    except:
        return []

if api_key:
    df, error = carica_dati_serie_a(api_key)
    
    if error:
        st.error(f"Impossibile recuperare i dati. Verifica l'API Key. ({error})")
    else:
        st.sidebar.success("Dati Serie A aggiornati!")
        
        # Creazione delle schede dell'applicazione
        tab1, tab2, tab3 = st.tabs(["🔮 Pronostico Partita", "⚔️ Confronto Head-to-Head (H2H)", "📊 Classifica & Statistiche"])
        
        squadre = sorted(df['Squadra'].tolist())
        
        # --- TAB 1: PRONOSTICO ---
        with tab1:
            st.header("Elaborazione Pronostico (Modello di Poisson)")
            col1, col2 = st.columns(2)
            with col1:
                sq_casa = st.selectbox("Squadra in Casa", squadre, index=0, key="p_casa")
            with col2:
                sq_trasferta_options = [s for s in squadre if s != sq_casa]
                sq_trasferta = st.selectbox("Squadra in Trasferta", sq_trasferta_options, index=0, key="p_trasferta")
                
            if st.button("Calcola Pronostico", key="btn_pronostico"):
                media_gf_casa = max(0.1, df['GF_Casa'].mean())
                media_gs_casa = max(0.1, df['GS_Casa'].mean())
                media_gf_trasferta = max(0.1, df['GF_Trasferta'].mean())
                media_gs_trasferta = max(0.1, df['GS_Trasferta'].mean())
                
                riga_casa = df[df['Squadra'] == sq_casa].iloc[0]
                riga_trasferta = df[df['Squadra'] == sq_trasferta].iloc[0]
                
                attacco_casa = riga_casa['GF_Casa'] / media_gf_casa
                difesa_casa = riga_casa['GS_Casa'] / media_gs_casa
                attacco_trasferta = riga_trasferta['GF_Trasferta'] / media_gf_trasferta
                difesa_trasferta = riga_trasferta['GS_Trasferta'] / media_gs_trasferta
                
                lambda_casa = attacco_casa * difesa_trasferta * media_gf_casa
                lambda_trasferta = attacco_trasferta * difesa_casa * media_gf_trasferta
                
                max_gol = 6
                matrice = np.zeros((max_gol, max_gol))
                for i in range(max_gol):
                    for j in range(max_gol):
                        matrice[i, j] = poisson.pmf(i, lambda_casa) * poisson.pmf(j, lambda_trasferta)
                        
                prob_1 = np.sum(np.tril(matrice, -1))
                prob_X = np.sum(np.diag(matrice))
                prob_2 = np.sum(np.triu(matrice, 1))
                
                gol_totali = np.add.outer(np.arange(max_gol), np.arange(max_gol))
                prob_under25 = np.sum(matrice[gol_totali < 2.5])
                prob_over25 = np.sum(matrice[gol_totali > 2.5])
                prob_nogoal = np.sum(matrice[0, :]) + np.sum(matrice[:, 0]) - matrice[0, 0]
                prob_goal = 1 - prob_nogoal
                
                st.subheader(f"📊 Risultato Analisi: {sq_casa} vs {sq_trasferta}")
                cA, cB = st.columns(2)
                cA.metric("Gol Attesi Casa", f"{lambda_casa:.2f}")
                cB.metric("Gol Attesi Trasferta", f"{lambda_trasferta:.2f}")
                
                st.markdown("---")
                st.write("### Probabilità Esiti 1X2")
                c1, c2, c3 = st.columns(3)
                c1.metric("1 (Casa)", f"{prob_1*100:.1f}%")
                c2.metric("X (Pareggio)", f"{prob_X*100:.1f}%")
                c3.metric("2 (Trasferta)", f"{prob_2*100:.1f}%")
                
                st.write("### Altri Mercati")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Under 2.5", f"{prob_under25*100:.1f}%")
                m2.metric("Over 2.5", f"{prob_over25*100:.1f}%")
                m3.metric("Goal", f"{prob_goal*100:.1f}%")
                m4.metric("NoGoal", f"{prob_nogoal*100:.1f}%")
                
                opzioni = {'1': prob_1, 'X': prob_X, '2': prob_2, 'Under 2.5': prob_under25, 'Over 2.5': prob_over25, 'Goal': prob_goal, 'NoGoal': prob_nogoal}
                miglior_esito = max(opzioni, key=opzioni.get)
                st.success(f"💡 **Miglior Pronostico Consigliato:** **{miglior_esito}** con il **{opzioni[miglior_esito]*100:.1f}%** di probabilità.")

        # --- TAB 2: HEAD-TO-HEAD (H2H) ---
        with tab2:
            st.header("Confronto Diretto Head-to-Head (H2H)")
            
            c_h2h_1, c_h2h_2 = st.columns(2)
            with c_h2h_1:
                t1_name = st.selectbox("Seleziona Prima Squadra", squadre, index=0, key="h2h_1")
            with c_h2h_2:
                t2_options = [s for s in squadre if s != t1_name]
                t2_name = st.selectbox("Seleziona Seconda Squadra", t2_options, index=0, key="h2h_2")
                
            team1 = df[df['Squadra'] == t1_name].iloc[0]
            team2 = df[df['Squadra'] == t2_name].iloc[0]
            
            st.markdown("### 📈 Confronto Rendimento Stagionale")
            
            comp_df = pd.DataFrame({
                'Statistica': ['Posizione in Classifica', 'Punti Totali', 'Partite Giocate', 'Vittorie', 'Pareggi', 'Sconfitte', 'Gol Fatti', 'Gol Subiti', 'Differenza Reti'],
                t1_name: [team1['Posizione'], team1['Punti'], team1['Giocate'], team1['Vinte'], team1['Pareggiate'], team1['Perse'], team1['GF_Totali'], team1['GS_Totali'], team1['DR']],
                t2_name: [team2['Posizione'], team2['Punti'], team2['Giocate'], team2['Vinte'], team2['Pareggiate'], team2['Perse'], team2['GF_Totali'], team2['GS_Totali'], team2['DR']]
            })
            
            st.dataframe(comp_df, use_container_width=True, hide_index=True)
            
            # Grafico comparativo
            st.markdown("### 📊 Grafico Comparativo Gol & Punti")
            chart_data = pd.DataFrame({
                'Squadra': [t1_name, t2_name],
                'Punti': [team1['Punti'], team2['Punti']],
                'Gol Fatti': [team1['GF_Totali'], team2['GF_Totali']],
                'Gol Subiti': [team1['GS_Totali'], team2['GS_Totali']]
            }).set_index('Squadra')
            
            st.bar_chart(chart_data)
            
            # Precedenti Scontri Diretti
            st.markdown("### 📜 Storico Precedenti Scontri Diretti")
            with st.spinner("Caricamento scontri diretti recenti..."):
                h2h_list = carica_h2h_dati(api_key, team1['ID'], team2['ID'])
                
            if h2h_list:
                df_h2h = pd.DataFrame(h2h_list)
                st.table(df_h2h)
            else:
                st.info(f"Nessun precedente recente registrato negli ultimi dati per {t1_name} vs {t2_name}.")

        # --- TAB 3: CLASSIFICA ---
        with tab3:
            st.header("Classifica Generale & Rendimento Casa/Trasferta")
            st.dataframe(
                df[['Posizione', 'Squadra', 'Punti', 'Giocate', 'Vinte', 'Pareggiate', 'Perse', 'GF_Totali', 'GS_Totali', 'DR', 'GF_Casa', 'GS_Casa', 'GF_Trasferta', 'GS_Trasferta']].sort_values('Posizione'),
                use_container_width=True,
                hide_index=True
            )
else:
    st.warning("Inserisci una API Key valida nella barra laterale o nei Secrets per iniziare.")
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
