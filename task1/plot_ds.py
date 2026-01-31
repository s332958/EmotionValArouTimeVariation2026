import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

# Parametri di soglia globali per il filtraggio e la visualizzazione
PHRASE_X_USER = 10
COUNT_WORD_IN_PHRASE = 10

def analizza_e_visualizza(file_path):
    # Creazione della directory per i grafici
    os.makedirs('task1/img', exist_ok=True)

    # 1. Caricamento del dataset
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Errore nel caricamento del file: {e}")
        return

    # --- PREPARAZIONE E ANALISI DATI ---
    
    # Normalizzazione colonna booleana per identificare liste di parole vs frasi
    df['is_words_bool'] = df['is_words'].astype(str).str.lower() == 'true'
    
    # 1.1 Conteggio tipologia righe (Data Distribution)
    conteggio_tipologia = df['is_words_bool'].value_counts()
    n_liste_parole = conteggio_tipologia.get(True, 0)
    n_frasi_normali = conteggio_tipologia.get(False, 0)

    # 1.2 Statistiche Utenti (Longitudinality)
    frasi_per_utente = df.groupby('user_id').size()
    u_meno_soglia = (frasi_per_utente < PHRASE_X_USER).sum()
    u_sopra_soglia = (frasi_per_utente >= PHRASE_X_USER).sum()

    # 1.3 Analisi Parole Target (Soggettività e Feeling Words)
    df_is_words_true = df[df['is_words_bool'] == True].copy()
    tutte_le_parole_target = []
    for text in df_is_words_true['text'].dropna():
        # Estrazione parole separate da virgola nelle righe di metadati
        tutte_le_parole_target.extend([p.strip().lower() for p in str(text).split(',') if p.strip()])
    
    set_parole_target = set(tutte_le_parole_target)
    frasi_corpus = df[df['is_words_bool'] == False]['text'].dropna().astype(str).tolist()
    
    # Conteggio occorrenze reali delle parole target nei testi scritti
    conteggio_nelle_frasi = Counter()
    for parola in set_parole_target:
        pattern = re.compile(rf'\b{re.escape(parola)}\b', re.IGNORECASE)
        for frase in frasi_corpus:
            conteggio_nelle_frasi[parola] += len(pattern.findall(frase))
    
    parole_counts = np.array(list(conteggio_nelle_frasi.values()))
    p_meno_soglia = sum(1 for p in set_parole_target if conteggio_nelle_frasi[p] < COUNT_WORD_IN_PHRASE)
    p_sopra_soglia = sum(1 for p in set_parole_target if conteggio_nelle_frasi[p] >= COUNT_WORD_IN_PHRASE)

    # --- FUNZIONI DI SUPPORTO GRAFICO ---
    def add_labels(ax):
        """Aggiunge il valore numerico sopra ogni barra del grafico."""
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='center', xytext=(0, 9), 
                        textcoords='offset points', fontsize=11, fontweight='bold')

    sns.set_theme(style="whitegrid")

    # --- GENERAZIONE GRAFICI ---

    # G1: Distribuzione Tipologia Righe
    plt.figure(figsize=(8, 6))
    ax0 = sns.barplot(x=['Essays/Phrases', 'Feeling Word Lists'], y=[n_frasi_normali, n_liste_parole], palette='coolwarm')
    add_labels(ax0)
    plt.title('Distribution of Row Types (Subtask 1)')
    plt.ylabel('Total Count')
    plt.savefig('task1/img/grafico_tipologia_righe.png', bbox_inches='tight')
    plt.close()

    # G2: Attività Utenti (Barre)
    plt.figure(figsize=(8, 6))
    ax1 = sns.barplot(x=[f'< {PHRASE_X_USER} texts', f'>= {PHRASE_X_USER} texts'], y=[u_meno_soglia, u_sopra_soglia], palette='viridis')
    add_labels(ax1)
    plt.title(f'User Longitudinal Activity (Threshold: {PHRASE_X_USER})')
    plt.ylabel('Number of Users')
    plt.savefig('task1/img/grafico_utenti_barre.png', bbox_inches='tight')
    plt.close()

    # G3: CDF Utenti (Cumulativo)
    plt.figure(figsize=(10, 6))
    sorted_u = np.sort(frasi_per_utente.values)
    cum_u = np.arange(1, len(sorted_u) + 1) / len(sorted_u)
    plt.step(sorted_u, cum_u, where='post', color='#2ecc71', linewidth=2, label='User CDF')
    plt.fill_between(sorted_u, cum_u, step="post", alpha=0.2, color='#2ecc71')
    plt.axvline(x=PHRASE_X_USER, color='red', linestyle='--', label=f'Chosen Threshold ({PHRASE_X_USER})')
    plt.title('Empirical Cumulative Distribution: Texts per User')
    plt.xlabel('Number of Texts')
    plt.ylabel('Cumulative Fraction of Users')
    plt.legend()
    plt.savefig('task1/img/grafico_cumulativo_utenti.png', bbox_inches='tight')
    plt.close()

    # G4: Frequenza Parole Target (Barre)
    plt.figure(figsize=(8, 6))
    ax2 = sns.barplot(x=[f'< {COUNT_WORD_IN_PHRASE} occ.', f'>= {COUNT_WORD_IN_PHRASE} occ.'], y=[p_meno_soglia, p_sopra_soglia], palette='magma')
    add_labels(ax2)
    plt.title(f'Target Words Frequency in Essays (Threshold: {COUNT_WORD_IN_PHRASE})')
    plt.ylabel('Number of Unique Words')
    plt.savefig('task1/img/grafico_parole_barre.png', bbox_inches='tight')
    plt.close()

    # G5: CDF Parole Target (Cumulativo con scala granulare 0-5-10...)
    plt.figure(figsize=(12, 6))
    if len(parole_counts) > 0:
        sorted_p = np.sort(parole_counts)
        cum_p = np.arange(1, len(sorted_p) + 1) / len(sorted_p)
        
        plt.step(sorted_p, cum_p, where='post', color='#e67e22', linewidth=2, label='Word CDF')
        plt.fill_between(sorted_p, cum_p, step="post", alpha=0.2, color='#e67e22')
        plt.axvline(x=COUNT_WORD_IN_PHRASE, color='red', linestyle='--', label=f'Word Threshold ({COUNT_WORD_IN_PHRASE})')
        
        # Modifica asse X: Scala granulare ogni 5 unità fino a 100
        max_x_view = 100 
        plt.xlim(0, max_x_view)
        plt.xticks(np.arange(0, max_x_view + 5, 5))
        
        plt.title('Empirical Cumulative Distribution: Target Word Occurrences')
        plt.xlabel('Number of Occurrences in Corpus')
        plt.ylabel('Cumulative Fraction of Words')
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.3)
    
    plt.savefig('task1/img/grafico_cumulativo_parole.png', bbox_inches='tight')
    plt.close()

    # G6: Heatmap Affective Space (Valence vs Arousal)
    plt.figure(figsize=(10, 8))
    # Raggruppamento per coordinate V-A
    heatmap_data = df.groupby(['arousal', 'valence']).size().unstack(fill_value=0)
    heatmap_data = heatmap_data.sort_index(ascending=False)
    sns.heatmap(data=heatmap_data, annot=True, fmt="d", cmap="YlGnBu", cbar_kws={'label': 'Number of Samples'})
    plt.title('Dataset Density: Valence vs Arousal Space')
    plt.xlabel('Valence Score')
    plt.ylabel('Arousal Score')
    plt.savefig('task1/img/heatmap_emozioni.png', bbox_inches='tight')
    plt.close()

    print("\n" + "="*40)
    print("📊 DATA ANALYSIS COMPLETED")
    print("="*40)
    print(f"Total Users:         {len(frasi_per_utente)}")
    print(f"Users with >= {PHRASE_X_USER} texts: {u_sopra_soglia} (Valid for longitudinal study)")
    print(f"Unique Target Words: {len(set_parole_target)}")
    print(f"Plots saved in:      task1/img/")
    print("="*40)

if __name__ == "__main__":
    # Assicurati che il percorso del file CSV sia corretto
    analizza_e_visualizza('datasets/train_subtask1.csv')