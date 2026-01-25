import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import os

PHRASE_X_USER = 10
COUNT_WORD_IN_PHRASE = 10

def analizza_e_visualizza(file_path):
    # Creazione cartella immagini se non esiste
    os.makedirs('task1/img', exist_ok=True)

    # 1. Caricamento del dataset
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Errore nel caricamento del file: {e}")
        return

    # --- ANALISI DATI ---
    
    # Pulizia e normalizzazione colonna is_words
    df['is_words_bool'] = df['is_words'].astype(str).str.lower() == 'true'
    
    # Conteggio tipologia righe (NUOVO)
    conteggio_tipologia = df['is_words_bool'].value_counts()
    n_liste_parole = conteggio_tipologia.get(True, 0)
    n_frasi_normali = conteggio_tipologia.get(False, 0)

    # Statistiche Utenti
    frasi_per_utente = df.groupby('user_id').size()
    u_meno_10 = (frasi_per_utente < PHRASE_X_USER).sum()
    u_10_piu = (frasi_per_utente >= PHRASE_X_USER).sum()

    # Analisi Parole Target (is_words == True)
    df_is_words_true = df[df['is_words_bool'] == True].copy()
    
    tutte_le_parole_target = []
    for text in df_is_words_true['text'].dropna():
        tutte_le_parole_target.extend([p.strip().lower() for p in str(text).split(',') if p.strip()])
    
    set_parole_target = set(tutte_le_parole_target)
    
    # Conteggio nelle frasi comuni (is_words == False)
    frasi_corpus = df[df['is_words_bool'] == False]['text'].dropna().astype(str).tolist()
    conteggio_nelle_frasi = Counter()
    for parola in set_parole_target:
        pattern = re.compile(rf'\b{re.escape(parola)}\b', re.IGNORECASE)
        for frase in frasi_corpus:
            trovati = pattern.findall(frase)
            if trovati:
                conteggio_nelle_frasi[parola] += len(trovati)
    
    p_meno_30 = sum(1 for p in set_parole_target if conteggio_nelle_frasi[p] < PHRASE_X_USER)
    p_30_piu = sum(1 for p in set_parole_target if conteggio_nelle_frasi[p] >= PHRASE_X_USER)

    # Funzione di supporto per aggiungere i numeri sopra le barre
    def add_labels(ax):
        for p in ax.patches:
            height = p.get_height()
            ax.annotate(f'{int(height)}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points',
                        fontsize=12, fontweight='bold')

    # --- GENERAZIONE E SALVATAGGIO GRAFICI SINGOLI ---
    sns.set_theme(style="whitegrid")

    # 1. Grafico Tipologia Righe (NUOVO)
    plt.figure(figsize=(8, 6))
    ax0 = sns.barplot(x=['Frasi Comuni', 'Liste Parole (is_word)'], y=[n_frasi_normali, n_liste_parole], palette='coolwarm')
    add_labels(ax0)
    plt.title('Distribuzione Tipologia Righe nel Dataset')
    plt.ylabel('Conteggio Totale')
    plt.savefig('task1/img/grafico_tipologia_righe.png', bbox_inches='tight')
    plt.show()

    # 2. Grafico Utenti
    plt.figure(figsize=(8, 6))
    ax1 = sns.barplot(x=[f'< {PHRASE_X_USER} righe', f'>= {PHRASE_X_USER} righe'], y=[u_meno_10, u_10_piu], palette='viridis')
    add_labels(ax1)
    plt.title(f'Attività Utenti (Soglia {PHRASE_X_USER})')
    plt.ylabel('Numero di Utenti')
    plt.savefig('task1/img/grafico_utenti.png', bbox_inches='tight')
    plt.show()

    # 3. Grafico Parole
    plt.figure(figsize=(8, 6))
    ax2 = sns.barplot(x=[f'< {COUNT_WORD_IN_PHRASE} occorr.', f'>= {COUNT_WORD_IN_PHRASE} occorr.'], y=[p_meno_30, p_30_piu], palette='magma')
    add_labels(ax2)
    plt.title(f'Frequenza Parole Target nelle Frasi (Soglia {COUNT_WORD_IN_PHRASE})')
    plt.ylabel('Numero di Parole')
    plt.savefig('task1/img/grafico_parole.png', bbox_inches='tight')
    plt.show()

    # 4. Heatmap Valence/Arousal
    plt.figure(figsize=(10, 8))
    heatmap_data = df.groupby(['arousal', 'valence']).size().unstack(fill_value=0)
    heatmap_data = heatmap_data.sort_index(ascending=False)
    sns.heatmap(data=heatmap_data, annot=True, fmt="d", cmap="YlGnBu", cbar=True)
    plt.title('Heatmap Emozioni (Valence vs Arousal)')
    plt.xlabel('Valence')
    plt.ylabel('Arousal')
    plt.savefig('task1/img/heatmap_emozioni.png', bbox_inches='tight')
    plt.show()

    print("\nAnalisi completata!")
    print("Immagini salvate in 'task1/img/':")
    print("- grafico_tipologia_righe.png\n- grafico_utenti.png\n- grafico_parole.png\n- heatmap_emozioni.png")

if __name__ == "__main__":
    analizza_e_visualizza('datasets/train_subtask1.csv')