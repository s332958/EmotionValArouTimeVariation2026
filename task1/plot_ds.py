import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def analizza_e_visualizza(file_path):
    # 1. Caricamento del dataset
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Errore nel caricamento del file: {e}")
        return

    # --- ANALISI DATI ---
    
    # Statistiche Utenti
    frasi_per_utente = df.groupby('user_id').size()
    u_meno_10 = (frasi_per_utente < 10).sum()
    u_10_piu = (frasi_per_utente >= 10).sum()

    # Analisi Parole Target (is_words == True)
    df['is_words'] = df['is_words'].astype(str).str.lower() == 'true'
    df_is_words_true = df[df['is_words'] == True].copy()
    
    tutte_le_parole_target = []
    for text in df_is_words_true['text'].dropna():
        tutte_le_parole_target.extend([p.strip().lower() for p in str(text).split(',') if p.strip()])
    
    set_parole_target = set(tutte_le_parole_target)
    
    # Conteggio nelle frasi comuni (is_words == False)
    frasi_corpus = df[df['is_words'] == False]['text'].dropna().astype(str).tolist()
    conteggio_nelle_frasi = Counter()
    for parola in set_parole_target:
        pattern = re.compile(rf'\b{re.escape(parola)}\b', re.IGNORECASE)
        for frase in frasi_corpus:
            trovati = pattern.findall(frase)
            if trovati:
                conteggio_nelle_frasi[parola] += len(trovati)
    
    p_meno_30 = sum(1 for p in set_parole_target if conteggio_nelle_frasi[p] < 30)
    p_30_piu = sum(1 for p in set_parole_target if conteggio_nelle_frasi[p] >= 30)

    # Funzione di supporto per aggiungere i numeri sopra le barre
    def add_labels(ax):
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha = 'center', va = 'center', 
                        xytext = (0, 9), 
                        textcoords = 'offset points',
                        fontsize=12, fontweight='bold')

    # --- GENERAZIONE E SALVATAGGIO GRAFICI SINGOLI ---
    sns.set_theme(style="whitegrid")

    # 1. Grafico Utenti
    plt.figure(figsize=(8, 6))
    ax1 = sns.barplot(x=['< 10 frasi', '>= 10 frasi'], y=[u_meno_10, u_10_piu], palette='viridis')
    add_labels(ax1)
    plt.title('Attività Utenti (Soglia 10)')
    plt.ylabel('Numero di Utenti')
    plt.savefig('task1/img/grafico_utenti.png', bbox_inches='tight')
    plt.show()

    # 2. Grafico Parole
    plt.figure(figsize=(8, 6))
    ax2 = sns.barplot(x=['< 30 occorr.', '>= 30 occorr.'], y=[p_meno_30, p_30_piu], palette='magma')
    add_labels(ax2)
    plt.title('Frequenza Parole Target (Soglia 30)')
    plt.ylabel('Numero di Parole')
    plt.savefig('task1/img/grafico_parole.png', bbox_inches='tight')
    plt.show()

    # 3. Heatmap Valence/Arousal
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
    print("Immagini salvate singolarmente: 'grafico_utenti.png', 'grafico_parole.png', 'heatmap_emozioni.png'")

if __name__ == "__main__":
    # Assicurati che il percorso sia corretto
    analizza_e_visualizza('datasets/train_subtask1.csv')