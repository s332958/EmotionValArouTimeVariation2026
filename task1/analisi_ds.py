import pandas as pd
import re
import sys
from collections import Counter

def analizza_dataset(file_path, output_file="task1/report/dataset_analisi.txt"):
    # Caricamento del dataset
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Errore nel caricamento del file: {e}")
        return

    # Apriamo il file in modalità scrittura
    with open(output_file, "w", encoding="utf-8") as f:
        # Funzione interna per stampare sia su console che su file
        def dual_print(text=""):
            print(text)
            f.write(str(text) + "\n")

        dual_print("="*60)
        dual_print("                REPORT DETTAGLIATO DATASET")
        dual_print("="*60)

        # 1. Statistiche Generali e Media Frasi per Utente
        n_utenti = df['user_id'].nunique()
        n_totale_frasi = len(df)
        frasi_per_utente_ser = df.groupby('user_id').size()
        media_frasi_utente = frasi_per_utente_ser.mean()

        utenti_meno_10 = (frasi_per_utente_ser < 10).sum()
        utenti_10_o_piu = (frasi_per_utente_ser >= 10).sum()

        dual_print(f"\n[1] STATISTICHE GENERALI:")
        dual_print(f"- Numero di utenti unici: {n_utenti}")
        dual_print(f"  > Utenti con meno di 10 frasi: {utenti_meno_10}")
        dual_print(f"  > Utenti con 10 o più frasi: {utenti_10_o_piu}")
        dual_print(f"- Numero totale di righe (frasi + liste): {n_totale_frasi}")
        dual_print(f"- Media righe per utente: {media_frasi_utente:.2f}")

        # 2. Numero di frasi per ogni utente
        dual_print("\n[2] DETTAGLIO RIGHE PER UTENTE:")
        df_utenti = frasi_per_utente_ser.reset_index(name='num_frasi')
        df_utenti = df_utenti.sort_values(by='num_frasi', ascending=False)
        dual_print(df_utenti.to_string(index=False))

        # 3. Analisi delle parole (is_words == True)
        df['is_words'] = df['is_words'].astype(str).str.lower() == 'true'
        df_is_words_true = df[df['is_words'] == True].copy()
        
        dati_parole_singole = []
        tutte_le_parole_in_liste = []

        for _, row in df_is_words_true.iterrows():
            parole = [p.strip().lower() for p in str(row['text']).split(',') if p.strip()]
            tutte_le_parole_in_liste.extend(parole)
            for p in parole:
                dati_parole_singole.append({
                    'parola': p,
                    'valence': row['valence'],
                    'arousal': row['arousal']
                })
        
        conteggio_in_liste = Counter(tutte_le_parole_in_liste)
        df_metriche_parole = pd.DataFrame(dati_parole_singole)
        
        if df_metriche_parole.empty:
            dual_print("\n[!] Nessuna parola trovata con is_words=True.")
        else:
            stats_parole = df_metriche_parole.groupby('parola').agg({
                'valence': 'mean',
                'arousal': 'mean'
            }).reset_index()
            
            set_parole_target = set(stats_parole['parola'])

            # 4. Conteggio occorrenze nelle frasi comuni (is_words == False)
            frasi_corpus = df[df['is_words'] == False]['text'].dropna().astype(str).tolist()
            conteggio_nelle_frasi = Counter()
            
            for parola in set_parole_target:
                pattern = re.compile(rf'\b{re.escape(parola)}\b', re.IGNORECASE)
                for frase in frasi_corpus:
                    trovati = pattern.findall(frase)
                    if trovati:
                        conteggio_nelle_frasi[parola] += len(trovati)

            # 5. Integrazione dati parole
            stats_parole['count_frasi (False)'] = stats_parole['parola'].map(conteggio_nelle_frasi).fillna(0).astype(int)
            stats_parole['count_liste (True)'] = stats_parole['parola'].map(conteggio_in_liste).fillna(0).astype(int)
            stats_parole['totale_occorrenze'] = stats_parole['count_frasi (False)'] + stats_parole['count_liste (True)']

            almeno_30 = stats_parole[stats_parole['count_frasi (False)'] >= 30]
            meno_di_30 = stats_parole[stats_parole['count_frasi (False)'] < 30]

            dual_print(f"\n[3] ANALISI FREQUENZA PAROLE TARGET:")
            dual_print(f"- Numero totale di parole target uniche identificate: {len(set_parole_target)}")
            dual_print(f"- Parole con >= 30 occorrenze nelle frasi comuni: {len(almeno_30)}")
            dual_print(f"- Parole con < 30 occorrenze nelle frasi comuni: {len(meno_di_30)}")

            dual_print("\n--- DETTAGLIO PAROLE (Ordinate per totale occorrenze) ---")
            dual_print(stats_parole.sort_values(by='totale_occorrenze', ascending=False).to_string(index=False, formatters={
                'valence': '{:,.2f}'.format,
                'arousal': '{:,.2f}'.format
            }))

        # 6. Distribuzione Arousal/Valence
        dual_print("\n[4] DISTRIBUZIONE COPPIE AROUSAL/VALENCE (Dataset completo):")
        coppie_count = df.groupby(['arousal', 'valence']).size().reset_index(name='quantita')
        dual_print(coppie_count.sort_values(by='quantita', ascending=False).to_string(index=False))
        dual_print("\n" + "="*60)
    
    print(f"\nAnalisi completata! Il report è stato salvato in: {output_file}")

if __name__ == "__main__":
    NOME_FILE = 'datasets/train_subtask1.csv' 
    analizza_dataset(NOME_FILE)