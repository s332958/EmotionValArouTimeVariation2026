import pandas as pd
import os

def analizza_risultati_modelli(file_path, output_txt="task1/report/analisi_aggregata_modelli.txt"):
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Errore nel caricamento del file: {e}")
        return None

    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    # 1. Gestione colonne mancanti (Mette NaN se non esistono)
    group_candidates = ['model_name', 'user_embedding_dim', 'filter_few_comments_user', 'augment_dataset']
    metrics_candidates = ['best_valence_value', 'best_arousal_value', 'best_mean_value', 
                          'best_valence_mae', 'best_arousal_mae', 'best_mean_mae']
    
    for col in metrics_candidates:
        if col not in df.columns:
            df[col] = float('nan')
    
    # Assicuriamoci che le colonne di gruppo esistano per evitare errori
    group_cols = [c for c in group_candidates if c in df.columns]

    # 2. Calcolo Medie Aggregate
    report_medie = df.groupby(group_cols, dropna=False)[metrics_candidates].mean().reset_index()
    report_medie = report_medie.round(4)

    # 3. Identificazione Best Configuration per ogni Modello
    # Cerchiamo il massimo di 'best_mean_value' per ogni model_name
    idx_best = report_medie.groupby('model_name')['best_mean_value'].idxmax()
    best_configs = report_medie.loc[idx_best]

    # 4. Salvataggio su File
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("="*90 + "\n")
        f.write("                 REPORT ANALISI PERFORMANCE LLM\n")
        f.write("="*90 + "\n\n")
        
        f.write("[1] MEDIE AGGREGATE PER CONFIGURAZIONE:\n")
        f.write(report_medie.to_string(index=False, na_rep='NaN'))
        f.write("\n\n" + "-"*90 + "\n\n")
        
        f.write("[2] MIGLIORI CONFIGURAZIONI (Basate su 'best_mean_value' massimo):\n")
        f.write("Nota: Mostra la combinazione vincente di parametri per ogni modello.\n\n")
        f.write(best_configs.to_string(index=False, na_rep='NaN'))
        f.write("\n\n" + "="*90 + "\n")

    print(f"Analisi completata. Report salvato in: {output_txt}")
    return report_medie, best_configs

if __name__ == "__main__":
    PATH_LOGS = "task1/report/report_training_go_twitter_models.csv"
    # PATH_LOGS = "task1/report/report_training_file_final.csv"
    analizza_risultati_modelli(PATH_LOGS)