import pandas as pd
import os

# Update this path with your actual log file path
LOGS_PATH = "task1/report/training_reports.csv"

def analyze_model_results(file_path, output_txt="task1/report/aggregated_model_analysis.txt"):
    """
    Analyzes the training report CSV and generates a summary text file with
    aggregated averages and best configurations per model.
    """
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

    os.makedirs(os.path.dirname(output_txt), exist_ok=True)

    # 1. Handle missing columns (Assign NaN if they don't exist)
    group_candidates = ['model_name', 'user_embedding_dim', 'filter_few_comments_user', 'augment_dataset']
    metrics_candidates = [
        'best_valence_value', 'best_arousal_value', 'best_mean_value', 
        'best_valence_mae', 'best_arousal_mae', 'best_mean_mae'
    ]
    
    for col in metrics_candidates:
        if col not in df.columns:
            df[col] = float('nan')
    
    # Ensure group columns exist to avoid grouping errors
    group_cols = [c for c in group_candidates if c in df.columns]

    # 2. Calculate Aggregated Averages
    # We group by model configuration and calculate the mean of the metrics
    aggregated_report = df.groupby(group_cols, dropna=False)[metrics_candidates].mean().reset_index()
    aggregated_report = aggregated_report.round(4)

    # 3. Identify Best Configuration for each Model
    # We look for the maximum 'best_mean_value' for each unique model_name
    idx_best = aggregated_report.groupby('model_name')['best_mean_value'].idxmax()
    best_configs = aggregated_report.loc[idx_best]

    # 4. Save to Text File
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write("="*90 + "\n")
        f.write("                       LLM PERFORMANCE ANALYSIS REPORT\n")
        f.write("="*90 + "\n\n")
        
        f.write("[1] AGGREGATED AVERAGES PER CONFIGURATION:\n")
        f.write(aggregated_report.to_string(index=False, na_rep='NaN'))
        f.write("\n\n" + "-"*90 + "\n\n")
        
        f.write("[2] BEST CONFIGURATIONS (Based on maximum 'best_mean_value'):\n")
        f.write("Note: Displays the winning parameter combination for each model.\n\n")
        f.write(best_configs.to_string(index=False, na_rep='NaN'))
        f.write("\n\n" + "="*90 + "\n")

    print(f"Analysis completed. Report saved to: {output_txt}")
    return aggregated_report, best_configs

if __name__ == "__main__":  
    analyze_model_results(LOGS_PATH)