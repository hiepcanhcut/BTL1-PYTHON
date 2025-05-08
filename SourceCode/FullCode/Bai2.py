import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

INPUT_CSV_FILE = 'ketqua.csv'
TOP_N = 3
OUTPUT_TOP_PLAYERS_FILE = 'top_3.txt'
OUTPUT_STATS_FILE = 'results2.csv'
OUTPUT_HISTOGRAM_DIR = 'histograms'
SELECTED_STATS = ['Gls', 'Ast', 'xG', 'CrdY', 'CrdR', 'PrgP']

def format_player_list(series, stat_name):
    output = ""
    for i, (index, value) in enumerate(series.items()):
        player_name = df.loc[index, 'Player']
        output += f"  {i+1}. {player_name} ({value})\n"
    return output

try:
    df = pd.read_csv(INPUT_CSV_FILE, encoding='utf-8')
except FileNotFoundError:
    print(f"Error: File '{INPUT_CSV_FILE}' not found. Please make sure it's in the same directory.")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

for col in SELECTED_STATS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

try:
    with open(OUTPUT_TOP_PLAYERS_FILE, 'w', encoding='utf-8') as f:
        for col in SELECTED_STATS:
            if col not in df.columns:
                f.write(f"Statistic: {col}\n")
                f.write(f"  (No data available for this statistic)\n\n")
                continue
            f.write(f"Statistic: {col}\n")
            f.write("-" * (len(col) + 11) + "\n")
            df_col_cleaned = df.dropna(subset=[col])
            if df_col_cleaned.empty:
                f.write(f"  (No valid data for this statistic)\n\n")
                continue
            top_highest = df_col_cleaned.nlargest(TOP_N, col)
            f.write(f"Top {TOP_N} Highest:\n")
            f.write(format_player_list(top_highest[col], col))
            top_lowest = df_col_cleaned.nsmallest(TOP_N, col)
            f.write(f"\nTop {TOP_N} Lowest:\n")
            f.write(format_player_list(top_lowest[col], col))
            f.write("\n" + "="*30 + "\n\n")
except Exception as e:
    print(f"Error writing to {OUTPUT_TOP_PLAYERS_FILE}: {e}")

try:
    overall_median = df[SELECTED_STATS].median()
    overall_mean = df[SELECTED_STATS].mean()
    overall_std = df[SELECTED_STATS].std()

    grouped_by_team = df.groupby('Squad')[SELECTED_STATS]
    team_median = grouped_by_team.median()
    team_mean = grouped_by_team.mean()
    team_std = grouped_by_team.std()

    results_list = []
    overall_row = {'Squad': 'all'}
    for col in SELECTED_STATS:
        overall_row[f'Median of {col}'] = overall_median.get(col)
        overall_row[f'Mean of {col}'] = overall_mean.get(col)
        overall_row[f'Std of {col}'] = overall_std.get(col)
    results_list.append(overall_row)

    for team in team_median.index:
        team_row = {'Squad': team}
        for col in SELECTED_STATS:
            team_row[f'Median of {col}'] = team_median.loc[team, col] if team in team_median.index else None
            team_row[f'Mean of {col}'] = team_mean.loc[team, col] if team in team_mean.index else None
            team_row[f'Std of {col}'] = team_std.loc[team, col] if team in team_std.index else None
        results_list.append(team_row)

    results_df = pd.DataFrame(results_list)
    results_df = results_df.set_index('Squad')

    ordered_columns = ['Squad']
    for col in SELECTED_STATS:
        ordered_columns.extend([f'Median of {col}', f'Mean of {col}', f'Std of {col}'])
    existing_ordered_columns = [col for col in ordered_columns if col in results_df.columns or col == 'Squad']
    results_df = results_df.reset_index()
    results_df = results_df[existing_ordered_columns]
    results_df = results_df.set_index('Squad')

    os.makedirs(os.path.dirname(OUTPUT_STATS_FILE) if os.path.dirname(OUTPUT_STATS_FILE) else '.', exist_ok=True)
    results_df.to_csv(OUTPUT_STATS_FILE, encoding='utf-8')
except Exception as e:
    print(f"Error calculating/saving descriptive statistics: {e}")
    results_df = pd.DataFrame()

if not os.path.exists(OUTPUT_HISTOGRAM_DIR):
    os.makedirs(OUTPUT_HISTOGRAM_DIR)

team_hist_dir = os.path.join(OUTPUT_HISTOGRAM_DIR, 'teams')
if not os.path.exists(team_hist_dir):
    os.makedirs(team_hist_dir)

for col in SELECTED_STATS:
    if col not in df.columns:
        continue
    plt.style.use('ggplot')
    plt.figure(figsize=(10, 6))
    df[col].dropna().hist(bins=20)
    plt.title(f'Distribution of {col} (All Players)')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.tight_layout()
    try:
        plt.savefig(os.path.join(OUTPUT_HISTOGRAM_DIR, f'{col}_all_players.png'))
    except Exception as e:
        print(f"Error saving plot for {col} (all players): {e}")
    plt.close()

    teams = df['Squad'].unique()
    for team in teams:
        plt.figure(figsize=(8, 5))
        team_data = df[df['Squad'] == team][col].dropna()
        if not team_data.empty:
            team_data.hist(bins=15)
            plt.title(f'Distribution of {col} ({team})')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.tight_layout()
            try:
                safe_team_name = "".join(c if c.isalnum() else "_" for c in team)
                plt.savefig(os.path.join(team_hist_dir, f'{col}_{safe_team_name}.png'))
            except Exception as e:
                print(f"Error saving plot for {col} ({team}): {e}")
        plt.close()

top_teams = {}
for col in SELECTED_STATS:
    mean_col_name = f'Mean of {col}'
    if mean_col_name in results_df.columns:
        teams_only_df = results_df.drop('all', errors='ignore')
        if not teams_only_df.empty:
            top_team = teams_only_df[mean_col_name].idxmax()
            top_score = teams_only_df[mean_col_name].max()
            top_teams[col] = (top_team, top_score)

if not results_df.empty:
    print("Identifying teams with highest average scores per statistic:")
    for col in SELECTED_STATS:
        mean_col_name = f'Mean of {col}'
        if mean_col_name in results_df.columns:
            teams_only_df = results_df.drop('all', errors='ignore')
            if not teams_only_df.empty:
                top_team = teams_only_df[mean_col_name].idxmax()
                top_score = teams_only_df[mean_col_name].max()
                top_teams[col] = (top_team, top_score)

    print("\nOverall Performance Analysis (Subjective based on selected stats):")
    if top_teams:
        team_mentions = pd.Series([team for team, score in top_teams.values()]).value_counts()
        print("  Teams mentioned most often as highest scorer:")
        print(team_mentions.head())

        top_scorer_team = top_teams.get('Gls', ('N/A', 0))[0]
        top_assist_team = top_teams.get('Ast', ('N/A', 0))[0]
        top_xg_team = top_teams.get('xG', ('N/A', 0))[0]

        print(f"\n  Based on key metrics:")
        print(f"  - Team with highest avg Goals (Gls): {top_scorer_team}")
        print(f"  - Team with highest avg Assists (Ast): {top_assist_team}")
        print(f"  - Team with highest avg Expected Goals (xG): {top_xg_team}")

        best_performing_team = team_mentions.idxmax() if not team_mentions.empty else "Undetermined"
        print(f"\n  Conclusion: Based on the analyzed statistics (especially {', '.join(SELECTED_STATS)}),")
        print(f"  '{best_performing_team}' appears to be performing strongly, frequently leading in average statistics.")
        print(f"  However, a comprehensive analysis would require more stats and context.")

print("\n--- Analysis Complete ---")
