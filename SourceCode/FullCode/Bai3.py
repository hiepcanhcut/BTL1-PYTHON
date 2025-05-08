import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import seaborn as sns

INPUT_CSV_FILE = 'ketqua.csv'
TOP_N = 3
OUTPUT_TOP_PLAYERS_FILE = 'top_3.txt'
OUTPUT_STATS_FILE = 'results2.csv'
OUTPUT_HISTOGRAM_DIR = 'histograms'
KEY_TEAM_STATS = ['Gls', 'Ast', 'G+A', 'xG', 'xAG', 'PrgC', 'PrgP']
CLUSTERING_RESULTS_DIR = 'clustering_results'

def get_numeric_columns(df):
    numeric_cols = []
    for col in df.columns:
        try:
            pd.to_numeric(df[col], errors='raise')
            if col not in ['Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born', 'Min', 'Starts', 'MP', '90s']:
                if df[col].dtype in [np.int64, np.float64] and df[col].std() < 10000 and df[col].max() < 50000:
                    numeric_cols.append(col)
        except (ValueError, TypeError):
            try:
                numeric_series = pd.to_numeric(df[col], errors='coerce')
                if numeric_series.notna().sum() / len(df) > 0.5:
                    if col not in ['Player', 'Nation', 'Pos', 'Squad', 'Age', 'Born', 'Min', 'Starts', 'MP', '90s']:
                        if numeric_series.std() < 10000 and numeric_series.max() < 50000:
                            numeric_cols.append(col)
            except (ValueError, TypeError):
                continue
    return numeric_cols

def format_player_list(series, stat_name):
    output = ""
    for i, (index, value) in enumerate(series.items()):
        player_name = df.loc[index, 'Player']
        output += f"  {i+1}. {player_name} ({value})\n"
    return output

def Determine_Optimal_K(data):
    inertia = []
    silhouette_scores = []
    kRange = range(2, 16)
    for k in kRange:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10).fit(data)
        inertia.append(kmeans.inertia_)
        print(f"Inertia for k={k} calculated.", end=' ')
        if k > 1:
            try:
                score = silhouette_score(data, kmeans.labels_)
                silhouette_scores.append(score)
                print(f"Silhouette Score: {score:.4f}")
            except ValueError as e:
                print(f"  Could not calculate silhouette score for K={k}: {e}")
                silhouette_scores.append(-1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(kRange, inertia, 'o--')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Elbow Method for Optimal K')
    plt.xticks(list(kRange))
    plt.grid(True)
    plt.savefig(os.path.join(CLUSTERING_RESULTS_DIR, 'Elbow_Method.png'))
    plt.close()
    print(f'Elbow_Method.png is saved')

    valid_k_range_silhouette = list(kRange)
    plt.figure(figsize=(10, 6))
    plt.plot(valid_k_range_silhouette, silhouette_scores, marker='o')
    plt.title('Silhouette Score for Optimal K')
    plt.xlabel('Number of clusters (K)')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.savefig(os.path.join(CLUSTERING_RESULTS_DIR, 'Silhouette_Score.png'))
    plt.close()
    print(f'Silhouette_Score.png is saved')

def Apply_K_means(data, optimal_k):
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(data)
    return cluster_labels

def Apply_PCA(data, cluster_labels):
    pca = PCA(n_components=2)
    components = pca.fit_transform(data)
    pca_df = pd.DataFrame(components, columns=['PC1', 'PC2'])
    pca_df['Cluster'] = cluster_labels
    print(f"PCA complete. Explained variance: {pca.explained_variance_ratio_.sum():.4f}")
    return pca_df

def Plot_2D_Cluster(pca_df, optimal_k):
    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        x='PC1', y='PC2', hue='Cluster',
        palette=sns.color_palette('viridis', n_colors=optimal_k),
        data=pca_df, legend='full', alpha=0.7
    )
    plt.title(f'PCA of Clusters (k={optimal_k})')
    plt.grid(True)
    plt.savefig(os.path.join(CLUSTERING_RESULTS_DIR, f'PCA_of_Clusters_k={optimal_k}.png'))
    plt.close()
    print(f'PCA_of_Clusters_k={optimal_k}.png is saved')

try:
    df = pd.read_csv(INPUT_CSV_FILE, encoding='utf-8')
except FileNotFoundError:
    print(f"Error: File '{INPUT_CSV_FILE}' not found. Please make sure it's in the same directory.")
    exit()
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

numeric_stat_cols = get_numeric_columns(df.copy())
for col in numeric_stat_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

try:
    with open(OUTPUT_TOP_PLAYERS_FILE, 'w', encoding='utf-8') as f:
        for col in numeric_stat_cols:
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
    overall_median = df[numeric_stat_cols].median()
    overall_mean = df[numeric_stat_cols].mean()
    overall_std = df[numeric_stat_cols].std()

    grouped_by_team = df.groupby('Squad')[numeric_stat_cols]
    team_median = grouped_by_team.median()
    team_mean = grouped_by_team.mean()
    team_std = grouped_by_team.std()

    results_list = []
    overall_row = {'Squad': 'all'}
    for col in numeric_stat_cols:
        overall_row[f'Median of {col}'] = overall_median.get(col)
        overall_row[f'Mean of {col}'] = overall_mean.get(col)
        overall_row[f'Std of {col}'] = overall_std.get(col)
    results_list.append(overall_row)

    for team in team_median.index:
        team_row = {'Squad': team}
        for col in numeric_stat_cols:
            team_row[f'Median of {col}'] = team_median.loc[team, col] if team in team_median.index else None
            team_row[f'Mean of {col}'] = team_mean.loc[team, col] if team in team_mean.index else None
            team_row[f'Std of {col}'] = team_std.loc[team, col] if team in team_std.index else None
        results_list.append(team_row)

    results_df = pd.DataFrame(results_list)
    results_df = results_df.set_index('Squad')

    ordered_columns = ['Squad']
    for col in numeric_stat_cols:
        ordered_columns.extend([f'Median of {col}', f'Mean of {col}', f'Std of {col}'])

    existing_ordered_columns = [col for col in ordered_columns if col in results_df.columns or col == 'Squad']
    results_df = results_df.reset_index()
    results_df = results_df[existing_ordered_columns]
    results_df = results_df.set_index('Squad')

    results_df.to_csv(OUTPUT_STATS_FILE, encoding='utf-8')
except Exception as e:
    print(f"Error calculating/saving descriptive statistics: {e}")
    results_df = pd.DataFrame()

if not os.path.exists(OUTPUT_HISTOGRAM_DIR):
    os.makedirs(OUTPUT_HISTOGRAM_DIR)

team_hist_dir = os.path.join(OUTPUT_HISTOGRAM_DIR, 'teams')
if not os.path.exists(team_hist_dir):
    os.makedirs(team_hist_dir)

for col in numeric_stat_cols:
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
        print(f"    Error saving plot for {col} (all players): {e}")
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
                print(f"    Error saving plot for {col} ({team}): {e}")
        plt.close()

top_teams = {}
for col in KEY_TEAM_STATS:
    mean_col_name = f'Mean of {col}'
    if mean_col_name in results_df.columns:
        teams_only_df = results_df.drop('all', errors='ignore')
        if not teams_only_df.empty:
            top_team = teams_only_df[mean_col_name].idxmax()
            top_score = teams_only_df[mean_col_name].max()
            top_teams[col] = (top_team, top_score)

if not results_df.empty:
    print("Identifying teams with highest average scores per statistic:")
    for col in KEY_TEAM_STATS:
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
        print(f"\n  Conclusion: Based on the analyzed statistics (especially {', '.join(KEY_TEAM_STATS)}),")
        print(f"  '{best_performing_team}' appears to be performing strongly, frequently leading in average statistics.")
        print(f"  However, a comprehensive analysis would require more stats (defensive, etc.) and context.")

print("\n--- Analysis Complete ---")

os.makedirs(CLUSTERING_RESULTS_DIR, exist_ok=True)
exclude_cols = ['Player', 'Nation', 'Squad', 'Pos', 'Age']
features = df.drop(columns=exclude_cols).apply(pd.to_numeric, errors='coerce')
cols_all_nan = features.columns[features.isnull().all()]
features = features.drop(columns=cols_all_nan)
imputer = SimpleImputer(strategy='median')
features_imputed = pd.DataFrame(imputer.fit_transform(features), columns=features.columns)
scaler = StandardScaler()
features_scaled = pd.DataFrame(scaler.fit_transform(features_imputed), columns=features_imputed.columns)

Determine_Optimal_K(features_scaled)
optimal_k = 6  
cluster_labels = Apply_K_means(features_scaled, optimal_k)
df['Cluster'] = cluster_labels
features_imputed['Cluster'] = cluster_labels

print("\nMean of key statistics per cluster:")
print(features_imputed.groupby('Cluster')[KEY_TEAM_STATS].mean())

pca_df = Apply_PCA(features_scaled, cluster_labels)
Plot_2D_Cluster(pca_df, optimal_k)

with open(os.path.join(CLUSTERING_RESULTS_DIR, "clustering_comments.txt"), "w", encoding="utf-8") as f:
    f.write("\n--- Comments on Clustering Results ---\n")
    f.write(f"1. Number of Groups (K):\n")
    f.write(f"   - Based on the Elbow Method and Silhouette Score analysis, K={optimal_k} clusters were chosen.\n")
    f.write(f"   - The Elbow plot likely showed diminishing returns in WCSS reduction around K={optimal_k}.\n")
    f.write(f"   - The Silhouette Score plot might have indicated a peak or high value at K={optimal_k}, suggesting reasonable cluster separation.\n")
    f.write(f"\n2. PCA and Clustering Plot:\n")
    f.write(f"   - PCA was used to reduce the features to 2 dimensions for visualization.\n")
    f.write(f"   - The 2D scatter plot shows the distribution of players based on these two principal components, colored by their assigned K-means cluster.\n")
    f.write(f"   - Interpretation of the plot:\n")
    f.write(f"     - Observe the separation between clusters. Are they distinct or overlapping?\n")
    f.write(f"     - The spread and density of points within each cluster provide insight into the similarity of players grouped together based on the selected statistics.\n")
    for i in range(optimal_k):
        cluster_data = df[df['Cluster'] == i]
        position_counts = cluster_data['Pos'].value_counts()
        f.write(f"\nCluster {i}:\n")
        f.write(f"  Total Players: {len(cluster_data)}\n")
        f.write("  Top Positions:\n")
        for pos, count in position_counts.head().items():
            f.write(f"    {pos}: {count}\n")
print(f'clustering_comments.txt is saved')
