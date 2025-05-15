import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the battle results
with open('dynamic_results.json', 'r') as f:
    results = json.load(f)

print(f"Loaded {len(results)} battle results.")

# Extract the model names
model_names = set()
for battle in results:
    model_names.add(battle["model1_name"])
    model_names.add(battle["model2_name"])

model_names = sorted(list(model_names))
print(f"Found {len(model_names)} unique models: {model_names}")

# Initialize a matrix to track wins and total matches between each pair of models
wins = {model1: {model2: 0 for model2 in model_names} for model1 in model_names}
matches = {model1: {model2: 0 for model2 in model_names} for model1 in model_names}

# Process all battles
valid_battles = 0
for battle in results:
    model1 = battle["model1_name"]
    model2 = battle["model2_name"]
    winner = battle["winning_model"]
    
    # Skip battles where both models are good or bad
    if winner in ["both_good", "both_bad"]:
        continue
    
    valid_battles += 1
    
    # Update win and match counts
    matches[model1][model2] += 1
    matches[model2][model1] += 1
    
    if winner == model1:
        wins[model1][model2] += 1
    elif winner == model2:
        wins[model2][model1] += 1

print(f"Processed {valid_battles} valid battles (ignoring 'both_good' and 'both_bad')")

# Calculate win rates
win_rates = {model1: {model2: 0.0 for model2 in model_names} for model1 in model_names}
for model1 in model_names:
    for model2 in model_names:
        if model1 != model2 and matches[model1][model2] > 0:
            win_rates[model1][model2] = wins[model1][model2] / matches[model1][model2]

# Create a DataFrame for the win rates
win_rate_df = pd.DataFrame(win_rates)

# Fill diagonal with NaN (model vs itself)
for model in model_names:
    win_rate_df.loc[model, model] = np.nan

# Create a heatmap visualization for win rates
plt.figure(figsize=(10, 8))
sns.heatmap(win_rate_df.T, annot=True, cmap="YlGnBu", fmt=".2f", linewidths=.5)
plt.title('Pairwise Win Rates Between Models (Row vs Column)', fontsize=16, pad=20)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('model_win_rates.png', dpi=300, bbox_inches='tight')
print("Win rate heatmap saved to model_win_rates.png")

# Create a DataFrame for the match counts
match_count_df = pd.DataFrame(matches)

# Since each match is counted twice (once for each model), divide by 2
for model1 in model_names:
    for model2 in model_names:
        match_count_df.loc[model1, model2] = int(match_count_df.loc[model1, model2] // 2)

# Fill diagonal with NaN (model vs itself)
for model in model_names:
    match_count_df.loc[model, model] = np.nan

# Create a heatmap visualization for match counts
plt.figure(figsize=(10, 8))
sns.heatmap(match_count_df.T, annot=True, cmap="Greens", fmt=".0f", linewidths=.5)
plt.title('Number of Matches Between Models (Row vs Column)', fontsize=16, pad=20)
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
plt.savefig('model_match_counts.png', dpi=300, bbox_inches='tight')
print("Match count heatmap saved to model_match_counts.png")

# Create a summary table with overall stats
summary = []
for model in model_names:
    total_wins = sum(wins[model].values())
    total_matches = sum(matches[model].values()) 
    win_rate = total_wins / total_matches if total_matches > 0 else 0
    
    # Calculate win rates against each opponent
    opponent_stats = []
    for opponent in model_names:
        if model != opponent and matches[model][opponent] > 0:
            rate = wins[model][opponent] / matches[model][opponent]
            opponent_stats.append(f"{opponent}: {rate:.2f} ({wins[model][opponent]}/{matches[model][opponent]})")
    
    summary.append({
        "Model": model,
        "Total Wins": total_wins,
        "Total Matches": total_matches,
        "Overall Win Rate": f"{win_rate:.2f}",
        "Win Rates vs Opponents": ", ".join(opponent_stats)
    })

summary_df = pd.DataFrame(summary)
summary_df = summary_df.sort_values(by="Overall Win Rate", ascending=False)

print("\nModel Performance Summary:")
print(summary_df[["Model", "Overall Win Rate", "Total Wins", "Total Matches"]])

# Save detailed results to CSV
summary_df.to_csv("model_performance_summary.csv", index=False)
win_rate_df.to_csv("pairwise_win_rates.csv")
match_count_df.to_csv("pairwise_match_counts.csv")
print("Summary data saved to model_performance_summary.csv, pairwise_win_rates.csv, and pairwise_match_counts.csv")