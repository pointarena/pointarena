import json
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math

# Constants
K_FACTOR = 2  # Changed from 32 to 3
INITIAL_RATING = 1000  # Initial rating for all models

def calculate_expected_score(rating_a, rating_b):
    """
    Calculate the expected score for player A in a match against player B.
    Expected score is a number between 0 and 1 (essentially a probability).
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))

def update_elo(rating_a, rating_b, score_a):
    """
    Update the Elo rating for player A based on their score against player B.
    Score is 1 for a win, 0 for a loss.
    """
    expected_a = calculate_expected_score(rating_a, rating_b)
    return rating_a + K_FACTOR * (score_a - expected_a)

def calculate_elo_confidence_interval(rating, num_games, confidence=0.95):
    """
    Calculate the confidence interval for an Elo rating.
    
    Args:
        rating: The Elo rating
        num_games: Number of games played
        confidence: Confidence level (default: 0.95 for 95% confidence)
        
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    # Z-score for 95% confidence interval is approximately 1.96
    z_score = 1.96
    
    # Standard error calculation for Elo
    # The formula is simplified as: SE = K * sqrt(N) / 2
    # where K is the K-factor and N is the number of games
    if num_games == 0:
        return rating, rating
    
    standard_error = (K_FACTOR * math.sqrt(num_games)) / 2
    margin_of_error = z_score * standard_error
    
    return rating - margin_of_error, rating + margin_of_error

def main():
    # Load the battle results
    with open('dynamic_results.json', 'r') as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} battle results.")
    
    # Initialize ratings dictionary with default rating
    ratings = {
        "Qwen/Qwen2.5-VL-7B-Instruct": INITIAL_RATING,
        "allenai/Molmo-7B-D-0924": INITIAL_RATING,
        "gpt-4o": INITIAL_RATING,
        "grok-2-vision-latest": INITIAL_RATING,
        "gemini-2.5-flash-preview-04-17": INITIAL_RATING
    }
    
    # Track wins, losses, and ties for each model
    stats = {model: {"wins": 0, "losses": 0, "ties": 0, "games": 0} for model in ratings.keys()}
    
    # Process all battles
    valid_battles = 0
    for battle in tqdm(results, desc="Processing battles"):
        model1 = battle["model1_name"]
        model2 = battle["model2_name"]
        winner = battle["winning_model"]
        
        # Skip battles where both models are good or bad
        if winner in ["both_good", "both_bad"]:
            continue
        
        valid_battles += 1
        
        # Update stats
        stats[model1]["games"] += 1
        stats[model2]["games"] += 1
        
        if winner == model1:
            stats[model1]["wins"] += 1
            stats[model2]["losses"] += 1
            
            # Update Elo ratings
            new_rating_1 = update_elo(ratings[model1], ratings[model2], 1)
            new_rating_2 = update_elo(ratings[model2], ratings[model1], 0)
            
            ratings[model1] = new_rating_1
            ratings[model2] = new_rating_2
            
        elif winner == model2:
            stats[model2]["wins"] += 1
            stats[model1]["losses"] += 1
            
            # Update Elo ratings
            new_rating_1 = update_elo(ratings[model1], ratings[model2], 0)
            new_rating_2 = update_elo(ratings[model2], ratings[model1], 1)
            
            ratings[model1] = new_rating_1
            ratings[model2] = new_rating_2
    
    print(f"Processed {valid_battles} valid battles (ignoring 'both_good' and 'both_bad')")
    
    # Create a DataFrame for the leaderboard
    leaderboard = []
    for model, rating in ratings.items():
        model_stats = stats[model]
        win_rate = model_stats["wins"] / (model_stats["wins"] + model_stats["losses"]) if (model_stats["wins"] + model_stats["losses"]) > 0 else 0
        
        # Calculate confidence intervals
        lower_ci, upper_ci = calculate_elo_confidence_interval(rating, model_stats["games"])
        
        leaderboard.append({
            "Model": model,
            "Elo Rating": round(rating, 1),
            "Wins": model_stats["wins"],
            "Losses": model_stats["losses"],
            "Games": model_stats["games"],
            "Win Rate": f"{win_rate:.2%}",
            "Lower CI": round(lower_ci, 1),
            "Upper CI": round(upper_ci, 1)
        })
    
    leaderboard_df = pd.DataFrame(leaderboard)
    leaderboard_df = leaderboard_df.sort_values(by="Elo Rating", ascending=False)
    
    print("\nElo Rating Leaderboard:")
    print(leaderboard_df)
    
    # Save leaderboard to CSV
    leaderboard_df.to_csv("elo_leaderboard.csv", index=False)
    print("Leaderboard saved to elo_leaderboard.csv")
    
    # Create a bar chart visualization
    plt.figure(figsize=(12, 8))
    models = leaderboard_df["Model"]
    ratings = leaderboard_df["Elo Rating"]
    
    # Create a color map
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(models)))
    
    plt.bar(models, ratings, color=colors)
    plt.axhline(y=INITIAL_RATING, color='r', linestyle='--', alpha=0.7, label='Initial Rating (1000)')
    
    plt.title('Model Elo Ratings Leaderboard', fontsize=16)
    plt.ylabel('Elo Rating', fontsize=14)
    plt.ylim(min(ratings) - 50, max(ratings) + 50)
    
    # Rotate model names for better readability
    plt.xticks(rotation=30, ha='right', fontsize=12)
    
    # Add rating values on top of bars
    for i, v in enumerate(ratings):
        plt.text(i, v + 5, str(round(v, 1)), ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.legend()
    plt.savefig('elo_ratings.png', dpi=300, bbox_inches='tight')
    print("Visualization saved to elo_ratings.png")
    
    # Create a chart with confidence intervals
    plt.figure(figsize=(14, 10))
    
    # Sort models by rating for better visualization
    sorted_data = leaderboard_df.sort_values(by="Elo Rating", ascending=False)
    models = sorted_data["Model"]
    ratings = sorted_data["Elo Rating"]
    lower_ci = sorted_data["Lower CI"]
    upper_ci = sorted_data["Upper CI"]
    
    # Calculate error bars
    yerr = np.array([ratings - lower_ci, upper_ci - ratings])
    
    # Create bar chart with error bars
    bars = plt.bar(models, ratings, color=colors, alpha=0.7)
    plt.errorbar(models, ratings, yerr=yerr, fmt='none', color='black', capsize=5)
    
    plt.axhline(y=INITIAL_RATING, color='r', linestyle='--', alpha=0.7, label='Initial Rating (1000)')
    
    plt.title('Model Elo Ratings with 95% Confidence Intervals', fontsize=16)
    plt.ylabel('Elo Rating', fontsize=14)
    
    # Set y-axis limits with some padding
    min_value = min(lower_ci) - 50
    max_value = max(upper_ci) + 50
    plt.ylim(min_value, max_value)
    
    # Rotate model names for better readability
    plt.xticks(rotation=30, ha='right', fontsize=12)
    
    # Add rating values on top of bars
    for i, (v, upper) in enumerate(zip(ratings, upper_ci)):
        plt.text(i, v, f"{round(v, 1)}", ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.legend()
    plt.savefig('elo_ratings_with_ci.png', dpi=300, bbox_inches='tight')
    print("Confidence interval visualization saved to elo_ratings_with_ci.png")

if __name__ == "__main__":
    main() 