import json
import pandas as pd
import numpy as np
import joblib
import pulp
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# --- Configuration ---
# Define the path to the directory where your saved model and data are stored
MODEL_DIR = "model" # Assuming you saved them in a 'model' folder

# Define paths to the saved files
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
FEATURE_MEDIANS_PATH = os.path.join(MODEL_DIR, "feature_medians.pkl")
PLAYER_ROLES_SEASON_PATH = os.path.join(MODEL_DIR, "player_roles_by_season.csv")
PLAYER_ROLES_GLOBAL_PATH = os.path.join(MODEL_DIR, "player_roles_global.csv")


# --- Load Model and Data ---
# Load the trained Random Forest model
try:
    rf_model = joblib.load(MODEL_PATH)
    print("Random Forest model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please ensure the model training and saving steps were completed.")
    rf_model = None # Set to None if loading fails

# Load the feature medians
try:
    feature_medians = joblib.load(FEATURE_MEDIANS_PATH)
    print("Feature medians loaded successfully.")
except FileNotFoundError:
    print(f"Error: Feature medians file not found at {FEATURE_MEDIANS_PATH}. Please ensure the feature engineering and saving steps were completed.")
    feature_medians = None # Set to None if loading fails

# Load the player roles dataframes
try:
    player_roles_season_df = pd.read_csv(PLAYER_ROLES_SEASON_PATH)
    player_roles_global_df = pd.read_csv(PLAYER_ROLES_GLOBAL_PATH)
    # Clean column names just in case
    player_roles_season_df.columns = player_roles_season_df.columns.str.strip().str.lower()
    player_roles_global_df.columns = player_roles_global_df.columns.str.strip().str.lower()
    print("Player roles dataframes loaded successfully.")
except FileNotFoundError:
    print(f"Error: Player roles files not found in {MODEL_DIR}. Please ensure the player roles processing and saving steps were completed.")
    player_roles_season_df = pd.DataFrame() # Empty dataframe if loading fails
    player_roles_global_df = pd.DataFrame() # Empty dataframe if loading fails


# # --- Data Preprocessing and Feature Engineering Function ---
# # This function encapsulates the logic from your notebook to prepare data for prediction
# def preprocess_match_data(match_data, historical_data_combined, player_roles_season_df, player_roles_global_df, feature_medians, X_train_cleaned_cols):
#     """
#     Preprocesses raw match data to create a DataFrame ready for prediction.

#     Args:
#         match_data (dict): Dictionary containing the loaded match JSON data.
#         historical_data_combined (pd.DataFrame): DataFrame containing combined historical data
#                                                with features (recency, rolling averages, etc.).
#                                                This should be pre-loaded.
#         player_roles_season_df (pd.DataFrame): DataFrame with seasonal player roles.
#         player_roles_global_df (pd.DataFrame): DataFrame with global player roles.
#         feature_medians (pd.Series): Series containing median values for imputation.
#         X_train_cleaned_cols (list): List of column names from the cleaned training data.

#     Returns:
#         pd.DataFrame: DataFrame for the squad with prepared features, or None if processing fails.
#         pd.DataFrame: Initial squad DataFrame with basic info, roles, and credits.
#     """
#     print("Starting data preprocessing...")
#     try:
#         # Extract relevant information from the match data
#         match_info = match_data["info"]
#         match_date_str = match_info["dates"][0]
#         match_date = pd.to_datetime(match_date_str)
#         match_season = match_info["season"]
#         match_registry = match_info["registry"]["people"]
#         squad_players_info = match_info.get('players', {})

#         squad_data = []
#         for team_name, players_list in squad_players_info.items():
#             for player_name in players_list:
#                 player_id = match_registry.get(player_name, player_name)
#                 squad_data.append({
#                     "player_id": player_id,
#                     "team": team_name,
#                     "match_date": match_date,
#                     "season": match_season,
#                 })

#         df_squad_real = pd.DataFrame(squad_data)
#         df_squad_real = df_squad_real.drop_duplicates(subset=['player_id', 'team']).reset_index(drop=True)

#         # Merge roles from loaded dataframes
#         df_squad_with_roles = pd.merge(
#             df_squad_real[['player_id', 'team', 'match_date', 'season']],
#             player_roles_season_df,
#             on=["player_id", "season"],
#             how="left",
#             suffixes=('', '_seasonal')
#         )
#         df_squad_with_roles = pd.merge(
#             df_squad_with_roles,
#             player_roles_global_df,
#             on="player_id",
#             how="left",
#             suffixes=('', '_global')
#         )
#         df_squad_with_roles['role'] = df_squad_with_roles['role_seasonal'].fillna(df_squad_with_roles['role_global']).fillna('BAT')
#         df_squad_with_roles = df_squad_with_roles.drop(columns=['role_seasonal', 'role_global'], errors='ignore')

#         # Merge credits - assuming df_combined has the credit for this match date
#         # This part needs to be robust. If df_combined doesn't have the exact date,
#         # you'd need a way to get the credit based on the last known match or median.
#         # For simplicity, we'll merge on player_id and match_date, and fill missing credits.
#         historical_player_info_for_merge = historical_data_combined[['player_id', 'match_date', 'credit']].copy()
#         df_squad_with_roles_credits = pd.merge(
#             df_squad_with_roles,
#             historical_player_info_for_merge,
#             on=['player_id', 'match_date'],
#             how='left'
#         )
#         # Fill missing credits with median credit from training data if feature_medians is available
#         if feature_medians is not None and 'credit' in feature_medians:
#              df_squad_with_roles_credits['credit'] = df_squad_with_roles_credits['credit'].fillna(feature_medians['credit'])
#         else:
#              # Fallback if medians not loaded or credit median missing
#              df_squad_with_roles_credits['credit'] = df_squad_with_roles_credits['credit'].fillna(7.75) # Use a default median credit


#         # Merge historical features (recency, rolling averages, team strength)
#         historical_features_for_merge = historical_data_combined[[
#             'player_id', 'match_date', 'recency', 'previous_matches',
#             'rolling_avg_3y', 'rolling_std_3y', 'team_rolling_avg_fp',
#             'opponent_rolling_avg_fp'
#         ]].copy()

#         df_squad_features = pd.merge(
#             df_squad_with_roles_credits,
#             historical_features_for_merge,
#             on=['player_id', 'match_date'],
#             how='left'
#         )

#         # Prepare features for prediction (X_squad)
#         X_squad = df_squad_features[[
#             'credit', 'season', 'recency', 'previous_matches',
#             'rolling_avg_3y', 'rolling_std_3y',
#             'team_rolling_avg_fp', 'opponent_rolling_avg_fp'
#         ]].copy()

#         # Add one-hot encoded roles
#         squad_roles_one_hot = pd.get_dummies(df_squad_features['role'], prefix='role', dummy_na=False)

#         # Ensure all role columns from training are present, fill missing with False
#         training_role_cols = [col for col in X_train_cleaned_cols if col.startswith('role_')]
#         for col in training_role_cols:
#             if col not in squad_roles_one_hot.columns:
#                 squad_roles_one_hot[col] = False

#         # Reorder one-hot encoded columns to match training data
#         squad_roles_one_hot = squad_roles_one_hot[training_role_cols]

#         # Concatenate features
#         X_squad = pd.concat([X_squad, squad_roles_one_hot], axis=1)

#         # Ensure the order of all columns matches the training data
#         X_squad = X_squad[X_train_cleaned_cols]


#         # Handle missing values using loaded medians
#         if feature_medians is not None:
#             X_squad = X_squad.fillna(feature_medians)
#         else:
#              # Fallback if medians not loaded
#              X_squad = X_squad.fillna(X_squad.median()) # Use squad medians as a last resort


#         print("Data preprocessing completed successfully.")
#         return X_squad, df_squad_with_roles_credits # Return both features and the squad info df

#     except Exception as e:
#         print(f"Error during data preprocessing: {e}")
#         return None, None
# --- Data Preprocessing and Feature Engineering Function ---
# This function encapsulates the logic from your notebook to prepare data for prediction
def preprocess_match_data(match_data, historical_data_combined, player_roles_season_df, player_roles_global_df, feature_medians, X_train_cleaned_cols):
    """
    Preprocesses raw match data to create a DataFrame ready for prediction.

    Args:
        match_data (dict): Dictionary containing the loaded match JSON data.
        historical_data_combined (pd.DataFrame): DataFrame containing combined historical data
                                               with features (recency, rolling averages, etc.).
                                               This should be pre-loaded.
        player_roles_season_df (pd.DataFrame): DataFrame with seasonal player roles.
        player_roles_global_df (pd.DataFrame): DataFrame with global player roles.
        feature_medians (pd.Series): Series containing median values for imputation.
        X_train_cleaned_cols (list): List of column names from the cleaned training data.

    Returns:
        pd.DataFrame: DataFrame for the squad with prepared features, or None if processing fails.
        pd.DataFrame: Initial squad DataFrame with basic info, roles, and credits.
    """
    print("Starting data preprocessing...")
    try:
        # Extract relevant information from the match data
        match_info = match_data["info"]
        match_date_str = match_info["dates"][0]
        match_date = pd.to_datetime(match_date_str)
        match_season = match_info["season"]
        match_registry = match_info["registry"]["people"]
        squad_players_info = match_info.get('players', {})

        squad_data = []
        for team_name, players_list in squad_players_info.items():
            for player_name in players_list:
                player_id = match_registry.get(player_name, player_name)
                squad_data.append({
                    "player_id": player_id,
                    "player_name": player_name,
                    "team": team_name,
                    "match_date": match_date,
                    "season": match_season,
                })

        df_squad_real = pd.DataFrame(squad_data)
        # Ensure unique players in the initial squad dataframe
        df_squad_real = df_squad_real.drop_duplicates(subset=['player_id']).reset_index(drop=True)


        # Merge roles from loaded dataframes
        # Use left merge to keep all players from the squad_real and add roles if available
        df_squad_with_roles = pd.merge(
            df_squad_real[['player_id', 'player_name', 'team', 'match_date', 'season']],
            player_roles_season_df[['player_id', 'season', 'role']].rename(columns={'role': 'role_seasonal'}), # Select and rename role column explicitly
            on=["player_id", "season"],
            how="left"
        )
        df_squad_with_roles = pd.merge(
            df_squad_with_roles,
            player_roles_global_df[['player_id', 'role']].rename(columns={'role': 'role_global'}), # Select and rename role column explicitly
            on="player_id",
            how="left"
        )

        # Combine roles: seasonal first, then global, then default to BAT
        # Check if seasonal role column exists before using it
        seasonal_role_col = 'role_seasonal' if 'role_seasonal' in df_squad_with_roles.columns else None
        # Check if global role column exists before using it
        global_role_col = 'role_global' if 'role_global' in df_squad_with_roles.columns else None

        if seasonal_role_col and global_role_col:
            df_squad_with_roles['role'] = df_squad_with_roles[seasonal_role_col].fillna(df_squad_with_roles[global_role_col]).fillna('BAT')
        elif seasonal_role_col:
             df_squad_with_roles['role'] = df_squad_with_roles[seasonal_role_col].fillna('BAT')
        elif global_role_col:
             df_squad_with_roles['role'] = df_squad_with_roles[global_role_col].fillna('BAT')
        else:
            df_squad_with_roles['role'] = 'BAT' # Default to BAT if neither column exists


        # Drop the temporary role columns if they exist
        cols_to_drop = []
        if seasonal_role_col and seasonal_role_col in df_squad_with_roles.columns:
            cols_to_drop.append(seasonal_role_col)
        if global_role_col and global_role_col in df_squad_with_roles.columns:
            cols_to_drop.append(global_role_col)

        df_squad_with_roles = df_squad_with_roles.drop(columns=cols_to_drop, errors='ignore')


        # Merge credits and historical features (recency, rolling averages, team strength)
        # This merge is likely where the duplication is happening.
        # We need to ensure we merge historical features correctly, linking them to the unique players in the squad.
        # Merging on ['player_id', 'match_date'] might bring in multiple rows if a player has multiple entries for the same date in df_combined (e.g., from different roles in the same match data processing).
        # A more robust approach is to merge historical features based on player_id and the *closest* historical match date *before or on* the current match date.
        # However, for simplicity and to match the notebook structure where df_combined already has features calculated per player per match,
        # we'll try to merge on ['player_id', 'match_date'] but handle potential duplicates if they arise from df_combined's structure.

        # Select necessary historical features (including credit and all other features)
        historical_features_for_merge = historical_data_combined[[
            'player_id', 'match_date', 'credit', 'recency', 'previous_matches',
            'rolling_avg_3y', 'rolling_std_3y', 'team_rolling_avg_fp',
            'opponent_rolling_avg_fp'
        ]].copy()

        # Ensure match_date is datetime in both DataFrames
        df_squad_with_roles['match_date'] = pd.to_datetime(df_squad_with_roles['match_date'])
        historical_features_for_merge['match_date'] = pd.to_datetime(historical_features_for_merge['match_date'])

        # Perform the merge. Use a left merge to keep all squad players.
        # If there are multiple matching rows in historical_features_for_merge for a player/date,
        # this will create duplicate rows in df_squad_features.
        df_squad_features = pd.merge(
            df_squad_with_roles,
            historical_features_for_merge,
            on=['player_id', 'match_date'],
            how='left',
            suffixes=('', '_hist') # Add suffix to avoid potential column name conflicts
        )

        # Handle potential duplicate rows resulting from the merge
        # If a player has multiple entries for the same match_date in df_combined,
        # we need to decide how to handle this. For prediction, we need one row per player.
        # A simple approach is to take the first duplicate or aggregate.
        # Let's assume for now that merging on player_id and match_date should ideally yield one row per player if df_combined is structured correctly.
        # If duplicates still occur, we need a more sophisticated approach (e.g., merging on nearest date or selecting one row per player per date).

        # Let's check for and handle duplicates if they occur after the merge
        # If duplicates exist, we'll keep the first occurrence based on the merge order
        df_squad_features = df_squad_features.drop_duplicates(subset=['player_id', 'match_date']).reset_index(drop=True)


        # Prepare features for prediction (X_squad)
        # Select the feature columns that were used for training the model (X_train_cleaned_cols)
        # Ensure the order of columns matches the training data
        # The columns in X_train_cleaned include:
        # 'credit', 'season', 'recency', 'previous_matches', 'rolling_avg_3y', 'rolling_std_3y',
        # 'team_rolling_avg_fp', 'opponent_rolling_avg_fp', 'role_BAT', 'role_BOWL', 'role_WK'

        # Create the feature set for the squad
        # Ensure all required feature columns exist in df_squad_features before selecting
        # If a column is missing after merges, add it and fill with median
        required_features = [col for col in X_train_cleaned_cols if not col.startswith('role_')]
        for col in required_features:
            if col not in df_squad_features.columns:
                 print(f"Warning: Feature column '{col}' missing after merges. Adding with median value.")
                 if feature_medians is not None and col in feature_medians:
                     df_squad_features[col] = feature_medians[col]
                 else:
                     df_squad_features[col] = 0 # Default to 0 if median not available

        X_squad = df_squad_features[required_features].copy()


        # Add one-hot encoded roles for the squad
        # Need to get all possible roles from the training data's X_train_cleaned to ensure consistency
        training_role_cols = [col for col in X_train_cleaned_cols if col.startswith('role_')]
        # Ensure 'role' column exists before one-hot encoding
        if 'role' not in df_squad_features.columns:
             print("Error: 'role' column not found in df_squad_features. Cannot create one-hot encoded roles.")
             return None, None # Cannot proceed without roles

        squad_roles_one_hot = pd.get_dummies(df_squad_features['role'], prefix='role', dummy_na=False)

        # Ensure all role columns from training are present in squad features, fill missing with False
        for col in training_role_cols:
            if col not in squad_roles_one_hot.columns:
                squad_roles_one_hot[col] = False

        # Reorder the one-hot encoded columns to match training data
        squad_roles_one_hot = squad_roles_one_hot[training_role_cols]

        # Concatenate the features
        X_squad = pd.concat([X_squad, squad_roles_one_hot], axis=1)

        # Ensure the order of all columns in X_squad matches X_train_cleaned
        X_squad = X_squad[X_train_cleaned_cols]


        # Handle any remaining missing values using loaded medians
        if feature_medians is not None:
            X_squad = X_squad.fillna(feature_medians)
        else:
             # Fallback if medians not loaded
             X_squad = X_squad.fillna(X_squad.median()) # Use squad medians as a last resort

        # Verify the number of rows in X_squad matches the number of players in the initial squad
        if len(X_squad) != len(df_squad_real):
             print(f"Warning: Number of rows in X_squad ({len(X_squad)}) does not match number of players in squad ({len(df_squad_real)}).")
             # This indicates a potential issue in merging or duplicate handling.
             # You might need to inspect df_squad_features and X_squad at this point if errors persist.


        print("Data preprocessing completed successfully.")
        return X_squad, df_squad_features # Return both features and the processed squad info df (df_squad_features now includes all merged info)

    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        # Temporarily re-raise the exception to see the full traceback
        raise e
        # return None, None # Comment out or remove this line for now


# --- Prediction Function ---
# This function uses the trained model to predict fantasy points
def predict_fantasy_points(X_squad, model):
    """
    Predicts fantasy points for the squad using the trained model.

    Args:
        X_squad (pd.DataFrame): DataFrame with prepared features for the squad.
        model: The trained machine learning model.

    Returns:
        np.ndarray: Array of predicted fantasy points.
    """
    print("Starting fantasy point prediction...")
    if model is None:
        print("Error: Model not loaded. Cannot predict.")
        return None
    try:
        predictions = model.predict(X_squad)
        print("Fantasy point prediction completed successfully.")
        return predictions
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None


# --- ILP Solver Function ---
# This function implements the ILP logic to select the optimal XI
def select_optimal_xi(df_squad_with_predictions):
    """
    Selects the optimal fantasy XI using Integer Linear Programming (ILP).

    Args:
        df_squad_with_predictions (pd.DataFrame): DataFrame for the squad
                                                  including 'player_id', 'team',
                                                  'role', 'credit', and 'predicted_fantasy_points'.

    Returns:
        pd.DataFrame: DataFrame containing the selected optimal XI, or None if no feasible solution.
    """
    print("Starting optimal XI selection using ILP...")
    if df_squad_with_predictions is None or df_squad_with_predictions.empty:
        print("Error: Squad DataFrame with predictions is empty or None. Cannot select XI.")
        return None

    try:
        # Create the ILP problem instance
        prob = pulp.LpProblem("Fantasy Team Selection", pulp.LpMaximize)

        # Define decision variables
        player_vars = pulp.LpVariable.dicts("Select", df_squad_with_predictions['player_id'], 0, 1, pulp.LpInteger)

        # Define the Objective Function (Maximize total predicted fantasy points)
        # Ensure predicted_fantasy_points is numeric and handle NaNs
        df_squad_with_predictions['predicted_fantasy_points'] = pd.to_numeric(df_squad_with_predictions['predicted_fantasy_points'], errors='coerce').fillna(0)

        prob += pulp.lpSum([df_squad_with_predictions.loc[df_squad_with_predictions['player_id'] == player_id, 'predicted_fantasy_points'].iloc[0] * player_vars[player_id] for player_id in df_squad_with_predictions['player_id']]), "Total Predicted Fantasy Points"

        # Add Constraints

        # Constraint 1: Total Players = 11
        prob += pulp.lpSum([player_vars[player_id] for player_id in df_squad_with_predictions['player_id']]) == 11, "Total Players = 11"

        # Constraint 2: Role Bounds
        # Ensure 'role' column exists before grouping
        if 'role' not in df_squad_with_predictions.columns:
             print("Error: 'role' column not found in squad data for ILP constraints.")
             return None # Cannot proceed without roles

        role_player_ids = df_squad_with_predictions.groupby('role')['player_id'].apply(list).to_dict()
        wk_players = role_player_ids.get('WK', [])
        bat_players = role_player_ids.get('BAT', [])
        ar_players = role_player_ids.get('AR', [])
        bowl_players = role_player_ids.get('BOWL', [])

        prob += pulp.lpSum([player_vars[player_id] for player_id in wk_players]) >= 1, "Min WK = 1"
        prob += pulp.lpSum([player_vars[player_id] for player_id in wk_players]) <= 4, "Max WK = 4"
        prob += pulp.lpSum([player_vars[player_id] for player_id in bat_players]) >= 3, "Min BAT = 3"
        prob += pulp.lpSum([player_vars[player_id] for player_id in bat_players]) <= 6, "Max BAT = 6"
        prob += pulp.lpSum([player_vars[player_id] for player_id in ar_players]) >= 1, "Min AR = 1"
        prob += pulp.lpSum([player_vars[player_id] for player_id in ar_players]) <= 4, "Max AR = 4"
        prob += pulp.lpSum([player_vars[player_id] for player_id in bowl_players]) >= 3, "Min BOWL = 3"
        prob += pulp.lpSum([player_vars[player_id] for player_id in bowl_players]) <= 6, "Max BOWL = 6"

        # Constraint 3: Team Cap (Max 7 players from any team)
        # Ensure 'team' column exists before grouping
        if 'team' not in df_squad_with_predictions.columns:
             print("Error: 'team' column not found in squad data for ILP constraints.")
             return None # Cannot proceed without teams

        team_player_ids = df_squad_with_predictions.groupby('team')['player_id'].apply(list).to_dict()
        for team_name, player_ids in team_player_ids.items():
            prob += pulp.lpSum([player_vars[player_id] for player_id in player_ids]) <= 7, f"Max 7 players from {team_name}"

        # Constraint 4: Both Teams Must Be Represented
        teams_in_squad = list(team_player_ids.keys())
        if len(teams_in_squad) == 2:
            team1_players = team_player_ids[teams_in_squad[0]]
            team2_players = team_player_ids[teams_in_squad[1]]
            prob += pulp.lpSum([player_vars[player_id] for player_id in team1_players]) >= 1, f"Min 1 player from {teams_in_squad[0]}"
            prob += pulp.lpSum([player_vars[player_id] for player_id in team2_players]) >= 1, f"Min 1 player from {teams_in_squad[1]}"


        # Solve the problem
        prob.solve()

        # Check solver status and extract result
        if prob.status == pulp.LpStatusOptimal:
            selected_xi_ids = [player_id for player_id in df_squad_with_predictions['player_id'] if player_vars[player_id].varValue == 1]
            df_selected_xi = df_squad_with_predictions[df_squad_with_predictions['player_id'].isin(selected_xi_ids)].copy()
            print("Optimal XI selection completed successfully.")
            return df_selected_xi
        else:
            print(f"ILP solver did not find an optimal solution. Status: {pulp.LpStatus[prob.status]}")
            return None

    except Exception as e:
        print(f"Error during ILP optimal XI selection: {e}")
        return None


# --- Flask Application ---
app = Flask(__name__)
CORS(app)

# Assume df_combined is loaded or accessible globally for historical features
# In a real application, you might load this once when the app starts
# or use a database. For this example, we assume it's in memory.
# You would need to load your full historical_data_combined DataFrame here.
# Replace this with your actual loading logic for df_combined
# Example: df_combined = pd.read_csv('path/to/your/combined_historical_data.csv')
# Or load from the source JSONs and process if that's faster/more feasible

# IMPORTANT: Ensure df_combined is available and properly loaded/processed
# before running the Flask app. If your df_combined is very large, consider
# optimizing how you access historical features (e.g., using a database).

# Define the path to the saved combined historical data CSV file
DF_COMBINED_PATH = os.path.join(MODEL_DIR, "df_combined.csv") # Assuming it's in the 'model' directory

# Load the combined historical data from the saved CSV file
try:
    df_combined = pd.read_csv(DF_COMBINED_PATH)
    df_combined['match_date'] = pd.to_datetime(df_combined['match_date'])
    print("Historical combined data loaded successfully.")
except FileNotFoundError:
    print(f"Error: Combined historical data file not found at {DF_COMBINED_PATH}.")
    print("Please ensure df_combined.csv is in the correct location.")
    df_combined = pd.DataFrame() # Empty dataframe if loading fails
except Exception as e:
    print(f"Error loading combined historical data: {e}")
    df_combined = pd.DataFrame() # Empty dataframe if loading fails


# Define the path to the saved feature column names file
FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "X_train_cleaned_cols.pkl")

# Load the list of feature column names
try:
    X_train_cleaned_cols = joblib.load(FEATURE_COLS_PATH)
    print("X_train_cleaned column names loaded successfully.")
except FileNotFoundError:
    print(f"Error: Feature column names file not found at {FEATURE_COLS_PATH}.")
    print("Please ensure X_train_cleaned_cols.pkl is in the correct location.")
    X_train_cleaned_cols = [] # Empty list if loading fails
except Exception as e:
    print(f"Error loading feature column names: {e}")
    X_train_cleaned_cols = [] # Empty list if loading fails

# Define the path to the saved feature medians file
FEATURE_MEDIANS_PATH = os.path.join(MODEL_DIR, "feature_medians.pkl")
# Load the feature medians
try:
    feature_medians = joblib.load(FEATURE_MEDIANS_PATH)
    print("Feature medians loaded successfully.")
except FileNotFoundError:
    print(f"Error: Feature medians file not found at {FEATURE_MEDIANS_PATH}. Please ensure the feature engineering and saving steps were completed.")
    feature_medians = None # Set to None if loading fails
except Exception as e:
    print(f"Error loading feature medians: {e}")
    feature_medians = None # Set to None if loading fails

# Define the path to the saved player roles files
PLAYER_ROLES_SEASON_PATH = os.path.join(MODEL_DIR, "player_roles_by_season.csv")
PLAYER_ROLES_GLOBAL_PATH = os.path.join(MODEL_DIR, "player_roles_global.csv")
# Load the player roles dataframes
try:
    player_roles_season_df = pd.read_csv(PLAYER_ROLES_SEASON_PATH)
    player_roles_global_df = pd.read_csv(PLAYER_ROLES_GLOBAL_PATH)
    # Clean column names just in case
    player_roles_season_df.columns = player_roles_season_df.columns.str.strip().str.lower()
    player_roles_global_df.columns = player_roles_global_df.columns.str.strip().str.lower()
    print("Player roles dataframes loaded successfully.")
except FileNotFoundError:
    print(f"Error: Player roles files not found in {MODEL_DIR}. Please ensure the player roles processing and saving steps were completed.")
    player_roles_season_df = pd.DataFrame() # Empty dataframe if loading fails
    player_roles_global_df = pd.DataFrame() # Empty dataframe if loading fails
except Exception as e:
    print(f"Error loading player roles dataframes: {e}")
    player_roles_season_df = pd.DataFrame() # Empty dataframe if loading fails
    player_roles_global_df = pd.DataFrame() # Empty dataframe if loading fails

# Define the path to the saved model file
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
# Load the trained Random Forest model
try:
    rf_model = joblib.load(MODEL_PATH)
    print("Random Forest model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please ensure the model training and saving steps were completed.")
    rf_model = None # Set to None if loading fails
except Exception as e:
    print(f"Error loading Random Forest model: {e}")
    rf_model = None # Set to None if loading fails


@app.route('/predict_team', methods=['POST'])
def predict_team():
    """
    Flask endpoint to receive match data, predict fantasy points,
    select the optimal XI, and return the result.
    """
    print("Received request to /predict_team")
    if not request.json:
        print("Error: Request body is not JSON.")
        return jsonify({"error": "Request body must be JSON"}), 415 # Unsupported Media Type

    match_data = request.json
    # Add a check to ensure match_data is a dictionary
    if not isinstance(match_data, dict):
        print("Error: Incoming data is not a valid JSON object (dictionary).")
        return jsonify({"error": "Invalid JSON data received. Expected a JSON object."}), 400 # Bad Request
    # --- Step 1: Preprocess the match data and engineer features ---
    # Requires df_combined to be available globally or loaded here
    # Ensure df_combined is loaded and accessible in your Flask app's environment
    if df_combined.empty: # Check if df_combined was loaded successfully
         print("Error: df_combined (historical data) is not loaded. Cannot preprocess.")
         return jsonify({"error": "Historical data not loaded. Please check server setup."}), 500

    if not X_train_cleaned_cols: # Check if feature columns were loaded successfully
         print("Error: Training feature columns not available. Cannot preprocess.")
         return jsonify({"error": "Training feature columns not available. Please check server setup."}), 500

    if feature_medians is None: # Check if feature medians were loaded successfully
         print("Error: Feature medians not loaded. Cannot preprocess.")
         return jsonify({"error": "Feature medians not loaded. Please check server setup."}), 500

    if player_roles_season_df.empty or player_roles_global_df.empty: # Check if player roles were loaded successfully
         print("Error: Player roles data not loaded. Cannot preprocess.")
         return jsonify({"error": "Player roles data not loaded. Please check server setup."}), 500

    if rf_model is None: # Check if model was loaded successfully
         print("Error: Model not loaded. Cannot predict.")
         return jsonify({"error": "Model not loaded. Please check server setup."}), 500


    X_squad, df_squad_info = preprocess_match_data(match_data, df_combined, player_roles_season_df, player_roles_global_df, feature_medians, X_train_cleaned_cols)

    if X_squad is None:
        return jsonify({"error": "Failed to preprocess match data"}), 500

    # --- Step 2: Predict fantasy points ---
    predictions = predict_fantasy_points(X_squad, rf_model)

    if predictions is None:
        return jsonify({"error": "Failed to predict fantasy points"}), 500

    # Add predictions to the squad info DataFrame
    # Ensure the index of predictions aligns with X_squad and df_squad_info
    # The length mismatch suggests X_squad and df_squad_info have different numbers of rows.
    # We need to ensure they are aligned correctly after preprocessing.
    # Since df_squad_info is returned by preprocess_match_data along with X_squad,
    # they should ideally have the same index and number of rows if preprocessing is correct.
    # Let's try to assign predictions based on the index of X_squad, assuming it aligns with df_squad_info
    if len(predictions) == len(df_squad_info):
        df_squad_with_predictions = df_squad_info.copy()
        df_squad_with_predictions['predicted_fantasy_points'] = predictions
    else:
        # If lengths still don't match, there's a deeper issue in preprocess_match_data
        # We should not proceed and indicate an error
        print(f"Error: Mismatch in lengths after preprocessing. Predictions length: {len(predictions)}, Squad info length: {len(df_squad_info)}")
        return jsonify({"error": "Internal error: Mismatch in data lengths after preprocessing."}), 500


    # --- Step 3: Select the optimal XI using ILP ---
    df_selected_xi = select_optimal_xi(df_squad_with_predictions)

    if df_selected_xi is None:
        return jsonify({"error": "Failed to select optimal XI using ILP. Check constraints or squad data."}), 500

    # --- Step 4: Prepare the response ---
    # Convert the selected XI DataFrame to a JSON friendly format
    # You might want to include additional player info here if needed for the frontend
    selected_xi_list = df_selected_xi[['player_id','player_name', 'team', 'role', 'credit', 'predicted_fantasy_points']].to_dict(orient='records')

    response_data = {
        "optimal_xi": selected_xi_list,
        "total_predicted_points": df_selected_xi['predicted_fantasy_points'].sum()
        # You could add more summary info here if desired
    }

    print("Successfully generated optimal XI.")
    return jsonify(response_data), 200 # OK

# --- Running the Flask app ---
# This part runs the Flask development server.
# In a production environment, you would use a production-ready WSGI server
# like Gunicorn or uWSGI.
if __name__ == '__main__':
    # Make sure the model and data files exist before starting the app
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}. Please train and save the model.")
    elif not os.path.exists(FEATURE_MEDIANS_PATH):
        print(f"Error: Feature medians file not found at {FEATURE_MEDIANS_PATH}. Please save the medians.")
    elif not os.path.exists(PLAYER_ROLES_SEASON_PATH) or not os.path.exists(PLAYER_ROLES_GLOBAL_PATH):
         print(f"Error: Player roles files not found in {MODEL_DIR}. Please process and save the roles.")
    elif df_combined.empty: # Check if df_combined was loaded successfully
         print("Error: df_combined (historical data) is empty after loading attempt.")
         print("Please check the loading path and file content.")
    elif not X_train_cleaned_cols: # Check if feature columns were loaded successfully
         print("Error: Training feature columns are empty after loading attempt.")
         print("Please check the loading path and file content for X_train_cleaned_cols.pkl.")
    else:
        print("Starting Flask development server...")
        # You can change the port and host here if needed
        app.run(debug=True, host='0.0.0.0', port=5001)


# # --- Flask Application ---
# app = Flask(__name__)

# # Assume df_combined is loaded or accessible globally for historical features
# # In a real application, you might load this once when the app starts
# # or use a database. For this example, we assume it's in memory.
# # You would need to load your full historical_data_combined DataFrame here.
# # Replace this with your actual loading logic for df_combined
# # Example: df_combined = pd.read_csv('path/to/your/combined_historical_data.csv')
# # Or load from the source JSONs and process if that's faster/more feasible

# # Placeholder for df_combined - YOU NEED TO LOAD YOUR ACTUAL df_combined HERE
# # For demonstration, let's create a dummy or assume it's available from a previous step
# # If df_combined is not loaded, the preprocessing function will likely fail.
# # In a real Flask app, you would load this data once when the app starts.
# # Example:
# # try:
# #     df_combined = pd.read_csv('path/to/your/combined_historical_data.csv')
# #     df_combined['match_date'] = pd.to_datetime(df_combined['match_date'])
# #     print("Historical combined data loaded successfully.")
# # except FileNotFoundError:
# #     print("Error: Combined historical data file not found. Preprocessing will fail.")
# #     df_combined = pd.DataFrame() # Empty dataframe if loading fails

# # IMPORTANT: Ensure df_combined is available and properly loaded/processed
# # before running the Flask app. If your df_combined is very large, consider
# # optimizing how you access historical features (e.g., using a database).

# # Define the path to the saved combined historical data CSV file
# DF_COMBINED_PATH = os.path.join(MODEL_DIR, "df_combined.csv") # Assuming it's in the 'model' directory

# # Load the combined historical data from the saved CSV file
# try:
#     df_combined = pd.read_csv(DF_COMBINED_PATH)
#     df_combined['match_date'] = pd.to_datetime(df_combined['match_date'])
#     print("Historical combined data loaded successfully.")
# except FileNotFoundError:
#     print(f"Error: Combined historical data file not found at {DF_COMBINED_PATH}.")
#     print("Please ensure df_combined.csv is in the correct location.")
#     df_combined = pd.DataFrame() # Empty dataframe if loading fails
# except Exception as e:
#     print(f"Error loading combined historical data: {e}")
#     df_combined = pd.DataFrame() # Empty dataframe if loading fails


# # Define the path to the saved feature column names file
# FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "X_train_cleaned_cols.pkl")

# # Load the list of feature column names
# try:
#     X_train_cleaned_cols = joblib.load(FEATURE_COLS_PATH)
#     print("X_train_cleaned column names loaded successfully.")
# except FileNotFoundError:
#     print(f"Error: Feature column names file not found at {FEATURE_COLS_PATH}.")
#     print("Please ensure X_train_cleaned_cols.pkl is in the correct location.")
#     X_train_cleaned_cols = [] # Empty list if loading fails
# except Exception as e:
#     print(f"Error loading feature column names: {e}")
#     X_train_cleaned_cols = [] # Empty list if loading fails

# # Define the path to the saved feature medians file
# FEATURE_MEDIANS_PATH = os.path.join(MODEL_DIR, "feature_medians.pkl")
# # Load the feature medians
# try:
#     feature_medians = joblib.load(FEATURE_MEDIANS_PATH)
#     print("Feature medians loaded successfully.")
# except FileNotFoundError:
#     print(f"Error: Feature medians file not found at {FEATURE_MEDIANS_PATH}. Please ensure the feature engineering and saving steps were completed.")
#     feature_medians = None # Set to None if loading fails
# except Exception as e:
#     print(f"Error loading feature medians: {e}")
#     feature_medians = None # Set to None if loading fails

# # Define the path to the saved player roles files
# PLAYER_ROLES_SEASON_PATH = os.path.join(MODEL_DIR, "player_roles_by_season.csv")
# PLAYER_ROLES_GLOBAL_PATH = os.path.join(MODEL_DIR, "player_roles_global.csv")
# # Load the player roles dataframes
# try:
#     player_roles_season_df = pd.read_csv(PLAYER_ROLES_SEASON_PATH)
#     player_roles_global_df = pd.read_csv(PLAYER_ROLES_GLOBAL_PATH)
#     # Clean column names just in case
#     player_roles_season_df.columns = player_roles_season_df.columns.str.strip().str.lower()
#     player_roles_global_df.columns = player_roles_global_df.columns.str.strip().str.lower()
#     print("Player roles dataframes loaded successfully.")
# except FileNotFoundError:
#     print(f"Error: Player roles files not found in {MODEL_DIR}. Please ensure the player roles processing and saving steps were completed.")
#     player_roles_season_df = pd.DataFrame() # Empty dataframe if loading fails
#     player_roles_global_df = pd.DataFrame() # Empty dataframe if loading fails
# except Exception as e:
#     print(f"Error loading player roles dataframes: {e}")
#     player_roles_season_df = pd.DataFrame() # Empty dataframe if loading fails
#     player_roles_global_df = pd.DataFrame() # Empty dataframe if loading fails

# # Define the path to the saved model file
# MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
# # Load the trained Random Forest model
# try:
#     rf_model = joblib.load(MODEL_PATH)
#     print("Random Forest model loaded successfully.")
# except FileNotFoundError:
#     print(f"Error: Model file not found at {MODEL_PATH}. Please ensure the model training and saving steps were completed.")
#     rf_model = None # Set to None if loading fails
# except Exception as e:
#     print(f"Error loading Random Forest model: {e}")
#     rf_model = None # Set to None if loading fails


# @app.route('/predict_team', methods=['POST'])
# def predict_team():
#     """
#     Flask endpoint to receive match data, predict fantasy points,
#     select the optimal XI, and return the result.
#     """
#     print("Received request to /predict_team")
#     if not request.json:
#         print("Error: Request body is not JSON.")
#         return jsonify({"error": "Request body must be JSON"}), 415 # Unsupported Media Type

#     match_data = request.json

#     # --- Step 1: Preprocess the match data and engineer features ---
#     # Requires df_combined to be available globally or loaded here
#     # Ensure df_combined is loaded and accessible in your Flask app's environment
#     if df_combined.empty: # Check if df_combined was loaded successfully
#          print("Error: df_combined (historical data) is not loaded. Cannot preprocess.")
#          return jsonify({"error": "Historical data not loaded. Please check server setup."}), 500

#     if not X_train_cleaned_cols: # Check if feature columns were loaded successfully
#          print("Error: Training feature columns not available. Cannot preprocess.")
#          return jsonify({"error": "Training feature columns not available. Please check server setup."}), 500

#     if feature_medians is None: # Check if feature medians were loaded successfully
#          print("Error: Feature medians not loaded. Cannot preprocess.")
#          return jsonify({"error": "Feature medians not loaded. Please check server setup."}), 500

#     if player_roles_season_df.empty or player_roles_global_df.empty: # Check if player roles were loaded successfully
#          print("Error: Player roles data not loaded. Cannot preprocess.")
#          return jsonify({"error": "Player roles data not loaded. Please check server setup."}), 500

#     if rf_model is None: # Check if model was loaded successfully
#          print("Error: Model not loaded. Cannot predict.")
#          return jsonify({"error": "Model not loaded. Please check server setup."}), 500


#     X_squad, df_squad_info = preprocess_match_data(match_data, df_combined, player_roles_season_df, player_roles_global_df, feature_medians, X_train_cleaned_cols)

#     if X_squad is None:
#         return jsonify({"error": "Failed to preprocess match data"}), 500

#     # --- Step 2: Predict fantasy points ---
#     predictions = predict_fantasy_points(X_squad, rf_model)

#     if predictions is None:
#         return jsonify({"error": "Failed to predict fantasy points"}), 500

#     # Add predictions to the squad info DataFrame
#     df_squad_with_predictions = df_squad_info.copy()
#     df_squad_with_predictions['predicted_fantasy_points'] = predictions

#     # --- Step 3: Select the optimal XI using ILP ---
#     df_selected_xi = select_optimal_xi(df_squad_with_predictions)

#     if df_selected_xi is None:
#         return jsonify({"error": "Failed to select optimal XI using ILP. Check constraints or squad data."}), 500

#     # --- Step 4: Prepare the response ---
#     # Convert the selected XI DataFrame to a JSON friendly format
#     # You might want to include additional player info here if needed for the frontend
#     selected_xi_list = df_selected_xi[['player_id', 'team', 'role', 'credit', 'predicted_fantasy_points']].to_dict(orient='records')

#     response_data = {
#         "optimal_xi": selected_xi_list,
#         "total_predicted_points": df_selected_xi['predicted_fantasy_points'].sum()
#         # You could add more summary info here if desired
#     }

#     print("Successfully generated optimal XI.")
#     return jsonify(response_data), 200 # OK

# # --- Running the Flask app ---
# # This part runs the Flask development server.
# # In a production environment, you would use a production-ready WSGI server
# # like Gunicorn or uWSGI.
# if __name__ == '__main__':
#     # Make sure the model and data files exist before starting the app
#     if not os.path.exists(MODEL_PATH):
#         print(f"Error: Model file not found at {MODEL_PATH}. Please train and save the model.")
#     elif not os.path.exists(FEATURE_MEDIANS_PATH):
#         print(f"Error: Feature medians file not found at {FEATURE_MEDIANS_PATH}. Please save the medians.")
#     elif not os.path.exists(PLAYER_ROLES_SEASON_PATH) or not os.path.exists(PLAYER_ROLES_GLOBAL_PATH):
#          print(f"Error: Player roles files not found in {MODEL_DIR}. Please process and save the roles.")
#     elif df_combined.empty: # Check if df_combined was loaded successfully
#          print("Error: df_combined (historical data) is empty after loading attempt.")
#          print("Please check the loading path and file content.")
#     elif not X_train_cleaned_cols: # Check if feature columns were loaded successfully
#          print("Error: Training feature columns are empty after loading attempt.")
#          print("Please check the loading path and file content for X_train_cleaned_cols.pkl.")
#     else:
#         print("Starting Flask development server...")
#         # You can change the port and host here if needed
#         app.run(debug=True, host='0.0.0.0', port=5001)


# # # --- Flask Application ---
# # app = Flask(__name__)

# # # Assume df_combined is loaded or accessible globally for historical features
# # # In a real application, you might load this once when the app starts
# # # or use a database. For this example, we assume it's in memory.
# # # You would need to load your full historical_data_combined DataFrame here.
# # # Replace this with your actual loading logic for df_combined
# # # Example: df_combined = pd.read_csv('path/to/your/combined_historical_data.csv')
# # # Or load from the source JSONs and process if that's faster/more feasible

# # # Placeholder for df_combined - YOU NEED TO LOAD YOUR ACTUAL df_combined HERE
# # # For demonstration, let's create a dummy or assume it's available from a previous step
# # # If df_combined is not loaded, the preprocessing function will likely fail.
# # # In a real Flask app, you would load this data once when the app starts.
# # # Example:
# # # try:
# # #     df_combined = pd.read_csv('path/to/your/combined_historical_data.csv')
# # #     df_combined['match_date'] = pd.to_datetime(df_combined['match_date'])
# # #     print("Historical combined data loaded successfully.")
# # # except FileNotFoundError:
# # #     print("Error: Combined historical data file not found. Preprocessing will fail.")
# # #     df_combined = pd.DataFrame() # Empty dataframe if loading fails

# # # IMPORTANT: Ensure df_combined is available and properly loaded/processed
# # # before running the Flask app. If your df_combined is very large, consider
# # # optimizing how you access historical features (e.g., using a database).

# # # Define the path to the saved combined historical data CSV file
# # DF_COMBINED_PATH = os.path.join(MODEL_DIR, "df_combined.csv") # Assuming it's in the 'model' directory

# # # Load the combined historical data from the saved CSV file
# # try:
# #     df_combined = pd.read_csv(DF_COMBINED_PATH)
# #     df_combined['match_date'] = pd.to_datetime(df_combined['match_date'])
# #     print("Historical combined data loaded successfully.")
# # except FileNotFoundError:
# #     print(f"Error: Combined historical data file not found at {DF_COMBINED_PATH}.")
# #     print("Please ensure df_combined.csv is in the correct location.")
# #     df_combined = pd.DataFrame() # Empty dataframe if loading fails
# # except Exception as e:
# #     print(f"Error loading combined historical data: {e}")
# #     df_combined = pd.DataFrame() # Empty dataframe if loading fails


# # # Define the path to the saved feature column names file
# # FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "X_train_cleaned_cols.pkl")

# # # Load the list of feature column names
# # try:
# #     X_train_cleaned_cols = joblib.load(FEATURE_COLS_PATH)
# #     print("X_train_cleaned column names loaded successfully.")
# # except FileNotFoundError:
# #     print(f"Error: Feature column names file not found at {FEATURE_COLS_PATH}.")
# #     print("Please ensure X_train_cleaned_cols.pkl is in the correct location.")
# #     X_train_cleaned_cols = [] # Empty list if loading fails
# # except Exception as e:
# #     print(f"Error loading feature column names: {e}")
# #     X_train_cleaned_cols = [] # Empty list if loading fails


# # @app.route('/predict_team', methods=['POST'])
# # def predict_team():
# #     """
# #     Flask endpoint to receive match data, predict fantasy points,
# #     select the optimal XI, and return the result.
# #     """
# #     print("Received request to /predict_team")
# #     if not request.json:
# #         print("Error: Request body is not JSON.")
# #         return jsonify({"error": "Request body must be JSON"}), 415 # Unsupported Media Type

# #     match_data = request.json

# #     # --- Step 1: Preprocess the match data and engineer features ---
# #     # Requires df_combined to be available globally or loaded here
# #     # Ensure df_combined is loaded and accessible in your Flask app's environment
# #     if df_combined.empty: # Check if df_combined was loaded successfully
# #          print("Error: df_combined (historical data) is not loaded. Cannot preprocess.")
# #          return jsonify({"error": "Historical data not loaded. Please check server setup."}), 500

# #     if not X_train_cleaned_cols: # Check if feature columns were loaded successfully
# #          print("Error: Training feature columns not available. Cannot preprocess.")
# #          return jsonify({"error": "Training feature columns not available. Please check server setup."}), 500


# #     X_squad, df_squad_info = preprocess_match_data(match_data, df_combined, player_roles_season_df, player_roles_global_df, feature_medians, X_train_cleaned_cols)

# #     if X_squad is None:
# #         return jsonify({"error": "Failed to preprocess match data"}), 500

# #     # --- Step 2: Predict fantasy points ---
# #     predictions = predict_fantasy_points(X_squad, rf_model)

# #     if predictions is None:
# #         return jsonify({"error": "Failed to predict fantasy points"}), 500

# #     # Add predictions to the squad info DataFrame
# #     df_squad_with_predictions = df_squad_info.copy()
# #     df_squad_with_predictions['predicted_fantasy_points'] = predictions

# #     # --- Step 3: Select the optimal XI using ILP ---
# #     df_selected_xi = select_optimal_xi(df_squad_with_predictions)

# #     if df_selected_xi is None:
# #         return jsonify({"error": "Failed to select optimal XI using ILP. Check constraints or squad data."}), 500

# #     # --- Step 4: Prepare the response ---
# #     # Convert the selected XI DataFrame to a JSON friendly format
# #     # You might want to include additional player info here if needed for the frontend
# #     selected_xi_list = df_selected_xi[['player_id', 'team', 'role', 'credit', 'predicted_fantasy_points']].to_dict(orient='records')

# #     response_data = {
# #         "optimal_xi": selected_xi_list,
# #         "total_predicted_points": df_selected_xi['predicted_fantasy_points'].sum()
# #         # You could add more summary info here if desired
# #     }

# #     print("Successfully generated optimal XI.")
# #     return jsonify(response_data), 200 # OK

# # # --- Running the Flask app ---
# # # This part runs the Flask development server.
# # # In a production environment, you would use a production-ready WSGI server
# # # like Gunicorn or uWSGI.
# # if __name__ == '__main__':
# #     # Make sure the model and data files exist before starting the app
# #     if not os.path.exists(MODEL_PATH):
# #         print(f"Error: Model file not found at {MODEL_PATH}. Please train and save the model.")
# #     elif not os.path.exists(FEATURE_MEDIANS_PATH):
# #         print(f"Error: Feature medians file not found at {FEATURE_MEDIANS_PATH}. Please save the medians.")
# #     elif not os.path.exists(PLAYER_ROLES_SEASON_PATH) or not os.path.exists(PLAYER_ROLES_GLOBAL_PATH):
# #          print(f"Error: Player roles files not found in {MODEL_DIR}. Please process and save the roles.")
# #     elif df_combined.empty: # Check if df_combined was loaded successfully
# #          print("Error: df_combined (historical data) is empty after loading attempt.")
# #          print("Please check the loading path and file content.")
# #     elif not X_train_cleaned_cols: # Check if feature columns were loaded successfully
# #          print("Error: Training feature columns are empty after loading attempt.")
# #          print("Please check the loading path and file content for X_train_cleaned_cols.pkl.")
# #     else:
# #         print("Starting Flask development server...")
# #         # You can change the port and host here if needed
# #         app.run(debug=True, host='0.0.0.0', port=5001)


# # # --- Flask Application ---
# # app = Flask(__name__)
# # # Load the combined historical data from the saved CSV file
# # try:
# #     # Define the path to the saved df_combined.csv file
# #     # Make sure this path is correct relative to your app.py file
# #     DF_COMBINED_PATH = os.path.join(MODEL_DIR, "df_combined.csv") # Assuming it's in the 'model' directory

# #     df_combined = pd.read_csv(DF_COMBINED_PATH)
# #     df_combined['match_date'] = pd.to_datetime(df_combined['match_date'])
# #     print("Historical combined data loaded successfully.")
# # except FileNotFoundError:
# #     print(f"Error: Combined historical data file not found at {DF_COMBINED_PATH}.")
# #     print("Please ensure df_combined.csv is in the correct location.")
# #     df_combined = pd.DataFrame() # Empty dataframe if loading fails
# # except Exception as e:
# #     print(f"Error loading combined historical data: {e}")
# #     df_combined = pd.DataFrame() # Empty dataframe if loading fails

# # # IMPORTANT: Ensure df_combined is available and properly loaded/processed
# # # before running the Flask app. If your df_combined is very large, consider
# # # optimizing how you access historical features (e.g., using a database).
# # # Assume df_combined is loaded or accessible globally for historical features
# # # In a real application, you might load this once when the app starts
# # # or use a database. For this example, we assume it's in memory.
# # # You would need to load your full historical_data_combined DataFrame here.
# # # Replace this with your actual loading logic for df_combined
# # # Example: df_combined = pd.read_csv('path/to/your/combined_historical_data.csv')
# # # Or load from the source JSONs and process if that's faster/more feasible

# # # Placeholder for df_combined - YOU NEED TO LOAD YOUR ACTUAL df_combined HERE
# # # For demonstration, let's create a dummy or assume it's available from a previous step
# # # If df_combined is not loaded, the preprocessing function will likely fail.
# # # In a real Flask app, you would load this data once when the app starts.
# # # Example:
# # # try:
# # #     df_combined = pd.read_csv('path/to/your/combined_historical_data.csv')
# # #     df_combined['match_date'] = pd.to_datetime(df_combined['match_date'])
# # #     print("Historical combined data loaded successfully.")
# # # except FileNotFoundError:
# # #     print("Error: Combined historical data file not found. Preprocessing will fail.")
# # #     df_combined = pd.DataFrame() # Empty dataframe if loading fails

# # # IMPORTANT: Ensure df_combined is available and properly loaded/processed
# # # before running the Flask app. If your df_combined is very large, consider
# # # optimizing how you access historical features (e.g., using a database).


# # # Get the list of feature columns from the cleaned training data
# # # This is needed to ensure consistency in feature order and presence
# # # Assuming X_train_cleaned is available from the notebook's training step
# # # If not, you would need to generate a dummy X_train_cleaned or save its columns
# # # in the saving step and load them here.
# # # try:
# # #     X_train_cleaned_cols = X_train_cleaned.columns.tolist()
# # #     print("X_train_cleaned columns obtained successfully.")
# # # except NameError:
# # #     print("Error: X_train_cleaned not found. Cannot get feature columns.")
# # #     print("Please ensure the model training step in the notebook was run.")
# # #     X_train_cleaned_cols = [] # Empty list if not available
# # # Define the path to the saved feature column names file

# # FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "X_train_cleaned_cols.pkl")
# # # Load the list of feature column names
# # try:
# #     X_train_cleaned_cols = joblib.load(FEATURE_COLS_PATH)
# #     print("X_train_cleaned column names loaded successfully.")
# # except FileNotFoundError:
# #     print(f"Error: Feature column names file not found at {FEATURE_COLS_PATH}.")
# #     print("Please ensure X_train_cleaned_cols.pkl is in the correct location.")
# #     X_train_cleaned_cols = [] # Empty list if loading fails
# # except Exception as e:
# #     print(f"Error loading feature column names: {e}")
# #     X_train_cleaned_cols = [] # Empty list if loading fails


# # @app.route('/predict_team', methods=['POST'])
# # def predict_team():
# #     """
# #     Flask endpoint to receive match data, predict fantasy points,
# #     select the optimal XI, and return the result.
# #     """
# #     print("Received request to /predict_team")
# #     if not request.json:
# #         print("Error: Request body is not JSON.")
# #         return jsonify({"error": "Request body must be JSON"}), 415 # Unsupported Media Type

# #     match_data = request.json

# #     # --- Step 1: Preprocess the match data and engineer features ---
# #     # Requires df_combined to be available globally or loaded here
# #     # Ensure df_combined is loaded and accessible in your Flask app's environment
# #     if 'df_combined' not in globals():
# #          print("Error: df_combined (historical data) is not loaded. Cannot preprocess.")
# #          return jsonify({"error": "Historical data not loaded. Please check server setup."}), 500

# #     if not X_train_cleaned_cols:
# #          print("Error: Training feature columns not available. Cannot preprocess.")
# #          return jsonify({"error": "Training feature columns not available. Please check server setup."}), 500


# #     X_squad, df_squad_info = preprocess_match_data(match_data, df_combined, player_roles_season_df, player_roles_global_df, feature_medians, X_train_cleaned_cols)

# #     if X_squad is None:
# #         return jsonify({"error": "Failed to preprocess match data"}), 500

# #     # --- Step 2: Predict fantasy points ---
# #     predictions = predict_fantasy_points(X_squad, rf_model)

# #     if predictions is None:
# #         return jsonify({"error": "Failed to predict fantasy points"}), 500

# #     # Add predictions to the squad info DataFrame
# #     df_squad_with_predictions = df_squad_info.copy()
# #     df_squad_with_predictions['predicted_fantasy_points'] = predictions

# #     # --- Step 3: Select the optimal XI using ILP ---
# #     df_selected_xi = select_optimal_xi(df_squad_with_predictions)

# #     if df_selected_xi is None:
# #         return jsonify({"error": "Failed to select optimal XI using ILP. Check constraints or squad data."}), 500

# #     # --- Step 4: Prepare the response ---
# #     # Convert the selected XI DataFrame to a JSON friendly format
# #     # You might want to include additional player info here if needed for the frontend

# #     # Replace NaN values with None in numeric columns before converting to dict
# #     # This is crucial for valid JSON output.
# #     # Identify numeric columns that might contain NaNs
# #     numeric_cols = df_selected_xi.select_dtypes(include=np.number).columns.tolist()
# #     for col in numeric_cols:
# #         if df_selected_xi[col].isnull().any():
# #             # Convert NaN to None (which jsonify handles as null)
# #             df_selected_xi[col] = df_selected_xi[col].replace({np.nan: None})

# #     # Add these print statements to inspect the DataFrame *after* NaN replacement
# #     print("\ndf_selected_xi after NaN replacement:")
# #     print(df_selected_xi.to_string()) # Print the full DataFrame to see all values

# #     # Check for NaNs specifically in the 'credit' column after replacement
# #     if 'credit' in df_selected_xi.columns:
# #         print(f"\nNaNs in 'credit' column after replacement: {df_selected_xi['credit'].isnull().sum()}")
# #         print("Credit column data types:", df_selected_xi['credit'].dtype)
# #         print("Sample credit values:", df_selected_xi['credit'].head())

# #     # Ensure all required columns exist in df_selected_xi before trying to access them
# #     required_output_cols = ['player_id', 'team', 'role', 'credit', 'predicted_fantasy_points']
# #     for col in required_output_cols:
# #         if col not in df_selected_xi.columns:
# #             print(f"Warning: Output column '{col}' not found in df_selected_xi. Adding with placeholder None.")
# #             df_selected_xi[col] = None # Add missing column with None


# #     selected_xi_list = df_selected_xi[required_output_cols].to_dict(orient='records')

# #     # Ensure None values in the list are handled correctly by jsonify (which they are by default)

# #     response_data = {
# #         "optimal_xi": selected_xi_list,
# #         # Ensure sum handles None values gracefully if needed, or calculate before conversion
# #         # Summing after replacing NaN with None might require adjusting sum logic if None is not treated as 0
# #         # Let's calculate total_predicted_points before replacing NaNs for safety
# #         "total_predicted_points": df_selected_xi['predicted_fantasy_points'].sum() if 'predicted_fantasy_points' in df_selected_xi.columns and not df_selected_xi['predicted_fantasy_points'].isnull().all() else 0
# #         # You could add more summary info here if desired
# #     }

# #     print("Successfully generated optimal XI.")
# #     return jsonify(response_data), 200 # OK

# # # --- Running the Flask app ---
# # # This part runs the Flask development server.
# # # In a production environment, you would use a production-ready WSGI server
# # # like Gunicorn or uWSGI.
# # if __name__ == '__main__':
# #     # Make sure the model and data files exist before starting the app
# #     if not os.path.exists(MODEL_PATH):
# #         print(f"Error: Model file not found at {MODEL_PATH}. Please train and save the model.")
# #     elif not os.path.exists(FEATURE_MEDIANS_PATH):
# #         print(f"Error: Feature medians file not found at {FEATURE_MEDIANS_PATH}. Please save the medians.")
# #     elif not os.path.exists(PLAYER_ROLES_SEASON_PATH) or not os.path.exists(PLAYER_ROLES_GLOBAL_PATH):
# #          print(f"Error: Player roles files not found in {MODEL_DIR}. Please process and save the roles.")
# #     elif 'df_combined' not in globals():
# #          print("Error: df_combined (historical data) is not loaded.")
# #          print("Please load your combined historical data DataFrame before running the app.")
# #     elif not X_train_cleaned_cols:
# #          print("Error: Training feature columns not available.")
# #          print("Please ensure the model training step in the notebook was run to define X_train_cleaned.")
# #     else:
# #         print("Starting Flask development server...")
# #         # You can change the port and host here if needed
# #         app.run(debug=True, host='0.0.0.0', port=5001)