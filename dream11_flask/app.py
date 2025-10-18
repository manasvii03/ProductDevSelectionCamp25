import json
import pandas as pd
import numpy as np
import joblib
import pulp
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

MODEL_DIR = "model" 

# Define paths to the saved files
MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
FEATURE_MEDIANS_PATH = os.path.join(MODEL_DIR, "feature_medians.pkl")
PLAYER_ROLES_SEASON_PATH = os.path.join(MODEL_DIR, "player_roles_by_season.csv")
PLAYER_ROLES_GLOBAL_PATH = os.path.join(MODEL_DIR, "player_roles_global.csv")


# Load the trained Random Forest model
try:
    rf_model = joblib.load(MODEL_PATH)
    print("Random Forest model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please ensure the model training and saving steps were completed.")
    rf_model = None

# Load the feature medians
try:
    feature_medians = joblib.load(FEATURE_MEDIANS_PATH)
    print("Feature medians loaded successfully.")
except FileNotFoundError:
    print(f"Error: Feature medians file not found at {FEATURE_MEDIANS_PATH}. Please ensure the feature engineering and saving steps were completed.")
    feature_medians = None 

# Load the player roles dataframes
try:
    player_roles_season_df = pd.read_csv(PLAYER_ROLES_SEASON_PATH)
    player_roles_global_df = pd.read_csv(PLAYER_ROLES_GLOBAL_PATH)
    player_roles_season_df.columns = player_roles_season_df.columns.str.strip().str.lower()
    player_roles_global_df.columns = player_roles_global_df.columns.str.strip().str.lower()
    print("Player roles dataframes loaded successfully.")
except FileNotFoundError:
    print(f"Error: Player roles files not found in {MODEL_DIR}. Please ensure the player roles processing and saving steps were completed.")
    player_roles_season_df = pd.DataFrame()
    player_roles_global_df = pd.DataFrame() 


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
        df_squad_with_roles = pd.merge(
            df_squad_real[['player_id', 'player_name', 'team', 'match_date', 'season']],
            player_roles_season_df[['player_id', 'season', 'role']].rename(columns={'role': 'role_seasonal'}), 
            on=["player_id", "season"],
            how="left"
        )
        df_squad_with_roles = pd.merge(
            df_squad_with_roles,
            player_roles_global_df[['player_id', 'role']].rename(columns={'role': 'role_global'}),
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

        # Select necessary historical features
        historical_features_for_merge = historical_data_combined[[
            'player_id', 'match_date', 'credit', 'recency', 'previous_matches',
            'rolling_avg_3y', 'rolling_std_3y', 'team_rolling_avg_fp',
            'opponent_rolling_avg_fp'
        ]].copy()

        # Ensure match_date is datetime in both DataFrames
        df_squad_with_roles['match_date'] = pd.to_datetime(df_squad_with_roles['match_date'])
        historical_features_for_merge['match_date'] = pd.to_datetime(historical_features_for_merge['match_date'])

        df_squad_features = pd.merge(
            df_squad_with_roles,
            historical_features_for_merge,
            on=['player_id', 'match_date'],
            how='left',
            suffixes=('', '_hist') 
        )

        # Handle potential duplicate rows resulting from the merge
        df_squad_features = df_squad_features.drop_duplicates(subset=['player_id', 'match_date']).reset_index(drop=True)


        # Prepare features for prediction
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
        return X_squad, df_squad_features 

    except Exception as e:
        print(f"Error during data preprocessing: {e}")
        raise e
        # return None, None 


def predict_fantasy_points(X_squad, model):

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



def select_optimal_xi(df_squad_with_predictions):
    print("Starting optimal XI selection using ILP...")
    if df_squad_with_predictions is None or df_squad_with_predictions.empty:
        print("Error: Squad DataFrame with predictions is empty or None. Cannot select XI.")
        return None

    try:
        prob = pulp.LpProblem("Fantasy Team Selection", pulp.LpMaximize)
        player_vars = pulp.LpVariable.dicts("Select", df_squad_with_predictions['player_id'], 0, 1, pulp.LpInteger)
        df_squad_with_predictions['predicted_fantasy_points'] = pd.to_numeric(df_squad_with_predictions['predicted_fantasy_points'], errors='coerce').fillna(0)

        prob += pulp.lpSum([df_squad_with_predictions.loc[df_squad_with_predictions['player_id'] == player_id, 'predicted_fantasy_points'].iloc[0] * player_vars[player_id] for player_id in df_squad_with_predictions['player_id']]), "Total Predicted Fantasy Points"
        # Constraint 1: Total Players = 11
        prob += pulp.lpSum([player_vars[player_id] for player_id in df_squad_with_predictions['player_id']]) == 11, "Total Players = 11"

        # Constraint 2: Role Bounds
        if 'role' not in df_squad_with_predictions.columns:
             print("Error: 'role' column not found in squad data for ILP constraints.")
             return None 

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

DF_COMBINED_PATH = os.path.join(MODEL_DIR, "df_combined.csv")

try:
    df_combined = pd.read_csv(DF_COMBINED_PATH)
    df_combined['match_date'] = pd.to_datetime(df_combined['match_date'])
    print("Historical combined data loaded successfully.")
except FileNotFoundError:
    print(f"Error: Combined historical data file not found at {DF_COMBINED_PATH}.")
    print("Please ensure df_combined.csv is in the correct location.")
    df_combined = pd.DataFrame() 
except Exception as e:
    print(f"Error loading combined historical data: {e}")
    df_combined = pd.DataFrame() 


FEATURE_COLS_PATH = os.path.join(MODEL_DIR, "X_train_cleaned_cols.pkl")

# Load the list of feature column names
try:
    X_train_cleaned_cols = joblib.load(FEATURE_COLS_PATH)
    print("X_train_cleaned column names loaded successfully.")
except FileNotFoundError:
    print(f"Error: Feature column names file not found at {FEATURE_COLS_PATH}.")
    print("Please ensure X_train_cleaned_cols.pkl is in the correct location.")
    X_train_cleaned_cols = []
except Exception as e:
    print(f"Error loading feature column names: {e}")
    X_train_cleaned_cols = [] 

FEATURE_MEDIANS_PATH = os.path.join(MODEL_DIR, "feature_medians.pkl")
# Load the feature medians
try:
    feature_medians = joblib.load(FEATURE_MEDIANS_PATH)
    print("Feature medians loaded successfully.")
except FileNotFoundError:
    print(f"Error: Feature medians file not found at {FEATURE_MEDIANS_PATH}. Please ensure the feature engineering and saving steps were completed.")
    feature_medians = None 
except Exception as e:
    print(f"Error loading feature medians: {e}")
    feature_medians = None 

PLAYER_ROLES_SEASON_PATH = os.path.join(MODEL_DIR, "player_roles_by_season.csv")
PLAYER_ROLES_GLOBAL_PATH = os.path.join(MODEL_DIR, "player_roles_global.csv")
try:
    player_roles_season_df = pd.read_csv(PLAYER_ROLES_SEASON_PATH)
    player_roles_global_df = pd.read_csv(PLAYER_ROLES_GLOBAL_PATH)
    player_roles_season_df.columns = player_roles_season_df.columns.str.strip().str.lower()
    player_roles_global_df.columns = player_roles_global_df.columns.str.strip().str.lower()
    print("Player roles dataframes loaded successfully.")
except FileNotFoundError:
    print(f"Error: Player roles files not found in {MODEL_DIR}. Please ensure the player roles processing and saving steps were completed.")
    player_roles_season_df = pd.DataFrame() 
    player_roles_global_df = pd.DataFrame() 
except Exception as e:
    print(f"Error loading player roles dataframes: {e}")
    player_roles_season_df = pd.DataFrame() 
    player_roles_global_df = pd.DataFrame() 


MODEL_PATH = os.path.join(MODEL_DIR, "rf_model.pkl")
# Load the trained Random Forest model
try:
    rf_model = joblib.load(MODEL_PATH)
    print("Random Forest model loaded successfully.")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please ensure the model training and saving steps were completed.")
    rf_model = None 
except Exception as e:
    print(f"Error loading Random Forest model: {e}")
    rf_model = None 


@app.route('/predict_team', methods=['POST'])
def predict_team():
    """
    Flask endpoint to receive match data, predict fantasy points,
    select the optimal XI, and return the result.
    """
    print("Received request to /predict_team")
    if not request.json:
        print("Error: Request body is not JSON.")
        return jsonify({"error": "Request body must be JSON"}), 415

    match_data = request.json
    if not isinstance(match_data, dict):
        print("Error: Incoming data is not a valid JSON object (dictionary).")
        return jsonify({"error": "Invalid JSON data received. Expected a JSON object."}), 400 
    # --- Step 1: Preprocess the match data and engineer features ---
    if df_combined.empty: 
         print("Error: df_combined (historical data) is not loaded. Cannot preprocess.")
         return jsonify({"error": "Historical data not loaded. Please check server setup."}), 500

    if not X_train_cleaned_cols: 
         print("Error: Training feature columns not available. Cannot preprocess.")
         return jsonify({"error": "Training feature columns not available. Please check server setup."}), 500

    if feature_medians is None: 
         print("Error: Feature medians not loaded. Cannot preprocess.")
         return jsonify({"error": "Feature medians not loaded. Please check server setup."}), 500

    if player_roles_season_df.empty or player_roles_global_df.empty: 
         print("Error: Player roles data not loaded. Cannot preprocess.")
         return jsonify({"error": "Player roles data not loaded. Please check server setup."}), 500

    if rf_model is None:
         print("Error: Model not loaded. Cannot predict.")
         return jsonify({"error": "Model not loaded. Please check server setup."}), 500


    X_squad, df_squad_info = preprocess_match_data(match_data, df_combined, player_roles_season_df, player_roles_global_df, feature_medians, X_train_cleaned_cols)

    if X_squad is None:
        return jsonify({"error": "Failed to preprocess match data"}), 500

    # --- Step 2: Predict fantasy points ---
    predictions = predict_fantasy_points(X_squad, rf_model)

    if predictions is None:
        return jsonify({"error": "Failed to predict fantasy points"}), 500

    if len(predictions) == len(df_squad_info):
        df_squad_with_predictions = df_squad_info.copy()
        df_squad_with_predictions['predicted_fantasy_points'] = predictions
    else:
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
    }

    print("Successfully generated optimal XI.")
    return jsonify(response_data), 200 

# --- Running the Flask app ---
if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}. Please train and save the model.")
    elif not os.path.exists(FEATURE_MEDIANS_PATH):
        print(f"Error: Feature medians file not found at {FEATURE_MEDIANS_PATH}. Please save the medians.")
    elif not os.path.exists(PLAYER_ROLES_SEASON_PATH) or not os.path.exists(PLAYER_ROLES_GLOBAL_PATH):
         print(f"Error: Player roles files not found in {MODEL_DIR}. Please process and save the roles.")
    elif df_combined.empty: 
         print("Error: df_combined (historical data) is empty after loading attempt.")
         print("Please check the loading path and file content.")
    elif not X_train_cleaned_cols: 
         print("Error: Training feature columns are empty after loading attempt.")
         print("Please check the loading path and file content for X_train_cleaned_cols.pkl.")
    else:
        print("Starting Flask development server...")
        app.run(debug=True, host='0.0.0.0', port=5001)
