from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import os

app = Flask(__name__)

# CONFIGURATION
# Update paths as needed, reduced data due to ram 
##FILE_2022 = r"C:\Personal_Projects\LEC_Website\Data\2022_LoL_esports_match_data_from_OraclesElixir.csv"
##FILE_2023 = r"C:\Personal_Projects\LEC_Website\Data\2023_LoL_esports_match_data_from_OraclesElixir.csv"
FILE_2024 = r"C:\Personal_Projects\LEC_Website\Data\2024_LoL_esports_match_data_from_OraclesElixir.csv"
FILE_2025 = r"C:\Personal_Projects\LEC_Website\Data\2025_LoL_esports_match_data_from_OraclesElixir.csv"

METRICS = ['golddiffat15', 'xpdiffat15', 'dpm', 'vspm', 'pool_depth']

# Weights for player rating (Recent years matter more)
YEAR_WEIGHTS = {2025: 1.0, 2024: 0.75, 2023: 0.5, 2022: 0.25}

LEAGUE_BONUS = {
    'LEC': 1.5, 'LCK': 2.0, 'LPL': 2.0, 'LCS': 1.2, 'MSI': 2.0, 'WLDs': 2.0, 
    'LCP': 1.0, 'LTA N': 1.0, 'LTA S': 1.0, 
    'LFL': 0.4, 'LVP SL': 0.4, 'LCKC': 0.4, 'NACL': 0.4, 'PRM': 0.4, 
    'NLC': 0.6, 'EM': 0.6, 'Default': 0.0
}

NAME_ALIASES = {'crownshot': 'crownie', 'thebausffs': 'baus', 'bausffs': 'baus', 'reckles': 'rekkles', '113': 'isma'}
LEGACY = {
    'razork': 1.1, 'upset': 1.1, 'vladi': 0.6, 'empyros': 0.6, 'lospa': 0.6,
    'humanoid': 0.1, 'skewmond': 0.9, 'caps': 0.9, 'hans sama': 0.8,
    'nemesis': 0.85, 'rekkles': 0.9, 'crownie': 0.7, 'baus': 0.7, 'velja': 0.5
}

ROSTERS = {
    "G2 Esports":     ['BrokenBlade', 'SkewMond', 'Caps', 'Hans Sama', 'Labrov'],
    "Fnatic":         ['Empyros', 'Razork', 'Vladi', 'Upset', 'Lospa'],
    "Team Vitality":  ['Naak Nako', 'Lyncas', 'Humanoid', 'Carzzy', 'Fleshy'],
    "Karmine Corp":   ['Canna', 'Yike', 'kyeahoo', 'Caliste', 'Busio'],
    "Team Heretics":  ['Tracyn', 'Sheo', 'Serin', 'Ice', 'Stend'],
    "GIANTX":         ['Lot', 'ISMA', 'Jackies', 'Noah', 'Jun'],
    "SK Gaming":      ['Wunder', 'Skeanz', 'LIDER', 'Jopa', 'Mikyx'],
    "Team BDS":       ['Rooster', 'Boukada', 'nuc', 'Paduck', 'Trymbi'],
    "Movistar KOI":   ['Myrwn', 'Elyoya', 'Jojopyun', 'Supa', 'Alvaro'],
    "Natus Vincere":  ['Maynter', 'Rhilech', 'Poby', 'Hans SamD', 'Parus'],
    "KC Blue":        ['Tao', 'Yukino', 'Kamiloo', 'Hazel', 'Prime'],
    "Los Ratones":    ['Baus', 'Velja', 'Nemesis', 'Crownie', 'Rekkles']
}

MODEL = None
DB = {}        
RAW_DB = {}    
CHAMP_DB = {}  
MATCHUP_DB = {} 

def clean_name(name):
    n = str(name).lower().strip()
    return NAME_ALIASES.get(n, n)

def get_league_bonus(league):
    return LEAGUE_BONUS.get(league, LEAGUE_BONUS['Default'])

def load_process(path, year, raw=False):
    """Loads a CSV, adds year info, and processes metrics."""
    if not os.path.exists(path): 
        print(f"⚠️ Warning: File not found {path}")
        return None
        
    df = pd.read_csv(path, low_memory=False)
    df = df[df['position'] != 'team'].copy()
    df['clean_name'] = df['playername'].apply(clean_name)
    df['year'] = year # Tag the year
    
    # Pool Depth Calculation
    pool_sizes = df.groupby(['clean_name', 'league'])['champion'].nunique().reset_index()
    pool_sizes.rename(columns={'champion': 'pool_depth'}, inplace=True)
    df = df.merge(pool_sizes, on=['clean_name', 'league'], how='left')

    if raw: return df
    
    # Calculate Standardized Scores (Z-Scores) per League
    for m in METRICS:
        league_stats = df.groupby('league')[m].transform(lambda x: (x - x.mean()) / x.std())
        weights = df['league'].apply(get_league_bonus)
        df[f'score_{m}'] = league_stats.fillna(0) + weights
        
    return df

def init_system():
    global MODEL, DB, RAW_DB, CHAMP_DB, MATCHUP_DB
    print("⏳ Initializing AI (2022-2025 Data Engine)...")
    
    # Load all datasets
    # We load them twice: once processed (with scores) for the AI, once raw for stats display
    ##d22 = load_process(FILE_2022, 2022)
    ##d23 = load_process(FILE_2023, 2023)
    d24 = load_process(FILE_2024, 2024)
    d25 = load_process(FILE_2025, 2025)
    
    ##raw_22 = load_process(FILE_2022, 2022, raw=True)
    ##raw_23 = load_process(FILE_2023, 2023, raw=True)
    raw_24 = load_process(FILE_2024, 2024, raw=True)
    raw_25 = load_process(FILE_2025, 2025, raw=True)

    # Combine into single lists for easier processing
    all_processed = [d for d in [ d24, d25] if d is not None]
    all_raw = [d for d in [raw_24, raw_25] if d is not None]

    if not all_processed:
        print("❌ CRITICAL ERROR: No data files loaded!")
        return

    # 1. Build Match Model (Player Vectors) using Weighted Average of Years
    # Combine all processed frames
    big_df = pd.concat(all_processed)
    
    # Group by Player and Year to get their average score for that year
    player_year_stats = big_df.groupby(['clean_name', 'year'])[[f'score_{m}' for m in METRICS]].mean().reset_index()
    
    for player, p_data in player_year_stats.groupby('clean_name'):
        vec = []
        for m in METRICS:
            col = f'score_{m}'
            # Weighted Average Calculation: Sum(Score * YearWeight) / Sum(YearWeights)
            total_score = 0
            total_weight = 0
            
            for _, row in p_data.iterrows():
                y = row['year']
                w = YEAR_WEIGHTS.get(y, 0.2) # Default weight if year unknown
                total_score += row[col] * w
                total_weight += w
            
            final_val = total_score / total_weight if total_weight > 0 else 0
            
            # Legacy Bonus
            if player in LEGACY: 
                final_val += LEGACY[player]
                
            vec.append(final_val)
        DB[player] = vec

    # 2. Build Draft & Matchup Databases (Using Raw Data)
    print("📚 Building Historical Knowledge Base...")
    full_history = pd.concat(all_raw)
    
    # Player Stats (Averaged across all years)
    raw_stats = full_history.groupby('clean_name')[METRICS].mean()
    RAW_DB = raw_stats.to_dict('index')

    # Champion Stats per Player
    for p_name, p_data in full_history.groupby('clean_name'):
        CHAMP_DB[p_name] = {}
        for champ, c_data in p_data.groupby('champion'):
            games = len(c_data)
            wins = c_data['result'].sum()
            kda = round((c_data['kills'].sum() + c_data['assists'].sum()) / max(1, c_data['deaths'].sum()), 2)
            wr = round((wins / games) * 100, 1)
            CHAMP_DB[p_name][champ] = {'games': games, 'win_rate': wr, 'kda': kda}

    # --- CHAMPION MATCHUP DATABASE ---
    # Separate Blue and Red sides
    blue = full_history[full_history['side'] == 'Blue'][['gameid', 'position', 'champion', 'result']]
    red = full_history[full_history['side'] == 'Red'][['gameid', 'position', 'champion', 'result']]
    
    matchups = pd.merge(blue, red, on=['gameid', 'position'], suffixes=('_b', '_r'))
    
    df1 = matchups[['champion_b', 'champion_r', 'result_b']].rename(
        columns={'champion_b': 'me', 'champion_r': 'enemy', 'result_b': 'win'}
    )
    df2 = matchups[['champion_r', 'champion_b', 'result_r']].rename(
        columns={'champion_r': 'me', 'champion_b': 'enemy', 'result_r': 'win'}
    )
    
    all_matchups = pd.concat([df1, df2])
    
    # Calculate global winrates for matchups
    stats = all_matchups.groupby(['me', 'enemy'])['win'].agg(['sum', 'count']).reset_index()
    
    for _, row in stats.iterrows():
        me, enemy, wins, games = row['me'], row['enemy'], row['sum'], row['count']
        if me not in MATCHUP_DB: MATCHUP_DB[me] = {}
        MATCHUP_DB[me][enemy] = {
            'wr': round((wins / games) * 100, 1),
            'games': int(games)
        }

    # 3. Train Model
    # Use big_df (which contains processed scores for 2022-2025)
    training_rows = []
    
    # Filter for complete games (5v5)
    valid_games = big_df['gameid'].value_counts()[lambda x: x == 10].index
    
    print(f"🤖 Training Model on {len(valid_games)} historical matches...")
    
    for _, match in big_df[big_df['gameid'].isin(valid_games)].groupby('gameid'):
        blue = match[match['side'] == 'Blue']
        red = match[match['side'] == 'Red']
        if len(blue) == 5 and len(red) == 5:
            feats = [blue[f'score_{m}'].sum() - red[f'score_{m}'].sum() for m in METRICS]
            training_rows.append(feats + [1 if blue.iloc[0]['result'] == 1 else 0])
    
    if training_rows:
        train_df = pd.DataFrame(training_rows, columns=METRICS + ['Win'])
        MODEL = LogisticRegression(C=1.0) 
        MODEL.fit(train_df[METRICS], train_df['Win'])
        print("✅ System Ready!")
    else:
        print("❌ Error: No valid training data found.")

def get_team_vector(team):
    vec = np.zeros(5)
    for p in ROSTERS[team]:
        n = clean_name(p)
        if n in DB: vec += DB[n]
    return vec

init_system()

# ROUTES
@app.route('/')
def home(): return render_template('index.html', teams=ROSTERS)

@app.route('/draft')
def draft_page():
    all_champs = set()
    for p_data in CHAMP_DB.values(): all_champs.update(p_data.keys())
    return render_template('draft.html', teams=ROSTERS, champions=sorted(list(all_champs)))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    blue, red = data['blue'], data['red']
    v1, v2 = get_team_vector(blue), get_team_vector(red)
    if MODEL:
        raw_prob = MODEL.predict_proba([(v1 - v2) / 5.0])[0][1]
        return jsonify({'winner': blue if raw_prob > 0.5 else red, 'blue_win_chance': round(raw_prob * 100, 1)})
    else:
        return jsonify({'error': 'Model not initialized'})

@app.route('/compare_players', methods=['POST'])
def compare_players():
    data = request.json
    p1, p2 = clean_name(data['player1']), clean_name(data['player2'])
    def get_stats(p): return {k: round(v, 1) for k, v in RAW_DB.get(p, {m: 0 for m in METRICS}).items()}
    return jsonify({'p1': get_stats(p1), 'p2': get_stats(p2)})

@app.route('/analyze_full_draft', methods=['POST'])
def analyze_full_draft():
    draft = request.json['draft']
    results = []
    
    for row in draft:
        role = row['role']
        p1, c1 = clean_name(row['blue_player']), row['blue_champ']
        p2, c2 = clean_name(row['red_player']), row['red_champ']
        
        # Player specific stats
        s1 = CHAMP_DB.get(p1, {}).get(c1, {'games': 0, 'win_rate': 0, 'kda': 0})
        s2 = CHAMP_DB.get(p2, {}).get(c2, {'games': 0, 'win_rate': 0, 'kda': 0})
        
        # Global Matchup stats
        matchup_data = MATCHUP_DB.get(c1, {}).get(c2, {'wr': 50.0, 'games': 0})
        
        results.append({
            'role': role,
            'blue': {'player': row['blue_player'], 'champ': c1, 'stats': s1},
            'red': {'player': row['red_player'], 'champ': c2, 'stats': s2},
            'matchup': matchup_data 
        })
        
    return jsonify({'matchups': results})

if __name__ == '__main__':
    app.run(debug=True)