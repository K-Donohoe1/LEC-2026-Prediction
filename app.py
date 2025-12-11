import os
import gc  # Garbage Collection to free memory
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILES = {
    2022: os.path.join(BASE_DIR, 'Data', '2022_LoL_esports_match_data_from_OraclesElixir.csv'),
    2023: os.path.join(BASE_DIR, 'Data', '2023_LoL_esports_match_data_from_OraclesElixir.csv'),
    2024: os.path.join(BASE_DIR, 'Data', '2024_LoL_esports_match_data_from_OraclesElixir.csv'),
    2025: os.path.join(BASE_DIR, 'Data', '2025_LoL_esports_match_data_from_OraclesElixir.csv'),
}

# Only load columns we actually need (Saves 60% memory)
USE_COLS = [
    'gameid', 'playername', 'position', 'teamname', 'champion', 
    'result', 'side', 'league', 'kills', 'deaths', 'assists',
    'golddiffat15', 'xpdiffat15', 'dpm', 'vspm'
]

# Force lower precision to save memory (float32 vs float64)
DTYPES = {
    'result': 'int8',
    'kills': 'int16', 
    'deaths': 'int16', 
    'assists': 'int16',
    'golddiffat15': 'float32',
    'xpdiffat15': 'float32',
    'dpm': 'float32',
    'vspm': 'float32'
}

METRICS = ['golddiffat15', 'xpdiffat15', 'dpm', 'vspm', 'pool_depth']
YEAR_WEIGHTS = {2025: 1.0, 2024: 0.75, 2023: 0.5, 2022: 0.25}

LEAGUE_BONUS = {
    'LEC': 1.5, 'LCK': 2.0, 'LPL': 2.0, 'LCS': 1.2, 'MSI': 2.0, 'WLDs': 2.0, 
    'LCP': 1.0, 'LTA N': 1.0, 'LTA S': 1.0, 'LFL': 0.4, 'LVP SL': 0.4, 
    'LCKC': 0.4, 'NACL': 0.4, 'PRM': 0.4, 'NLC': 0.6, 'EM': 0.6, 'Default': 0.0
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

def init_system():
    global MODEL, DB, RAW_DB, CHAMP_DB, MATCHUP_DB
    print("⏳ Initializing AI (Memory Optimized Mode)...")
    
    # Temp storage for yearly player stats
    player_yearly_scores = {} # {player: {2022: [scores], 2023: [scores]}}
    training_rows = []
    
    # PROCESS ONE YEAR AT A TIME to save RAM
    for year, filepath in FILES.items():
        if not os.path.exists(filepath):
            print(f"⚠️ Skipping missing file: {filepath}")
            continue
            
        print(f"📂 Processing {year}...")
        
        # 1. Load optimized CSV
        df = pd.read_csv(filepath, usecols=USE_COLS, dtype=DTYPES, low_memory=True)
        df = df[df['position'] != 'team'].copy()
        df['clean_name'] = df['playername'].apply(clean_name)
        
        # 2. Calculate Metrics (Z-Scores)
        # Pool Depth (Count unique champs per player in this year)
        pool = df.groupby(['clean_name', 'league'])['champion'].nunique().reset_index(name='pool_depth')
        df = df.merge(pool, on=['clean_name', 'league'], how='left')
        
        for m in METRICS:
            # Handle standard metrics
            if m in df.columns:
                league_stats = df.groupby('league')[m].transform(lambda x: (x - x.mean()) / x.std())
                weights = df['league'].apply(get_league_bonus)
                df[f'score_{m}'] = league_stats.fillna(0) + weights
            # Handle pool_depth specially (it was just added)
            elif m == 'pool_depth':
                league_stats = df.groupby('league')[m].transform(lambda x: (x - x.mean()) / x.std())
                weights = df['league'].apply(get_league_bonus)
                df[f'score_{m}'] = league_stats.fillna(0) + weights

        # 3. Store Player Stats for this Year (for final aggregation)
        year_stats = df.groupby('clean_name')[[f'score_{m}' for m in METRICS]].mean()
        for player, row in year_stats.iterrows():
            if player not in player_yearly_scores: player_yearly_scores[player] = {}
            player_yearly_scores[player][year] = row.values

        # 4. Update RAW_DB and CHAMP_DB (Incremental update)
        # We process raw stats here and add them to the global dicts
        
        # Raw Stats (accumulate totals)
        # Note: For simplicity in memory-mode, we will just take the latest year's raw stats for display
        # or you can implement a complex running average. For now, we overwrite to keep it fast.
        current_raw = df.groupby('clean_name')[['golddiffat15', 'xpdiffat15', 'dpm', 'vspm']].mean().to_dict('index')
        RAW_DB.update(current_raw) # This will mostly keep 2025 stats for display, which is fine

        # Champion Stats
        for p_name, p_data in df.groupby('clean_name'):
            if p_name not in CHAMP_DB: CHAMP_DB[p_name] = {}
            
            # This is slightly expensive, so we do a quick aggregation
            champ_agg = p_data.groupby('champion').agg({
                'result': ['count', 'sum'],
                'kills': 'sum', 'deaths': 'sum', 'assists': 'sum'
            })
            
            for champ, agg in champ_agg.iterrows():
                games = agg['result']['count']
                wins = agg['result']['sum']
                k = agg['kills']['sum']
                d = agg['deaths']['sum']
                a = agg['assists']['sum']
                
                # Check if we already have data for this champ (from previous years)
                if champ not in CHAMP_DB[p_name]:
                    CHAMP_DB[p_name][champ] = {'games': 0, 'wins': 0, 'k': 0, 'd': 0, 'a': 0}
                
                c_entry = CHAMP_DB[p_name][champ]
                c_entry['games'] += games
                c_entry['wins'] += wins
                c_entry['k'] += k
                c_entry['d'] += d
                c_entry['a'] += a

        # 5. Build Matchups (Blue vs Red)
        valid_games = df['gameid'].value_counts()
        complete_games = valid_games[valid_games == 10].index
        
        # Filter for only complete games to save iteration time
        game_df = df[df['gameid'].isin(complete_games)]
        
        # Extract Matchups
        blue = game_df[game_df['side'] == 'Blue'][['gameid', 'position', 'champion', 'result']]
        red = game_df[game_df['side'] == 'Red'][['gameid', 'position', 'champion', 'result']]
        merged = pd.merge(blue, red, on=['gameid', 'position'], suffixes=('_b', '_r'))
        
        # Update Matchup DB
        for _, row in merged.iterrows():
            # Blue POV
            b_champ, r_champ = row['champion_b'], row['champion_r']
            b_win = row['result_b']
            
            if b_champ not in MATCHUP_DB: MATCHUP_DB[b_champ] = {}
            if r_champ not in MATCHUP_DB[b_champ]: MATCHUP_DB[b_champ][r_champ] = {'wins': 0, 'games': 0}
            MATCHUP_DB[b_champ][r_champ]['games'] += 1
            MATCHUP_DB[b_champ][r_champ]['wins'] += b_win
            
            # Red POV
            if r_champ not in MATCHUP_DB: MATCHUP_DB[r_champ] = {}
            if b_champ not in MATCHUP_DB[r_champ]: MATCHUP_DB[r_champ][b_champ] = {'wins': 0, 'games': 0}
            MATCHUP_DB[r_champ][b_champ]['games'] += 1
            MATCHUP_DB[r_champ][b_champ]['wins'] += (1 - b_win)

        # 6. Prepare Training Data
        # Group by game to get team vectors
        for gid, match in game_df.groupby('gameid'):
            blue_team = match[match['side'] == 'Blue']
            red_team = match[match['side'] == 'Red']
            
            if len(blue_team) == 5 and len(red_team) == 5:
                # Sum the scores we calculated earlier
                b_score = blue_team[[f'score_{m}' for m in METRICS]].sum().values
                r_score = red_team[[f'score_{m}' for m in METRICS]].sum().values
                
                diff = b_score - r_score
                label = 1 if blue_team.iloc[0]['result'] == 1 else 0
                training_rows.append(np.append(diff, label))

        # FREE MEMORY
        print(f"🗑️ Cleaning up {year}...")
        del df, pool, game_df, blue, red, merged
        gc.collect() # Force Python to release RAM

    # --- FINAL AGGREGATION ---
    print("⚙️ Finalizing Models...")
    
    # 1. Finalize Matchup DB (Calc Percentages)
    for me in MATCHUP_DB:
        for enemy in MATCHUP_DB[me]:
            d = MATCHUP_DB[me][enemy]
            d['wr'] = round((d['wins'] / d['games']) * 100, 1)

    # 2. Finalize Champ DB (Calc KDA/WR)
    for p in CHAMP_DB:
        for c in CHAMP_DB[p]:
            d = CHAMP_DB[p][c]
            d['win_rate'] = round((d['wins'] / d['games']) * 100, 1)
            d['kda'] = round((d['k'] + d['a']) / max(1, d['d']), 2)

    # 3. Finalize Player Vectors (Weighted Average)
    for player, years_data in player_yearly_scores.items():
        total_score = np.zeros(len(METRICS))
        total_weight = 0
        for y, scores in years_data.items():
            w = YEAR_WEIGHTS.get(y, 0.2)
            total_score += scores * w
            total_weight += w
        
        final_vec = total_score / total_weight if total_weight > 0 else total_score
        
        # Legacy Bonus
        if player in LEGACY: final_vec += LEGACY[player]
        DB[player] = final_vec

    # 4. Train Model
    if training_rows:
        print(f"🤖 Training on {len(training_rows)} matches...")
        train_data = np.array(training_rows)
        X = train_data[:, :-1] # Features
        y = train_data[:, -1]  # Target
        
        MODEL = LogisticRegression(C=1.0, solver='liblinear') # Liblinear is memory efficient
        MODEL.fit(X, y)
        print("✅ AI Online!")
    else:
        print("❌ No training data found.")

def get_team_vector(team):
    vec = np.zeros(len(METRICS))
    for p in ROSTERS[team]:
        n = clean_name(p)
        if n in DB: vec += DB[n]
    return vec

# Initialize on startup
init_system()

# ==========================================
# ROUTES
# ==========================================
@app.route('/')
def home(): return render_template('index.html', teams=ROSTERS)

@app.route('/draft')
def draft_page():
    all_champs = set()
    for p_data in CHAMP_DB.values(): all_champs.update(p_data.keys())
    return render_template('draft.html', teams=ROSTERS, champions=sorted(list(all_champs)))

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL: return jsonify({'error': 'Model loading...'})
    data = request.json
    v1 = get_team_vector(data['blue'])
    v2 = get_team_vector(data['red'])
    diff = (v1 - v2) / 5.0
    prob = MODEL.predict_proba([diff])[0][1]
    return jsonify({'winner': data['blue'] if prob > 0.5 else data['red'], 'blue_win_chance': round(prob * 100, 1)})

@app.route('/analyze_full_draft', methods=['POST'])
def analyze_full_draft():
    draft = request.json.get('draft', [])
    results = []
    for row in draft:
        p1, c1 = clean_name(row['blue_player']), row['blue_champ']
        p2, c2 = clean_name(row['red_player']), row['red_champ']
        
        s1 = CHAMP_DB.get(p1, {}).get(c1, {'games': 0, 'win_rate': 0, 'kda': 0})
        s2 = CHAMP_DB.get(p2, {}).get(c2, {'games': 0, 'win_rate': 0, 'kda': 0})
        
        mu = MATCHUP_DB.get(c1, {}).get(c2, {'wr': 50.0, 'games': 0})
        
        results.append({
            'role': row['role'],
            'blue': {'player': row['blue_player'], 'champ': c1, 'stats': s1},
            'red': {'player': row['red_player'], 'champ': c2, 'stats': s2},
            'matchup': mu 
        })
    return jsonify({'matchups': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)