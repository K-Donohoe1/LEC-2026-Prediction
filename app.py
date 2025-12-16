import os
import gc
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
    2024: os.path.join(BASE_DIR, 'Data', '2024_LoL_esports_match_data_from_OraclesElixir.csv'),
    2025: os.path.join(BASE_DIR, 'Data', '2025_LoL_esports_match_data_from_OraclesElixir.csv'),
}

USE_COLS = ['gameid', 'playername', 'position', 'teamname', 'champion', 'result', 'side', 'league', 'kills', 'deaths', 'assists', 'golddiffat15', 'xpdiffat15', 'dpm', 'vspm']
DTYPES = {'result': 'int8', 'kills': 'int16', 'deaths': 'int16', 'assists': 'int16', 'golddiffat15': 'float32', 'xpdiffat15': 'float32', 'dpm': 'float32', 'vspm': 'float32'}
METRICS = ['golddiffat15', 'xpdiffat15', 'dpm', 'vspm', 'pool_depth']
YEAR_WEIGHTS = {2025: 1.0, 2024: 0.75}

# MEMORY FIX: Only keep Tier 1 leagues for general training.
# (Rookie stats will still be loaded via the 'target_players' check below)
TRAINING_LEAGUES = ['LEC', 'LCS', 'LCK', 'LPL', 'MSI', 'WLDs']

LEAGUE_BONUS = {'LEC': 1.5, 'LCK': 2.0, 'LPL': 2.0, 'LCS': 1.2, 'MSI': 2.0, 'WLDs': 2.5, 'LFL': 0.5, 'Default': 0.4}
NAME_ALIASES = {'crownshot': 'crownie', 'thebausffs': 'baus', 'bausffs': 'baus', 'reckles': 'rekkles', '113': 'isma'}
LEGACY = {'razork': 1.1, 'upset': 1.1, 'caps': 0.9, 'hans sama': 0.8, 'humanoid': 0.1}

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

def init_system():
    global MODEL, DB, RAW_DB, CHAMP_DB, MATCHUP_DB
    
    print("⏳ Starting System (Optimized)...")
    
    # 1. Identify Target Players (So we grab their ERL games)
    target_players = set()
    for roster in ROSTERS.values():
        for p in roster:
            target_players.add(clean_name(p))
            
    print(f"📋 Tracking {len(target_players)} specific players + Major Regions")

    player_yearly_scores = {}
    training_rows = []
    
    for year, filepath in FILES.items():
        if not os.path.exists(filepath): continue
        print(f"📂 Loading {year}...")
        
        try:
            # --- SMART CHUNKING ---
            # Read file in small pieces
            chunk_iterator = pd.read_csv(filepath, usecols=USE_COLS, dtype=DTYPES, chunksize=10000)
            filtered_chunks = []
            
            for chunk in chunk_iterator:
                # OPTIMIZATION: Drop 'team' rows immediately to save 17% RAM
                chunk = chunk[chunk['position'] != 'team'].copy()
                
                # Normalize names
                chunk['clean_name'] = chunk['playername'].apply(lambda x: str(x).lower().strip())
                chunk['clean_name'] = chunk['clean_name'].map(lambda x: NAME_ALIASES.get(x, x))
                
                # FILTER: Keep if (Major League) OR (Is one of our Rookies)
                mask = (chunk['league'].isin(TRAINING_LEAGUES)) | (chunk['clean_name'].isin(target_players))
                filtered_chunks.append(chunk[mask])
            
            # Combine
            if filtered_chunks:
                df = pd.concat(filtered_chunks)
            else:
                continue

            del filtered_chunks, chunk_iterator
            gc.collect() 
            
            # --- PROCESSING ---
            
            # Metrics
            pool = df.groupby(['clean_name', 'league'])['champion'].nunique().reset_index(name='pool_depth')
            df = df.merge(pool, on=['clean_name', 'league'], how='left')
            
            league_weights = df['league'].map(LEAGUE_BONUS).fillna(LEAGUE_BONUS['Default'])
            for m in METRICS:
                if m in df.columns or m == 'pool_depth':
                    league_stats = df.groupby('league')[m].transform(lambda x: (x - x.mean()) / x.std())
                    df[f'score_{m}'] = league_stats.fillna(0) + league_weights

            # Store Scores (AI Model)
            year_stats = df.groupby('clean_name')[[f'score_{m}' for m in METRICS]].mean()
            for player, row in year_stats.iterrows():
                if player not in player_yearly_scores: player_yearly_scores[player] = {}
                player_yearly_scores[player][year] = row.values

            # Store Raw Stats
            current_raw = df.groupby('clean_name')[['golddiffat15', 'xpdiffat15', 'dpm', 'vspm']].mean().to_dict('index')
            RAW_DB.update(current_raw)

            # Champion Stats
            champ_agg = df.groupby(['clean_name', 'champion']).agg({
                'result': ['count', 'sum'], 'kills': 'sum', 'deaths': 'sum', 'assists': 'sum'
            }).reset_index()
            
            for _, row in champ_agg.iterrows():
                p_name, champ = row['clean_name'], row['champion']
                games, wins = row['result']['count'], row['result']['sum']
                k, d, a = row['kills']['sum'], row['deaths']['sum'], row['assists']['sum']
                
                if p_name not in CHAMP_DB: CHAMP_DB[p_name] = {}
                if champ not in CHAMP_DB[p_name]: CHAMP_DB[p_name][champ] = {'games': 0, 'wins': 0, 'k': 0, 'd': 0, 'a': 0}
                
                c = CHAMP_DB[p_name][champ]
                c['games'] += games; c['wins'] += wins; c['k'] += k; c['d'] += d; c['a'] += a

            # Training Data
            valid_games = df['gameid'].value_counts()
            game_df = df[df['gameid'].isin(valid_games[valid_games == 10].index)]
            
            for gid, match in game_df.groupby('gameid'):
                blue = match[match['side'] == 'Blue']
                red = match[match['side'] == 'Red']
                if len(blue) == 5 and len(red) == 5:
                    b_score = blue[[f'score_{m}' for m in METRICS]].sum().values
                    r_score = red[[f'score_{m}' for m in METRICS]].sum().values
                    training_rows.append(np.append(b_score - r_score, 1 if blue.iloc[0]['result'] == 1 else 0))
            
            # Matchups
            blue = game_df[game_df['side'] == 'Blue'][['gameid', 'position', 'champion', 'result']]
            red = game_df[game_df['side'] == 'Red'][['gameid', 'position', 'champion', 'result']]
            merged = pd.merge(blue, red, on=['gameid', 'position'], suffixes=('_b', '_r'))
            
            for _, row in merged.iterrows():
                b_c, r_c, res = row['champion_b'], row['champion_r'], row['result_b']
                if b_c not in MATCHUP_DB: MATCHUP_DB[b_c] = {}
                if r_c not in MATCHUP_DB[b_c]: MATCHUP_DB[b_c][r_c] = {'wins': 0, 'games': 0}
                MATCHUP_DB[b_c][r_c]['games'] += 1
                MATCHUP_DB[b_c][r_c]['wins'] += res
                
                if r_c not in MATCHUP_DB: MATCHUP_DB[r_c] = {}
                if b_c not in MATCHUP_DB[r_c]: MATCHUP_DB[r_c][b_c] = {'wins': 0, 'games': 0}
                MATCHUP_DB[r_c][b_c]['games'] += 1
                MATCHUP_DB[r_c][b_c]['wins'] += (1 - res)

            del df, pool, game_df, blue, red, merged, league_weights
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error loading {year}: {e}")

    # Finalize
    for p in CHAMP_DB:
        for c in CHAMP_DB[p]:
            d = CHAMP_DB[p][c]
            d['win_rate'] = round((d['wins'] / d['games']) * 100, 1)
            d['kda'] = round((d['k'] + d['a']) / max(1, d['d']), 2)
    
    for me in MATCHUP_DB:
        for en in MATCHUP_DB[me]:
            d = MATCHUP_DB[me][en]
            d['wr'] = round((d['wins'] / d['games']) * 100, 1)

    # Build DB
    for player, years in player_yearly_scores.items():
        total, weight = np.zeros(len(METRICS)), 0
        for y, scores in years.items():
            w = YEAR_WEIGHTS.get(y, 0.2)
            total += scores * w
            weight += w
        if weight > 0:
            DB[player] = (total / weight) + LEGACY.get(player, 0)

    # Train Model
    if training_rows:
        print(f"🤖 Training on {len(training_rows)} matches...")
        train_data = np.array(training_rows)
        MODEL = LogisticRegression(C=1.0, solver='liblinear').fit(train_data[:, :-1], train_data[:, -1])
    else:
        print("❌ CRITICAL: No training data found.")

    print("✅ System Fully Loaded!")

def get_team_vector(team):
    vec = np.zeros(len(METRICS))
    for p in ROSTERS[team]:
        n = clean_name(p)
        if n in DB: 
            vec += DB[n]
        else:
            # Safe Fallback: If player has NO history, assume average
            pass
    return vec

# ROUTES
@app.route('/')
def home():
    return render_template('index.html', teams=ROSTERS)

@app.route('/draft')
def draft_page():
    all_champs = set()
    for p_data in CHAMP_DB.values(): all_champs.update(p_data.keys())
    return render_template('draft.html', teams=ROSTERS, champions=sorted(list(all_champs)))

@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL: return jsonify({'error': 'System loading...'})
    data = request.json
    v1 = get_team_vector(data['blue'])
    v2 = get_team_vector(data['red'])
    prob = MODEL.predict_proba([(v1 - v2) / 5.0])[0][1]
    return jsonify({'winner': data['blue'] if prob > 0.5 else data['red'], 'blue_win_chance': round(prob * 100, 1)})

@app.route('/compare_players', methods=['POST'])
def compare_players():
    data = request.json
    p1, p2 = clean_name(data['player1']), clean_name(data['player2'])
    def get_stats(p): return {k: round(v, 1) for k, v in RAW_DB.get(p, {m: 0 for m in METRICS}).items()}
    return jsonify({'p1': get_stats(p1), 'p2': get_stats(p2)})

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
    init_system()
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)