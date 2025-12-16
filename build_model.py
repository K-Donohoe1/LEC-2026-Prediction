import os
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.linear_model import LogisticRegression

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Ensure your Data folder has ALL 4 files!
FILES = {
    2022: os.path.join(BASE_DIR, 'Data', '2022_LoL_esports_match_data_from_OraclesElixir.csv'),
    2023: os.path.join(BASE_DIR, 'Data', '2023_LoL_esports_match_data_from_OraclesElixir.csv'),
    2024: os.path.join(BASE_DIR, 'Data', '2024_LoL_esports_match_data_from_OraclesElixir.csv'),
    2025: os.path.join(BASE_DIR, 'Data', '2025_LoL_esports_match_data_from_OraclesElixir.csv'),
}

USE_COLS = ['gameid', 'playername', 'position', 'teamname', 'champion', 'result', 'side', 'league', 'kills', 'deaths', 'assists', 'golddiffat15', 'xpdiffat15', 'dpm', 'vspm']
METRICS = ['golddiffat15', 'xpdiffat15', 'dpm', 'vspm', 'pool_depth']

# Weight recent years higher for player stats
YEAR_WEIGHTS = {
    2025: 1.0,
    2024: 0.8,
    2023: 0.5,
    2022: 0.3
}

# Leagues to keep for model training (Major Regions + International)
TRAINING_LEAGUES = ['LEC', 'LCS', 'LCK', 'LPL', 'MSI', 'WLDs'] 
LEAGUE_BONUS = {'LEC': 1.5, 'LCK': 2.0, 'LPL': 2.0, 'LCS': 1.2, 'MSI': 2.0, 'WLDs': 2.0, 'Default': 0.5}
NAME_ALIASES = {'crownshot': 'crownie', 'thebausffs': 'baus', 'bausffs': 'baus', 'reckles': 'rekkles', '113': 'isma'}
LEGACY = {'razork': 1.1, 'upset': 1.1, 'caps': 0.9, 'hans sama': 0.8, 'humanoid': 0.1}

# Your Rosters
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

def clean_name(name):
    n = str(name).lower().strip()
    return NAME_ALIASES.get(n, n)

def run():
    print("🚀 Starting Local Build Process (4-Year Mode)...")
    
    # 1. Identify Target Players (To keep their stats even from minor leagues)
    target_players = set()
    for roster in ROSTERS.values():
        for p in roster:
            target_players.add(clean_name(p))
            
    print(f"📋 Tracking {len(target_players)} specific players.")

    # Data Holders
    DB = {}         # Player Vectors (AI)
    RAW_DB = {}     # Raw Stats (Display)
    CHAMP_DB = {}   # Champ Stats (Draft)
    MATCHUP_DB = {} # Matchup Stats (Draft)
    
    player_yearly_scores = {}
    training_rows = []

    # 2. Process Files
    for year, filepath in FILES.items():
        if not os.path.exists(filepath):
            print(f"⚠️ Missing: {filepath}")
            continue
            
        print(f"📂 Processing {year}...")
        
        # Load FULL file
        df = pd.read_csv(filepath, usecols=USE_COLS, low_memory=False)
        
        # Remove duplicate columns if they exist
        df = df.loc[:, ~df.columns.duplicated()]
        
        df = df[df['position'] != 'team'].copy()
        
        # Normalize names
        df['clean_name'] = df['playername'].apply(lambda x: str(x).lower().strip())
        df['clean_name'] = df['clean_name'].map(lambda x: NAME_ALIASES.get(x, x))
        
        # Filter: Keep Major Leagues OR Target Players
        mask = (df['league'].isin(TRAINING_LEAGUES)) | (df['clean_name'].isin(target_players))
        df = df[mask].copy()
        
        # Calculate Metrics
        pool = df.groupby(['clean_name', 'league'])['champion'].nunique().reset_index(name='pool_depth')
        df = df.merge(pool, on=['clean_name', 'league'], how='left')
        
        league_weights = df['league'].map(LEAGUE_BONUS).fillna(LEAGUE_BONUS['Default'])
        for m in METRICS:
            if m in df.columns or m == 'pool_depth':
                league_stats = df.groupby('league')[m].transform(lambda x: (x - x.mean()) / x.std())
                df[f'score_{m}'] = league_stats.fillna(0) + league_weights

        # Store Scores (for weighted average later)
        year_stats = df.groupby('clean_name')[[f'score_{m}' for m in METRICS]].mean()
        for player, row in year_stats.iterrows():
            if player not in player_yearly_scores: player_yearly_scores[player] = {}
            player_yearly_scores[player][year] = row.values

        # Store Raw Stats (Keep updating so we get the LATEST stats for display)
        current_raw = df.groupby('clean_name')[['golddiffat15', 'xpdiffat15', 'dpm', 'vspm']].mean().to_dict('index')
        RAW_DB.update(current_raw)

        # Champion Stats
        champ_agg = df.groupby(['clean_name', 'champion']).agg({
            'result': ['count', 'sum'], 'kills': 'sum', 'deaths': 'sum', 'assists': 'sum'
        })
        champ_agg.columns = ['_'.join(col).strip() for col in champ_agg.columns.values]
        champ_agg = champ_agg.reset_index()
        
        for _, row in champ_agg.iterrows():
            p_name = row['clean_name']
            champ = row['champion']
            
            if p_name not in CHAMP_DB: CHAMP_DB[p_name] = {}
            if champ not in CHAMP_DB[p_name]: CHAMP_DB[p_name][champ] = {'games': 0, 'wins': 0, 'k': 0, 'd': 0, 'a': 0}
            
            c = CHAMP_DB[p_name][champ]
            c['games'] += row['result_count']
            c['wins'] += row['result_sum']
            c['k'] += row['kills_sum']
            c['d'] += row['deaths_sum']
            c['a'] += row['assists_sum']

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
            
            # Red Perspective
            if r_c not in MATCHUP_DB: MATCHUP_DB[r_c] = {}
            if b_c not in MATCHUP_DB[r_c]: MATCHUP_DB[r_c][b_c] = {'wins': 0, 'games': 0}
            MATCHUP_DB[r_c][b_c]['games'] += 1
            MATCHUP_DB[r_c][b_c]['wins'] += (1 - res)

    print("💾 Saving Data...")
    
    # Finalize DB Vectors (Weighted Average)
    for player, years in player_yearly_scores.items():
        total, weight = np.zeros(len(METRICS)), 0
        for y, scores in years.items():
            w = YEAR_WEIGHTS.get(y, 0.2)
            total += scores * w
            weight += w
        
        if weight > 0:
            final_vec = (total / weight) + LEGACY.get(player, 0)
            DB[player] = final_vec.tolist()

    # Finalize Champ Stats
    for p in CHAMP_DB:
        for c in CHAMP_DB[p]:
            d = CHAMP_DB[p][c]
            d['win_rate'] = round((d['wins'] / d['games']) * 100, 1)
            d['kda'] = round((d['k'] + d['a']) / max(1, d['d']), 2)

    # Finalize Matchups
    for me in MATCHUP_DB:
        for en in MATCHUP_DB[me]:
            d = MATCHUP_DB[me][en]
            d['wr'] = round((d['wins'] / d['games']) * 100, 1)

    # Train Model
    print(f"🤖 Training Model on {len(training_rows)} matches...")
    if training_rows:
        train_data = np.array(training_rows)
        model = LogisticRegression(C=1.0, solver='liblinear')
        model.fit(train_data[:, :-1], train_data[:, -1])

        # Save to files
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
            
        stats_data = {
            'DB': DB,
            'RAW_DB': RAW_DB,
            'CHAMP_DB': CHAMP_DB,
            'MATCHUP_DB': MATCHUP_DB
        }
        with open('stats.json', 'w') as f:
            json.dump(stats_data, f)

        print("✅ DONE! Created 'model.pkl' and 'stats.json'")
    else:
        print("❌ Error: No training data found!")

if __name__ == "__main__":
    run()