import os
import pickle
import json
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
STATS_PATH = os.path.join(BASE_DIR, 'stats.json')

# LOAD PRE-BUILT DATA
print("⏳ Loading Pre-Built Data...")
with open(MODEL_PATH, 'rb') as f:
    MODEL = pickle.load(f)

with open(STATS_PATH, 'r') as f:
    STATS = json.load(f)

DB = STATS['DB']
RAW_DB = STATS['RAW_DB']
CHAMP_DB = STATS['CHAMP_DB']
MATCHUP_DB = STATS['MATCHUP_DB']
print("✅ System Ready!")

NAME_ALIASES = {'crownshot': 'crownie', 'thebausffs': 'baus', 'bausffs': 'baus', 'reckles': 'rekkles', '113': 'isma'}

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

def get_team_vector(team):
    vec = np.zeros(5) # 5 metrics
    for p in ROSTERS[team]:
        n = clean_name(p)
        if n in DB:
            vec += np.array(DB[n])
    return vec

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
    data = request.json
    v1 = get_team_vector(data['blue'])
    v2 = get_team_vector(data['red'])
    prob = MODEL.predict_proba([(v1 - v2) / 5.0])[0][1]
    return jsonify({'winner': data['blue'] if prob > 0.5 else data['red'], 'blue_win_chance': round(prob * 100, 1)})

@app.route('/compare_players', methods=['POST'])
def compare_players():
    data = request.json
    p1, p2 = clean_name(data['player1']), clean_name(data['player2'])
    # Metrics order: golddiffat15, xpdiffat15, dpm, vspm
    def get_stats(p): return RAW_DB.get(p, {'golddiffat15':0, 'xpdiffat15':0, 'dpm':0, 'vspm':0})
    
    # Round values for display
    s1 = {k: round(v, 1) for k,v in get_stats(p1).items()}
    s2 = {k: round(v, 1) for k,v in get_stats(p2).items()}
    return jsonify({'p1': s1, 'p2': s2})

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
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)