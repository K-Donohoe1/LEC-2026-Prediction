# LEC-2026-Prediction
Python-based predictive analytics for the 2026 LEC season using historical match data and player performance metrics.

This project uses a Machine Learning model (Logistic Regression) to analyze player statistics from the 2022, 2023, 2024 and, 2025 seasons, apply contextual adjustments (tier weights, legacy buffs), and predict match outcomes with realistic probabilities.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Flask-Web%20Framework-green)
![Scikit-Learn](https://img.shields.io/badge/AI-Scikit--Learn-orange)

## Features

* ** Advanced Prediction Model:** Uses Logistic Regression trained on Gold, XP, Damage, and Vision differentials.
* ** Context-Aware Scoring:** Applies "League Bonuses" to differentiate between LEC, LFL, and NLC performance (tier 1 leagues and tier 2 leagues).
* ** Legacy & Smurf Detection:** Custom logic to correctly rate outliers like **Los Ratones** (multiple ex-tier 1 pros) or benched veterans like **Humanoid**.
* ** Head-to-Head Comparison:** Visual stats comparison for any two players in any role.
* ** Realism Engine:** Includes logic to dampen probabilities, preventing unrealistic "100% win rate" predictions.
* ** Draft Tool:** Shows specific player winrates on certain champions and historical champion matchups per role.

## Tech Stack
* **Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn (Logistic Regression / Random Forest / XGBoost)
* **Data Sources:** Oracle's Elixir

## Installation & Setup

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/K-Donohoe1/LEC-2026-prediction.git](https://github.com/K-Donohoe1/LEC-2026-prediction.git)
    cd LEC-2026-prediction
    ```

2.  **Install Requirements**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Add Data Files**
    Ensure the following CSV files (from Oracle's Elixir) are in the root directory:
    * `2022_LoL_esports_match_data_from_OraclesElixir.csv`
    * `2023_LoL_esports_match_data_from_OraclesElixir.csv`
    * `2024_LoL_esports_match_data_from_OraclesElixir.csv`
    * `2025_LoL_esports_match_data_from_OraclesElixir.csv`

5.  **Run the App**
    ```bash
    python app.py
    ```
    Open your browser to `http://127.0.0.1:5000`.

## Deployment (Render/Cloud)

This app is production-ready for platforms like Render or PythonAnywhere.
* **Build Command:** `pip install -r requirements.txt`
* **Start Command:** `gunicorn app:app`

## How It Works

The ML Model calculates a **Power Score** for every player based on 4 metrics:
1.  **Gold Diff @ 15:** Laning prowess.
2.  **XP Diff @ 15:** Lane control.
3.  **Damage Per Minute (DPM):** Carry potential.
4.  **Vision Score Per Minute:** Map awareness.

It then sums these scores for a 5-man roster and compares the total team vectors. A Logistic Regression model (trained on historical win/loss data) calculates the probability of Team A beating Team B based on that stat difference.

## Legal stuff

LEC 2026 Oracle is a fan project and is not endorsed by Riot Games.
It does not reflect the views or opinions of Riot Games or anyone officially involved in producing or managing Riot Games properties.
Riot Games, and all associated properties are trademarks or registered trademarks of Riot Games, Inc.

## Credits

* **Data:** Tim Sevenhuysen [Oracle's Elixir](https://oracleselixir.com/)   Game statistics are the property of Riot Games, and any usage of such data must follow Riot Games' terms and policies.
* **Tech Stack:** Python, Flask, Pandas, Scikit-Learn

URL to website using Render: https://lec-2026-prediction.onrender.com (site currently doesn't work!)

---
*Created for the 2026 LEC Season Simulation Project.*
