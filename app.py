from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

df_bowling = pd.read_csv('all_season_bowling_card.csv')  
df_batting = pd.read_csv('all_season_batting_card.csv')
data = pd.read_csv('all_season_summary.csv')

def get_bowler_stats(bowler_name):
    bowler_data = df_bowling[df_bowling['fullName'] == bowler_name]

    if bowler_data.empty:
        return None, f"No data found for bowler: {bowler_name}"

    total_wickets = bowler_data['wickets'].sum()
    total_runs_conceded = bowler_data['conceded'].sum()
    bowling_avg = total_runs_conceded / total_wickets if total_wickets > 0 else 0
    total_maidens = bowler_data['maidens'].sum()

    best_figures = bowler_data.loc[bowler_data['wickets'].idxmax()]
    best_wickets = best_figures['wickets']
    best_runs_conceded = best_figures['conceded']

    total_fours_conceded = int(bowler_data['foursConceded'].sum())
    total_sixes_conceded = int(bowler_data['sixesConceded'].sum())
    total_overs_bowled = bowler_data['overs'].sum()
    economy_rate = total_runs_conceded / total_overs_bowled if total_overs_bowled > 0 else 0
    economy_rate = round(float(economy_rate), 2)

    stats = {
        "Total Wickets": total_wickets,
        "Bowling Average": round(bowling_avg, 2),
        "Total Maidens": total_maidens,
        "Best Figures": f"{best_wickets}/{best_runs_conceded}",
        "Economy Rate": economy_rate,
        "Total Runs Conceded": total_runs_conceded,
        "Fours Conceded": total_fours_conceded,
        "Sixes Conceded": total_sixes_conceded
    }

    return stats, None

def get_player_stats(player_name):
    global df_batting  


    df_batting['runs'] = pd.to_numeric(df_batting['runs'], errors='coerce')
    df_batting['ballsFaced'] = pd.to_numeric(df_batting['ballsFaced'], errors='coerce')
    df_batting['strikeRate'] = pd.to_numeric(df_batting['strikeRate'], errors='coerce')
    df_batting['fours'] = pd.to_numeric(df_batting['fours'], errors='coerce')
    df_batting['sixes'] = pd.to_numeric(df_batting['sixes'], errors='coerce')

    df_batting = df_batting.dropna(subset=['runs', 'ballsFaced', 'strikeRate'])

    player_data = df_batting[df_batting['fullName'] == player_name]

    if player_data.empty:
        return None, f"No data found for player: {player_name}"

    runs_scored = player_data['runs'].sum()
    highest_score = player_data['runs'].max()
    matches_played = player_data['match_id'].nunique()
    average_score = player_data['runs'].mean()
    strike_rate = player_data['strikeRate'].mean()

    total_50s = len(player_data[(player_data['runs'] >= 50) & (player_data['runs'] < 100)])
    total_100s = len(player_data[player_data['runs'] >= 100])

    boundary_runs = (player_data['fours'].sum() * 4) + (player_data['sixes'].sum() * 6)
    boundary_percentage = (boundary_runs / runs_scored) * 100 if runs_scored > 0 else 0

    stats = {
        "Runs Scored": runs_scored,
        "Highest Score": highest_score,
        "Matches Played": matches_played,
        "Average Score": round(average_score, 2) if matches_played > 0 else 0,
        "Strike Rate": round(strike_rate, 2),
        "50s": total_50s,
        "100s": total_100s,
        "Boundary Percentage": f"{round(boundary_percentage, 2)}%"
    }

    return stats, None



def get_top_batsmen_stats():
    try:
        batsman_stats = df_batting.groupby('fullName').agg(
            Runs_Scored=('runs', 'sum'),
            Highest_Score=('runs', 'max'),
            Matches_Played=('match_id', 'nunique'),
            Average_Score=('runs', 'mean'),
            Balls_Faced=('ballsFaced', 'sum')
        ).reset_index()

        batsman_stats['Strike_Rate'] = (batsman_stats['Runs_Scored'] / batsman_stats['Balls_Faced']) * 100
        batsman_stats['Strike_Rate'].fillna(0, inplace=True)  

        top_batsmen = batsman_stats.sort_values(by='Runs_Scored', ascending=False).head(10)

        output = []
        for index, row in top_batsmen.iterrows():
            output.append({
                "Batsman": row['fullName'],
                "Runs_Scored": row['Runs_Scored'],
                "Highest_Score": row['Highest_Score'],
                "Matches_Played": row['Matches_Played'],
                "Average_Score": round(row['Average_Score'], 2) if row['Matches_Played'] > 0 else 0,
                "Strike_Rate": round(row['Strike_Rate'], 2)
            })

        return output

    except Exception as e:
        print(f"Error in get_top_batsmen_stats: {e}")
        return []

def get_top_bowlers_stats():

    df_bowling['overs'] = pd.to_numeric(df_bowling['overs'], errors='coerce')
    df_bowling['conceded'] = pd.to_numeric(df_bowling['conceded'], errors='coerce')
    df_bowling['economyRate'] = pd.to_numeric(df_bowling['economyRate'], errors='coerce')
    df_bowling['wickets'] = pd.to_numeric(df_bowling['wickets'], errors='coerce')

    bowler_stats = df_bowling.groupby('fullName').agg(
        Wickets_Taken=('wickets', 'sum'),
        Overs_Bowled=('overs', 'sum'),
        Runs_Conceded=('conceded', 'sum'),
        Economy_Rate=('economyRate', 'mean'),
    ).reset_index()

    top_bowlers = bowler_stats.sort_values(by='Wickets_Taken', ascending=False).head(10)

    output = []
    for index, row in top_bowlers.iterrows():
        output.append({
            "Bowler": row['fullName'],
            "Wickets_Taken": row['Wickets_Taken'],
            "Overs_Bowled": round(row['Overs_Bowled'], 1),
            "Runs_Conceded": row['Runs_Conceded'],
            "Economy_Rate": round(row['Economy_Rate'], 2),
        })

    return output

def get_highest_score(team_name):
    """Calculate the highest score for the given team."""
    home_highest_score = data[data['home_team'] == team_name]['home_runs'].max()

    away_highest_score = data[data['away_team'] == team_name]['away_runs'].max()
    
    return max(home_highest_score, away_highest_score)

def get_performance_summary(team_name):
    """Calculate performance against other teams."""
    performance_summary = []

    opponents = set(data['home_team'].unique()).union(set(data['away_team'].unique())) - {team_name}

    for opponent in opponents:
        matches = data[((data['home_team'] == team_name) & (data['away_team'] == opponent)) |
                       ((data['away_team'] == team_name) & (data['home_team'] == opponent))]

        matches_played = matches.shape[0]
        wins = matches[matches['winner'] == team_name].shape[0]
        losses = matches_played - wins if matches_played > 0 else 0

        team_highest_score = matches.apply(lambda row: row['home_runs'] if row['home_team'] == team_name else row['away_runs'], axis=1).max() if matches_played > 0 else 0

        if matches_played > 0:  
            performance_summary.append({
                'Opponent': opponent,
                'Matches Played': matches_played,
                'Wins': wins,
                'Losses': losses,
                'Team Highest Score': team_highest_score
            })

    performance_summary_df = pd.DataFrame(performance_summary)
    
    return performance_summary_df

def get_top_batsmen(team_name, season):
    team_data = df_batting[(df_batting['current_innings'] == team_name) & (df_batting['season'] == season)]

    runs_by_player = team_data.groupby('fullName')['runs'].sum().sort_values(ascending=False)

    top_10_batsmen = runs_by_player.head(10)
    return top_10_batsmen .to_dict()


def calculate_home_win_rate(home_team):
    total_home_matches = len(data[data['home_team'] == home_team])
    home_wins = len(data[(data['home_team'] == home_team) & (data['winner'] == home_team)])
    
    if total_home_matches > 0:
        win_rate = home_wins / total_home_matches
    else:
        win_rate = 0.5  
    return win_rate

def calculate_toss_win_rate(toss_won_team, decision):
    total_matches = len(data[(data['toss_won'] == toss_won_team) & (data['decision'] == decision)])
    wins_after_toss = len(data[(data['toss_won'] == toss_won_team) & (data['decision'] == decision) & (data['winner'] == toss_won_team)])
    
    if total_matches > 0:
        win_rate = wins_after_toss / total_matches
    else:
        win_rate = 0.5 
    return win_rate

@app.route('/', methods=['GET'])
def index():
    top_batsmen = get_top_batsmen_stats()  # Call the function to get top batsmen
    top_bowlers = get_top_bowlers_stats()  # Call the function to get top bowlers
    return render_template('home.html', top_batsmen=top_batsmen, top_bowlers=top_bowlers)

@app.route('/analysis', methods=['POST'])
def team_analysis():
    team_name = request.form.get('team_name')

    highest_score = get_highest_score(team_name)
    performance_summary = get_performance_summary(team_name)


    return render_template('analysis.html', team_name=team_name, highest_score=highest_score, performance_summary=performance_summary)

@app.route('/top_batsmen_form', methods=['GET'])
def show_top_batsmen_form():
    return render_template('top_batsmen_form.html', team_name=None, top_batsmen=None)

@app.route('/top_batsmen', methods=['POST'])
def process_top_batsmen():
    team_name = request.form['team_name']
    season = int(request.form['season'])

    top_batsmen = get_top_batsmen(team_name, season)  

    return render_template('top_batsmen_form.html', team_name=team_name, top_batsmen=top_batsmen)


@app.route('/index', methods=['GET'])
def show_bowler_form():
    return render_template('index.html', bowler_name=None, stats=None, error=None)


@app.route('/get_bowler_stats', methods=['POST'])
def process_bowler_stats():
    bowler_name = request.form['bowler_name']
    stats, error = get_bowler_stats(bowler_name)
    return render_template('index.html', bowler_name=bowler_name, stats=stats, error=error)

@app.route('/batsmen', methods=['GET']) 
def show_batsmen_form():
    return render_template('batsmen.html', player_name=None, stats=None, error=None)

@app.route('/get_player_stats', methods=['POST'])
def process_player_stats():
    player_name = request.form['player_name']
    stats, error = get_player_stats(player_name)

    if stats is None:
        error = f"No data found for player: {player_name}"
    
    return render_template('batsmen.html', player_name=player_name, stats=stats, error=error)

@app.route('/team_prediction', methods=['POST'])
def team_prediction():
    features = ['home_team', 'away_team', 'toss_won', 'decision']

    encoders = joblib.load('label_encoder.pkl')  

    home_team = request.form['home_team']
    away_team = request.form['away_team']
    toss_won = request.form['toss_won']
    decision = request.form['decision']

    decision_mapping = {'Bat': 'BAT FIRST', 'Field': 'BOWL FIRST'}
    decision = decision_mapping.get(decision, decision)  

    input_data = pd.DataFrame([[home_team, away_team, toss_won, decision]], columns=features)

    for col in input_data.columns:
        input_data[col] = encoders[col].transform(input_data[col])

    model = joblib.load('match_winner_model.pkl')

    predicted_winner = model.predict(input_data)[0]
    predicted_winner = encoders['away_team'].inverse_transform([predicted_winner])[0]


    return render_template('prediction.html', predicted_winner=predicted_winner,home_team=home_team,
    away_team=away_team,
    toss_won=toss_won,
    decision=decision)

if __name__ == '__main__':
    app.run(debug=True)
