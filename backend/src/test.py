
from Model import Game
from Database import Season
import sys
import time
import pandas as pd

# Usage
# predictor = BracketPredictor(season)
# results = predictor.predict_bracket()
# print(results)

def predict_bracket(season):
    season = Season(season)
    seeds = season.get_seeds()
    tourney = season.get_tourney()
    
    
    sorting = [0, 7, 4, 3, 2, 5, 6, 1]
    teams = []
    for i in range(4):  # Assuming 4 regions
        region_teams = seeds.iloc[i*16:(i+1)*16].reset_index()  # Extract the teams for the region
        region = []
        for idx in sorting:
            region.append(region_teams.iloc[idx]['TeamName'])
            region.append(region_teams.iloc[-(idx+1)]['TeamName'])
        teams.append(region)
    
    correct = 0
    
    first_points = 0
    second_round = []
    # print('\nFirst Round')
    curr_games = tourney.iloc[4:36]
    for i in range(len(teams)):
        # print('Region', i+1)
        region = []
        for j in range(0, len(teams[i]), 2):
            # Predict game
            game = Game(season, teams[i][j], teams[i][j+1])
            # print(game)
            # print(game.predict_game())
            game.predict_game()
            # Determine if prediction is correct
            predicted_winner = game.get_winner()
            actual_game = curr_games[(curr_games['WTeamName'] == predicted_winner) | (curr_games['LTeamName'] == predicted_winner)]
            if not actual_game.empty:
                actual_winner = actual_game.iloc[0]['WTeamName']
                if predicted_winner == actual_winner:
                    first_points += 10
                    correct += 1
                    # print('Points for', predicted_winner)
            # Append for bracket
            region.append(predicted_winner)
        second_round.append(region)
        # print()
        
    # print('Points after First Round:', points)
    
    # print('\nSecond Round')
    second_points = 0
    sweet_16 = []
    curr_games = tourney.iloc[36:52]
    for i in range(len(second_round)):
        # print('Region', i+1)
        region = []
        for j in range(0, len(second_round[i]), 2):
            game = Game(season, second_round[i][j], second_round[i][j+1])
            # print(game)
            # print(game.predict_game())
            game.predict_game()
            predicted_winner = game.get_winner()
            actual_game = curr_games[(curr_games['WTeamName'] == predicted_winner) | (curr_games['LTeamName'] == predicted_winner)]
            if not actual_game.empty:
                actual_winner = actual_game.iloc[0]['WTeamName']
                if predicted_winner == actual_winner:
                    second_points += 20
                    correct += 1
                    # print('Points for', predicted_winner)
            # Append for bracket
            region.append(predicted_winner)
        sweet_16.append(region)
        # print()
    
    # print('Points after Second Round:', points)
    
    # print('\nSweet 16')
    elite_8 = []
    curr_games = tourney.iloc[52:60]
    sweet_points = 0
    for i in range(len(sweet_16)):
        # print('Region', i+1)
        region = []
        for j in range(0, len(sweet_16[i]), 2):
            game = Game(season, sweet_16[i][j], sweet_16[i][j+1])
            # print(game)
            # print(game.predict_game())
            game.predict_game()
            predicted_winner = game.get_winner()
            actual_game = curr_games[(curr_games['WTeamName'] == predicted_winner) | (curr_games['LTeamName'] == predicted_winner)]
            if not actual_game.empty:
                actual_winner = actual_game.iloc[0]['WTeamName']
                if predicted_winner == actual_winner:
                    sweet_points += 40
                    correct += 1
                    # print('Points for', predicted_winner)
            # Append for bracket
            region.append(predicted_winner)
        elite_8.append(region)
        # print()
        
    # print('Points after Sweet 16:', points)
    
    # print('\nElite 8')
    final_4 = []
    curr_games = tourney.iloc[60:64]
    elite_points = 0
    for i in range(len(elite_8)):
        # print('Region', i+1)
        region = []
        for j in range(0, len(elite_8[i]), 2):
            game = Game(season, elite_8[i][j], elite_8[i][j+1])
            # print(game)
            game.predict_game()
            # print(game.predict_game())
            predicted_winner = game.get_winner()
            actual_game = curr_games[(curr_games['WTeamName'] == predicted_winner) | (curr_games['LTeamName'] == predicted_winner)]
            if not actual_game.empty:
                actual_winner = actual_game.iloc[0]['WTeamName']
                if predicted_winner == actual_winner:
                    elite_points += 80
                    correct += 1
                    # print('Points for', predicted_winner)
            # Append for bracket
            region.append(predicted_winner)
        final_4.append(region)
        # print()
    
    # print('Points after Elite 8:', points)
    
    final_4 = [team for region in final_4 for team in region]
    # print('\nFinal 4')
    championship = []
    curr_games = tourney.iloc[64:66]
    final_points = 0
    for i in range(0, len(final_4), 2):
        game = Game(season, final_4[i], final_4[i+1])
        # print(game)
        game.predict_game()
        # print(game.predict_game())
        predicted_winner = game.get_winner()
        actual_game = curr_games[(curr_games['WTeamName'] == predicted_winner) | (curr_games['LTeamName'] == predicted_winner)]
        if not actual_game.empty:
            actual_winner = actual_game.iloc[0]['WTeamName']
            if predicted_winner == actual_winner:
                final_points += 160
                correct += 1
                # print('Points for', predicted_winner)
        # Append for bracket
        championship.append(predicted_winner)
    
    # print('\nPoints after Final Four:', points)
        
    # print('\nChampionship')
    curr_games = tourney.iloc[66:]
    champ_points = 0
    game = Game(season, championship[0], championship[1])
    # print(game)
    # print(game.predict_game())
    game.predict_game()
    predicted_winner = game.get_winner()
    actual_game = curr_games[(curr_games['WTeamName'] == predicted_winner) | (curr_games['LTeamName'] == predicted_winner)]
    if not actual_game.empty:
        actual_winner = actual_game.iloc[0]['WTeamName']
        if predicted_winner == actual_winner:
            champ_points += 320
            correct += 1
            # print('Points for', predicted_winner)
    # Append for bracket
    championship.append(predicted_winner)
    
    # print('\nTotal Points:', points)
    return first_points, second_points, sweet_points, elite_points, final_points, champ_points, correct

def run_sims():
    start_time = time.time()
    df = pd.DataFrame(columns=['Season', 'First Round', 'Second Round', 'Sweet 16', 'Elite 8', 'Final 4', 'Championship', 'Correct Picks', 'Total Points'])
    for season in range(2003, 2024):
        if season != 2020:
            points = predict_bracket(season)
            season_data = pd.DataFrame([(season,) + points + (sum(points[:-1]),)], 
                                    columns=df.columns)
            df.loc[season - 2003] = season_data.iloc[0]
    df.set_index('Season', inplace=True)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(df)
    print('\nStatistics', df.describe(), sep='\n')
    print(f"\nTime elapsed: {round(elapsed_time, 2)} s")
       
if __name__ == '__main__':
    # run_sims()
    season = int(sys.argv[1])
    predict_bracket(season)

    
"""
To do:
3. Add massey rankings
4. Add kenpom rankings
5. Speed up predict_bracket, shorten code for readability
"""


