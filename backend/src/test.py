from Model import Model, Game
from Database import Season
import sys
import time
import pandas as pd

# Usage
# predictor = BracketPredictor(season)
# results = predictor.predict_bracket()
# print(results)

def predict_bracket(season):
    """
    Used to predict all brackets without prints.
    """
    model = Model(season)
    model.initialize()
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
    x, y = 4, 36 if season.get_season() != 2021 else 35
    curr_games = tourney.iloc[x:y]
    for i in range(len(teams)):
        region = []
        for j in range(0, len(teams[i]), 2):
            # Predict game
            game = Game(model, season, teams[i][j], teams[i][j+1])
            game.predict_game()
            # Determine if prediction is correct
            predicted_winner = game.get_winner()
            actual_game = curr_games[(curr_games['WTeamName'] == predicted_winner) | (curr_games['LTeamName'] == predicted_winner)]
            if not actual_game.empty:
                actual_winner = actual_game.iloc[0]['WTeamName']
                if predicted_winner == actual_winner:
                    first_points += 10
                    correct += 1
            # Append for bracket
            region.append(predicted_winner)
        second_round.append(region)
        
    second_points = 0
    sweet_16 = []
    x, y = y, y + 16
    curr_games = tourney.iloc[x:y]
    for i in range(len(second_round)):
        region = []
        for j in range(0, len(second_round[i]), 2):
            game = Game(model, season, second_round[i][j], second_round[i][j+1])
            game.predict_game()
            predicted_winner = game.get_winner()
            actual_game = curr_games[(curr_games['WTeamName'] == predicted_winner) | (curr_games['LTeamName'] == predicted_winner)]
            if not actual_game.empty:
                actual_winner = actual_game.iloc[0]['WTeamName']
                if predicted_winner == actual_winner:
                    second_points += 20
                    correct += 1
            # Append for bracket
            region.append(predicted_winner)
        sweet_16.append(region)
    
    elite_8 = []
    x, y = y, y + 8
    curr_games = tourney.iloc[x:y]
    sweet_points = 0
    for i in range(len(sweet_16)):
        region = []
        for j in range(0, len(sweet_16[i]), 2):
            game = Game(model, season, sweet_16[i][j], sweet_16[i][j+1])
            game.predict_game()
            predicted_winner = game.get_winner()
            actual_game = curr_games[(curr_games['WTeamName'] == predicted_winner) | (curr_games['LTeamName'] == predicted_winner)]
            if not actual_game.empty:
                actual_winner = actual_game.iloc[0]['WTeamName']
                if predicted_winner == actual_winner:
                    sweet_points += 40
                    correct += 1
            # Append for bracket
            region.append(predicted_winner)
        elite_8.append(region)
    
    final_4 = []
    x, y = y, y + 4
    curr_games = tourney.iloc[x:y]
    elite_points = 0
    for i in range(len(elite_8)):
        region = []
        for j in range(0, len(elite_8[i]), 2):
            game = Game(model, season, elite_8[i][j], elite_8[i][j+1])
            game.predict_game()
            predicted_winner = game.get_winner()
            actual_game = curr_games[(curr_games['WTeamName'] == predicted_winner) | (curr_games['LTeamName'] == predicted_winner)]
            if not actual_game.empty:
                actual_winner = actual_game.iloc[0]['WTeamName']
                if predicted_winner == actual_winner:
                    elite_points += 80
                    correct += 1
            # Append for bracket
            region.append(predicted_winner)
        final_4.append(region)
    
    final_4 = [team for region in final_4 for team in region]
    championship = []
    x, y = y, y + 2
    curr_games = tourney.iloc[x:y]
    final_points = 0
    for i in range(0, len(final_4), 2):
        game = Game(model, season, final_4[i], final_4[i+1])
        game.predict_game()
        predicted_winner = game.get_winner()
        actual_game = curr_games[(curr_games['WTeamName'] == predicted_winner) | (curr_games['LTeamName'] == predicted_winner)]
        if not actual_game.empty:
            actual_winner = actual_game.iloc[0]['WTeamName']
            if predicted_winner == actual_winner:
                final_points += 160
                correct += 1
        # Append for bracket
        championship.append(predicted_winner)
        
    x = y
    curr_games = tourney.iloc[x:]
    champ_points = 0
    game = Game(model, season, championship[0], championship[1])
    game.predict_game()
    predicted_winner = game.get_winner()
    actual_game = curr_games[(curr_games['WTeamName'] == predicted_winner) | (curr_games['LTeamName'] == predicted_winner)]
    if not actual_game.empty:
        actual_winner = actual_game.iloc[0]['WTeamName']
        if predicted_winner == actual_winner:
            champ_points += 320
            correct += 1
    # Append for bracket
    championship.append(predicted_winner)
    
    return first_points, second_points, sweet_points, elite_points, final_points, champ_points, correct

def predict_season(season):
    model = Model(season)
    model.initialize()
    s = season
    season = Season(s)
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
    x = 4
    y = 36 if s != 2021 else 35
    second_round = []
    print('\nFirst Round')
    curr_games = tourney.iloc[x:y]
    for i in range(len(teams)):
        print('Region', i+1)
        region = []
        for j in range(0, len(teams[i]), 2):
            # Predict game
            game = Game(model, season, teams[i][j], teams[i][j+1])
            print(game)
            print(game.predict_game())
            # Determine if prediction is correct
            predicted_winner = game.get_winner()
            actual_game = curr_games[(curr_games['WTeamName'] == predicted_winner) | (curr_games['LTeamName'] == predicted_winner)]
            if not actual_game.empty:
                actual_winner = actual_game.iloc[0]['WTeamName']
                if predicted_winner == actual_winner:
                    first_points += 10
                    correct += 1
                    print('Points for', predicted_winner)
            # Append for bracket
            region.append(predicted_winner)
        second_round.append(region)
        print()
        
    print('Points after First Round:', first_points)
    
    print('\nSecond Round')
    second_points = 0
    x, y = y, y + 16
    sweet_16 = []
    curr_games = tourney.iloc[x:y]
    for i in range(len(second_round)):
        print('Region', i+1)
        region = []
        for j in range(0, len(second_round[i]), 2):
            game = Game(model, season, second_round[i][j], second_round[i][j+1])
            print(game)
            print(game.predict_game())
            predicted_winner = game.get_winner()
            actual_game = curr_games[(curr_games['WTeamName'] == predicted_winner) | (curr_games['LTeamName'] == predicted_winner)]
            if not actual_game.empty:
                actual_winner = actual_game.iloc[0]['WTeamName']
                if predicted_winner == actual_winner:
                    second_points += 20
                    correct += 1
                    print('Points for', predicted_winner)
            # Append for bracket
            region.append(predicted_winner)
        sweet_16.append(region)
        print()
    
    print('Points after Second Round:', first_points + second_points)
    
    print('\nSweet 16')
    elite_8 = []
    x, y = y, y + 8
    curr_games = tourney.iloc[x:y]
    sweet_points = 0

    for i in range(len(sweet_16)):
        print('Region', i+1)
        region = []
        for j in range(0, len(sweet_16[i]), 2):
            game = Game(model, season, sweet_16[i][j], sweet_16[i][j+1])
            print(game)
            print(game.predict_game())
            predicted_winner = game.get_winner()
            actual_game = curr_games[(curr_games['WTeamName'] == predicted_winner) | (curr_games['LTeamName'] == predicted_winner)]
            if not actual_game.empty:
                actual_winner = actual_game.iloc[0]['WTeamName']
                if predicted_winner == actual_winner:
                    sweet_points += 40
                    correct += 1
                    print('Points for', predicted_winner)
            # Append for bracket
            region.append(predicted_winner)
        elite_8.append(region)
        print()
        
    print('Points after Sweet 16:', first_points + second_points + sweet_points)
    
    print('\nElite 8')
    final_4 = []
    x, y = y, y + 4
    curr_games = tourney.iloc[x:y]
    elite_points = 0
    for i in range(len(elite_8)):
        print('Region', i+1)
        region = []
        for j in range(0, len(elite_8[i]), 2):
            game = Game(model, season, elite_8[i][j], elite_8[i][j+1])
            print(game)
            print(game.predict_game())
            predicted_winner = game.get_winner()
            actual_game = curr_games[(curr_games['WTeamName'] == predicted_winner) | (curr_games['LTeamName'] == predicted_winner)]
            if not actual_game.empty:
                actual_winner = actual_game.iloc[0]['WTeamName']
                if predicted_winner == actual_winner:
                    elite_points += 80
                    correct += 1
                    print('Points for', predicted_winner)
            # Append for bracket
            region.append(predicted_winner)
        final_4.append(region)
        print()
    
    print('Points after Elite 8:', first_points + second_points + sweet_points + elite_points)
    
    final_4 = [team for region in final_4 for team in region]
    print('\nFinal 4')
    championship = []
    x, y = y, y + 2
    curr_games = tourney.iloc[x:y]
    final_points = 0
    for i in range(0, len(final_4), 2):
        game = Game(model, season, final_4[i], final_4[i+1])
        print(game)
        print(game.predict_game())
        predicted_winner = game.get_winner()
        actual_game = curr_games[(curr_games['WTeamName'] == predicted_winner) | (curr_games['LTeamName'] == predicted_winner)]
        if not actual_game.empty:
            actual_winner = actual_game.iloc[0]['WTeamName']
            if predicted_winner == actual_winner:
                final_points += 160
                correct += 1
                print('Points for', predicted_winner)
        # Append for bracket
        championship.append(predicted_winner)
    
    print('\nPoints after Final Four:', first_points + second_points + sweet_points + elite_points + final_points)
        
    print('\nChampionship')
    x = y
    curr_games = tourney.iloc[x:]
    champ_points = 0
    game = Game(model, season, championship[0], championship[1])
    print(game)
    print(game.predict_game())
    predicted_winner = game.get_winner()
    actual_game = curr_games[(curr_games['WTeamName'] == predicted_winner) | (curr_games['LTeamName'] == predicted_winner)]
    if not actual_game.empty:
        actual_winner = actual_game.iloc[0]['WTeamName']
        if predicted_winner == actual_winner:
            champ_points += 320
            correct += 1
            print('Points for', predicted_winner)
    # Append for bracket
    championship.append(predicted_winner)
    
    print('\nTotal Points:', first_points + second_points + sweet_points + elite_points + final_points + champ_points)
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
    # Simulation
    run_sims()
    
    # Get specific season
    # season = int(sys.argv[1])
    # res = predict_season(season)
    # print(res)
    
    # Get game
    # season = 2024
    # game = Game(Season(season), 'Houston', 'Auburn')
    # print(game.predict_game())
    

    
"""
To do:
1. Speed up predict_bracket, shorten code for readability
"""


