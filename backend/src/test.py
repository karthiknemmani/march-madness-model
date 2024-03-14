
from Model import Game
from Database import Season
import sys

def predict_bracket(season):
    season = Season(season)
    seeds = season.get_seeds()
    
    sorting = [0, 7, 4, 3, 2, 5, 6, 1]
    teams = []
    for i in range(4):  # Assuming 4 regions
        region_teams = seeds.iloc[i*16:(i+1)*16].reset_index()  # Extract the teams for the region
        region = []
        for idx in sorting:
            region.append(region_teams.iloc[idx]['TeamName'])
            region.append(region_teams.iloc[-(idx+1)]['TeamName'])
        teams.append(region)
        
    second_round = []
    print('\nFirst Round')
    for i in range(len(teams)):
        print('Region', i+1)
        region = []
        for j in range(0, len(teams[i]), 2):
            game = Game(season, teams[i][j], teams[i][j+1])
            print(game.predict_game())
            region.append(game.get_winner())
        second_round.append(region)
        print()
    
    print('Second Round')
    sweet_16 = []
    for i in range(len(second_round)):
        print('Region', i+1)
        region = []
        for j in range(0, len(second_round[i]), 2):
            game = Game(season, second_round[i][j], second_round[i][j+1])
            print(game.predict_game())
            region.append(game.get_winner())
        sweet_16.append(region)
        print()
        
    print('Sweet 16')
    elite_8 = []
    for i in range(len(sweet_16)):
        print('Region', i+1)
        region = []
        for j in range(0, len(sweet_16[i]), 2):
            game = Game(season, sweet_16[i][j], sweet_16[i][j+1])
            print(game.predict_game())
            region.append(game.get_winner())
        elite_8.append(region)
        print()
    
    print('Elite 8')
    final_4 = []
    for i in range(len(elite_8)):
        print('Region', i+1)
        region = []
        for j in range(0, len(elite_8[i]), 2):
            game = Game(season, elite_8[i][j], elite_8[i][j+1])
            print(game.predict_game())
            region.append(game.get_winner())
        final_4.append(region)
        print()
    
    final_4 = [team for region in final_4 for team in region]
    print('Final 4')
    championship = []
    for i in range(0, len(final_4), 2):
        game = Game(season, final_4[i], final_4[i+1])
        print(game.predict_game())
        championship.append(game.get_winner())
        
    print('\nChampionship')
    game = Game(season, championship[0], championship[1])
    print(game.predict_game())
    
if __name__ == '__main__':
    start, end = int(sys.argv[1]), int(sys.argv[2])
    for season in range(start, end + 1):
        if season == 2020:
            continue
        with open(f'../tests/test{season}_seeds.txt', 'w') as file:
            original_stdout = sys.stdout
            sys.stdout = file
            predict_bracket(season)
            sys.stdout = original_stdout
