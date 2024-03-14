from Database import TeamDatabase, Season, Team
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from joblib import dump
import pandas as pd
import time


class Model:
    
    X_train_scaled, X_test_scaled, y_train, y_test = None, None, None, None
    lr = LogisticRegression(random_state=1)
    scaler = MinMaxScaler()
    
    @classmethod
    def initialize(cls):
        # split and scale data
        cls.prepare()
        # fit the models
        cls.lr.fit(cls.X_train_scaled, cls.y_train)
        

    @classmethod
    def prepare(cls):
        # generate model
        tourney_games = TeamDatabase.get_tourney_stats()
        reg_season = TeamDatabase.get_team_stats().reset_index()

        # Merge tourney games with reg_season for both teams
        # This requires reg_season to have 'Season' and 'TeamID' columns after reset_index
        team1_stats = pd.merge(tourney_games, reg_season, left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left')
        team2_stats = pd.merge(tourney_games, reg_season, left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left')

        # Calculate differences between team stats, excluding non-numeric and non-relevant columns
        diff_columns = [col for col in reg_season.columns if col not in ['Season', 'TeamID', 'TeamName']]
        tourney_stats = pd.DataFrame()

        for col in diff_columns:
            tourney_stats[col] = team1_stats[col] - team2_stats[col]

        # Calculate seed difference and include game result
        tourney_stats['Seed'] = tourney_games['Seed1'] - tourney_games['Seed2']
        tourney_stats['Result'] = tourney_games['Result']
    
        # get database
        X = tourney_stats.drop('Result', axis=1)
        y = tourney_stats['Result']
        
        # split data
        X_train, X_test, cls.y_train, cls.y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        
        # scale data
        cls.X_train_scaled = cls.scaler.fit_transform(X_train)
        cls.X_test_scaled = cls.scaler.transform(X_test)
    
    @classmethod
    def score(cls):
        return cls.lr.score(cls.X_test_scaled, cls.y_test)
    
    @classmethod
    def cross_val(cls, folds):
        return cross_val_score(cls.lr, cls.X_train_scaled, cls.y_train, cv=folds)
    
    @classmethod
    def classification_report(cls):
        y_pred = cls.lr.predict(cls.X_test_scaled)
        return classification_report(cls.y_test, y_pred)
    
class Game:
    def __init__(self, season: Season, team1: str, team2: str):
        self.season = season
        self.teams = (team1, team2)
        self.team1 = Team(team1, self.season)
        self.team2 = Team(team2, self.season)
        self.winner = None
    
    def get_season(self): return self.season.get_season()
    def get_team1(self): return self.teams[0]
    def get_team2(self): return self.teams[1]
    
    def get_winner(self):
        assert self.winner is not None
        return self.winner
    
    def display_stats(self):
        print(f'{self.teams[0]} Stats:', repr(self.team1.get_stats()), sep='\n')
        print(f'{self.teams[1]} Stats:', repr(self.team2.get_stats()), sep='\n')
    
    def predict_game(self):
        # Get stats
        team1_stats = self.team1.get_stats().drop(['TeamID', 'TeamName'], axis=1)
        team2_stats = self.team2.get_stats().drop(['TeamID', 'TeamName'], axis=1)
        
        # Get seeds
        team1_stats['Seed'] = self.team1.get_seed()
        team2_stats['Seed'] = self.team2.get_seed()
        # team1_stats['Seed'] = 0
        # team2_stats['Seed'] = 0
        # Scale data
        diff = pd.DataFrame(team1_stats.values - team2_stats.values, columns=team1_stats.columns.values)
        diff_scaled = Model.scaler.transform(diff)
        
        # Predict
        prob = Model.lr.predict_proba(diff_scaled)
        
        team1_prob = prob[0][1]
        team2_prob = prob[0][0]
        
        self.winner = self.teams[0] if team1_prob > team2_prob else self.teams[1]
        winning_prob = team1_prob if team1_prob > team2_prob else team2_prob
        
        return self.winner, round(winning_prob, 4)
        
        
    

start_time = time.time()
Model.initialize()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Successfully initialized model in {int(elapsed_time * 1000)} ms")


# game = Game(2023, 'Purdue', 'Virginia')
# print(game.predict_game())

# g = Game(2023, 'Texas', 'Houston')
# print(g.predict_game())

# print(Model.score(Model.rf))
# print(Model.score(Model.lr))
# print(Model.score(Model.nb))
# print(Model.cross_val(Model.rf, 5))
# print(Model.cross_val(Model.lr, 5))
# print(Model.cross_val(Model.nb, 5))
# print(Model.classification_report(Model.rf))
# print(Model.classification_report(Model.lr))
# print(Model.classification_report(Model.nb))