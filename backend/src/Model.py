from Database import TeamDatabase, Season, Team
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from joblib import dump
import pandas as pd
import time


class Model:
    
    def __init__(self, season):
        self.X_train_scaled, self.X_test_scaled, self.y_train, self.y_test = None, None, None, None
        
    
    # logistic_params = {
    #     'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
    #     'penalty': ['l1', 'l2'],  # Norm used in the penalization
    #     'solver': ['liblinear', 'saga'],  # Algorithm to use in the optimization problem
    #     'max_iter': [100, 200, 500]  # Maximum number of iterations taken for the solvers to converge
    # }
    # random_forest_params = {
    #     'n_estimators': [100, 200],  # Number of trees
    #     'max_depth': [None, 10, 20],  # Maximum depth of the tree
    #     'min_samples_split': [2, 5],  # Minimum number of samples required to split an internal node
    #     'min_samples_leaf': [1, 4],  # Minimum number of samples required to be at a leaf node
    #     'bootstrap': [True, False],  # Method for sampling data points
    # }
    
        self.lr = LogisticRegression(C=1, max_iter=100, penalty='l1', solver='liblinear')
        # self.lr = LogisticRegression(solver='liblinear')
        self.nb = GaussianNB()
        self.rf = RandomForestClassifier()
        # self.rf = RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1, bootstrap=True)
        # rf_grid = GridSearchCV(rf, random_forest_params, cv=5, n_jobs=-1)
        
        # lr_grid = GridSearchCV(lr, logistic_params, cv=5, n_jobs=-1)
        self.scaler = StandardScaler()
        
        self.season = season
    
    def initialize(self):
        # split and scale data
        # print('preparing')
        self.prepare()
        # fit the models
        self.lr.fit(self.X_train_scaled, self.y_train)
        self.rf.fit(self.X_train_scaled, self.y_train)
        self.nb.fit(self.X_train_scaled, self.y_train)
        

    def prepare(self):
        tourney_games = TeamDatabase.get_tourney_stats()
        reg_season = TeamDatabase.get_team_stats().reset_index()
        
        diff_columns = [col for col in reg_season.columns if col not in ['Season', 'TeamID', 'TeamName']]

        # Optimizing the merge by reducing it to necessary columns first
        necessary_columns = ['Season', 'TeamID'] + diff_columns
        reg_season_reduced = reg_season[necessary_columns]

        # Merge tourney games with reg_season for both teams in one go if possible
        # Assuming this cannot be further optimized without changing data structure
        team1_stats = pd.merge(tourney_games, reg_season_reduced, left_on=['Season', 'Team1'], right_on=['Season', 'TeamID'], how='left')
        team2_stats = pd.merge(tourney_games, reg_season_reduced, left_on=['Season', 'Team2'], right_on=['Season', 'TeamID'], how='left')

        # Calculating differences in one go
        tourney_stats = (team1_stats[diff_columns].values - team2_stats[diff_columns].values)
        tourney_stats_df = pd.DataFrame(tourney_stats, columns=diff_columns)
        tourney_stats_df['Seed'] = tourney_games['Seed1'] - tourney_games['Seed2']
        tourney_stats_df['Result'] = tourney_games['Result']
        tourney_stats_df['Season'] = tourney_games['Season']
        
        # Exclude the current season's data for training
        training_data = tourney_stats_df[tourney_stats_df['Season'] != self.season]
        X = training_data.drop(['Result', 'Season'], axis=1)
        y = training_data['Result']
        
        # Split and scale
        X_train, X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=1)
        self.X_train_scaled = self.scaler.fit_transform(X_train)
        self.X_test_scaled = self.scaler.transform(X_test)

        
        # print('before fit')
        # cls.rf_grid.fit(cls.X_train_scaled, cls.y_train)
        # print('best params: ', cls.rf_grid.best_params_)
    
    def score(self):
        return self.lr.score(self.X_test_scaled, self.y_test), self.rf.score(self.X_test_scaled, self.y_test), self.nb.score(cls.X_test_scaled, cls.y_test)

    def cross_val(self, folds):
        return cross_val_score(self.lr, self.X_train_scaled, self.y_train, cv=folds)
    
    def classification_report(self):
        y_pred = self.lr.predict(self.X_test_scaled)
        return classification_report(self.y_test, y_pred)
    
class Game:
    def __init__(self, model: Model, season: Season, team1: str, team2: str):
        self.season = season
        self.teams = (team1, team2)
        self.team1 = Team(team1, self.season)
        self.team2 = Team(team2, self.season)
        self.model = model
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
        team1_stats['Seed'] = self.team1.get_seed() if self.season.get_season() not in {2020, 2024} else 0
        team2_stats['Seed'] = self.team2.get_seed() if self.season.get_season() not in {2020, 2024} else 0
        # team1_stats['Seed'] = 0
        # team2_stats['Seed'] = 0
        # Scale data
        diff = pd.DataFrame(team1_stats.values - team2_stats.values, columns=team1_stats.columns.values)
        diff_scaled = self.model.scaler.transform(diff)
        
        # Predict
        prob = self.model.lr.predict_proba(diff_scaled)
        
        team1_prob = prob[0][1]
        team2_prob = prob[0][0]
        
        self.winner = self.teams[0] if team1_prob > team2_prob else self.teams[1]
        winning_prob = team1_prob if team1_prob > team2_prob else team2_prob
        
        if team1_prob > team2_prob:
            upset = self.team1.get_seed() > self.team2.get_seed()
        else:
            upset = self.team2.get_seed() > self.team1.get_seed()
            
        
        return self.winner, round(winning_prob, 4), upset

    def __str__(self):
        return f'{self.teams[0]} vs. {self.teams[1]}'
        
        
    

# start_time = time.time()
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Successfully initialized model in {int(elapsed_time * 1000)} ms")


# game = Game(2023, 'Purdue', 'Virginia')
# print(game.predict_game())

# g = Game(2023, 'Texas', 'Houston')
# print(g.predict_game())

# print(Model.score(Model.rf))
# print(Model.score())
# print(Model.score(Model.nb))
# print(Model.cross_val(Model.rf, 5))
# print(Model.cross_val(Model.lr, 5))
# print(Model.cross_val(Model.nb, 5))
# print(Model.classification_report(Model.rf))
# print(Model.classification_report(Model.lr))
# print(Model.classification_report(Model.nb))