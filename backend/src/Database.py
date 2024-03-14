import numpy as np
import pandas as pd
import time


class TeamDatabase:
    """ 
    Used to store and access data about teams, seasons, and tournaments.
    """
    folder = '../../data/'
    seeds = pd.read_csv(folder + 'MNCAATourneySeeds.csv').set_index(['Season', 'TeamID'])
    conferences = pd.read_csv(folder + 'MTeamConferences.csv')
    tourney_compact = pd.read_csv(folder + 'MNCAATourneyCompactResults.csv')
    teams = pd.read_csv(folder + 'MTeams.csv')
    regular_season = None
    team_stats = None
    tourney_stats = None
    
    @classmethod
    def initialize(cls):
        cls.regular_season = cls.create_results(cls.folder + 'MRegularSeasonDetailedResults.csv')
        cls.team_stats = cls.generate_team_stats()
        cls.tourney_stats = cls.generate_tourney_stats()

    @staticmethod
    def create_results(file):
        df = pd.read_csv(file)
        
        # Assuming TeamDatabase.teams is accessible and formatted as expected
        team_id_to_name = TeamDatabase.teams.set_index('TeamID')['TeamName'].to_dict()
        
        # Map team IDs to team names directly
        df['WTeamName'] = df['WTeamID'].map(team_id_to_name)
        df['LTeamName'] = df['LTeamID'].map(team_id_to_name)
        
        return df
    
    @classmethod
    def get_seeds(cls):
        return cls.seeds
    
    @classmethod
    def get_regular_season(cls):
        return cls.regular_season
    
    @classmethod
    def get_team_stats(cls):
        return cls.team_stats
    
    @classmethod
    def get_tourney_stats(cls):
        return cls.tourney_stats
    
    @classmethod
    def generate_team_stats(cls):
        """
        Generate team stats from regular season results.
        """
        # Generate total team results dataframe
        WinTeams = pd.DataFrame()
        LossTeams = pd.DataFrame()

        columns = ['Season', 'TeamID', 'Points', 'OppPoints',
            'Loc', 'NumOT', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',
            'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'OppFGM', 'OppFGA',
            'OppFGM3', 'OppFGA3', 'OppFTM', 'OppFTA', 'OppOR', 'OppDR', 'OppAst', 'OppTO',
            'OppStl', 'OppBlk', 'OppPF']
        
        # set up wins dataframe
        WinTeams[columns] = cls.regular_season[['Season', 'WTeamID', 'WScore', 'LScore',
            'WLoc', 'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA',
            'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA',
            'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO',
            'LStl', 'LBlk', 'LPF']]
        WinTeams['Wins'] = 1
        WinTeams['Losses'] = 0

        # set up losses dataframe
        LossTeams[columns] = cls.regular_season[['Season', 'LTeamID', 'LScore', 'WScore',
            'WLoc', 'NumOT', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA',
            'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF', 'WFGM', 'WFGA',
            'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO',
            'WStl', 'WBlk', 'WPF']]
        LossTeams['Loc'] = LossTeams['Loc'].apply(lambda loc: 'A' if loc == 'H' else ('H' if loc == 'A' else 'N'))
        LossTeams['Wins'] = 0
        LossTeams['Losses'] = 1

        # set up full df
        AggTeams = pd.concat([WinTeams, LossTeams]).groupby(['Season', 'TeamID']).sum()
        AggTeams['NumGames'] = AggTeams['Wins'] + AggTeams['Losses']
        AggTeams = AggTeams.reset_index()
        
        AggTeams = pd.merge(AggTeams, cls.teams[['TeamID', 'TeamName']], on='TeamID', how='left')
        AggTeams = AggTeams.set_index(['Season', 'TeamID'])
        # Standardize it for per game stats
        RegSeasonStats = pd.DataFrame(index=AggTeams.index)
        RegSeasonStats['TeamName'] = AggTeams['TeamName']
        
        # Basic stats that depend on the total number of games
        total_stats = {
            'WinPct': ('Wins', 'NumGames'),
            'PointsPerGame': ('Points', 'NumGames'),
            'OppPointsPerGame': ('OppPoints', 'NumGames'),
            'PointsRatio': ('Points', 'OppPoints'),
            'OTPerGame': ('NumOT', 'NumGames'),
        }

        # Per game stats and percentages
        in_game = {
            'FG': ('FGM', 'FGA'),
            'OppFG': ('OppFGM', 'OppFGA'),
            '3Pt': ('FGM3', 'FGA3'),
            'Opp3Pt': ('OppFGM3', 'OppFGA3'),
            'FT': ('FTM', 'FTA'),
            'OR': ('OR', 'NumGames'),
            'DR': ('DR', 'NumGames'),
            'Ast': ('Ast', 'NumGames'),
            'TO': ('TO', 'NumGames'),
            'Stl': ('Stl', 'NumGames'),
            'Blk': ('Blk', 'NumGames'),
            'PF': ('PF', 'NumGames'),
        }

        # Calculate basic stats
        for stat, (numerator, denominator) in total_stats.items():
            RegSeasonStats[stat] = AggTeams[numerator] / AggTeams[denominator]

        # Calculate per game stats and percentages
        for stat_prefix, (made, attempted) in in_game.items():
            if 'NumGames' in attempted:  # For per game calculations
                RegSeasonStats[f'{stat_prefix}PerGame'] = AggTeams[made] / AggTeams[attempted]
            else:
                RegSeasonStats[f'{stat_prefix}PerGame'] = AggTeams[made] / AggTeams['NumGames']
                RegSeasonStats[f'{stat_prefix}Pct'] = AggTeams[made] / AggTeams[attempted]

        RegSeasonStats['OppFTPerGame'] = AggTeams['OppFTM'] / AggTeams['NumGames']
        
        return RegSeasonStats
    

    @classmethod
    def generate_tourney_stats(cls):
        """
        Generate tournament stats from tournament results.
        """
        # set winners
        winners =  pd.DataFrame()
        winners[['Season', 'Team1', 'Team2']] = cls.tourney_compact[['Season', 'WTeamID', 'LTeamID']]
        winners['Result'] = 1

        # set losers
        losers =  pd.DataFrame()
        losers[['Season', 'Team1', 'Team2']] = cls.tourney_compact[['Season', 'LTeamID', 'WTeamID']]
        losers['Result'] = 0

        TourneyGames = pd.concat([winners, losers])
        TourneyGames = TourneyGames[TourneyGames['Season'] >= 2003].reset_index(drop=True)
        
        # Get seeds and regions
        cls.seeds['SeedNum'] = cls.seeds['Seed'].str[1:3].astype(int)
        cls.seeds['Region'] = cls.seeds['Seed'].str[0]
    
        TourneyGames['Seed1'] = cls.seeds.loc[TourneyGames[['Season', 'Team1']].apply(tuple, axis=1), 'SeedNum'].values
        TourneyGames['Seed2'] = cls.seeds.loc[TourneyGames[['Season', 'Team2']].apply(tuple, axis=1), 'SeedNum'].values
        
        TourneyGames['Team1Region'] = cls.seeds.loc[TourneyGames[['Season', 'Team1']].apply(tuple, axis=1), 'Region'].values
        TourneyGames['Team2Region'] = cls.seeds.loc[TourneyGames[['Season', 'Team2']].apply(tuple, axis=1), 'Region'].values
        
        TourneyGames['Region'] = np.where(TourneyGames['Team1Region'] == TourneyGames['Team2Region'],
                                        TourneyGames['Team1Region'], 'FF')
        
        cls.seeds = cls.seeds.drop(columns=['SeedNum', 'Region'])
        TourneyGames = TourneyGames.drop(columns=['Team1Region', 'Team2Region'])
        
        return TourneyGames
    
    # Season queries
    @classmethod
    def season_results(cls, season):
        return cls.regular_season[cls.regular_season['Season'] == season]
    
    @classmethod
    def season_stats(cls, season):
        raw_df = cls.team_stats.reset_index()
        return raw_df[raw_df['Season'] == season].drop(columns=['Season'])



class Season:
    """
    Used to store all data about a season. Useful for generating predictions for one season.
    """
    def __init__(self, season: int):
        def create_season():
            # get results
            season_results = TeamDatabase.season_results(season)
            
            # get team stats
            season_stats = TeamDatabase.season_stats(season)
            
            return season_results, season_stats
        
        self.season = season
        self.results, self.stats = create_season()
    
    def get_season(self) -> int: return self.season    
    def get_results(self) -> pd.DataFrame: return self.results
    def get_stats(self) -> pd.DataFrame: return self.stats
    
    def get_seeds(self) -> pd.DataFrame:
        df = TeamDatabase.get_seeds().loc[self.season]
        df = df[~df['Seed'].str.contains('b')] # remove play-in games
        df['Region'] = df['Seed'].str[0]
        df['Seed'] = df['Seed'].str[1:3].astype(int)
        df = pd.merge(df, TeamDatabase.teams[['TeamID', 'TeamName']], on='TeamID', how='left')
        return df
    
"""
Used to store all data about a team in a season. Useful for generating predictions for one team in one season.
"""
class Team:
    def __init__(self, name: str, season: Season):
        def create_team():
            # get results
            season_results = season.get_results()
            team_results = season_results[(season_results['WTeamName'] == name) | (season_results['LTeamName'] == name)]
            # get team stats
            season_stats = season.get_stats()
            team_stats = season_stats[season_stats['TeamName'] == name]
            
            return team_results, team_stats
        
        self.name = name
        self.season = season
        self.results, self.stats = create_team()
    
    def get_name(self): return self.name
    def get_season(self): return self.season
    def get_results(self): return self.results
    def get_stats(self): return self.stats
    
    def get_seed(self):
        seed = self.season.get_seeds()[self.season.get_seeds()['TeamName'] == self.name]['Seed']
        return seed.values[0] if not seed.empty else 'X17' # placeholder
    
    def sparse_results(self):
        return self.results[['WTeamName', 'LTeamName', 'WScore', 'LScore']]      

start_time = time.time()
TeamDatabase.initialize()
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Successfully initialized database in {int(elapsed_time * 1000)} ms")