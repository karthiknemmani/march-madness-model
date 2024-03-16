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
    kenpom = pd.read_csv(folder + 'MKenpomData.csv')
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
    def setup_team_stats(cls):
        columns = ['Season', 'TeamID', 'Points', 'OppPoints',
            'Loc', 'NumOT', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA',
            'OR', 'DR', 'Ast', 'TO', 'Stl', 'Blk', 'PF', 'OppFGM', 'OppFGA',
            'OppFGM3', 'OppFGA3', 'OppFTM', 'OppFTA', 'OppOR', 'OppDR', 'OppAst', 'OppTO',
            'OppStl', 'OppBlk', 'OppPF']
        
        WinTeams, LossTeams = cls.prepare_team_dataframes(columns)

        # Aggregate WinTeams and LossTeams into AggTeams
        AggTeams = pd.concat([WinTeams, LossTeams]).groupby(['Season', 'TeamID']).sum()
        AggTeams['NumGames'] = AggTeams['Wins'] + AggTeams['Losses']
        AggTeams.reset_index(inplace=True)
        
        # AggTeams = cls.calculate_strength_metrics(AggTeams)

        return AggTeams
    
    @classmethod
    def prepare_team_dataframes(cls, columns):
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
        
        return WinTeams, LossTeams
    
    @classmethod
    def calculate_strength_metrics(cls, AggTeams):
        # Strength of schedule raw data
        matches = pd.concat([
            cls.regular_season[['Season', 'WTeamID', 'LTeamID']].rename(columns={'WTeamID': 'TeamID', 'LTeamID': 'OpponentID'}),
            cls.regular_season[['Season', 'LTeamID', 'WTeamID']].rename(columns={'LTeamID': 'TeamID', 'WTeamID': 'OpponentID'})
        ])

        matches_with_stats = matches.merge(
            AggTeams, how='left', left_on=['Season', 'OpponentID'], right_on=['Season', 'TeamID']
        ).rename(columns={'Wins': 'OppWins', 'NumGames': 'OppNumGames'})

        opp_stats = matches_with_stats.groupby(['Season', 'TeamID_x']).agg(
            OppWins=pd.NamedAgg(column='OppWins', aggfunc='sum'),
            OppGames=pd.NamedAgg(column='OppNumGames', aggfunc='sum')
        ).reset_index()

        AggTeams = pd.merge(
            AggTeams, opp_stats, how='left', left_on=['Season', 'TeamID'], right_on=['Season', 'TeamID_x']
        ).drop(columns=['TeamID_x'])
        
        # Strength of victory raw data
        victory_matches = cls.regular_season[['Season', 'WTeamID', 'LTeamID']].rename(columns={'WTeamID': 'TeamID', 'LTeamID': 'OpponentID'})

        victory_matches_with_stats = victory_matches.merge(
            AggTeams[['Season', 'TeamID', 'Wins', 'NumGames']], how='left', left_on=['Season', 'OpponentID'], right_on=['Season', 'TeamID']
        ).rename(columns={'Wins': 'OppVictoryWins', 'NumGames': 'OppVictoryNumGames'})

        victory_stats = victory_matches_with_stats.groupby(['Season', 'TeamID_x']).agg(
            OppVictoryWins=pd.NamedAgg(column='OppVictoryWins', aggfunc='sum'),
            OppVictoryGames=pd.NamedAgg(column='OppVictoryNumGames', aggfunc='sum')
        ).reset_index()


        AggTeams = pd.merge(
            AggTeams, victory_stats, how='left', left_on=['Season', 'TeamID'], right_on=['Season', 'TeamID_x']
        ).drop(columns=['TeamID_x'])
        
        return AggTeams
        
    @classmethod
    def advanced_metrics(cls, AggTeams):
        RegSeasonStats = pd.DataFrame(index=AggTeams.index)
        RegSeasonStats['TeamName'] = AggTeams['TeamName']
        
        # Basic stats that depend on the total number of games
        total_stats = {
            'WinPct': ('Wins', 'NumGames'),
            'PointsPerGame': ('Points', 'NumGames'),
            'OppPointsPerGame': ('OppPoints', 'NumGames'),
            'PointsRatio': ('Points', 'OppPoints'),
            'OTPerGame': ('NumOT', 'NumGames'),
            # 'StrengthOfSchedule': ('OppWins', 'OppGames'),
            # 'StrengthOfVictory': ('OppVictoryWins', 'OppVictoryGames')
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
    def generate_team_stats(cls):
        """
        Generate team stats from regular season results.
        """
        AggTeams = cls.setup_team_stats()
        
        AggTeams = pd.merge(AggTeams, cls.teams[['TeamID', 'TeamName']], on='TeamID', how='left')
        
        # add kenpom rank

        AggTeams = pd.merge(AggTeams, cls.kenpom[['Season', 'Team', 'AdjEM', 'AdjO', 'AdjD', 'AdjT', 'Luck', 'AdjSOS', 'OppO', 'OppD', 'NCSOS']], left_on=['Season', 'TeamName'], right_on=['Season', 'Team'], how='left')

        AggTeams.drop(columns=['Team'], inplace=True)
        AggTeams = AggTeams.set_index(['Season', 'TeamID'])
        # Standardize it for per game stats
        RegSeasonStats = cls.advanced_metrics(AggTeams)
        
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
    
    @classmethod
    def season_tourney(cls, season):
        df = cls.tourney_compact[cls.tourney_compact['Season'] == season]

        # Winning teams
        df = pd.merge(df, cls.teams[['TeamID', 'TeamName']], left_on='WTeamID', right_on='TeamID', how='left')
        df.rename(columns={'TeamName': 'WTeamName'}, inplace=True)
        df.drop('TeamID', axis=1, inplace=True)

        # Losing teams
        df = pd.merge(df, cls.teams[['TeamID', 'TeamName']], left_on='LTeamID', right_on='TeamID', how='left')
        df.rename(columns={'TeamName': 'LTeamName'}, inplace=True)
        df.drop('TeamID', axis=1, inplace=True)

        return df



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
        self.tourney = TeamDatabase.season_tourney(season)
    
    def get_season(self) -> int: return self.season    
    def get_results(self) -> pd.DataFrame: return self.results
    def get_stats(self) -> pd.DataFrame: return self.stats
    def get_tourney(self) -> pd.DataFrame: return self.tourney
    
    def get_seeds(self) -> pd.DataFrame:
        df = TeamDatabase.get_seeds().loc[self.season].reset_index()
        # write code here
        play_in_teams = self.tourney.head(4)['LTeamID'].tolist()

        df = df[~df['TeamID'].isin(play_in_teams)]
        df['Region'] = df['Seed'].str[0]
        df['Seed'] = df['Seed'].str[1:3].astype(int)
        df = pd.merge(df, TeamDatabase.teams[['TeamID', 'TeamName']], on='TeamID', how='left')
        return df
    

class Team:
    """
    Used to store all data about a team in a season. Useful for generating predictions for one team in one season.
    """
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

# start_time = time.time()
TeamDatabase.initialize()
# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Successfully initialized database in {int(elapsed_time * 1000)} ms")