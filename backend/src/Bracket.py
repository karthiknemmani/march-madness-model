from Database import Season
from Model import Game

class Region:
    """
    Generates a bracket for a region for a given season, with predictions.
    """
    def __init__(self, season: Season, region: str):
        self.season = season
        self.region = region