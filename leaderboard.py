'''This module defines a Leaderboard class that loads player CSV statistics
for a season from FanGraphs leaderboard CSV exports.'''

import csv

class Leaderboard:
    '''Defines an object Leaderboard for given season and level.'''

    def __init__(self, year, level):
        '''Initializes an instance of object Leaderboard by loading a
        leaderboard CSV in the 'data/' folder.'''

        self.year = year
        self.level = level

        filename = f"data/{year}{level}.csv"
        with open(file=filename, mode="r", encoding="utf-8-sig") as file:
            self.file = list(csv.DictReader(file))

        self.players_by_id = {row["PlayerId"]: row for row in self.file}
        self.players_by_name = {row["Name"]: row for row in self.file}

    def find_player(self, player_id):
        '''Return player statistics for a given PlayerId.'''

        return self.players_by_id.get(player_id)

    def find_player_by_name(self, name):
        '''Return player statistics for a given player name.'''

        return self.players_by_name.get(name)

    def __str__(self):
        '''Return a formatted table of players with columns Name, Age, and PA.'''

        string =  f'{"Name": 30}{"Age": 10}{"PA"}\n'
        for row in self.file:
            string += f'{row["Name"]: 30}{row["Age"]: 10}{row["PA"]}\n'
        return string
