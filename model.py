'''This module trains a PyTorch neural network to project MLB hitting outcomes
from Minor League statistics using historical player data and allows for predictions
given singular players.'''

import random
import torch
import torch.nn as nn
import torch.optim as optim
from leaderboard import Leaderboard

class Model:
    '''Defines an object Model for a given number of years, a given MiLB level,
    and a given number of simulations.'''

    def __init__(self, years, level, num_sims):
        '''Initializes an instance of object Model with neural network
        settings and parameters being set.'''

        self.years = years
        self.level = level
        self.num_sims = num_sims
        self.milb_leaderboards, self.mlb_leaderboards = self.init_leaderboards()

        self.model = nn.Sequential(
            nn.Linear(7, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 6)
        )

        self.training_mean = None
        self.training_std = None
        self.loss_fx = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.train_model()

    def init_leaderboards(self):
        '''Creates a list with all applicable MiLB leaderboards and a list
        with all applicable MLB leaderboards.'''

        START_YEAR = 2015
        CURRENT_YEAR = 2025

        milb_leaderboards = []
        for year in range(START_YEAR, CURRENT_YEAR - self.years):
            if year == 2020:
                continue
            milb_leaderboards.append(Leaderboard(year, self.level))

        mlb_leaderboards = []
        for year in range(START_YEAR + self.years, CURRENT_YEAR):
            mlb_leaderboards.append(Leaderboard(year, "MLB"))

        return milb_leaderboards, mlb_leaderboards

    def train_model(self):
        '''Trains the neural network for a specified number of epochs.'''
        self.model.train()

        for epoch in range(self.num_sims):
            input_tensor, output_tensor = self.build_tensors()

            if input_tensor is None:
                continue

            prediction = self.model(input_tensor)
            prediction = torch.softmax(prediction, dim=1)
            loss = self.loss_fx(prediction, output_tensor)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 1000 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item(): .6f}")

    def predict(self, name, board):
        '''Prints the MLB statline of a provided player using the neural
        network.'''

        self.model.eval()

        player = board.find_player_by_name(name)
        input_tensor = [
            float(player["Age"]), float(player["GB/FB"]),
            float(player["LD%"]), float(player["HR/FB"]),
            float(player["SwStr%"]), float(player["BB%"]),
            float(player["K%"])
        ]

        input_tensor = torch.tensor([input_tensor], dtype=torch.float32)
        input_tensor = self.standardize_milb_stats(input_tensor, True)

        with torch.no_grad():
            output_tensor = self.model(input_tensor)
            output_tensor = torch.softmax(output_tensor, dim=1)

        avg, obp, slg, ops, hr = self.adjust_output_tensor(output_tensor.squeeze().tolist())

        print(f"{str(name).upper()}\nAVG: {avg: .3f}\nOBP: {obp: .3f}")
        print(f"SLG: {slg: .3f}\nOPS: {ops: .3f}\nHR: {hr: .0f}")

    def build_tensors(self):
        '''Builds an input and output tensor, with the input consisting of
        MiLB statistics and the output consisting of the target MLB statistics.'''

        random_milb_board, random_mlb_board = self.create_leaderboard_pair()
        players_in = []
        players_out = []

        for row in random_milb_board.file:
            target_row = random_mlb_board.find_player(row["PlayerId"])

            if target_row is not None:
                target_row = self.adjust_mlb_stats(target_row)
                players_in.append(
                    [
                        float(row["Age"]), float(row["GB/FB"]),
                        float(row["LD%"]), float(row["HR/FB"]),
                        float(row["SwStr%"]), float(row["BB%"]),
                        float(row["K%"])
                    ]
                )

                players_out.append(
                    [
                        target_row["1B%"], target_row["2B%"],
                        target_row["3B%"], target_row["HR%"],
                        target_row["BB%"], target_row["Out%"]
                    ]
                )

        if not players_in or not players_out:
            return None, None

        season_in = torch.tensor(players_in, dtype=torch.float32)
        season_out = torch.tensor(players_out, dtype=torch.float32)
        season_in = self.standardize_milb_stats(season_in)

        return season_in, season_out

    def create_leaderboard_pair(self):
        '''Creates a pairing of an MiLB leaderboard and a MLB leaderboard.'''

        random_milb_board = random.choice(self.milb_leaderboards)

        for random_mlb_board in self.mlb_leaderboards:
            if random_mlb_board.year == random_milb_board.year + self.years:
                return random_milb_board, random_mlb_board

        raise ValueError("No matching MLB leaderboard found.")

    def standardize_milb_stats(self, milb_stats, is_eval=False):
        '''Places age and GB/FB on a curve and standardizes all statistics.'''

        age = milb_stats[:,0]
        gb_fb = milb_stats[:,1]
        ld = milb_stats[:,2]
        hr_fb = milb_stats[:,3]
        sw_str = milb_stats[:,4]
        bb = milb_stats[:,5]
        k = milb_stats[:,6]

        PRIME_AGE = 27
        OPT_GB_FB = 1
        age_curve = (age - PRIME_AGE) ** 2
        gb_fb_curve = (gb_fb - OPT_GB_FB) ** 2

        new_stats = torch.stack([age_curve, gb_fb_curve, ld, hr_fb,
                                 sw_str, bb, k], dim=1)

        if not is_eval:
            self.training_mean = new_stats.mean(dim=0)
            self.training_std = new_stats.std(dim=0)
            self.training_std[self.training_std == 0] = 1

        z_new_stats = (new_stats - self.training_mean) / self.training_std

        return z_new_stats

    def adjust_mlb_stats(self, season):
        '''Transforms counting statistics from an MLB leaderboard into rate
        statistics, with all uncategorizable outcomes falling under Out%.'''

        pa = float(season["PA"])
        singles = float(season["1B"])
        doubles = float(season["2B"])
        triples = float(season["3B"])
        hr = float(season["HR"])
        bb = float(season["BB"])

        adjusted_season = {
            "Name": season["Name"],
            "1B%": singles / pa,
            "2B%": doubles / pa,
            "3B%": triples / pa,
            "HR%": hr / pa,
            "BB%": bb / pa,
            "Out%": (pa - singles - doubles - triples - hr - bb) / pa
        }

        return adjusted_season

    def adjust_output_tensor(self, tensor):
        '''Transforms the probability of plate appearance outcomes into
        easily interpretable counting statistics.'''

        PA = 650
        bb_rate = tensor[4]
        ab = PA * (1 - bb_rate)

        single_rate = tensor[0]
        singles = single_rate * PA
        double_rate = tensor[1]
        doubles = double_rate * PA
        triple_rate = tensor[2]
        triples = triple_rate * PA
        hr_rate = tensor[3]
        hr = hr_rate * PA

        avg = (singles + doubles + triples + hr) / ab
        obp = single_rate + double_rate + triple_rate + hr_rate + bb_rate
        slg = (singles + 2 * doubles + 3 * triples + 4 * hr) / ab
        ops = obp + slg

        return avg, obp, slg, ops, hr
