'''This module runs the program by providing a command-line interface
that allows the user to input parameters for training the neural network and
predict the stats of a MiLB player from 2015-2025 according to the neural network.'''

from leaderboard import Leaderboard
from model import Model

def main():
    '''Runs command-line interface for training and player projection.'''
    is_valid_input = False

    while not is_valid_input:
        try:
            years = int(input("To how many years should the model project out? (0-3)\n>> ").strip())
            if years < 0 or years > 3:
                raise ValueError
            is_valid_input = True
        except ValueError:
            print("Please input a valid amount of years.")

    is_valid_input = False
    while not is_valid_input:
        level = input("From which starting MiLB level?\n>> ").strip().upper()
        if level in ["AA","AAA"]:
            is_valid_input = True
        else:
            print("Please input a valid MiLB level (e.g. AA, AAA).")

    is_valid_input = False
    while not is_valid_input:
        try:
            num_sims = int(input("How many simulations?\n>> ").strip())
            if num_sims <= 0:
                raise ValueError
            is_valid_input = True
        except ValueError:
            print("Please input a valid number of simulations.")

    print(f"Initializing {years}-year model for {level} to MLB...")
    model = Model(years, level, num_sims)

    command = ""
    while command.lower().strip() != "exit":
        try:
            command = input(f"Pick one {level} season between 2015 and 2025.\n>> ")
            if command.lower().strip() == "exit":
                continue
            board = Leaderboard(int(command), level)
        except ValueError:
            print("Please input a valid season between 2015 and 2025.")
            continue
        except FileNotFoundError:
            print("Please input a valid season between 2015 and 2025.")
            continue

        try:
            print(board)
            command = input("Pick a name from this leaderboard to analyze.\n>> ").strip()
            if command.lower().strip() == "exit":
                continue
            model.predict(command, board)
        except TypeError:
            print("Name not found in leaderboard.")

if __name__ == "__main__":
    main()
