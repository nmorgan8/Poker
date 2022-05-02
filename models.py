import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import argparse

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor
from matplotlib import cm

PLAYER_NAME = 'IlxxxlI'

def write_data():
    files = ['data/File194.txt', 'data/File195.txt', 'data/File196.txt', 'data/File198.txt', 'data/File199.txt', 'data/File200.txt', 'data/File201.txt', 'data/File203.txt', 'data/File204.txt']
    with open('combined_data.txt', 'w') as combined:
        for file in files:
            f = open(file, 'r')
            for line in f:
                combined.write(line)

def getgames(file):
    """
    Helper method that groups data into each game
    """
    for match in re.finditer(r'Game started at.*?Game ended at.*?\n\n', file, re.MULTILINE + re.DOTALL):
        yield match.group(0)
# next().group(0)

# isolate into part
def separate(game):
    """
    Helper method that seperates the data from each game
    """
    match = re.match(r'Game started at: (.*)Game ID: (.*?)\n(.*)(Player.*)------ Summary ------(.*)Game ended at: (.*)', game, re.MULTILINE + re.DOTALL)
    try:
        start, gameid, playerstartmoney, actions, summary, end = match.groups()
        return start.strip(), gameid.strip(), playerstartmoney, actions, summary, end.strip()
    except AttributeError:
        return game, "", "", "", "", ""

def process_data(file='combined_data.txt'):
    files_games = []
    with open(file, 'r') as f:
        games = getgames(''.join(f.readlines()))
        file_games = []
        for game in games:
            parts = separate(game)
            parts = list(parts)
            parts.append(f.name)
            file_games.append(parts)
    files_games.extend(file_games)
    df = (pd.DataFrame(files_games, columns=['Start', 'ID', 'Money', 'Actions', 'Summary', 'End', 'File'])
            .assign(NPlayer=lambda df:df.Money.str.count('Seat')-1)
        )
    return df

def extract_data(df):
    card_to_idx = {
      '2': 0,
      '3': 1,
      '4': 2,
      '5': 3,
      '6': 4, 
      '7': 5, 
      '8': 6, 
      '9': 7, 
      '10': 8, 
      'J': 9, 
      'Q': 10, 
      'K': 11,
      'A': 12 
    }
    x = []
    y = []
    no_flop, games = 0, 0
    for i, (start, id, money, actions, summary, end, file, nplayer) in df.iterrows():
        cards = re.findall('^.*\*\*\*.*$', money, re.MULTILINE)
        player_info = re.findall(f'^.*{PLAYER_NAME}.*$', money,re.MULTILINE) # list of str
        player_sum = re.findall(f'^.*{PLAYER_NAME}.*$', summary,re.MULTILINE)[0] # str
        final_board = re.findall('\[.*\]', summary, re.MULTILINE) # list of str
        # players_start_count = re.findall('Seat.*$', money, re.MULTILINE)
        # button_num = int(players_start_count[0][5:7])
        # num_players = len(players_start_count) - 1

        sample = []

        

        # get starting hands
        info = " ".join(player_info)
        cards = re.findall('\[(.*?)\]', info)
        if cards: 
            suits = []
            for card in cards:
                suit = card[-1]
                val = card[:-1]
                sample.append(val)
                suits.append(suit)
            if suits[0] == suits[1]:
                sample.append(1)
            else:
                sample.append(0)

            # get outcome of each hand
            res = re.findall('Wins: .*$|Loses: .*$', player_sum)[0].split(' ')

            x.append(sample)

            if 'L' in res[0]: # lost money
                y.append(float(res[1][0:-1]) * -1)
            elif 'W' in res[0]: # won money
                y.append(float(res[1][0:-1]))
            elif float(res[1]) == 0.0: # if fold without losing money
                x.pop()

            if not final_board:
                no_flop += 1
            games+=1
            
    return x, y, card_to_idx

def tokenize_cards(x, card_to_idx):
    X = []
    for row in x:
        X.append([card_to_idx[row[0]], card_to_idx[row[1]], row[2]])
    X = np.array(X)
    return X
    
class Regression():
    def __init__(self, df):
        self.X, self.y, self.word_to_ix = extract_data(df)
        self.X = tokenize_cards(self.X, self.word_to_ix)
        self.y = np.array(self.y)
        self.model = LinearRegression()
        self.train()

    def train(self):
        self.model.fit(self.X, self.y)

    def evaluate(self):
        start_hands = []
        start_hands_num = []
        poss_cards = list(self.word_to_ix.keys())
        poss_cards_num = list(self.word_to_ix.values())
        for i in range(len(poss_cards)):
            for j in range(len(poss_cards)):
                for suit in range(2):
                    if suit == 1 and poss_cards_num[i] == poss_cards_num[j]:
                        continue
                    else:
                        start_hands.append([poss_cards[i], poss_cards[j], suit])
                        start_hands_num.append([poss_cards_num[i], poss_cards_num[j], suit])

        self.hand_expected_values = {}
        for hand, num in zip(start_hands, start_hands_num):
            self.hand_expected_values[str(hand)] = self.model.predict([num]) * 100
        return self.hand_expected_values

    def plot(self):
        for key in self.hand_expected_values:
            key_values = key.strip("'").strip('][').split(', ')
            self.hand_expected_values[key]



class KMeansModel():
    def __init__(self, df, clusters=3):
        self.X, self.y, self.word_to_ix = extract_data(df)
        self.X = tokenize_cards(self.X, self.word_to_ix)
        self.y = np.array(self.y)
        self.data = np.c_[self.X, self.y]
        self.model = KMeans(n_clusters=clusters)
        self.train()

    def train(self):
        self.model.fit(self.data)

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(45, 60)
        plot_data = self.data.T
        labels = self.model.labels_

        labeled_data = np.vstack([plot_data, labels])

        ax.scatter(labeled_data[0], labeled_data[1], labeled_data[3], c=labeled_data[4])
        plt.show()

class MLP():
    def __init__(self, df):
        self.X, self.y, self.word_to_ix = extract_data(df)
        self.X = tokenize_cards(self.X, self.word_to_ix)
        self.y = np.array(self.y)
        self.model = MLPRegressor()
        self.train()

    def train(self):
        self.model.fit(self.X, self.y)
        return

    def evaluate(self):
        start_hands = []
        start_hands_num = []
        poss_cards = list(self.word_to_ix.keys())
        poss_cards_num = list(self.word_to_ix.values())
        for i in range(len(poss_cards)):
            for j in range(len(poss_cards)):
                for suit in range(2):
                    if suit == 1 and poss_cards_num[i] == poss_cards_num[j]:
                        continue
                    else:
                        start_hands.append([poss_cards[i], poss_cards[j], suit])
                        start_hands_num.append([poss_cards_num[i], poss_cards_num[j], suit])

        self.hand_expected_values = {}
        for hand, num in zip(start_hands, start_hands_num):
            self.hand_expected_values[str(hand)] = self.model.predict([num]) * 100
        return self.hand_expected_values

    def plot(self):

        for key in self.hand_expected_values:
            key_values = key.strip("'").strip('][').split(', ')
            self.hand_expected_values[key]

        rows, cols = 13, 13
        fig, ax = plt.subplots(rows, cols,
                            sharex='col', 
                            sharey='row')

        poss_cards = list(self.word_to_ix.keys())

        for row in reversed(range(rows)):
            for col in reversed(range(cols)):

                ax[row, col].text(0.5, 0.5, 
                                f'{poss_cards[row]}{poss_cards[col]}',
                                color="green",
                                fontsize=8, 
                                ha='center')


        plt.show()

def main(args):
    # Extract and preprocess data
    df = process_data()
    
    if args.m == 'reg':
        reg = Regression(df)
        print(reg.evaluate())
    
    if args.m == 'kmeans':
        kmean = KMeansModel(df)
        kmean.plot()

    if args.m == 'mlp':
        mlp = MLP(df)
        mlp.plot()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', default='kmeans', choices=['reg', 'kmeans', 'mlp'])

    args = parser.parse_args()
    main(args)

    