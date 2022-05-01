import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import argparse

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans, MeanShift

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

        for hand, num in zip(start_hands, start_hands_num):
            print(hand, self.model.predict([num]) * 100)
        return

    def plot():
        pass


class KMeansModel():
    def __init__(self, df, clusters=3):
        self.X, self.y, self.word_to_ix = extract_data(df)
        self.X = tokenize_cards(self.X, self.word_to_ix)
        self.y = np.array(self.y)
        self.model = KMeans(n_clusters=clusters)
        self.train()

    def train(self):
        print(self.X)
        print(self.y)
        pass

    def evaluate(self):
        print('kmeans eval')

class DensityEstimation():
    def __init__(self, df):
        self.df = df

    def evaluate(self):
        print('density eval')

def main(args):
    # Extract and preprocess data
    df = process_data()
    
    if args.m == 'reg':
        model = Regression(df)
        model.evaluate()
    
    if args.m == 'kmeans':
        model = KMeansModel(df)
        model.evaluate

    if args.m == 'density':
        model = DensityEstimation(df)
        model.evaluate



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', default='all', choices=['reg', 'kmeans', 'density', 'all'])

    args = parser.parse_args()
    main(args)

    