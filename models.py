import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import argparse

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.neural_network import MLPRegressor

PLAYER_NAME = 'IlxxxlI' # Player username

def write_data():
    """
    Helper method to combine all raw data files into single file and write to 'combined_data.txt'

        Params:
            None

        Returns:
            None
    """
    files = ['data/File194.txt', 'data/File195.txt', 'data/File196.txt', 'data/File198.txt', 'data/File199.txt', 'data/File200.txt', 'data/File201.txt', 'data/File203.txt', 'data/File204.txt']
    with open('combined_data.txt', 'w') as combined:
        for file in files:
            f = open(file, 'r')
            for line in f:
                combined.write(line)

def getgames(file):
    """
    Helper method that groups data into each hand

        Params:
            file (str): raw string data from 'combined_data.txt'
        
        Returns:
            None
    """
    for match in re.finditer(r'Game started at.*?Game ended at.*?\n\n', file, re.MULTILINE + re.DOTALL):
        yield match.group(0)
# next().group(0)

# isolate into part
def separate(game):
    """
    Helper method that seperates each hand into partitioned components and make more usable

        Params:
            game (str): raw string data for single hand

        Returns:
            start (str): date and time of start of the game
            gameid (str): unique id for the particular game
            playerstartmoney (str): seat numbers of the players and starting stack counts
            actions (str): different actions (cards dealt, players fold/bet, etc) from start fo end of hand
            summary (str): summary of all players winnings/lossings for the hand
            end (str): date and time of end of the game
    """
    match = re.match(r'Game started at: (.*)Game ID: (.*?)\n(.*)(Player.*)------ Summary ------(.*)Game ended at: (.*)', game, re.MULTILINE + re.DOTALL)
    try:
        start, gameid, playerstartmoney, actions, summary, end = match.groups()
        return start.strip(), gameid.strip(), playerstartmoney, actions, summary, end.strip()
    except AttributeError:
        return game, "", "", "", "", ""

def process_data(file='combined_data.txt'):
    """
    Method to extract data from text file and format into a pandas DataFrame

        Params:
            file (str): name of file to be processed

        Returns:
            df (pandas DataFrame): DataFrame containing all data from seperate()
    """
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
    """
    Method to extract data from preprocessed text

        Params:
            df (pandas DataFrame): DataFrame containing all returned data from seperate()

        Returns:
            x (numpy ndarray): data used to train/test various models
            y (numpy ndarray): label data for the expected net gain/loss
            card_to_idx (dict): dictionary mapping card string value to number
    """
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
    """
    Method to convert cards from str to numerical representation

        Params:
            x (numpy ndarray): numpy array containing all sample data
            card_to_idx (dict): dictionary mapping card string value to number

        Returns:
            X (numpy ndarray): numpy array containing all converted sample data
    """
    X = []
    for row in x:
        X.append([card_to_idx[row[0]], card_to_idx[row[1]], row[2]])
    X = np.array(X)
    return X
    
class Regression():
    """
    Regression class that builds and models baseline Linear Regression
    """
    def __init__(self, df):
        """
        Constructor to instantiate necessary information and data
        """
        self.X, self.y, self.word_to_ix = extract_data(df)
        self.X = tokenize_cards(self.X, self.word_to_ix)
        self.y = np.array(self.y)
        self.model = LinearRegression()
        self.train()

    def train(self):
        """
        Method to fit the model
        """
        self.model.fit(self.X, self.y)

    def evaluate(self):
        """
        Method to format and predict outcomes based on trained model
        """
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
        """
        Plot expected outcome of various hands 
        """

        self.evaluate()
        plt.axis('off')

        # print(self.hand_expected_values)
        # for key in self.hand_expected_values:
        #     key_values = key.strip("'").strip('][').split(', ')
        #     print(key, self.hand_expected_values[key])

        rows, cols = 13, 13
        fig, ax = plt.subplots(rows, cols,
                            sharex='col', 
                            sharey='row')

        poss_cards = list(self.word_to_ix.keys())

        for row in range(rows):
            for col in range(cols):
                ax[row, col].spines['top'].set_visible(False)
                ax[row, col].spines['right'].set_visible(False)
                ax[row, col].spines['bottom'].set_visible(False)
                ax[row, col].spines['left'].set_visible(False)

                ax[row, col].set_xticks([])
                ax[row, col].set_yticks([])

                if row == 0:
                    ax[row, col].set_title(poss_cards[12 - col])
                if col == 0:
                    ax[row, col].set_ylabel(poss_cards[12 - row])

                key = [poss_cards[12 - row], poss_cards[12 - col]]
                if row >= col: # below y = -x

                    key.append(0)
                    if self.hand_expected_values[str(key)][0] > 10:
                        color = 'green'
                    elif self.hand_expected_values[str(key)][0] < -10:
                        color = 'red'
                    else:
                        color = 'orange'

                    ax[row, col].set_facecolor(color)
                    ax[row, col].text(0.5, 0.15, 
                                    f'{poss_cards[12 - row]}{poss_cards[12 - col]}o\n{round(self.hand_expected_values[str(key)][0], 1)}',
                                    # color=color,
                                    fontsize=8, 
                                    ha='center')
                else: # above y = -x

                    key.append(1)
                    if self.hand_expected_values[str(key)][0] > 10:
                        color = 'green'
                    elif self.hand_expected_values[str(key)][0] < -10:
                        color = 'red'
                    else:
                        color = 'orange'

                    ax[row, col].set_facecolor(color)
                    ax[row, col].text(0.5, 0.15, 
                                    f'{poss_cards[12 - row]}{poss_cards[12 - col]}s\n{round(self.hand_expected_values[str(key)][0], 1)}',
                                    # color=color,
                                    fontsize=8, 
                                    ha='center')

        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()


class KMeansModel():
    """
    Regression class that builds and models KMeans model
    """
    def __init__(self, df, clusters=3):
        self.X, self.y, self.word_to_ix = extract_data(df)
        self.X = tokenize_cards(self.X, self.word_to_ix)
        self.y = np.array(self.y)
        self.data = np.c_[self.X, self.y]
        self.model = KMeans(n_clusters=clusters)
        self.train()

    def train(self):
        self.model.fit(self.data)
        return

    def evaluate(self):
        print('kmeans eval')

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(45, 60)
        plot_data = self.data.T
        labels = self.model.labels_

        labeled_data = np.vstack([plot_data, labels])

        ax.scatter(labeled_data[0], labeled_data[1], labeled_data[3], c=labeled_data[4])
        plt.show()

class MLP(Regression):
    """
    MLP class that builds and models baseline MLPRegressor model,
    inherits most functionality from Regression
    """
    def __init__(self, df):
        """
        Constructor to instantiate necessary information and data
        """
        self.X, self.y, self.word_to_ix = extract_data(df)
        self.X = tokenize_cards(self.X, self.word_to_ix)
        self.y = np.array(self.y)
        self.model = MLPRegressor(random_state=321)
        self.train()

def main(args):
    """
    Main Method to orchestrate all the code
    """
    # Extract and preprocess data
    df = process_data()
    
    if args.m == 'reg':
        reg = Regression(df)
        reg.plot()
    
    if args.m == 'kmeans':
        kmean = KMeansModel(df)
        kmean.plot()

    if args.m == 'mlp':
        mlp = MLP(df)
        mlp.plot()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', default='reg', choices=['reg', 'kmeans', 'mlp'])

    args = parser.parse_args()
    main(args)