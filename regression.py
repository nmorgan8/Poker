import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression

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
    card_to_idx = {}
    x = []
    y = []
    for i, (start, id, money, actions, summary, end, file, nplayer) in df.iterrows():
        out = False
        cards = re.findall('^.*\*\*\*.*$', money, re.MULTILINE)
        player_info = re.findall(f'^.*{PLAYER_NAME}.*$', money,re.MULTILINE) # list of str
        player_sum = re.findall(f'^.*{PLAYER_NAME}.*$', summary,re.MULTILINE)[0] # str
        final_board = re.findall('\[.*\]', summary, re.MULTILINE) # list of str

        # get starting hands
        info = " ".join(player_info)
        cards = re.findall('\[(.*?)\]', info)
        if cards:
            for card in cards:
                card_to_idx.setdefault(card, len(card_to_idx))
            x.append(cards)
        else:
            out = True

        # get outcome of each hand
        res = re.findall('Wins: .*$|Loses: .*$', player_sum)[0].split(' ')
        if 'L' in res[0] and not out:
            y.append(float(res[1][0:-1]) * -1)
        elif 'W' in res[0] and not out:
            y.append(float(res[1][0:-1]))

    return x, y, card_to_idx

def tokenize_cards(x, card_to_idx):
    X = []
    for row in x:
        X.append([card_to_idx[row[0]], card_to_idx[row[1]]])
    X = np.array(X)
    return X


if __name__ == '__main__':
    # Extract and preprocess data
    df = process_data()
    x, y, word_to_ix = extract_data(df)
    X = tokenize_cards(x, word_to_ix)
    y = np.array(y)

    # Build linear regression model
    model = LinearRegression().fit(X, y)

    # What is the expected profit for Jd, Jc
    start_hands = []
    start_hands_num = []
    poss_cards = list(word_to_ix.keys())
    poss_cards_num = list(word_to_ix.values())
    for i in range(len(poss_cards) - 1):
        for j in range(i+1, len(poss_cards)):
            start_hands.append([poss_cards[i], poss_cards[j]])
            start_hands_num.append([poss_cards_num[i], poss_cards_num[j]])

    for hand, num in zip(start_hands, start_hands_num):
        print(hand, model.predict([num]))

    