
###

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as spicystats
import sys

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, roc_curve

from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from pandas.plotting import scatter_matrix
import statsmodels.api as sm
from statsmodels.stats.proportion import proportions_ztest


from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import f1_score, log_loss

sys.path.insert(0, '../src')
sys.path.insert(0, '../data')


import chess

###

# Read in dataset and drop columns containing immaterial data.
def read_file(csv_file):
    chess_games = pd.read_csv(csv_file)
    chess_games.drop(['rated', 'id', 'created_at', 'last_move_at', 'increment_code', 'white_id', 'black_id', 'opening_eco'], axis=1, inplace=True)
    return chess_games

### 

# Grab and count the chess victory status, broken down by category. Make pie chart to depict the delineations.

def chess_victories(chess_df, path="images/chess_outcomes_breakdown.png"):
     victory_status = chess_df['victory_status'].value_counts()

     fig, ax = plt.subplots(figsize = (10,6))

     sizes = victory_status.values
     labels = ['Resignations: 11,147', 'Mates: 6,325', 'Out of Time: 1,680', 'Draws: 906']
     explode = (0.1, 0.1, 0.2, 0.2)
     c = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
     plt.pie(sizes, explode=explode, colors=c, 
        autopct='%1.1f%%', shadow=True, startangle=140)
     ax.legend( labels, title="Victory Status",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))
     ax.set_title("Victory Status Breakdown")
     plt.tight_layout()
     plt.axis('equal')
     plt.savefig(path)
     return print({ 'Chess Outcomes Breakdown': victory_status})

###

# Look at games won, by mate, out of time, and resignation, and compare to draws. Make pie charts for both delineations. 

def wins_versus_draws(chess_games, path="images/wins_versus_draws.png"):
    game_status = chess_games['victory_status'].value_counts('draw')
    game_status_percent = game_status * 100
    victories_percent = str(sum(game_status_percent.iloc[:3]))
    draws_percent = str(sum(game_status_percent.iloc[-1:]))
    game_wins = sum(game_status.iloc[:3])
    game_draws = sum(game_status.iloc[-1:])
    
    fig, ax = plt.subplots(figsize = (10,6))

    sizes = victories_percent, draws_percent
    labels = ['Wins: 19,152', 'Draws: 906']
    explode = (0.2, 0.2)
    c = ['#1f77b4', '#ff7f0e']
    plt.pie(sizes, explode=explode, colors=c, 
        autopct='%1.1f%%', shadow=True, startangle=140)
    ax.legend(labels, title="Wins & Draws",
          loc="center left",
          bbox_to_anchor=(1, 0, 0.5, 1))
    ax.set_title("Wins Versus Draws")
    plt.tight_layout()
    plt.axis('equal')
    plt.savefig(path)
    return print({ '% of Victories': victories_percent, '% of Draws': draws_percent})

###

# Null Hypothesis: The rating differential between black and white has no significant impact on which player wins the game.
# Alternate Hypothesis: The rating differential between black and white has a significant impact on which player wins the game.

###

# Look at stronger white players, greater than 100 ELO, and map wins to 1 and draws or losses to 0.
# Perform t-test to analyze the p-value and ultimately the null & alternate hypotheses for white differentials.
# Histogram pot displaying white wins versus white draws when white is the superior opponent.

def chess_differentials_white(chess_df, path="images/white_wins_vs_draws_&_losses.png"):
    chess_df = chess_df.copy()
    chess_df['Rating Differential White'] = (chess_df['white_rating'] - chess_df['black_rating']).astype(int)
    black_wins = chess_df[(chess_df['Rating Differential White'] > 100) & (chess_df['winner'] == 'black')].value_counts()
    white_greater_100 = chess_df[(chess_df['Rating Differential White'] > 100)]
    
    def victory_status(s):
        if s == 'black' or s == 'draw' :
            return 0

        else:
            return 1
    
    white_greater_100['winner'] = white_greater_100['winner'].apply(victory_status).astype(int)
    
    white_ttest = spicystats.ttest_ind(white_greater_100['Rating Differential White'], white_greater_100['winner'], equal_var = False)
    
    fig, ax = plt.subplots()

    white_victory = white_greater_100[white_greater_100['winner'] == 1]
    white_draw_or_loss = white_greater_100[white_greater_100['winner'] == 0]
    

    ax.hist(white_victory['winner'], color = 'b', alpha = 0.5, label = 'White Wins: 4,110 [72.54%]')
    ax.hist(white_draw_or_loss['winner'], color = 'g', alpha = 0.5, label = 'White Draws or Loses: 1,556 [27.46%]')
    ax.set_title('White Wins Versus Draws + Losses')
    ax.set_ylabel('# of Wins & Draws + Losses')
    ax.set_xlabel('Wins & Draws + Losses: Total Games [5,666]')

    ax.legend()
    plt.savefig(path)
    white_win_percentage = (4110 /(4110+1556))*100
    white_draw_or_loss_percentage = (1556 /(4110+1556))*100
    print({ 'White Superior Games': white_greater_100, 'White T-test':white_ttest, 'White Victories': white_victory, 'White Draws or Losses': white_draw_or_loss, 'White Win Pct': white_win_percentage, 'White Draw or Loss Pct': white_draw_or_loss_percentage, 'Black Wins': black_wins})
    return white_greater_100

###

# Look at stronger black players, greater than 100 ELO, and map black wins to 1 and black draws or losses to 0.
# Perform t-test to analyze the p-value and ultimately the null & alternate hypotheses for black differentials.
# Histogram plot displaying black wins versus black draws and losses when black is the superior opponent.

def chess_differentials_black(chess_df, path="images/black_wins_vs_draws_&_losses.png"):
    chess_df = chess_df.copy()
    chess_df['Rating Differential Black'] = (chess_df['black_rating'] - chess_df['white_rating']).astype(int)
    white_wins = chess_df[(chess_df['Rating Differential Black'] > 100) & (chess_df['winner'] == 'white')].value_counts()
    black_greater_100 = chess_df[(chess_df['Rating Differential Black'] > 100)]

    def victory_status(s):
        if s == 'white' or s == 'draw' :
            return 0

        else:
            return 1
    
    black_greater_100['winner'] = black_greater_100['winner'].apply(victory_status).astype(int)
    
    black_ttest = spicystats.ttest_ind(black_greater_100['Rating Differential Black'], black_greater_100['winner'], equal_var = False)
    
    fig,ax = plt.subplots()
    
    black_victory = black_greater_100[black_greater_100['winner'] == 1]
    black_draw_or_loss = black_greater_100[black_greater_100['winner'] == 0]
    
    ax.hist(black_victory['winner'], color = 'b', alpha = 0.5, label = 'Black Wins: 3,623 [69.27%]')
    ax.hist(black_draw_or_loss['winner'], color = 'g', alpha = 0.5, label = 'Black Draws or Loses: 1,607 [30.73%]')
    ax.set_title('Black Wins Versus Draws + Losses')
    ax.set_ylabel('# of Wins & Draws + Losses')
    ax.set_xlabel('Wins & Draws + Losses: Total Games [5,230]')

    ax.legend()
    plt.savefig(path)
    black_win_percentage = (3623 /(3623+1607))*100
    black_draw_or_loss_percentage = (1607 /(3623+1607))*100
    print({' Black Superior Games': black_greater_100, 'Black T-test': black_ttest, 'Black Victories': black_victory, 'Black Draws or Losses': black_draw_or_loss, 'Black Win Pct': black_win_percentage, 'Black Draw or Loss Pct': black_draw_or_loss_percentage, 'White Wins': white_wins})
    return black_greater_100
###

# Perform nonlinear Spearman correlation to see potential correlations between rating differentals, fow white and black respectively, and their victory status.
def chess_correlations(white, black, path="images/correlations.png"):
    white = white.copy()
    black = black.copy()
    white_correlation_spearman = white['Rating Differential White'].corr(white['winner'], method ='spearman')

    black_correlation_spearman = black['Rating Differential Black'].corr(black['winner'], method ='spearman')

    white_correlation_pearson = white['Rating Differential White'].corr(white['winner'], method ='pearson')

    black_correlation_pearson = black['Rating Differential Black'].corr(black['winner'], method ='pearson')

    fig,ax = plt.subplots()
    x_axis = [white_correlation_pearson, black_correlation_pearson, white_correlation_spearman, black_correlation_spearman]
    x_axis_names = ['White Correlation Pearon', 'Black Correlation Pearson', 'White Correlation Spearman', 'Black Correlation Spearman']
    c = ['silver', 'black', 'silver', 'black']
    
    ax.set_ylim(0.15, 0.20)
    plt.bar(x_axis_names, x_axis, color=c)
    ax.set_title('Black and White Correlations')
    ax.set_ylabel('Correlation Values')
    ax.set_xlabel('Correlations')
    plt.tight_layout()
    fig.set_size_inches(16, 10)
    plt.savefig(path)
    white_pearson_legend = mpatches.Patch(color= 'silver', label='White Correlation Pearon: 0.1668')
    black_pearson_legend = mpatches.Patch(color= 'black', label='Black Correlation Pearson: 0.1778')
    white_spearman_legend = mpatches.Patch(color= 'silver', label='White Correlation Spearman: 0.1785')
    black_spearman_legend = mpatches.Patch(color= 'black', label='Black Correlation Spearman: 0.1902')
    plt.legend(handles=[white_pearson_legend, black_pearson_legend, white_spearman_legend, black_spearman_legend])

    return print({ 'Pearson White Correlation': white_correlation_pearson, 'Pearson Black Correlation': black_correlation_pearson, 'Spearman White Correlation': white_correlation_spearman, 'Spearman Black Correlation': black_correlation_spearman})



# There is no true correlation between white or black being a significantly stronger opponent, in terms of ELO rating, 
# and winning games versus losing or drawing games, as seen from the linear, Pearson correlations and the nonlinear, Spearman correlations below.


# The p-value is less than 0.05, so we reject the null hypothesis and there is significant difference, or impact, 
# between the black rating differentials, with black being the superior opponent, and who wins the game.
# When only considering wins and draws plus losses of a higher-ranked black opponent, black wins 69.27% of the time and draws or loses 30.73% of the time over 5,230 games.

# The p-value is less than 0.05, so we reject the null hypothesis and there is significant difference, or impact, 
# between the white rating differentials, with white being the superior opponent, and who wins the game.
# When only considering wins and draws plus losses of a higher-ranked white opponent, white wins 72.54% of the time and draws or loses 27.46% of the time over 5,666 games.

###

# Logistic Regression for white with 5 folds to determine true positive rate, or probability of detection, as a fucntion of false positive rate, or probability of false alarm. 
# Also, across all 5 folds, the average accuracy, precision, recall, F1 Score, and log loss were determined for white.

def test_train_white(chess_df, path="images/roc_curve_white.png"):
    chess_df = chess_df.copy()
    chess_df['Rating Differential White'] = (chess_df['white_rating'] - chess_df['black_rating']).astype(int)
    white_greater_100 = chess_df[(chess_df['Rating Differential White'] > 100)]
    
    def victory_status(s):
        if s == 'black' or s == 'draw' :
            return 0

        else:
            return 1
    
    white_greater_100['winner'] = white_greater_100['winner'].apply(victory_status).astype(int)


    X = white_greater_100[['turns', 'opening_ply', 'Rating Differential White']].astype(int)
    y = white_greater_100['winner']

    random_seed = 8
   
    
    def cross_val_linear(X, y, k):    
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)
    
        kf = KFold(k)
    
        kf.get_n_splits(X_train)

        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_score_list = []
        log_loss_list = []
    
        for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        
            X_train_kfold = X_train.iloc[train_index]
            y_train_kfold = y_train.iloc[train_index]
            X_test_kfold = X_train.iloc[test_index]
            y_test_kfold = y_train.iloc[test_index]

            log_model = LogisticRegression(random_state=random_seed).fit(X_train_kfold, y_train_kfold)
            y_pred = log_model.predict(X_test_kfold)
            y_prob = log_model.predict_proba(X_test_kfold)
            y_hat = log_model.decision_function(X_test_kfold)

            fpr, tpr, thresholds = metrics.roc_curve(y_test_kfold, y_hat)
            roc_auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, label='ROC curve ' + f'{i+1}' + ' (area = %0.2f)' % roc_auc)
        
            accuracy_list.append(metrics.accuracy_score(y_test_kfold, y_pred))
            precision_list.append(metrics.precision_score(y_test_kfold, y_pred))
            recall_list.append(metrics.recall_score(y_test_kfold,y_pred))
            f1_score_list.append(f1_score(y_test_kfold, y_pred, average='binary', zero_division='warn'))
            log_loss_list.append(log_loss(y_test_kfold, y_prob, normalize = True))
        plt.plot([0,1], [0,1])
        plt.legend()
        plt.title('ROC Curve: White')
        plt.xlabel('False Positive Rate: White')
        plt.ylabel('True Positive Rate: White')
        plt.savefig(path)
        return print({'Mean Accuracy List [White]': np.mean(accuracy_list), 'Mean Precision List [White]': np.mean(precision_list), 'Mean Recall List [White]': np.mean(recall_list), 'Mean F1 Score [White]': np.mean(f1_score_list), 'Mean Log Loss [White]': np.mean(log_loss_list)})
        


    cross_val_linear(X, y, 5)

###

# Logistic Regression for black with 5 folds to determine true positive rate, or probability of detection, as a fucntion of false positive rate, or probability of false alarm. 
# Also, across all 5 folds, the average accuracy, precision, recall, F1 Score, and log loss were determined for black.

def test_train_black(chess_df, path='images/roc_curve_black.png'):
    chess_df = chess_df.copy()
    chess_df['Rating Differential Black'] = (chess_df['black_rating'] - chess_df['white_rating']).astype(int)
    black_greater_100 = chess_df[(chess_df['Rating Differential Black'] > 100)]

    def victory_status(s):
        if s == 'white' or s == 'draw' :
            return 0

        else:
            return 1
    
    black_greater_100['winner'] = black_greater_100['winner'].apply(victory_status).astype(int)


    X = black_greater_100[['turns', 'opening_ply', 'Rating Differential Black']].astype(int)
    y = black_greater_100['winner']

    random_seed = 8
   
    
    def cross_val_linear(X, y, k):    
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)
    
        kf = KFold(k)
    
        kf.get_n_splits(X_train)

        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_score_list = []
        log_loss_list = []
    
        for i, (train_index, test_index) in enumerate(kf.split(X_train)):
        
            X_train_kfold = X_train.iloc[train_index]
            y_train_kfold = y_train.iloc[train_index]
            X_test_kfold = X_train.iloc[test_index]
            y_test_kfold = y_train.iloc[test_index]

            log_model = LogisticRegression(random_state=random_seed).fit(X_train_kfold, y_train_kfold)
            y_pred = log_model.predict(X_test_kfold)
            y_prob = log_model.predict_proba(X_test_kfold)
            y_hat = log_model.decision_function(X_test_kfold)

            fpr, tpr, thresholds = metrics.roc_curve(y_test_kfold, y_hat)
            roc_auc = metrics.auc(fpr, tpr)
            plt.plot(fpr, tpr, label='ROC curve ' + f'{i+1}' + ' (area = %0.2f)' % roc_auc)
        
            accuracy_list.append(metrics.accuracy_score(y_test_kfold, y_pred))
            precision_list.append(metrics.precision_score(y_test_kfold, y_pred))
            recall_list.append(metrics.recall_score(y_test_kfold,y_pred))
            f1_score_list.append(f1_score(y_test_kfold, y_pred, average='binary', zero_division='warn'))
            log_loss_list.append(log_loss(y_test_kfold, y_prob, normalize = True))
        plt.plot([0,1], [0,1])
        plt.legend()
        plt.title('ROC Curve: Black')
        plt.xlabel('False Positive Rate: Black')
        plt.ylabel('True Positive Rate: Black')
        plt.savefig(path)
        return print({'Mean Accuracy List [Black]': np.mean(accuracy_list), 'Mean Precision List [Black]': np.mean(precision_list), 'Mean Recall List [Black]': np.mean(recall_list), 'Mean F1 Score [Black]': np.mean(f1_score_list), 'Mean Log Loss [Black]': np.mean(log_loss_list)})
        


    cross_val_linear(X, y, 5)


# Respective functions listed below to test outputs to terminal and images directory.

if __name__ == "__main__":

    chess_data = read_file("data/games.csv")

    # victories_breakdown = chess_victories(chess_data)

    # wins_and_draws = wins_versus_draws(chess_data)

    # chess_outcomes_white = chess_differentials_white(chess_data)

    # chess_outcomes_black = chess_differentials_black(chess_data)

    # chess_corr = chess_correlations(chess_differentials_white(chess_data), chess_differentials_black(chess_data))

    log_reg_white = test_train_white(chess_data)

    # log_reg_black = test_train_black(chess_data)