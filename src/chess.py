
###

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import scipy.stats as spicystats
import sys
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, roc_curve
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from pandas.plotting import scatter_matrix
from statsmodels.stats.proportion import proportions_ztest
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import f1_score, log_loss
from tabulate import tabulate

sys.path.insert(0, '../src')
sys.path.insert(0, '../data')

import chess

###

# Read in dataset and drop columns containing immaterial data.
def read_file(csv_file):

    '''Inputs: Excel File: Chess Dataset'''

    '''Outputs: Filtered Chess Dataframe'''

    chess_games = pd.read_csv(csv_file)

    chess_games.drop(['rated', 'id', 'created_at', 'last_move_at', 'increment_code', 'white_id', 'black_id', 'opening_eco'], axis=1, inplace=True)
    return chess_games

### 

# Grab and count the chess victory status, broken down by category. Make pie chart to depict the delineations.

def chess_victories(chess_df, path="images/chess_outcomes_breakdown.png"):

    '''Inputs: Filtered Chess Dataframe, Pathfile for Chess Outcomes Pie Plot '''

    '''Outputs: Numerical Breakdown of Victory Statuses [Resignations, Mates, Out of Time, & Draws], Chess Outcomes Pie Plot'''

    victory_status = chess_df['victory_status'].value_counts()

    fig, ax = plt.subplots(figsize = (6,6))

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

    '''Inputs: Filtered Chess Dataframe, Pathfile for Wins Vs Draws Pie Plot '''

    '''Outputs: % of Victories, % of Draws, Pie Plot of Wins Vs Draws'''

    game_status = chess_games['victory_status'].value_counts('draw')

    game_status_percent = game_status * 100

    victories_percent = str(sum(game_status_percent.iloc[:3]))

    draws_percent = str(sum(game_status_percent.iloc[-1:]))

    
    
    fig, ax = plt.subplots(figsize = (6,6))

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

# Look at stronger white players, greater than 100 ELO difference, and map wins to 1 and draws or losses to 0.
# Perform two-sample independent t-tests to analyze the p-values and ultimately the null & alternative hypotheses for white rating differentials, 
# number of consecutive moves where a player follows an optimal book and number of turns in the game.
# Histogram plot displaying white wins versus white draws and losses when white is the superior opponent.

def chess_differentials_white(chess_df, path1='images/white_t_tests.png', path2='images/white_wins_vs_draws_&_losses.png'):
    
    '''Inputs: Filtered Chess Dataframe, Pathfile for White T-tests Table, Pathfile for White Wins Vs Draws Plot'''

    '''Outputs: White T-Tests Table, White Wins Vs Draws or Losses Plot, White WIn %, White Loss or Draw %, 
    White T-test Values [t-statistic & p-value]: Rating Differential, Opening Play, and # of Turns'''

    chess_df = chess_df.copy()

    chess_df['Rating Differential White'] = (chess_df['white_rating'] - chess_df['black_rating']).astype(int)

    
    white_greater_100 = chess_df[(chess_df['Rating Differential White'] > 100)]
    
    def victory_status(s):

        '''Inputs: String [White Winner Column]'''

        '''Outputs: Updated White Winner Column with Binary Values.'''

        if s == 'black' or s == 'draw' :
            return 0

        else:
            return 1
    
    white_greater_100['winner'] = white_greater_100['winner'].apply(victory_status).astype(int)
    
    white_ttest_rating_diff = spicystats.ttest_ind(white_greater_100['Rating Differential White'], white_greater_100['winner'], equal_var = False)
    white_ttest_opening_play = spicystats.ttest_ind(white_greater_100['opening_ply'], white_greater_100['winner'], equal_var = False)
    white_ttest_num_turns = spicystats.ttest_ind(white_greater_100['turns'], white_greater_100['winner'], equal_var = False)

    tests = [white_ttest_rating_diff, white_ttest_opening_play, white_ttest_num_turns]
    rows = ['White t-test Rating Differential', 'White t-test Opening Play', 'White t-test # Turns']
    columns = ['t-statistic', 'p-value']

    print({'White t-test Rating Differential': white_ttest_rating_diff, 'White t-test Opening Play': white_ttest_opening_play, 'White t-test # Turns': white_ttest_num_turns})
    
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_title('White t-tests Breakdown')

    table = ax.table(
    cellText = tests, 
    rowLabels = rows, 
    colLabels = columns,
    rowColours =["palegreen"] * 10, 
    colColours =["palegreen"] * 10,
    cellLoc ='center', 
    loc ='upper left')

    fig.tight_layout()

    plt.savefig(path1)

    plt.clf()

    fig,ax = plt.subplots()

    white_victory = white_greater_100[white_greater_100['winner'] == 1]

    white_draw_or_loss = white_greater_100[white_greater_100['winner'] == 0]
    

    ax.hist(white_victory['winner'], color = 'b', alpha = 0.5, label = 'White Wins: 4,110 [72.54%]')
    ax.hist(white_draw_or_loss['winner'], color = 'g', alpha = 0.5, label = 'White Draws or Loses: 1,556 [27.46%]')
    ax.set_title('White Wins Versus Draws + Losses')
    ax.set_ylabel('# of Wins & Draws + Losses')
    ax.set_xlabel('Wins & Draws + Losses: Total Games [5,666]')

    ax.legend()

    plt.savefig(path2)
    
    white_win_percentage = (4110 /(4110+1556))*100

    white_draw_or_loss_percentage = (1556 /(4110+1556))*100

    print ({'White Win Pct': white_win_percentage, 'White Draw or Loss Pct': white_draw_or_loss_percentage})

    return white_greater_100

###

# Look at stronger black players, greater than 100 ELO difference, and map wins to 1 and draws or losses to 0.
# Perform two-sample independent t-tests to analyze the p-values and ultimately the null & alternative hypotheses for black rating differentials, 
# number of consecutive moves where a player follows an optimal book and number of turns in the game.
# Histogram plot displaying black wins versus black draws and losses when black is the superior opponent.

def chess_differentials_black(chess_df, path1='images/black_t_tests.png', path2='images/black_wins_vs_draws_&_losses.png'):

    '''Inputs: Filtered Chess Dataframe, Pathfile for Black T-tests Table, Pathfile for Black Wins Vs Draws Plot'''

    '''Outputs: Black T-Tests Table, Black Wins Vs Draws or Losses Plot, Black WIn %, Black Loss or Draw %, 
    Black T-test Values [t-statistic & p-value]: Rating Differential, Opening Play, and # of Turns'''

    chess_df = chess_df.copy()

    chess_df['Rating Differential Black'] = (chess_df['black_rating'] - chess_df['white_rating']).astype(int)

    black_greater_100 = chess_df[(chess_df['Rating Differential Black'] > 100)]

    def victory_status(s):

        '''Inputs: String [Black Winner Column]'''

        '''Outputs: Updated Black Winner Column with Binary Values.'''

        if s == 'white' or s == 'draw' :
            return 0

        else:
            return 1
    
    black_greater_100['winner'] = black_greater_100['winner'].apply(victory_status).astype(int)
    
    black_ttest_rating_diff = spicystats.ttest_ind(black_greater_100['Rating Differential Black'], black_greater_100['winner'], equal_var = False)
    black_ttest_opening_play = spicystats.ttest_ind(black_greater_100['opening_ply'], black_greater_100['winner'], equal_var = False)
    black_ttest_num_turns = spicystats.ttest_ind(black_greater_100['turns'], black_greater_100['winner'], equal_var = False)
    
    tests = [black_ttest_rating_diff, black_ttest_opening_play, black_ttest_num_turns]
    rows = ['Black t-test Rating Differential', 'Black t-test Opening Play', 'Black t-test # Turns']
    columns = ['t-statistic', 'p-value']
    print({'Black t-test Rating Differential': black_ttest_rating_diff, 'Black t-test Opening Play': black_ttest_opening_play, 'Black t-test # Turns': black_ttest_num_turns})
    
    fig, ax = plt.subplots()
    ax.set_axis_off()
    ax.set_title('Black t-tests Breakdown')

    table = ax.table(
    cellText = tests, 
    rowLabels = rows, 
    colLabels = columns,
    rowColours =["palegreen"] * 10, 
    colColours =["palegreen"] * 10,
    cellLoc ='center', 
    loc ='upper left')

    fig.tight_layout()

    plt.savefig(path1) 

    plt.clf()

    fig,ax = plt.subplots()
    
    black_victory = black_greater_100[black_greater_100['winner'] == 1]

    black_draw_or_loss = black_greater_100[black_greater_100['winner'] == 0]
    
    ax.hist(black_victory['winner'], color = 'b', alpha = 0.5, label = 'Black Wins: 3,623 [69.27%]')
    ax.hist(black_draw_or_loss['winner'], color = 'g', alpha = 0.5, label = 'Black Draws or Loses: 1,607 [30.73%]')
    ax.set_title('Black Wins Versus Draws + Losses')
    ax.set_ylabel('# of Wins & Draws + Losses')
    ax.set_xlabel('Wins & Draws + Losses: Total Games [5,230]')

    ax.legend()

    plt.savefig(path2)
   
    black_win_percentage = (3623 /(3623+1607))*100

    black_draw_or_loss_percentage = (1607 /(3623+1607))*100

    print({'Black Win Pct': black_win_percentage, 'Black Draw or Loss Pct': black_draw_or_loss_percentage})

    return black_greater_100
   
###

# Perform linear, Pearson correlations and nonlinear, Spearman correlation to see potential correlations between rating differentals, for white and black respectively, and their victory status.
def chess_correlations(white, black, path="images/correlations.png"):

    '''Inputs: Filtered White Dataframe [chess_differentials_white function], 
    Filtered Black Dataframe [chess_differentials_black function], Pathfile for Correlations Plot.'''

    '''Outputs: White Correlation Pearon, Black Correlation Pearson, White Correlation Spearman, Black Correlation Spearman, & Correlations Plot.'''

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
    ax.set_title('Black and White Correlations', fontsize = 16)
    ax.set_ylabel('Correlation Values', fontsize = 16)
    ax.set_xlabel('Correlations', fontsize = 16)
    plt.tight_layout()
    fig.set_size_inches(16, 10)

    plt.savefig(path)

    white_pearson_legend = mpatches.Patch(color= 'silver', label='White Correlation Pearon: 0.1668')

    black_pearson_legend = mpatches.Patch(color= 'black', label='Black Correlation Pearson: 0.1778')

    white_spearman_legend = mpatches.Patch(color= 'silver', label='White Correlation Spearman: 0.1785')

    black_spearman_legend = mpatches.Patch(color= 'black', label='Black Correlation Spearman: 0.1902')

    plt.legend(handles=[white_pearson_legend, black_pearson_legend, white_spearman_legend, black_spearman_legend])

    return print({ 'Pearson White Correlation': white_correlation_pearson, 'Pearson Black Correlation': black_correlation_pearson, 'Spearman White Correlation': white_correlation_spearman, 'Spearman Black Correlation': black_correlation_spearman})

###

# Logistic Regression for white with 5 folds, with a data split of 80% for train and 20% for test, to determine true positive rate, or probability of detection, as a fucntion of false positive rate, or probability of false alarm.
# Also, across all 5 folds, the average accuracy, precision, recall, F1 Score, log loss, and log loss probability were determined for white.

def test_train_white(chess_df, path1='images/roc_curve_white.png', path2='images/stats_model_white.png'):

    '''Inputs: Filtered Chess Dataframe, Pathfile for White ROC Curve, Pathfile for White Stats Model.'''

    chess_df = chess_df.copy()

    chess_df['Rating Differential White'] = (
        chess_df['white_rating'] - chess_df['black_rating']).astype(int)

    white_greater_100 = chess_df[(chess_df['Rating Differential White'] > 100)]

    def victory_status(s):

        '''Inputs: String [White Winner Column]'''

        '''Outputs: Updated White Winner Column with Binary Values.'''

        if s == 'black' or s == 'draw':
            return 0

        else:
            return 1

    white_greater_100['winner'] = white_greater_100['winner'].apply(
        victory_status).astype(int)

    X = white_greater_100[['turns', 'opening_ply',
                           'Rating Differential White']].astype(int)
    y = white_greater_100['winner']

    random_seed = 8

    def cross_val_linear(X, y, k):

        '''Inputs: X(Features [White]), y(Target [White]), amd k(# of Folds [White])'''

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_seed, stratify=y)

        log_reg = sm.Logit(y_train, X_train).fit()

        print(log_reg.summary())

        plt.rc('figure', figsize=(12, 7))

        plt.text(0.01, 0.05, str(log_reg.summary()), {'fontsize': 10}, fontproperties = 'monospace') 
        plt.title('Logistic Regression Stats Model: White')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(path2)

        plt.clf()

        kf = KFold(k)

        kf.get_n_splits(X_train)

        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_score_list = []
        log_loss_list = []
    
        for i, (train_index, test_index) in enumerate(kf.split(X_train)):

            '''Inputs: Training Data [White], Test Data [White], Split k Folds [White]'''
            
            '''Outputs: Mean Accuracy List [White], Mean Precision List [White], Mean Recall List [White], Mean F1 Score [White], 
             Mean Log Loss [White], Mean Log Loss Probability [White], & Stats Table [Black]'''
        
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

            log_loss_prob = (-1*np.log(log_loss_list))
        
        white_stats = [['White Average Accuracy', 'White Average Precision', 'White Average Recall', 'White Average F1 Score', 'White Average Log Loss', 'White Average Log Loss Probability'], [np.mean(accuracy_list), np.mean(precision_list), np.mean(recall_list), np.mean(f1_score_list), np.mean(log_loss_list), np.mean(log_loss_prob)]]
        white_stats_table = ((tabulate(white_stats, headers='firstrow', tablefmt='grid')))
        textFilePath = "../images/white_stats_table.txt"
        with open(textFilePath, 'w') as f:
            f.write(white_stats_table)


        plt.plot([0,1], [0,1])
        plt.legend()
        plt.title('ROC Curve: White')
        plt.xlabel('False Positive Rate: White')
        plt.ylabel('True Positive Rate: White')
        plt.tight_layout()

        plt.savefig(path1)

        return print({'Mean Accuracy List [White]': np.mean(accuracy_list), 'Mean Precision List [White]': np.mean(precision_list), 'Mean Recall List [White]': np.mean(recall_list), 'Mean F1 Score [White]': np.mean(f1_score_list), 'Mean Log Loss [White]': np.mean(log_loss_list), 'Mean Log Loss Probability [White]': np.mean(log_loss_prob)})
        
    cross_val_linear(X, y, 5)

###

# Logistic Regression for black with 5 folds, with a data split of 80% for train and 20% for test, to determine true positive rate, or probability of detection, as a fucntion of false positive rate, or probability of false alarm. 
# Also, across all 5 folds, the average accuracy, precision, recall, F1 Score, log loss, and log loss probability were determined for black.

def test_train_black(chess_df, path1='images/roc_curve_black.png', path2='images/stats_model_black.png'):

    '''Inputs: Filtered Chess Dataframe, Pathfile for Black ROC Curve, Pathfile for Black Stats Model.'''

    chess_df = chess_df.copy()

    chess_df['Rating Differential Black'] = (chess_df['black_rating'] - chess_df['white_rating']).astype(int)

    black_greater_100 = chess_df[(chess_df['Rating Differential Black'] > 100)]

    def victory_status(s):

        '''Inputs: String [Black Winner Column]'''

        '''Outputs: Updated Black Winner Column with Binary Values.'''

        if s == 'white' or s == 'draw' :
            return 0

        else:
            return 1
    
    black_greater_100['winner'] = black_greater_100['winner'].apply(victory_status).astype(int)


    X = black_greater_100[['turns', 'opening_ply', 'Rating Differential Black']].astype(int)

    y = black_greater_100['winner']

    random_seed = 8
   
    
    def cross_val_linear(X, y, k):    
        
        '''Inputs: X(Features [Black]), y(Target [Black]), amd k(# of Folds [Black])'''

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)

        log_reg = sm.Logit(y_train, X_train).fit()

        print(log_reg.summary())

        plt.rc('figure', figsize=(12, 7))

        plt.text(0.01, 0.05, str(log_reg.summary()), {'fontsize': 10}, fontproperties = 'monospace') 
        plt.title('Logistic Regression Stats Model: Black')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(path2)
        plt.clf()
    
        kf = KFold(k)
    
        kf.get_n_splits(X_train)

        accuracy_list = []
        precision_list = []
        recall_list = []
        f1_score_list = []
        log_loss_list = []
    
        for i, (train_index, test_index) in enumerate(kf.split(X_train)):

            '''Inputs: Training Data [Black], Test Data [Black], Split k Folds [Black]'''
        
            '''Outputs: Mean Accuracy List [Black], Mean Precision List [Black], Mean Recall List [Black], Mean F1 Score [Black], 
            Mean Log Loss [Black], Mean Log Loss Probability [Black], & Stats Table [Black]'''

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

            log_loss_prob = (-1*np.log(log_loss_list))
        
        black_stats = [['Black Average Accuracy', 'Black Average Precision', 'Black Average Recall', 'Black Average F1 Score', 'Black Average Log Loss', 'Black Average Log Loss Probability'], [np.mean(accuracy_list), np.mean(precision_list), np.mean(recall_list), np.mean(f1_score_list), np.mean(log_loss_list), np.mean(log_loss_prob)]]
        black_stats_table = ((tabulate(black_stats, headers='firstrow', tablefmt='grid')))
        textFilePath = "../images/black_stats_table.txt"
        with open(textFilePath, 'w') as f:
            f.write(black_stats_table)

        plt.plot([0,1], [0,1])
        plt.legend()
        plt.title('ROC Curve: Black')
        plt.xlabel('False Positive Rate: Black')
        plt.ylabel('True Positive Rate: Black')
        plt.tight_layout()

        plt.savefig(path1)

        return print({'Mean Accuracy List [Black]': np.mean(accuracy_list), 'Mean Precision List [Black]': np.mean(precision_list), 'Mean Recall List [Black]': np.mean(recall_list), 'Mean F1 Score [Black]': np.mean(f1_score_list), 'Mean Log Loss [Black]': np.mean(log_loss_list), 'Mean Log Loss Probability [Black]': np.mean(log_loss_prob)})
            
    cross_val_linear(X, y, 5)

###

# Respective functions listed below to test outputs to terminal and images directory.

if __name__ == "__main__":

    chess_data = read_file("data/games.csv")

    # victories_breakdown = chess_victories(chess_data)

    # wins_and_draws = wins_versus_draws(chess_data)

    # chess_outcomes_white = chess_differentials_white(chess_data)

    chess_outcomes_black = chess_differentials_black(chess_data)

    # chess_corr = chess_correlations(chess_differentials_white(chess_data), chess_differentials_black(chess_data))

    # log_reg_white = test_train_white(chess_data)

    # log_reg_black = test_train_black(chess_data)