
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

import chess 

sys.path.insert(0, '../src')
sys.path.insert(0, '../data')

###

# Grab and count the chess victory status, broken down by category. Drop columns containing immaterial data.

def read_file(csv_file):

    '''Inputs: Excel File: Chess Dataset.'''

    '''Outputs: Filtered Chess Dataframe.'''

    chess_games = pd.read_csv(csv_file)

    chess_games.drop(['rated', 'id', 'created_at', 'last_move_at', 'increment_code', 'white_id', 'black_id', 'opening_eco'], axis=1, inplace=True)
    
    return chess_games

### 

# Make pie chart to depict the victory status categories.

def chess_victories(chess_df):

    '''Inputs: Filtered Chess Dataframe.'''

    '''Outputs: Numerical Breakdown of Victory Statuses [Resignations, Mates, Out of Time, & Draws].'''

    victory_status = chess_df['victory_status'].value_counts().values

    return victory_status

###

# Pie plot of the breakdown of chess outcomes for all games.

def victory_status_pie_chart(victory_status, path='images/chess_outcomes_breakdown.png'):

    '''Inputs: Numerical Breakdown of Victory Statuses [Resignations, Mates, Out of Time, & Draws], Pathfile for Victory Status Pie Plot.'''

    '''Outputs: Victory Status Pie Plot.'''

    fig, ax = plt.subplots(figsize = (6,6))
     
    sizes = victory_status
     
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

###

# Look at games won, by mate, out of time, and resignation, and compare to draws.

def wins_versus_draws(chess_df):

    '''Inputs: Filtered Chess Dataframe.'''

    '''Outputs: % of Victories, % of Draws.'''
    
    game_status = chess_df['victory_status'].value_counts('draw')
    
    game_status_percent = game_status * 100
    
    victories_percent = str(sum(game_status_percent.iloc[:3]))
    
    draws_percent = str(sum(game_status_percent.iloc[-1:]))
    
    return victories_percent, draws_percent

###

# Pie plot of the total wins versus draws.

def wins_versus_draws_pie_chart(wins_and_draws, path='images/wins_versus_draws.png'):

    '''Inputs: % of Victories & % of Draws, Pathfile for Wins Vs Draws Pie Plot.'''

    '''Outputs: Wins Vs Draws Pie Plot.'''
    
    fig, ax = plt.subplots(figsize = (6,6))

    sizes = wins_and_draws
    
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

###

# Obtain filtered white dataset where white is superior, or greater than 100 rating than black, and parse by white wins versus draws plus losses.
# Map white wins to 1 and white draws or losses to 0.

def chess_differentials_white(chess_df):
    
    '''Inputs: Filtered Chess Dataframe.'''

    '''Outputs: Dataset Where White is Superior [> 100 Rating than Black] & Parsed by Wins Vs Draws + Losses.'''

    chess_df = chess_df.copy()

    chess_df['Rating Differential White'] = (chess_df['white_rating'] - chess_df['black_rating']).astype(int)

    white_greater_100 = chess_df[(chess_df['Rating Differential White'] > 100)]

    def victory_status(s):
        if s == 'black' or s == 'draw' :
            return 0

        else:
            return 1

    white_greater_100['winner'] = white_greater_100['winner'].apply(victory_status).astype(int)

    return white_greater_100

###

# Obtain filtered black dataset where black is superior, or greater than 100 rating than white, and parse by black wins versus draws plus losses.
# Map black wins to 1 and black draws or losses to 0.

def chess_differentials_black(chess_df):

    '''Inputs: Filtered Chess Dataframe.''' 
   
    '''Outputs: Dataset Where Black is Superior [> 100 Rating than White] & Parsed by Wins Vs Draws + Losses.'''

    chess_df = chess_df.copy()
    
    chess_df['Rating Differential Black'] = (chess_df['black_rating'] - chess_df['white_rating']).astype(int)
    
    black_greater_100 = chess_df[(chess_df['Rating Differential Black'] > 100)]

    def victory_status(s):
        if s == 'white' or s == 'draw' :
            return 0

        else:
            return 1
    
    black_greater_100['winner'] = black_greater_100['winner'].apply(victory_status).astype(int)

    return black_greater_100
   
### 

# For both white and black datasets where they are superior, perform two-sample independent t-tests to analyze the p-value and ultimately the null & alternative hypotheses for rating differentials, 
# number of consecutive moves where a player follows an optimal book, and number of turns in the game. Generate table of results for both white and black.

def t_tests(white_greater, black_greater):  

    '''Inputs: Filtered Chess Dataframes where White and Black are Superior & Parsed by Wins Vs Draws + Losses.'''

    '''Outputs: White and Black t-tests for Rating Differential, Opening Plays, & # of Turns.'''
    
    white_average_rating_diff = np.mean(white_greater['Rating Differential White'])

    greater_avg_rating_diff = white_greater['winner'][white_greater['Rating Differential White']>= white_average_rating_diff]

    less_avg_rating_diff = white_greater['winner'][white_greater['Rating Differential White']< white_average_rating_diff]

    global white_ttest_rating_diff

    white_ttest_rating_diff = spicystats.ttest_ind(greater_avg_rating_diff, less_avg_rating_diff, equal_var=False)


    white_average_plays = np.mean(white_greater['opening_ply'])

    greater_avg_plys = white_greater['winner'][white_greater['opening_ply']>= white_average_plays]
    
    less_avg_plys = white_greater['winner'][white_greater['opening_ply']< white_average_plays]

    global white_ttest_opening_play
    
    white_ttest_opening_play = spicystats.ttest_ind(greater_avg_plys, less_avg_plys, equal_var=False)
    

    white_average_turns = np.mean(white_greater['turns'])

    greater_avg_turns = white_greater['winner'][white_greater['turns']>= white_average_turns]

    less_avg_turns = white_greater['winner'][white_greater['turns']< white_average_turns]

    global white_ttest_num_turns

    white_ttest_num_turns = spicystats.ttest_ind(greater_avg_turns, less_avg_turns, equal_var = False)


    black_average_rating_diff = np.mean(black_greater['Rating Differential Black'])

    greater_avg_rating_diff = black_greater['winner'][black_greater['Rating Differential Black']>= black_average_rating_diff]

    less_avg_rating_diff = black_greater['winner'][black_greater['Rating Differential Black']< black_average_rating_diff]

    global black_ttest_rating_diff

    black_ttest_rating_diff = spicystats.ttest_ind(greater_avg_rating_diff, less_avg_rating_diff, equal_var=False)

    
    black_average_plays = np.mean(black_greater['opening_ply'])

    greater_avg_plys = black_greater['winner'][black_greater['opening_ply']>= black_average_plays]
    
    less_avg_plys = black_greater['winner'][black_greater['opening_ply']< black_average_plays]

    global black_ttest_opening_play
    
    black_ttest_opening_play = spicystats.ttest_ind(greater_avg_plys, less_avg_plys, equal_var=False)
    

    black_average_turns = np.mean(black_greater['turns'])

    greater_avg_turns = black_greater['winner'][black_greater['turns']>= black_average_turns]

    less_avg_turns = black_greater['winner'][black_greater['turns']< black_average_turns]

    global black_ttest_num_turns

    black_ttest_num_turns = spicystats.ttest_ind(greater_avg_turns, less_avg_turns, equal_var = False)

    return white_ttest_rating_diff, white_ttest_opening_play, white_ttest_num_turns, black_ttest_rating_diff, black_ttest_opening_play, black_ttest_num_turns 

###

# White and black t-test tables for rating differential, opening play, and # of turns.

def t_tests_tables(tests, path1='images/white_t_tests.png', path2='images/black_t_tests.png'):

    '''Inputs: Filtered Chess Dataframes where White and Black are Superior & Parsed by Wins Vs Draws + Losses, 
    Pathfile for White t-tests Table, Pathfile for Black t-tests Table.'''

    '''Outputs: White and Black t-tests tables for Rating Differential, Opening Plays, & # of Turns.'''
    
    tests = [white_ttest_rating_diff, white_ttest_opening_play, white_ttest_num_turns]

    rows = ['White t-test: Average Rating Differential', 'White t-test: Average Opening Play', 'White t-test: Average # Turns']

    columns = ['t-statistic', 'p-value']
    
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

    tests = [black_ttest_rating_diff, black_ttest_opening_play, black_ttest_num_turns]

    rows = ['Black t-test: Average Rating Differential', 'Black t-test: Average Opening Play', 'Black t-test: Average # Turns']

    columns = ['t-statistic', 'p-value']
    
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

    plt.savefig(path2)

###

# Histogram plot displaying wins versus draws and losses, for both white and black, when white and black is the superior opponent [parsed datasets].

def wins_vs_losses_and_draws_plots(white_greater_100, black_greater_100, path1='images/white_wins_vs_draws_&_losses.png', path2='images/black_wins_vs_draws_&_losses.png'):

    '''Inputs: Filtered Chess Dataframes where White and Black are Superior & Parsed by Wins Vs Draws + Losses, 
    Pathfile for White Wins Vs Draws & Losses Plot, Pathfile for Black Wins Vs Draws & Losses Plot.'''

    '''Outputs: White and Black Histogram Plots for Wins Vs Draws & Losses.'''
    
    white_victory = white_greater_100[white_greater_100['winner'] == 1]

    counted_white_victories = white_victory['winner'].value_counts()

    white_draw_or_loss = white_greater_100[white_greater_100['winner'] == 0]

    white_draw_or_loss_count = white_draw_or_loss['winner'].value_counts()

    white_win_percentage = (counted_white_victories.values /(counted_white_victories.values + white_draw_or_loss_count.values))*100

    white_draw_or_loss_percentage = (white_draw_or_loss_count.values /(counted_white_victories.values + white_draw_or_loss_count.values))*100
    
    fig,ax = plt.subplots()

    ax.hist(white_victory['winner'], color = 'b', alpha = 0.5, label = 'White Wins: 4,110 [72.54%]')

    ax.hist(white_draw_or_loss['winner'], color = 'g', alpha = 0.5, label = 'White Draws or Loses: 1,556 [27.46%]')

    ax.set_title('White Draws + Losses Versus Wins', fontsize = 16)

    ax.set_ylabel('# of Draws + Losses & Wins', fontsize = 16)

    fig.canvas.draw()

    labels = [item.get_text() for item in ax.get_xticklabels()]

    labels[1] = ''
    labels[2] = ''
    labels[4] = ''
    labels[5] = ''
    labels[6] = ''
    labels[8] = ''
    labels[9] = ''
    labels[3] = 'Draws + Losses'
    labels[7] = 'Wins'

    ax.set_xticklabels(labels)

    ax.set_xlabel('Draws + Losses & Wins: Total Games [5,666]', fontsize = 16)

    ax.legend(fontsize = 10)

    plt.savefig(path1)

    fig,ax = plt.subplots()
    
    black_victory = black_greater_100[black_greater_100['winner'] == 1]

    counted_black_victories = black_victory['winner'].value_counts()

    black_draw_or_loss = black_greater_100[black_greater_100['winner'] == 0]

    black_draw_or_loss_count = black_draw_or_loss['winner'].value_counts()

    black_win_percentage = (counted_black_victories.values /(counted_black_victories.values + black_draw_or_loss_count.values))*100
    
    black_draw_or_loss_percentage = (black_draw_or_loss_count.values /(counted_black_victories.values + black_draw_or_loss_count.values))*100
    
    ax.hist(black_victory['winner'], color = 'b', alpha = 0.5, label = 'Black Wins: 3,623 [69.27%]')

    ax.hist(black_draw_or_loss['winner'], color = 'g', alpha = 0.5, label = 'Black Draws or Loses: 1,607 [30.73%]')

    ax.set_title('Black Draws + Losses Versus Wins', fontsize = 16)

    ax.set_ylabel('# of Draws + Losses & Wins', fontsize = 16)

    fig.canvas.draw()

    labels = [item.get_text() for item in ax.get_xticklabels()]
    
    labels[1] = ''
    labels[2] = ''
    labels[4] = ''
    labels[5] = ''
    labels[6] = ''
    labels[8] = ''
    labels[9] = ''
    labels[3] = 'Draws + Losses'
    labels[7] = 'Wins'

    ax.set_xticklabels(labels)

    ax.set_xlabel('Draws + Losses & Wins: Total Games [5,230]', fontsize = 16)

    ax.legend(fontsize = 10)
    
    plt.savefig(path2)

###

# Perform linear, Pearson correlations and nonlinear, Spearman correlations to see potential correlations between rating differentals, for white and black respectively, and their victory status.

def chess_correlations(white, black):

    '''Inputs: Filtered Chess Dataframes where White and Black are Superior & Parsed by Wins Vs Draws + Losses.'''

    '''Outputs: White Correlation Pearon, Black Correlation Pearson, White Correlation Spearman, Black Correlation Spearman, & Correlations Plot.'''

    white = white.copy()

    black = black.copy()

    global white_correlation_spearman

    white_correlation_spearman = white['Rating Differential White'].corr(white['winner'], method ='spearman')

    global black_correlation_spearman

    black_correlation_spearman = black['Rating Differential Black'].corr(black['winner'], method ='spearman')

    global white_correlation_pearson

    white_correlation_pearson = white['Rating Differential White'].corr(white['winner'], method ='pearson')

    global black_correlation_pearson

    black_correlation_pearson = black['Rating Differential Black'].corr(black['winner'], method ='pearson')

    return white_correlation_pearson, black_correlation_pearson, white_correlation_spearman, black_correlation_spearman

###

# Plot the linear, Pearson correlations and nonliear, Spearman correlations for both white and black 
# [Parsed datasets for superior opponents and aggregation of wins vs losses plus draws.]

def correlations_plot(chess_df, path="images/correlations.png"):
    
    '''Inputs: White Correlation Pearon, Black Correlation Pearson, White Correlation Spearman, Black Correlation Spearman, & Correlations Plot.'''

    '''Outputs: Plot of correlations.'''

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

    white_pearson_legend = mpatches.Patch(color= 'silver', label='White Correlation Pearon: 0.1668')

    black_pearson_legend = mpatches.Patch(color= 'black', label='Black Correlation Pearson: 0.1778')

    white_spearman_legend = mpatches.Patch(color= 'silver', label='White Correlation Spearman: 0.1785')

    black_spearman_legend = mpatches.Patch(color= 'black', label='Black Correlation Spearman: 0.1902')

    plt.legend(handles=[white_pearson_legend, black_pearson_legend, white_spearman_legend, black_spearman_legend], fontsize = 16)

    plt.savefig(path)

# Logistic Regression for white with 5 folds, with a data split of 80% for train and 20% for test, to determine true positive rate, or probability of detection, as a fucntion of false positive rate, or probability of false alarm.
# Also, across all 5 folds, the average accuracy, precision, recall, F1 Score, log loss, and log loss probability were determined for white.

def test_train_white(white_greater_100, path1='images/roc_curve_white.png', path2='images/stats_model_white.png', path3='images/white_stats_table.png'):

    '''Inputs: Filtered White Chess Dataframe, Pathfile for White ROC Curve, Pathfile for White Stats Model.'''

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

        plt.plot([0,1], [0,1])
        plt.legend(fontsize = 16)
        plt.title('ROC Curve: White', fontsize = 16)
        plt.xlabel('False Positive Rate: White', fontsize = 16)
        plt.ylabel('True Positive Rate: White', fontsize = 16)
        plt.tight_layout()

        plt.savefig(path1)

        plt.rcParams["figure.figsize"] = [14, 6]
        plt.rcParams["figure.autolayout"] = True

        fig, axs = plt.subplots(1, 1)

        data = [(np.mean(accuracy_list), np.mean(precision_list), np.mean(recall_list), np.mean(f1_score_list), np.mean(log_loss_list), np.mean(log_loss_prob))]
        
        columns = ("White Average Accuracy", "White Average Precision", "White Average Recall", "White Average F1 Score", "White Average Log Loss", "White Average Log Loss Probability")
        
        axs.set_title('White Stats Table', fontsize = 16)
        axs.axis('tight')
        axs.axis('off')
        
        white_stats_table = axs.table(cellText=data, colLabels=columns, loc='center')
        
        fontsize = 24
        white_stats_table.set_fontsize(fontsize)
        
        plt.tight_layout()
        
        plt.savefig(path3)
             
    cross_val_linear(X, y, 5)

###

# Logistic Regression for black with 5 folds, with a data split of 80% for train and 20% for test, to determine true positive rate, or probability of detection, as a fucntion of false positive rate, or probability of false alarm. 
# Also, across all 5 folds, the average accuracy, precision, recall, F1 Score, log loss, and log loss probability were determined for black.

def test_train_black(black_greater_100, path1='images/roc_curve_black.png', path2='images/stats_model_black.png', path3='images/black_stats_table.png'):

    '''Inputs: Filtered Chess Dataframe, Pathfile for Black ROC Curve, Pathfile for Black Stats Model.'''

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

        plt.plot([0,1], [0,1])
        plt.legend(fontsize = 16)
        plt.title('ROC Curve: Black', fontsize = 16)
        plt.xlabel('False Positive Rate: Black', fontsize = 16)
        plt.ylabel('True Positive Rate: Black', fontsize = 16)
        plt.tight_layout()

        plt.savefig(path1)

        plt.rcParams["figure.figsize"] = [14, 6]
        plt.rcParams["figure.autolayout"] = True

        fig, axs = plt.subplots(1, 1)

        data = [(np.mean(accuracy_list), np.mean(precision_list), np.mean(recall_list), np.mean(f1_score_list), np.mean(log_loss_list), np.mean(log_loss_prob))]
        
        columns = ("Black Average Accuracy", "Black Average Precision", "Black Average Recall", "Black Average F1 Score", "Black Average Log Loss", "Black Average Log Loss Probability")
        
        axs.set_title('Black Stats Table', fontsize = 16)
        axs.axis('tight')
        axs.axis('off')
        
        white_stats_table = axs.table(cellText=data, colLabels=columns, loc='center')
        
        fontsize = 24
        white_stats_table.set_fontsize(fontsize)
        
        plt.tight_layout()
        
        plt.savefig(path3) 
            
    cross_val_linear(X, y, 5)

###

# Respective functions listed below to test outputs to terminal and images directory.

# For log_reg_white & log_reg_black, only run with chess_data uncommented and 
# specific log_reg function(white or black) uncommented(not both) or images will not be properly generated.

if __name__ == "__main__":
    
    # For log_reg_white & log_reg_black, only run with chess_data uncommented and 
    # specific log_reg function(white or black) uncommented(not both) or images will not be properly generated.

    chess_data = read_file("data/games.csv")

    # victories_breakdown = chess_victories(chess_data)

    # victory_status_pie = victory_status_pie_chart(chess_victories(chess_data))

    # wins_and_draws = wins_versus_draws(chess_data)

    # wins_and_draws_pie = wins_versus_draws_pie_chart(wins_versus_draws(chess_data))

    # chess_outcomes_white = chess_differentials_white(chess_data)

    # chess_outcomes_black = chess_differentials_black(chess_data)

    # t_testing = t_tests(chess_outcomes_white, chess_outcomes_black)

    # t_testing_tables = t_tests_tables(t_testing)

    # win_loss_draw_plot = wins_vs_losses_and_draws_plots(chess_outcomes_white, chess_outcomes_black)

    # chess_corr = chess_correlations(chess_outcomes_white, chess_outcomes_black)

    # white_correlation_pearson, black_correlation_pearson, white_correlation_spearman, black_correlation_spearman = chess_correlations(chess_differentials_white(chess_data), chess_differentials_black(chess_data))

    # chess_corr = correlations_plot(chess_data)

    # log_reg_white = test_train_white(chess_differentials_white(chess_data))

    # log_reg_black = test_train_black(chess_differentials_black(chess_data))
