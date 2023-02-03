
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

# Logistic Regression for white with 5 folds, with a data split of 80% for train and 20% for test, to determine true positive rate, or probability of detection, as a fucntion of false positive rate, or probability of false alarm.
# Also, across all 5 folds, the average accuracy, precision, recall, F1 Score, log loss, and log loss probability were determined for white.

def test_train_white(chess_df, path1='images/roc_curve_white.png', path2='images/stats_model_white.png'):
    chess_df = chess_df.copy()
    chess_df['Rating Differential White'] = (
        chess_df['white_rating'] - chess_df['black_rating']).astype(int)
    white_greater_100 = chess_df[(chess_df['Rating Differential White'] > 100)]

    def victory_status(s):
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
        plt.legend()
        plt.title('ROC Curve: White')
        plt.xlabel('False Positive Rate: White')
        plt.ylabel('True Positive Rate: White')
        plt.tight_layout()
        plt.savefig(path1)
        return print({'Mean Accuracy List [White]': np.mean(accuracy_list), 'Mean Precision List [White]': np.mean(precision_list), 'Mean Recall List [White]': np.mean(recall_list), 'Mean F1 Score [White]': np.mean(f1_score_list), 'Mean Log Loss [White]': np.mean(log_loss_list), 'Mean Log Loss Probability [White]': np.mean(log_loss_prob)})
        


    cross_val_linear(X, y, 5)


# The average accuracy and average precision across all 5 folds for white are both respectable and nearly the same with the average accuracy being approximately 72.42% and the average precision being approximately 72.70%.
# Accuracy is a function of true positives, true negatives, false positives, or probability of false alarms, and false negatives, or probability of incorrectly identifying that an attribute is absent.
# We can ascertain that our logistic regression model is being reduced due to the reasonable amount of false positives that are present and is supported by our precision also being reduced due to the same false positive generation.
# Precision is a function of true positives, or probability of detection, and false positives, which is the probability of a false alarm. 
# We can ascertain that our logistic regression model has a reasonable amount of false positives, or incorrectly predicting the positive class of 1, when it should be predicting the negative class of 0, hence false positives are generated. 
# The average recall across all 5 folds for white was excellent with a result of approximately 99.24%.
# Recall is a function of true positives, or probability of detection, and false negatives, which is the probability of incorrectly identifying that an attribute is absent.
# We can ascertain that our logistic regression model has very few false negatives, which would be missing the ability to successsfully predict the correct, positive class of 1 and instead predicting a negative class of 0.
# The average F1 Score across all 5 folds for white was very good with a result of approximately 83.91%.
# The average F1 score is a function of the average precision and average recall. 
# The relatively good mean precision of 72.70% and stellar mean recall of 99.24% is yielding the great F1 score of approximately 83.91% 
# The average log loss across all 5 folds for white was only average with a result of approximately 0.5660, where 0 is optimal. 
# The associated average probability of the log loss across all 5 folds for white was determined to be approximately 0.5696, which effectively denotes the probability of the logistic regression model predicting the proper class, 0 for draws & losses, and 1 for wins, respectively.
# The mediocre result for average log loss and average log loss probability could be due to an imbalanced dataset that truly contains significantly more wins, or binary values of 1, in comparsion to draws & losses, or binary values of 0.

###

# Logistic Regression for black with 5 folds, with a data split of 80% for train and 20% for test, to determine true positive rate, or probability of detection, as a fucntion of false positive rate, or probability of false alarm. 
# Also, across all 5 folds, the average accuracy, precision, recall, F1 Score, log loss, and log loss probability were determined for black.

def test_train_black(chess_df, path1='images/roc_curve_black.png', path2='images/stats_model_black.png'):
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
        plt.legend()
        plt.title('ROC Curve: Black')
        plt.xlabel('False Positive Rate: Black')
        plt.ylabel('True Positive Rate: Black')
        plt.tight_layout()
        plt.savefig(path1)
        return print({'Mean Accuracy List [Black]': np.mean(accuracy_list), 'Mean Precision List [Black]': np.mean(precision_list), 'Mean Recall List [Black]': np.mean(recall_list), 'Mean F1 Score [Black]': np.mean(f1_score_list), 'Mean Log Loss [Black]': np.mean(log_loss_list), 'Mean Log Loss Probability [Black]': np.mean(log_loss_prob)})
        


    cross_val_linear(X, y, 5)


# The average accuracy and average precision across all 5 folds for black are both respectable and nearly the same with the average accuracy being approximately 69.38% and the average precision being approximately 69.39%.
# Accuracy is a function of true positives, true negatives, false positives, or probability of false alarms, and false negatives, or probability of incorrectly identifying that an attribute is absent.
# We can ascertain that our logistic regression model is being reduced due to the reasonable amount of false positives that are present and is supported by our precision also being reduced due to the same false positive generation.
# Precision is a function of true positives, or probability of detection, and false positives, which is the probability of a false alarm. 
# We can ascertain that our logistic regression model has a reasonable amount of false positives, or incorrectly predicting the positive class of 1, when it should be predicting the negative class of 0, hence false positives are generated. 
# The average recall across all 5 folds for black was excellent with a result of approximately 99.83%.
# Recall is a function of true positives, or probability of detection, and false negatives, which is the probability of incorrectly identifying that an attribute is absent.
# We can ascertain that our logistic regression model has very few false negatives, which would be missing the ability to successsfully predict the correct, positive class of 1 and instead predicting a negative class of 0.
# The average F1 Score across all 5 folds for black was very good with a result of approximately 81.86%.
# The average F1 score is a function of the average precision and average recall. 
# The relatively good mean precision of 69.39% and stellar mean recall of 99.83% is yielding the great F1 score of approximately 81.86% 
# The average log loss across all 5 folds for black was only average with a result of approximately 0.5969, where 0 is optimal. 
# The associated average probability of the log loss across all 5 folds for black was determined to be approximately 0.5162, which effectively denotes the probability of the logistic regression model predicting the proper class, 0 for draws & losses, and 1 for wins, respectively.
# The mediocre result for average log loss and average log loss probability could be due to an imbalanced dataset that truly contains significantly more wins, or binary values of 1, in comparsion to draws & losses, or binary values of 0.

###

# Conclusion
# When comparing filtered games where white is the superior opponent with black being the superior opponent, the total games in each respective database very similar with comparable wins versus draws and losses. 
# When running one version of a logistic regression model for both white and black games, we see very similar average metrics for the accuracy, precision, recall, F1 score, log loss, and log loss probability. 
# However, when utilizing a different logistic regression statistical model, a worthwhile not is looking at the respective inputs, or features, for our white and black chess games, respectively. 
# The three features utilized for both black and white games were rating differential, number of turns in the game, and opening play, which is the consecutive moves from the initiation of the game that opponents stick to an optimal, book opening. 
# In general, typically skilled players, even with reasonable rating differentials, will have games with a substantial amount of moves because they avoid suboptimal moves and outright blunders.
# More specifically, with the white games, our three features all had p-values of 0.000, which are significant since they are less than 0.05, and thus are a good choice to be incorporated into predicting our target, or binary output of wins versus draws and losses.
# Regarding the black games, the rating differential feature was determined to be significant at 0.000, but interestingly enough the number of terms and opening play were didn't yeild significant p-values, or greater than 0.05, at 0.225 and 0.372, respectively.
# There has been extensive, cumulative analysis on chess throughout the years to determine that white has an innate advantage simply by being the first player to make a move.
# The respective difference in significances of our features between white and black games might indicate the fact that due to the inherent disadvantage of black being the responsive player, there is no significance on the number of moves and sticking to an optimal book opening. 
# Essentially, the disadvantage of moving second might trump the ability to follow optimal book openings and play in technically sound games that contain a large number of moves.

###

# Respective functions listed below to test outputs to terminal and images directory.

if __name__ == "__main__":

    chess_data = read_file("data/games.csv")

    # victories_breakdown = chess_victories(chess_data)

    # wins_and_draws = wins_versus_draws(chess_data)

    # chess_outcomes_white = chess_differentials_white(chess_data)

    # chess_outcomes_black = chess_differentials_black(chess_data)

    # chess_corr = chess_correlations(chess_differentials_white(chess_data), chess_differentials_black(chess_data))

    # log_reg_white = test_train_white(chess_data)

    # log_reg_black = test_train_black(chess_data)