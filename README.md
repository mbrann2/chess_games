# chess_games
<br />

# Chess
![](images/chess.png)

***
<br />

# Table of Contents
- [Introduction](#introduction)
- [Background Information](#background-information)
- [Overall Information](#overall)
- [Data Analysis & Vizualization](#data-analysis--vizualization)
    - [Chess Outcomes Breakdown](#chess-outcomes-breakdown)
    - [Hypothesis Testings](#hypothesis-testings)
    - [Correlations: Pearson & Spearman](#correlations-pearson--spearman)
    - [Logistic Regression](#logistic-regression)
- [Conclusion](#conclusion)
<br />

***
## **Introduction**
***
This dataset, which can be found [here](https://www.kaggle.com/datasets/datasnaek/chess), is a set of just over 20,000 chess games collected from a selection of users on the site Lichess.org.

Lots of information is contained within a single chess game, let alone a full dataset of multiple games. It is primarily a game of patterns, and data science is all about detecting patterns in data, which is why chess has been one of the most invested in areas of AI in the past.

<br />

***
## **Background Information**
***
The information conatined in the chess dataset is comprised of the following factors:

- Game ID

- Rated (T/F)

- Start Time

- End Time

- Number of Turns

- Game Status

- Winner

- Time Increment

- White Player ID

- White Player Rating

- Black Player ID

- Black Player Rating

- All Moves in Standard Chess Notation

- Opening Eco (Standardised Code for any given opening, list here)

- Opening Name

- Opening Ply (Number of moves in the opening phase)

<br />
<br />

***
## **Overall**
***
The dataset provides a multitude of factors pertaining to chess games. After analyzing and cleaning the data [Chess Games](data/games.csv), it was initially determined that a logical focus would be on the outcomes of the [games](#chess-outcomes-breakdown) and more specifically the partitioning of [wins versus draws](#chess-outcomes-breakdown). Delving deeper, the overarching goal was to implement various methods to statistically analyze games where white and black are superior, with supporting supplemental features, and draw conclusions on our findings as it relates to wins versus draws and losses along with determining if white has an innate advantage by moving first.
<br />

***
## Data Analysis & Vizualization
Listed below is the data analysis and vizualization aspects for the chess games dataset.
***

### Chess Outcomes Breakdown
***
After analyzing the outcomes of all of the chess games, it was revealed that a majority of the games ended in resignations and the least amount of games ended in draws.

![](images/chess_outcomes_breakdown.png) <br />
***
The next step was to aggregate all forms of wins, which are resignations, out of time, and mates, and perform a direct comparison against the amount of draws.
***
![](images/wins_versus_draws.png) <br />

***
### Hypothesis Testings
***
After seeing the disparity in overall wins versus losses, the next approach was to partition the dataset into two separate datasets for white and black based on the respective player being the superior opponent, or greater than 100 ELO difference, and mapped wins to 1 and draws or losses to 0. Then, for both datasets, we generated our null and alernative hypothesis, which are depcited below, and performed two-sample independent t-tests to analyze the p-values and ultimately the null & alternative hypotheses for rating differentials, number of consecutive moves where a player follows an optimal book, and number of turns in the game.  <br />
***
Null Hypothesis 1: The rating differential between black and white has no significance on which player wins the game. <br />

Alternative Hypothesis 1: The rating differential between black and white has a significance on which player wins the game. <br />
***
Null Hypothesis 2: The number of consecutive moves where a player follows an optimal book opening has no significance on which player wins the game. <br />

Alternative Hypothesis 2: The number of consecutive moves where a player follows an optimal book opening has a significance on which player wins the game. <br />
***
Null Hypothesis 3: The number of turns in a game has no significance on which player wins the game. <br />

Alternative Hypothesis 3: The number of turns in a game has significance on which player wins the game. <br />
***

A summary of the two-sample independent t-test results for black and white are found below. Additionally, histogram plots were generated to display player wins versus draws and losses when that respective player is the superior opponent.
***

White Games [Superior Opponent]:

As we can see, the p-value for rating differential, number of turns, and opening play is less than 0.05, with values of 0.00, so we reject all the null hypotheses and there is significant difference, or impact, between: the white rating differentials, the consecutive moves used from an optimal book opening, the number of turns in the game, all in relation to who wins the game. Additionally, the high t-statistic of approximately 116.77, 109.70, and 129.08 for rating differentials, opening book play, and number of turns simply indicates the confidence in the predictor coefficient, since it's very large, and further supports the decision to reject the null hypothesis.![](images/white_t_tests.png) <br />

When only considering wins and draws plus losses of a higher-ranked white opponent, white wins 72.54% of the time and draws or loses 27.46% of the time over 5,666 games.![](images/white_wins_vs_draws_&_losses.png) <br />

***
Black Games [Superior Opponent]:

The p-value for rating differential, number of turns, and opening play is less than 0.05, with values of 0.00, so we reject all the null hypotheses and there is significant difference, or impact, between: the black rating differentials, the consecutive moves used from an optimal book opening, the number of turns in the game, all in relation to who wins the game. Additionally, the high t-statistic of approximately 114.47, 99.36, and 127.15 for rating differentials, opening book play, and number of turns simply indicates the confidence in the predictor coefficient, since it's very large, and further supports the decision to reject the null hypothesis.![](images/black_t_tests.png) <br />

When only considering wins and draws plus losses of a higher-ranked black opponent, black wins 69.27% of the time and draws or loses 30.73% of the time over 5,230 games.![](images/black_wins_vs_draws_&_losses.png) <br />

***
### Correlations: Pearson & Spearman
***
Next, perform linear, Pearson correlations and nonlinear, Spearman correlation to see potential correlations between rating differentials, for white and black respectively, and their victory status.![](images/correlations.png) Shockingly, there is no true correlation between white or black being a significantly stronger opponent, in terms of ELO rating, and winning games versus losing or drawing games, as seen from the linear, Pearson correlations and the nonlinear, Spearman correlations below. Intuitively, one would think that over a continuous span of gradually increasing ELO differentials, there would be a correlation to the majority, binary class of 1, which is wins. Despite all of the players having respectable ratings at the very least, this counterintuition could be due to players still being considerably lower rated than grandmasters which leads to miscalculations and outright blunders, despite being the higher-rated player.   <br />

***
### Logistic Regression
***
Since the dataset contains continuous numerical inputs, or features, in the form of: rating differentials, consecutive moves where a player follows an optimal book opening, and number of turns in the game along with a transformation of the target, or wins versus draws and losses, to binary values, a logistic regression model was selected. The logistic regression for both white and black utilized 5 folds, or subsets, with an 80% split of the samples into the training set and the remaining 20% of the data was held out for the test set. A graphical output was generated for the true positive rate, or probability of detection, as a function of false positive rate, or probability of false alarm, on a Receiver Operating Characteristic (ROC) plot with curves for all 5 folds. Additionally, an Area Under the Curve (AUC) was generated for our ROC curves to determine the model's predictive accuracy and found that for all 5 folds, for both black and white, the AUC was bound between 0.61 and 0.64, which is indicative of poor predictive accuarcy for our model as 1.0 would be a perfect prediction. It's believed that the model's predictive accuracy would increase if additional quantitative features were provided in the dataset that factor into winning a chess game versus losing or drawing. Some, but not all, of the excluded features are: hours of individual study/preparation, hours of analysis of the opponent's playstyle and openings selection, hours of sleep preceding the game, number of times the opponents have previously faced each other. ![](images/roc_curve_white.png) <br />

![](images/roc_curve_black.png) <br />
***
Also, across all 5 folds, the average accuracy, precision, recall, F1 Score, log loss, and log loss probability were determined for white and black. 
***
White Games [Superior Opponent]:

![](images/white_stats_table.png)

The average accuracy and average precision across all 5 folds for white are both respectable and nearly the same with the average accuracy being approximately 72.42% and the average precision being approximately 72.70%. Accuracy is a function of true positives, true negatives, false positives, or probability of false alarms, and false negatives, or probability of incorrectly identifying that an attribute is absent. One can ascertain that our logistic regression model is being reduced due to the reasonable amount of false positives that are present and is supported by our precision also being reduced due to the same false positive generation. Precision is a function of true positives, or probability of detection, and false positives, which is the probability of a false alarm. One can ascertain that our logistic regression model has a reasonable amount of false positives, or incorrectly predicting the positive class of 1, when it should be predicting the negative class of 0, hence false positives are generated. The average recall across all 5 folds for white was excellent with a result of approximately 99.24%. Recall is a function of true positives, or probability of detection, and false negatives, which is the probability of incorrectly identifying that an attribute is absent. One can ascertain that our logistic regression model has very few false negatives, which would be missing the ability to successsfully predict the correct, positive class of 1 and instead predicting a negative class of 0. The average F1 Score across all 5 folds for white was very good with a result of approximately 83.91%. The average F1 score is a function of the average precision and average recall. The relatively good mean precision of 72.70% and stellar mean recall of 99.24% is yielding the great F1 score of approximately 83.91%. The average log loss across all 5 folds for white was only average with a result of approximately 0.5660, where 0 is optimal. The associated average probability of the log loss across all 5 folds for white was determined to be approximately 0.5696, which effectively denotes the probability of the logistic regression model predicting the proper class, 0 for draws & losses, and 1 for wins, respectively. The mediocre result for average log loss and average log loss probability could be due to an imbalanced dataset that truly contains significantly more wins, or binary values of 1, in comparsion to draws & losses, or binary values of 0.![]() <br />

Black Games [Superior Opponent]:
![](images/black_stats_table.png)

The average accuracy and average precision across all 5 folds for black are both respectable and nearly the same with the average accuracy being approximately 69.38% and the average precision being approximately 69.39%. Accuracy is a function of true positives, true negatives, false positives, or probability of false alarms, and false negatives, or probability of incorrectly identifying that an attribute is absent. One can ascertain that our logistic regression model is being reduced due to the reasonable amount of false positives that are present and is supported by our precision also being reduced due to the same false positive generation. Precision is a function of true positives, or probability of detection, and false positives, which is the probability of a false alarm. One can ascertain that our logistic regression model has a reasonable amount of false positives, or incorrectly predicting the positive class of 1, when it should be predicting the negative class of 0, hence false positives are generated. The average recall across all 5 folds for black was excellent with a result of approximately 99.83%. Recall is a function of true positives, or probability of detection, and false negatives, which is the probability of incorrectly identifying that an attribute is absent. One can ascertain that our logistic regression model has very few false negatives, which would be missing the ability to successsfully predict the correct, positive class of 1 and instead predicting a negative class of 0. The average F1 Score across all 5 folds for black was very good with a result of approximately 81.86%. The average F1 score is a function of the average precision and average recall. The relatively good mean precision of 69.39% and stellar mean recall of 99.83% is yielding the great F1 score of approximately 81.86%. The average log loss across all 5 folds for black was only average with a result of approximately 0.5969, where 0 is optimal. The associated average probability of the log loss across all 5 folds for black was determined to be approximately 0.5162, which effectively denotes the probability of the logistic regression model predicting the proper class, 0 for draws & losses, and 1 for wins, respectively. The mediocre result for average log loss and average log loss probability could be due to an imbalanced dataset that truly contains significantly more wins, or binary values of 1, in comparsion to draws & losses, or binary values of 0. <br />

***
## Conclusion
***
When comparing filtered games where white is the superior opponent with black being the superior opponent, the total games in each respective database very similar with comparable wins versus draws and losses. When running one version of a logistic regression model for both white and black games, one see very similar average metrics for the accuracy, precision, recall, F1 score, log loss, and log loss probability. However, when utilizing a different logistic regression statistical model, a worthwhile not is looking at the respective inputs, or features, for our white and black chess games, respectively. The three features utilized for both black and white games were rating differential, number of turns in the game, and opening play, which is the consecutive moves from the initiation of the game that opponents stick to an optimal, book opening. In general, typically skilled players, even with reasonable rating differentials, will have games with a substantial amount of moves because they avoid suboptimal moves and outright blunders. More specifically, with the white games, our three features all had p-values of 0.000, which are significant since they are less than 0.05, and thus are a good choice to be incorporated into predicting our target, or binary output of wins versus draws and losses. Regarding the black games, the rating differential feature was determined to be significant at 0.000, but interestingly enough the number of terms and opening play were didn't yield significant p-values, or greater than 0.05, at 0.225 and 0.372, respectively.
There has been extensive, cumulative analysis on chess throughout the years to determine that white has an innate advantage simply by being the first player to make a move. The respective difference in significances of our features between white and black games might indicate the fact that due to the inherent disadvantage of black being the responsive player, there is no significance on the number of moves and sticking to an optimal book opening. Essentially, the disadvantage of moving second might trump the ability to follow optimal book openings and play in technically sound games that contain a large number of moves.![](images/stats_model_white.png) <br />

![](images/stats_model_black.png)