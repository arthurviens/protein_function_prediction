# PREDICTION OF PROTEIN FUNCTION

Dear participants,

the goal of this data challenge is to resolve a use case in data analysis
which is inspired from a real scientific project (and real data in a
simplified form). You will be given features extracted from proteins. The task
is to predict whether the protein has a certain function. You are free to use
any approach and technique you want. 

The challenge provides three sets of data:

    - a training dataset (data/X_train.csv, data/y_train.csv): this is what
      you should use for training your model
    - two test sets (valid, test) for which you only have the feature matrix.
      This two datasets are going to be used for scoring your methods.


Each row corresponds to a protein in a given organism. Each column (except the
This challenge aims to put in practice first) corresponds to a protein feature
(di-amino acid composition, physico-chemical features, etc). The goal of this
challenge is to predict whether the proteins have a specific function (linked
to secretion systems). It is thus a classification problem. 

## How is this going to work ?

1. You can decide to work in teams (max two people) or alone.
2. Register on the website and use this link to access the data, starting kit,
and to submit your results.
3. Work on the model. You are encourage to explore:

    - Various transformations of the features, 
    - Various machine learning algorithms and their hyperparameters

4. Submit as many models as you wish, in a limit of 5 a day, to the
leaderboard, to see their performance on the public validation set.
5. Submit the predictions made by 2 optimized final models to the leaderboard.
Those are the models you believe should win the challenge.

## How is validation performed?

The validation is performed using the average of the recall of the positive
class and the recall of the negative class.

```python
score_pos = recall_score(y_true, y_pred)
score_neg = recall_score(1-y_true, 1-y_pred)
score = (score_pos + score_neg) / 2
```

## Important dates

- Beginning of the challenge: Tuesday, January 4th
- End of the first phase: Wednesday, January, 26th
- End of the second phase: Thursday, January, 27th
- Final reports & code due: January, 28th, 2022

## Evaluation

Evaluation will be done on (1) a final report (max 4 pages); (3) validity of
the approach chosen; (3) reproducibility of the results. Bonus points will be
given depending on the final rankings.


**Final report** The report is to be deposited at
https://cloud.univ-grenoble-alpes.fr/index.php/s/JAP9MbBqBJtm2fy no later than
January, 28th, 2022, at 23.59.

- Please name your report file “Lastname1Initial_Lastname2Initial.pdf”
  (supposing you are a team of 2 people). If Jane Smith and Sarah Martin work
  together, their report should be named MartinS_SmithJ.pdf.
- Your report should be no more than 3 pages long.


Your report should contain the following elements:

- Your full names and the name of your team name on the codalab plateform.
- A discussion of feature processing. Did you standardize the data, chose
  alternative rep- resentations for some features, discarded other features,
  and why?
- The cross-validated performance, on the training data, of the algorithms you
  explored. You are strongly encouraged to explore the space of parameters for
  each of these al- gorithms. Briefly explain how you did it. Discuss which
  algorithms/parameters work best.
- The performance, on the validation data (visible part of the leaderboard),
  of one model of each of the five families. Discuss whether the results match
  your expectations.
- A discussion of additional models you have tried, insights you have gained
  (_e.g._, “This method works well but is difficult to fit” or “This method is
  not very accurate but is really fast to train”).
- A discussion of your choice of final model(s). You can submit up to 2 final
  models. What are these models, how did you construct them, why do you expect
  them to be your best proposals? Include tables or figures as you see fit.
- Link to the code for the two models submitted.


## Final words

We hope you will enjoy the challenge!

Good luck!
-- The protein function prediction team.
