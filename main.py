import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

countries = pd.read_csv("teams.csv")

#in order to reduce noise, rows that don't have a value for "prev_medals" will be dropped
countries = countries.dropna()

#Narrowing the data to the variables that have higher correlation to "medals"
   
countries.select_dtypes('number').corr()["medals"]

'''
year            -0.034870
events           0.770646
athletes         0.839909
age              0.023231
height           0.141946
weight           0.089701
medals           1.000000
prev_medals      0.920048
prev_3_medals    0.918438
'''

countries = countries[["team", "country", "year", "events", "athletes", "medals", "prev_medals"]]

#creating our training and test sets dividing them by year, this gives a roughly 80/20 split
train = countries[countries["year"] < 2012].copy()
test = countries[countries["year"] >= 2012].copy()

#fitting model
reg = LinearRegression()
predictors = ["athletes", "prev_medals", "events"]

reg.fit(train[predictors], train["medals"])
predictions = reg.predict(test[predictors])
test["predictions"] = predictions

#formatting the predictions
test.loc[test["predictions"] < 0, "predictions"] = 0
test["predictions"] = test["predictions"].round()

error = mean_absolute_error(test["medals"], test["predictions"])

error #the prediction error is 3.269, this is far below the standard deviation

####analysing the errors in order to determine the effectiveness of the model

absolute_error = (test["medals"] - test["predictions"]).abs()
absolute_error

'''
6       1.0
7       1.0
24      0.0
25      0.0
37      1.0
       ...
2111    0.0
2131    0.0
2132    0.0
2142    2.0
2143    3.0
'''

#finding the error by team to get a more specific picture
team_error = absolute_error.groupby(test["team"]).mean()
#mean number of medals per team
medals = test["medals"].groupby(test["team"]).mean()

error_ratio = team_error / medals

'''
AFG    2.0
ALB    NaN
ALG    1.0
AND    NaN
ANG    inf
      ...
VIE    1.0
VIN    NaN
YEM    NaN
ZAM    NaN
ZIM    inf
Length: 204
'''

#There are NaN values (for teams that won 0 medals and their team_error is 0),
#as well as inf (for teams whose team_error is 1 and medals is 0), these values have to be removed

error_ratio = error_ratio[~pd.isnull(error_ratio)]
error_ratio = error_ratio[np.isfinite(error_ratio)]
'''
AFG    2.000000
ALG    1.000000
ARG    0.951220
ARM    1.000000
AUS    0.331633
         ...
UKR    0.317073
USA    0.109375
UZB    1.000000
VEN    1.000000
VIE    1.000000
Length: 97
'''

#plotting the results
error_ratio.plot.hist()
plt.show()

#this graph shows that our predictions are very close to the actual number of medals the majority of the time
#this results could be further interpreted by seeing which countries have the lowest error rates and viceversa

print(error_ratio.sort_values())

'''
FRA    0.028090
RUS    0.050980
ETH    0.066667
NZL    0.095238
HUN    0.104167
         ...
UGA    2.000000
GAB    2.000000
AFG    2.000000
BOT    3.000000
UAE    4.000000
Length: 97

After looking at the amount of medals of the teams with highest and lowest error ratios, we can conclude that
the model is more effective at predicting amount of medals for teams that regurlaly win more medals.

This model could potentially be further improved by analyzing data of individual athletes, predicting how many medals they would win and adding
up this amount to the team total medal count. This could be done by using the original dataset.

'''
