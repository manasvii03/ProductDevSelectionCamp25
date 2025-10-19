# Working Model 
The following Google Drive link has a video of the working model and interface
https://drive.google.com/drive/folders/1qUYpgHgD-wU3RcsiqMTfizyj-XNXn3nl?usp=drive_link

# Tech Stack
Frontend: Flutter, Dart
Backend: Python, Flask, Pandas, Numpy, PuLP
ML: Scikit-learn, Joblib

# Dependancies
fl_chart: This is what I used to implement pie charts for role counts and team splits in the final model
file_picker: In the final model, a button is present, which, when u click, opens up the files explorer on the user's device. This was done using this dependency.

# Challenges Faced
1. When data is going from Flask to Flutter, Python objects are converted into JSON. During training and prediction of the model, players with a limited history or those who did not bat or bowl have values for certain features that are missing, which appear as NaN (Python). JSON doesn't recognise NaN. It only recognises None. I did modify the code in app.py to try to convert NaN to None whenever it appears in the data frame, but it wasnt happening. Because of this issue, all those JSON files that ended up giving an output having NaN values for any features did not get printed on the interface.
2. Due to a lack of time, I wasn't able to integrate the backend with the frontend of the UI pages.
3. I tried to implement permutation bars to show the contribution of each feature for the fantasy points prediction of each player. I ended up executing it on Google Colab during the training and testing of my model. But I wasnt able to integrate it with the frontend.

# USP



# Model
## Model Features
credits, season, no of days since last match (recency), number of previous matches played, rolling average and standard deviation of fantasy points over the last 5 years, team strength and opponent strength
## Model Selection
I ended up going with *Random Forest* to train the model
## Picking the Best Team
The end goal of the model is to predict the best team of 11 players. But a user can't just pick the top players with the highest fantasy points because certain constraints are imposed. 
Integer Linear Programming is what I used to predict an optimal XI by maximising the fantasy points and following constraints like having only 11 members on the team, role bounds, team cap and the fact that both teams have to be satisfied.
PuLP is what we used to solve this optimization problem, giving us the single best 11-player combination.
