# Working Model 
The following Google Drive link has a video of the working model and interface  
https://drive.google.com/drive/folders/1qUYpgHgD-wU3RcsiqMTfizyj-XNXn3nl?usp=drive_link

# Tech Stack
- **Frontend:** Flutter, Dart  
- **Backend:** Python, Flask, Pandas, NumPy, PuLP  
- **Machine Learning:** Scikit-learn, Joblib

# Dependancies
fl_chart: This is what I used to implement pie charts for role counts and team splits in the final model  
file_picker: In the final model, a button is present, which, when u click, opens up the files explorer on the user's device. This was done using this dependency.

## **Model Features**
- **credits**
- **season**
- **recency:** Number of days since last match
- **previous_matches:** Number of previous matches played
- **Performance Trends:** Rolling average and standard deviation of fantasy points over the last 5 years
- **Team Strength** and **Opponent Strength**

## **Model Selection**
I ended up going with **Random Forest** to train the model because it performs well on tabular data and handles non-linear relationships effectively.

## **Picking the Best Team**
The end goal of the model is to **predict the best team of 11 players**.  
However, simply picking the top players based on predicted fantasy points isn’t valid because of constraints like:
- Only **11 members** allowed per team
- **Role bounds**
- **Team cap**
- Players must be selected from **both teams**

  
This optimisation problem is solved by **Integer Linear Programming (ILP)** with the **PuLP** library. 
This gives the **single best 11-player combination** that maximizes predicted fantasy points.

# Model Choice Explanation

I started off by using **Linear Regression** to train my model, but the predictions were not very accurate.  
This is because Linear Regression tends to **oversimplify the data** and fails to capture **complex or non-linear relationships** between features.
Next, I tried using a **Random Forest** model, which is known to perform well on **tabular data** - like the player performance spreadsheets that was given.  
Random Forests can find deeper patterns in the data by combining multiple decision trees, and they gave me noticeably **better predictions in a short amount of time**.
I chose **not to use neural networks**, even though they might have achieved slightly higher accuracy, because they require much more **computational power and time** to train and predict.  
Random Forest turned out to be the right balance between **accuracy, speed, and interpretability** for this task.


# USP
### **Neat and Simple UI**
The frontend was built using **Flutter**.  
- Used a **DataTable** to neatly organize and display player information.  
- Integrated **`fl_chart`** to visualize **role counts** and **team splits** through pie charts.  
- Implemented a **consistent theme** across all pages to give the web app a clean look.

### **Optimal Team Selection**
Instead of using inefficient loops to try out every possible team combination, I used **Integer Linear Programming (ILP)** through the **PuLP** library.  
This approach allows the algorithm to:
- Apply **constraints** to immediately reject a large number of invalid or suboptimal team combinations.
- Focus only on a **small, optimal subset** of combinations where the best solution is guaranteed to exist.
- Achieve **much faster** and more efficient team selection compared to brute force methods.

# How to Run
### 1. Backend (Flask)
```bash
cd dream11_flask
pip install -r req.txt
python3 app.py
```
This will give you the Flask endpoint URL

### 1. Frontend (Flutter)
```bash
cd dream11_flutter
```
Paste the Flask URL in main.dart  
Select Chrome (Web) as the emulator
```bash
flutter run
```
Or press the run button in main.dart


# Challenges Faced
1. When data is going from Flask to Flutter, Python objects are converted into JSON. During training and prediction of the model, players with a limited history or those who did not bat or bowl have values for certain features that are missing, which appear as NaN (Python). JSON doesn't recognise NaN. It only recognises None. I did modify the code in app.py to try to convert NaN to None whenever it appears in the data frame, but it wasnt happening. Because of this issue, all those JSON files that ended up giving an output having NaN values for any features did not get printed on the interface.
2. Due to a lack of time, I wasn't able to integrate the backend with the frontend of the login/signup pages.
3. I tried to implement permutation bars to show the contribution of each feature for the fantasy points prediction of each player. I ended up executing it on Google Colab during the training and testing of my model. But I wasnt able to integrate it with the frontend.


