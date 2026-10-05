import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
df = pd.read_csv("advertising.csv")
9/28/26, 7:44 PM CodeAlpha Data Science Internship Guide
https://gemini.google.com/app/a74e08b8eaed3b7b 4/7 # Visualization: Impact of TV ads on sales
sns.regplot(data=df, x="TV", y="Sales", color="teal")
plt.title("TV Advertising Spend vs Sales")
plt.show()
# Model
X = df[["TV", "Radio", "Newspaper"]]
y = df["Sales"]
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42
)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print("Model R2 Score:", r2_score(y_test, y_pred))