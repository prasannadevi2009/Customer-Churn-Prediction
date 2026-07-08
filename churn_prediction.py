import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_excel("Telco_customer_churn.xlsx")

df["Churn Label"].value_counts().plot(kind="bar")

plt.title("Customer Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.show()

churn_by_contract = pd.crosstab(
    df["Contract"],
    df["Churn Label"],
    normalize="index"
) * 100

print(churn_by_contract)

churn_by_contract.plot(kind="bar")
plt.title("Churn Rate by Contract Type")
plt.ylabel("Percentage")
plt.show()

df.boxplot(column="Tenure Months", by="Churn Label")
plt.show()

df.boxplot(column="Monthly Charges", by="Churn Label")
plt.show()

print(df.isnull().sum())

df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
print(df["Total Charges"].dtype)

df["Avg_Monthly_Spend"] = (
    df["Total Charges"] /
    (df["Tenure Months"] + 1)
)

print(df[["Total Charges", "Tenure Months", "Avg_Monthly_Spend"]].head())

df["Long_Term_Customer"] = (
    df["Tenure Months"] > 24
).astype(int)

df["High_Value_Customer"] = (
    df["Monthly Charges"] > df["Monthly Charges"].median()
).astype(int)

print(df[[
    "Long_Term_Customer",
    "High_Value_Customer"
]].head())


features = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Churn Score",
    "CLTV",
    "Long_Term_Customer",
    "High_Value_Customer"
]

X = df[features]

print(X.isnull().sum())
X = X.fillna(X.median())
df["Total Charges"] = df["Total Charges"].fillna(
    df["Total Charges"].median()
)
features = [
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Churn Score",
    "CLTV",
    "Long_Term_Customer",
    "High_Value_Customer"
]

X = df[features]
print(X.isnull().sum())

y = df["Churn Value"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(X_train.shape)
print(X_test.shape)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print(cm)

print(classification_report(y_test, y_pred))

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("Random Forest Accuracy:",
      accuracy_score(y_test, rf_pred))

df.to_csv("churn_cleaned1.csv", index=False)