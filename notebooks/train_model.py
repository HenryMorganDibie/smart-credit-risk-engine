import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
df = pd.read_csv("data/loan_applications.csv")

# Feature Engineering
df["ApplicationDate"] = pd.to_datetime(df["ApplicationDate"])
df["EmploymentStartDate"] = pd.to_datetime(df["EmploymentStartDate"])
df["TenureMonths"] = ((df["ApplicationDate"] - df["EmploymentStartDate"]).dt.days / 30).astype(int)
df["DebtToIncome"] = df["OutstandingDebt"] / df["AnnualIncome"]

# Encode target and categorical variable
df["LoanOutcome"] = df["LoanOutcome"].map({"Approved": 1, "Denied": 0})
df["EmploymentStatus"] = LabelEncoder().fit_transform(df["EmploymentStatus"])

# Define features and label
features = ["CreditScore", "AnnualIncome", "LoanAmount", "OutstandingDebt", "DebtToIncome", "TenureMonths", "EmploymentStatus"]
X = df[features]
y = df["LoanOutcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
with open("src/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Random Forest Classifier model trained and saved as src/model.pkl")
