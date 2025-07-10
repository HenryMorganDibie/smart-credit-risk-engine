import pandas as pd

def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)

    # Convert dates to datetime objects
    df["ApplicationDate"] = pd.to_datetime(df["ApplicationDate"])
    df["EmploymentStartDate"] = pd.to_datetime(df["EmploymentStartDate"])

    # Calculate job tenure
    df["TenureMonths"] = ((df["ApplicationDate"] - df["EmploymentStartDate"]).dt.days / 30).astype(int)

    # Compute Debt-to-Income Ratio
    df["DebtToIncome"] = df["OutstandingDebt"] / df["AnnualIncome"]

    return df
