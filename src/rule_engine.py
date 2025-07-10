def score_application(row):
    if row["CreditScore"] < 600:
        return "Denied"
    elif row["DebtToIncome"] > 0.35:
        return "Denied"
    elif row["EmploymentStatus"] == "Unemployed" and row["LoanAmount"] > 1000000:
        return "Denied"
    elif row["TenureMonths"] < 12:
        return "Denied"
    else:
        return "Approved"
