# 🏦 Smart Credit Risk Decision Engine

A transparent and interactive machine learning-powered application to assess and automate credit risk decisions for loan applicants — built for financial services teams and fintechs.

---

## 📌 Features

- ✅ Rule-based credit approval engine
- 🤖 Machine Learning predictions with 90%+ accuracy
- 🧠 SHAP model explainability (why a loan was approved/denied)
- 📄 Auto-generated PDF reports for each applicant
- 🚨 Risk flag tagging for high-risk applicants
- 📊 Feature importance visualization
- 💻 Streamlit-powered UI for non-technical users

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone [https://github.com/HenryMorganDibie/smart-credit-risk-engine.git]
cd smart_credit_risk_engine
```

### 2. Create a virtual environment and install dependencies

python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
pip install -r requirements.txt

### 3. Add dataset
```
data/loan_applications.csv
```
### 4. Train the ML model
```
python notebooks/train_model.py 
```
This saves the trained model to src/model.pkl.

### 5. Launch the Streamlit app
```
streamlit run app/streamlit_app.py
```
## 🧠 How It Works

### 🔁 Pipeline Overview
- Load & preprocess loan applicant data
- Apply rule-based decision logic
- Apply trained Random Forest ML model
- Compare ML vs. rule engine vs. actual outcome
- Generate risk flag & PDF report
- Explain predictions using SHAP

## Project Structure
<pre lang="markdown">
smart_credit_risk_engine/
│
├── app/
│   └── streamlit_app.py          # Main Streamlit dashboard
│
├── data/
│   └── loan_applications.csv     # Your dataset
│
├── notebooks/
│   └── train_model.py            # ML training script
│
├── outputs/
│   └── reports/                  # Auto-generated PDF reports
│
├── src/
│   ├── utils.py                  # Data loading & feature prep
│   ├── rule_engine.py           # Rule-based logic
│   ├── pdf_generator.py         # PDF generation function
│   └── model.pkl                # Trained ML model
│
├── README.md                     # This file
└── requirements.txt              # Python dependencies
</pre>

## 🖼️ Screenshots

![alt text](<model comparison and generation of high risk applicants pdf.png>) ![alt text](<Application decisions.png>) ![alt text](<ML modelfeature Importance.png>) ![alt text](<Shap Explanation for individual predictions.png>) ![alt text](<ML model predictions vs rule based vs actual.png>) ![alt text](<downloadable decision reports.png>)

## 👨‍💼 Use Cases
- Fintech companies automating credit approvals

- Lending teams evaluating risky applicants

- Compliance teams demanding explainability

- AI/ML product demos for banks and credit unions

- Teaching tool for ML fairness and transparency

## 🛠️ Tech Stack

🐍 Python 3.9+

🎯 Scikit-learn (Random Forest)

🧠 SHAP (model interpretability)

📊 Plotly (interactive visuals)

📄 FPDF (PDF generation)

⚙️ Streamlit (dashboard UI)

## 🙌 Credits
Built with ❤️ by Henry C. Dibie
Inspired by real-world use cases in credit risk and AI transparency.

## 📬 Contact
Got questions, feedback, or partnership interest?

📧 henrymorgan273@yahoo.com
