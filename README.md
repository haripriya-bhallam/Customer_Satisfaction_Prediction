# Author: Haripriya BhallamAdd commentMore actions
# Date: 2025
# Customer_Satisfaction_Prediction
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

warnings.filterwarnings("ignore")
sns.set(style='whitegrid')

# Load Dataset
data_path = "data/customer_support_tickets.csv"
df = pd.read_csv(data_path)

# Data Cleaning
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(subset=['Customer Satisfaction Rating'], inplace=True)
df['Date of Purchase'] = pd.to_datetime(df['Date of Purchase'], errors='coerce')

# Feature Engineering
df['Purchase_Year'] = df['Date of Purchase'].dt.year
bins = [0, 20, 30, 40, 50, 60, 70, 100]
labels = ['<20', '21-30', '31-40', '41-50', '51-60', '61-70', '70+']
df['Age Group'] = pd.cut(df['Customer Age'], bins=bins, labels=labels)

# Encode categorical features
categorical_cols = ['Customer Gender', 'Product Purchased', 'Ticket Type', 'Ticket Status',
                    'Ticket Priority', 'Ticket Channel', 'Age Group']
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    le_dict[col] = le

# NLP - TF-IDF on Ticket Description
tfidf = TfidfVectorizer(max_features=100, stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['Ticket Description'].fillna('')).toarray()
tfidf_df = pd.DataFrame(tfidf_matrix, columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])

# Combine with other features
drop_cols = ['Ticket ID', 'Customer Name', 'Customer Email', 'Ticket Subject',
             'Resolution', 'Date of Purchase', 'First Response Time', 'Time to Resolution',
             'Ticket Description']
df.drop(columns=drop_cols, inplace=True)
X = pd.concat([df.drop('Customer Satisfaction Rating', axis=1).reset_index(drop=True), tfidf_df], axis=1)
y = df['Customer Satisfaction Rating'].astype(int)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Model Training: RandomForest
rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Model Training: GradientBoosting
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(X_train, y_train)
gb_pred = gb_model.predict(X_test)

# Evaluation Function
def evaluate_model(name, y_true, y_pred):
    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 Score (macro):", f1_score(y_true, y_pred, average='macro'))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))

# Evaluate both models
evaluate_model("Random Forest", y_test, rf_pred)
evaluate_model("Gradient Boosting", y_test, gb_pred)

# Feature Importance Plot for RF
feat_importance = pd.Series(rf_model.feature_importances_, index=X.columns)
plt.figure(figsize=(12, 8))
feat_importance.sort_values(ascending=False)[:20].plot(kind='barh', color='darkcyan')
plt.title("Top 20 Feature Importances - Random Forest")
plt.xlabel("Importance Score")
plt.gca().invert_yaxis()
os.makedirs("reports", exist_ok=True)
plt.savefig("reports/customer_satisfaction_feature_importance.png")
plt.show()

# Save processed data summary to Excel
excel_path = "reports/customer_summary_report.xlsx"
with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
    df['Customer Satisfaction Rating'].value_counts().sort_index().to_frame('Count').to_excel(writer, sheet_name='Satisfaction Distribution')
    df.groupby('Customer Gender').size().to_frame('Total Tickets').to_excel(writer, sheet_name='Tickets by Gender')
    df.groupby('Product Purchased')['Customer Satisfaction Rating'].mean().to_frame('Avg Rating').to_excel(writer, sheet_name='Satisfaction by Product')
    df['Ticket Priority'].value_counts().to_frame('Count').to_excel(writer, sheet_name='Priority Breakdown')
print(f"\n✅ Excel report saved to: {excel_path}")

Customer Satisfaction Queries:


-- 1. Top 10 Most Frequent Ticket Subjects
SELECT "Ticket Subject", COUNT(*) AS issue_count
FROM customer_support_tickets
GROUP BY "Ticket Subject"
ORDER BY issue_count DESC
LIMIT 10;

-- 2. Average Satisfaction Rating by Product
SELECT "Product Purchased", ROUND(AVG("Customer Satisfaction Rating"), 2) AS avg_rating
FROM customer_support_tickets
WHERE "Customer Satisfaction Rating" IS NOT NULL
GROUP BY "Product Purchased"
ORDER BY avg_rating DESC;

-- 3. Average Satisfaction by Ticket Channel
SELECT "Ticket Channel", ROUND(AVG("Customer Satisfaction Rating"), 2) AS avg_rating
FROM customer_support_tickets
GROUP BY "Ticket Channel"
ORDER BY avg_rating DESC;

-- 4. Ticket Volume by Priority Level
SELECT "Ticket Priority", COUNT(*) AS ticket_count
FROM customer_support_tickets
GROUP BY "Ticket Priority"
ORDER BY ticket_count DESC;

-- 5. Satisfaction by Gender
SELECT "Customer Gender", ROUND(AVG("Customer Satisfaction Rating"), 2) AS avg_rating
FROM customer_support_tickets
GROUP BY "Customer Gender"
ORDER BY avg_rating DESC;

-- 6. Monthly Ticket Volume Trend
SELECT TO_CHAR("Date of Purchase", 'YYYY-MM') AS month, COUNT(*) AS ticket_count
FROM customer_support_tickets
GROUP BY month
ORDER BY month;

-- 7. Top 5 Products with Most Tickets
SELECT "Product Purchased", COUNT(*) AS ticket_volume
FROM customer_support_tickets
GROUP BY "Product Purchased"
ORDER BY ticket_volume DESC
LIMIT 5;

-- 8. Most Common Ticket Types
SELECT "Ticket Type", COUNT(*) AS total
FROM customer_support_tickets
GROUP BY "Ticket Type"
ORDER BY total DESC;

-- 9. Average First Response Time by Priority
SELECT "Ticket Priority", ROUND(AVG(EXTRACT(EPOCH FROM ("Time to Resolution"::timestamp - "First Response Time"::timestamp))/3600, 2) AS avg_response_hrs
FROM customer_support_tickets
WHERE "First Response Time" IS NOT NULL AND "Time to Resolution" IS NOT NULL
GROUP BY "Ticket Priority"
ORDER BY avg_response_hrs ASC;

-- 10. Satisfaction by Age Group
SELECT
  CASE
    WHEN "Customer Age" < 20 THEN '<20'
    WHEN "Customer Age" BETWEEN 20 AND 30 THEN '21-30'
    WHEN "Customer Age" BETWEEN 31 AND 40 THEN '31-40'
    WHEN "Customer Age" BETWEEN 41 AND 50 THEN '41-50'
    WHEN "Customer Age" BETWEEN 51 AND 60 THEN '51-60'
    WHEN "Customer Age" BETWEEN 61 AND 70 THEN '61-70'
    ELSE '70+'
  END AS age_group,
  ROUND(AVG("Customer Satisfaction Rating"), 2) AS avg_rating
FROM customer_support_tickets
WHERE "Customer Satisfaction Rating" IS NOT NULL
GROUP BY age_group
ORDER BY age_group;

