import pickle
import pandas as pd
import seaborn as sns
from Data_Preprocessing import preprocess_data, income_category_mapping
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

csv_path = "Data/BankChurners.csv"
data = pd.read_csv(csv_path)
ccdata_processed = preprocess_data(data)

# This is our optimal model, if you would like to try another one, change the 
# parameters at Model_Config.py and run ML_Models.py
with open('optimal_rf_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

print("Model downloaded successfully")

with open('train_test_data.pkl', 'rb') as file:
    X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test = pickle.load(file)
print("Train and Test data downloaded successfully")

# Model's performance for each Age Group
'''
age_group_metrics = {}

for age_group in X_test['Age_Group'].unique():
    X_test_age = X_test_scaled[X_test['Age_Group'] == age_group]
    y_test_age = y_test[X_test['Age_Group'] == age_group]
    if len(X_test_age) > 0:
        y_pred_age = loaded_model.predict(X_test_age)
        metrics = {
            'Accuracy': accuracy_score(y_test_age, y_pred_age),
            'Precision': precision_score(y_test_age, y_pred_age),
            'Recall': recall_score(y_test_age, y_pred_age),
            'F1': f1_score(y_test_age, y_pred_age),
            'ROC AUC': roc_auc_score(y_test_age, y_pred_age)
        }
        age_group_metrics[age_group] = metrics

# Library to Df
age_group_metrics_df = pd.DataFrame.from_dict(age_group_metrics, orient='index')
print(age_group_metrics_df)
'''


'''
reverse_income_category_mapping = {v: k for k, v in income_category_mapping.items()}

# Collect metrics into a list of tuples
income_metrics_list = []

for label in sorted(ccdata_processed['Income_Category_Label'].unique()):
    X_test_income = X_test_scaled[X_test['Income_Category_Label'] == label]
    y_test_income = y_test[X_test['Income_Category_Label'] == label]
    
    # Evaluate the model
    y_pred_income = loaded_model.predict(X_test_income, y_test_income)
    #income_metrics = evaluate_model(optimal_rf_model, X_test_income, y_test_income)
    
    # Add to list with the label
    income_metrics_list.append((label, income_metrics))

# Sort by the label (which is the first item in each tuple)
income_metrics_list.sort(key=lambda x: x[0])

# Print the metrics with the category name
for label, metrics in income_metrics_list:
    category_name = reverse_income_category_mapping[label]
    print(f"Income Category '{category_name}' Metrics:", metrics)
'''

'''
reverse_income_category_mapping = {v: k for k, v in income_category_mapping.items()}
# Collect metrics into a list of tuples
income_metrics_list = []

#income_metrics_data = []

for label, metrics in income_metrics_list:
    category_name = reverse_income_category_mapping[label]
    income_metrics_list.append({
        'Income Category': category_name,
        'Accuracy': round(metrics['Accuracy'], 2),
        'Precision': round(metrics['Precision'], 2),
        'Recall': round(metrics['Recall'], 2),
        'F1': round(metrics['F1'], 2),
        'ROC AUC': round(metrics['ROC AUC'], 2)
    })

# Create DataFrame from the data
income_metrics_df = pd.DataFrame(income_metrics_list)

# Print the DataFrame
print(income_metrics_df)
'''


'''
income_metrics_list = []
reverse_income_category_mapping = {v: k for k, v in income_category_mapping.items()}
for label in sorted(ccdata_processed['Income_Category_Label'].unique()):
    X_test_income = X_test_scaled[X_test['Income_Category_Label'] == label]
    y_test_income = y_test[X_test['Income_Category_Label'] == label]
    
    # Evaluate the model
    y_pred_income = loaded_model.predict(X_test_income)
    income_metrics = {
        'Accuracy': accuracy_score(y_test_income, y_pred_income),
        'Precision': precision_score(y_test_income, y_pred_income),
        'Recall': recall_score(y_test_income, y_pred_income),
        'F1': f1_score(y_test_income, y_pred_income),
        'ROC AUC': roc_auc_score(y_test_income, y_pred_income)
    }
    
    # Add to list with the label
    income_metrics_list.append((label, income_metrics))

# Sort by the label (which is the first item in each tuple)
income_metrics_list.sort(key=lambda x: x[0])


for label, metrics in income_metrics_list:
    category_name = reverse_income_category_mapping[label]
    print(f"Income Category '{category_name}' Metrics:", metrics)

'''

# Confusion Matrix for Income 120$+
# Reason recall for Income 120$+ was the lowest, we would like to see the detail
'''
# Filter the test set for the '$120K +' income category
label_for_120k_plus = income_category_mapping['$120K +']
X_test_120k_plus = X_test_scaled[X_test['Income_Category_Label'] == label_for_120k_plus]
y_test_120k_plus = y_test[X_test['Income_Category_Label'] == label_for_120k_plus]

# Predict using the optimal model
y_pred_120k_plus = loaded_model.predict(X_test_120k_plus)

# Generate the confusion matrix
cm = confusion_matrix(y_test_120k_plus, y_pred_120k_plus)

# Plot the confusion matrix
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix for $120K + Income Category')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('Outputs/confusion_matrix_120k_plus.png', bbox_inches='tight')
plt.show()
'''

# Feature Importances
'''
# Extract feature importances from the Random Forest model
feature_importances = loaded_model.feature_importances_

# Match feature names with their importance scores
features = X_train.columns
importances_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})

# Sort the DataFrame by importance
importances_df = importances_df.sort_values(by='Importance', ascending=False)

# Plotting
plt.figure(figsize=(12, 8))
plt.barh(importances_df['Feature'], importances_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.gca().invert_yaxis()  # To display the highest importance at the top
plt.savefig('Outputs/Feature_Importances.png', bbox_inches='tight')
plt.show()
'''



# Correlation Matrix
'''

# Select only numerical columns for correlation matrix
numerical_columns = ccdata_processed.select_dtypes(include=['int64', 'float64'])

# Calculate correlation matrix for numerical columns only
correlation_matrix = numerical_columns.corr()

# Exclude 'Attrition_Flag_Label' when examining correlations with other features
feature_correlations = correlation_matrix.drop('Attrition_Flag_Label', axis=0)['Attrition_Flag_Label'].sort_values()

# Define colors based on correlation values
colors = ['green' if i in feature_correlations.nlargest(2).index else 
          'red' if x > 0 else 'blue' for i, x in feature_correlations.items()]

# Plotting
plt.figure(figsize=(12, 8))
plt.barh(feature_correlations.index, feature_correlations.values, color=colors)
plt.title('Feature Correlations with Customer Attrition (Numerical Features Only)')
plt.xlabel('Correlation coefficient')
plt.ylabel('Feature')
plt.savefig('Outputs/Correlation_Matrix.png', bbox_inches='tight')
plt.show()
'''

# Risk Segmentation
'''
feature_columns = ['Age_Group', 'Dependent_count', 'Months_on_book', 
                   'Total_Relationship_Count', 'Months_Inactive_12_mon', 
                   'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 
                   'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 
                   'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                   'Gender_Label', 'Education_Level_Label', 'Marital_Status_Label', 
                   'Income_Category_Label', 'Card_Category_Label']
X = ccdata_processed[feature_columns]
ccdata_processed['Attrition_Probability'] = loaded_model.predict_proba(X)[:, 1]

ccdata_processed['Risk_Segment'] = pd.cut(ccdata_processed['Attrition_Probability'],
                                           bins=[0, 0.33, 0.66, 1],
                                           labels=['Low Risk', 'Mid Risk', 'High Risk'],
                                           right=False)

segment_summary = ccdata_processed.groupby('Risk_Segment').agg({
    'Total_Trans_Amt': ['mean', 'std'],
    'Total_Trans_Ct': ['mean', 'std'],
    'Contacts_Count_12_mon': ['mean', 'std'],
    'Months_Inactive_12_mon': ['mean', 'std'],
    'Total_Ct_Chng_Q4_Q1': ['mean', 'std'],
    'Total_Revolving_Bal': ['mean', 'std'],
    'Total_Relationship_Count': ['mean', 'std']
})
'''

# Segmentation by Age Group and Income Category
'''
age_income_risk = ccdata_processed.groupby(['Age_Group', 'Income_Category_Label'])['Attrition_Probability'].mean()


age_income_pivot = age_income_risk.reset_index().pivot(index='Age_Group', columns='Income_Category_Label', values='Attrition_Probability')


plt.figure(figsize=(10, 8))
sns.heatmap(age_income_pivot, annot=True, fmt=".2f", cmap="Reds")
plt.title('Risk Distribution Across Age Groups and Income Categories')
plt.ylabel('Age Group')
plt.xlabel('Income Category Label')
plt.savefig('Outputs/Risk_Distribution_Age_Income.png', bbox_inches='tight')
plt.show()
'''

# Column-based exploration by risk segments
# Must do the risk segmentation first
'''
# Flatten the multi-level column index
segment_summary.columns = ['_'.join(col).strip() for col in segment_summary.columns.values]

# Reset the index to turn the Risk_Segment into a column
segment_summary_reset = segment_summary.reset_index()

# First 3 subplot 
plt.figure(figsize=(14, 7))
plt.subplot(1, 3, 1)
sns.barplot(x='Risk_Segment', y='Total_Trans_Amt_mean', data=segment_summary_reset)
plt.title('Average Total Transaction Amount by Risk Segment')

plt.subplot(1, 3, 2)
sns.barplot(x='Risk_Segment', y='Total_Trans_Ct_mean', data=segment_summary_reset)
plt.title('Average Total Transaction Count by Risk Segment')

plt.subplot(1, 3, 3)
sns.barplot(x='Risk_Segment', y='Contacts_Count_12_mon_mean', data=segment_summary_reset)
plt.title('Average Contacts Count by Risk Segment')
plt.tight_layout()
plt.savefig('Outputs/Column_Based_Risk1.png', bbox_inches='tight')
plt.show()

# Second 3 subplot
plt.figure(figsize=(14, 7))
plt.subplot(1, 3, 1)
sns.barplot(x='Risk_Segment', y='Months_Inactive_12_mon_mean', data=segment_summary_reset)
plt.title('Average Months Inactive by Risk Segment')

plt.subplot(1, 3, 2)
sns.barplot(x='Risk_Segment', y='Total_Ct_Chng_Q4_Q1_mean', data=segment_summary_reset)
plt.title('Average Change in Transaction Count Q4 to Q1 by Risk Segment')

plt.subplot(1, 3, 3)
sns.barplot(x='Risk_Segment', y='Total_Revolving_Bal_mean', data=segment_summary_reset)
plt.title('Average Total Revolving Balance by Risk Segment')
plt.tight_layout()
plt.savefig('Outputs/Column_Based_Risk2.png', bbox_inches='tight')
plt.show()
'''