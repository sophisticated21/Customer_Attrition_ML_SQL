feature_columns = ['Age_Group', 'Dependent_count', 'Months_on_book', 
                   'Total_Relationship_Count', 'Months_Inactive_12_mon', 
                   'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 
                   'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 
                   'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
                   'Gender_Label', 'Education_Level_Label', 'Marital_Status_Label', 
                   'Income_Category_Label', 'Card_Category_Label']

target_column = ['Attrition_Flag_Label']

lr_params = {
    'C': [10, 50, 100],
    'penalty': ['l1'],
    'solver': ['liblinear']
}


dt_params = {
    'max_depth': [None, 5, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [4, 6, 8, 10],
    'max_features': [None, 'sqrt']
}

# RF grid search takes about 10 minutes, you may want to use direct params(below)
'''
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt']
}
'''

rf_params = {
    'n_estimators': [200],
    'max_depth': [20],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'max_features': ['sqrt']
}