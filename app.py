import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from datetime import datetime
import uuid
import os

# Set random seed for reproducibility
np.random.seed(42)

# 1. Data Loading and Preprocessing
def load_and_preprocess_data(file_path):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Basic data cleaning
    # Handle missing values
    df = df.dropna()
    
    # Convert categorical variables
    categorical_columns = ['BusinessTravel', 'Department', 'EducationField', 
                         'Gender', 'JobRole', 'MaritalStatus', 'Over18', 'OverTime']
    
    # Create label encoders dictionary
    le_dict = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        le_dict[col] = le
    
    # Map numeric ratings to meaningful labels
    education_map = {1: 'Below College', 2: 'College', 3: 'Bachelor', 
                    4: 'Master', 5: 'Doctor'}
    satisfaction_map = {1: 'Low', 2: 'Medium', 3: 'High', 4: 'Very High'}
    performance_map = {1: 'Low', 2: 'Good', 3: 'Excellent', 4: 'Outstanding'}
    worklife_map = {1: 'Bad', 2: 'Good', 3: 'Better', 4: 'Best'}
    
    df['Education_Level'] = df['Education'].map(education_map)
    df['EnvironmentSatisfaction_Level'] = df['EnvironmentSatisfaction'].map(satisfaction_map)
    df['JobInvolvement_Level'] = df['JobInvolvement'].map(satisfaction_map)
    df['JobSatisfaction_Level'] = df['JobSatisfaction'].map(satisfaction_map)
    df['PerformanceRating_Level'] = df['PerformanceRating'].map(performance_map)
    df['RelationshipSatisfaction_Level'] = df['RelationshipSatisfaction'].map(satisfaction_map)
    df['WorkLifeBalance_Level'] = df['WorkLifeBalance'].map(worklife_map)
    
    return df, le_dict

# 2. Data Warehouse Features
def add_warehouse_features(df):
    # Add time-invariant features
    df['EmployeeID'] = [str(uuid.uuid4()) for _ in range(len(df))]
    df['RecordCreationDate'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    df['LastUpdatedDate'] = df['RecordCreationDate']
    
    # Add derived features
    df['TenurePerCompany'] = df['TotalWorkingYears'] / (df['NumCompaniesWorked'] + 1)
    df['IncomePerYearOfService'] = df['MonthlyIncome'] / (df['YearsAtCompany'] + 1)
    
    return df

# 3. Exploratory Data Analysis
def perform_eda(df, output_dir='eda_plots'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Attrition distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Attrition', data=df)
    plt.title('Attrition Distribution')
    plt.savefig(f'{output_dir}/attrition_distribution.png')
    plt.close()
    
    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    sns.heatmap(df[numeric_cols].corr(), annot=False, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig(f'{output_dir}/correlation_heatmap.png')
    plt.close()
    
    # Job Satisfaction vs Attrition
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Attrition', y='JobSatisfaction', data=df)
    plt.title('Job Satisfaction vs Attrition')
    plt.savefig(f'{output_dir}/job_satisfaction_vs_attrition.png')
    plt.close()

# 4. Model Development
def develop_models(df):
    # Prepare features and target
    features = ['Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
                'Education', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
                'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'MaritalStatus',
                'MonthlyIncome', 'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
                'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel',
                'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
                'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion',
                'YearsWithCurrManager']
    
    X = df[features]
    y = df['Attrition'].map({'Yes': 1, 'No': 0})
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale numerical features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest Classifier with balanced class weights
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X_train, y_train)  # Random Forest doesn't need scaled data
    rf_predictions = rf_model.predict(X_test)
    
    # Logistic Regression with scaled data and balanced class weights
    lr_model = LogisticRegression(max_iter=2000, random_state=42, class_weight='balanced')
    lr_model.fit(X_train_scaled, y_train)
    lr_predictions = lr_model.predict(X_test_scaled)
    
    # Print model performance
    print("Random Forest Classification Report:")
    print(classification_report(y_test, rf_predictions))
    
    print("\nLogistic Regression Classification Report:")
    print(classification_report(y_test, lr_predictions))
    
    # Feature importance from Random Forest
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance.head(10))
    plt.title('Top 10 Feature Importance for Attrition Prediction')
    plt.savefig('feature_importance.png')
    plt.close()
    
    return rf_model, lr_model, feature_importance

# 5. Management Reports
def generate_management_reports(df, output_dir='reports'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # i) Sum of monthly income, average job satisfaction by Job Role and Education Field
    income_satisfaction = df.groupby(['JobRole', 'EducationField']).agg({
        'MonthlyIncome': 'sum',
        'JobSatisfaction': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='JobRole', y='MonthlyIncome', hue='EducationField', data=income_satisfaction)
    plt.title('Total Monthly Income by Job Role and Education Field')
    plt.xticks(rotation=45)
    plt.savefig(f'{output_dir}/income_by_jobrole_education.png')
    plt.close()
    
    # ii) Gender wise, job role wise average JobSatisfaction and Environment satisfaction
    gender_satisfaction = df.groupby(['Gender', 'JobRole']).agg({
        'JobSatisfaction': 'mean',
        'EnvironmentSatisfaction': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='JobRole', y='JobSatisfaction', hue='Gender', data=gender_satisfaction)
    plt.title('Average Job Satisfaction by Gender and Job Role')
    plt.xticks(rotation=45)
    plt.savefig(f'{output_dir}/job_satisfaction_by_gender_jobrole.png')
    plt.close()
    
    # iii) EducationField wise, JobRole wise average Hourly Rate, monthly income, JobSatisfaction
    education_metrics = df.groupby(['EducationField', 'JobRole']).agg({
        'HourlyRate': 'mean',
        'MonthlyIncome': 'mean',
        'JobSatisfaction': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='JobRole', y='MonthlyIncome', hue='EducationField', data=education_metrics)
    plt.title('Average Monthly Income by Education Field and Job Role')
    plt.xticks(rotation=45)
    plt.savefig(f'{output_dir}/monthly_income_by_education_jobrole.png')
    plt.close()
    
    # iv) Department wise, gender wise average job satisfaction
    dept_gender_satisfaction = df.groupby(['Department', 'Gender']).agg({
        'JobSatisfaction': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Department', y='JobSatisfaction', hue='Gender', data=dept_gender_satisfaction)
    plt.title('Average Job Satisfaction by Department and Gender')
    plt.savefig(f'{output_dir}/job_satisfaction_by_dept_gender.png')
    plt.close()
    
    # v) Average DistanceFromHome by gender, department, and JobRole
    distance_breakdown = df.groupby(['Gender', 'Department', 'JobRole']).agg({
        'DistanceFromHome': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x='JobRole', y='DistanceFromHome', hue='Gender', data=distance_breakdown)
    plt.title('Average Distance From Home by Gender and Job Role')
    plt.xticks(rotation=45)
    plt.savefig(f'{output_dir}/distance_by_gender_jobrole.png')
    plt.close()
    
    # vi) Average monthly income by education and attrition
    income_by_education_attrition = df.groupby(['Education_Level', 'Attrition']).agg({
        'MonthlyIncome': 'mean'
    }).reset_index()
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Education_Level', y='MonthlyIncome', hue='Attrition', 
                data=income_by_education_attrition)
    plt.title('Average Monthly Income by Education and Attrition')
    plt.savefig(f'{output_dir}/income_by_education_attrition.png')
    plt.close()

# Main execution
if __name__ == '__main__':
    # Specify the dataset path
    file_path = r"C:\Users\bijay\Documents\DIY2_ABHINAV_KUMAR\DIY_2_EmployeeAnalysis\HR Employee Attrition.csv"
    
    # Load and preprocess data
    df, le_dict = load_and_preprocess_data(file_path)
    
    # Add data warehouse features
    df = add_warehouse_features(df)
    
    # Perform EDA
    perform_eda(df)
    
    # Develop models
    rf_model, lr_model, feature_importance = develop_models(df)
    
    # Generate management reports
    generate_management_reports(df)
    
    # Save processed dataset
    df.to_csv('processed_employee_data.csv', index=False)
    
    # Print feature importance
    print("\nTop 10 Features Contributing to Attrition:")
    print(feature_importance.head(10))
    
    # Save feature importance
    feature_importance.to_csv('feature_importance.csv', index=False)