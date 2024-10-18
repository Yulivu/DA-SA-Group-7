import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau, pointbiserialr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import statsmodels.api as sm

# Load data
data = pd.read_csv(r'C:\Users\debuf\Desktop\china\project\dataset\Data.csv')
data_cleaned = data[['Education level', 'Are you disgusted with the content of Chinese mainstream culture on the Douyin platform?', 'gender']].dropna()
data_cleaned.columns = ['Education Level', 'Aversion to Mainstream Content', 'Gender']

# Mapping labels for education and gender
education_mapping = {
    1: 'Primary School',
    2: 'Junior High School',
    3: 'High School',
    4: 'Undergraduate',
    5: 'Postgraduate',
    6: 'PhD Student'
}

gender_mapping = {
    1: 'Man',
    2: 'Woman',
    3: 'Third Gender'
}

# Apply mappings to the data
data_cleaned['Education Level Label'] = data_cleaned['Education Level'].map(education_mapping)
data_cleaned['Gender Label'] = data_cleaned['Gender'].map(gender_mapping)

# Regression Model
X = pd.get_dummies(data_cleaned[['Education Level', 'Gender']], drop_first=True)
y = data_cleaned['Aversion to Mainstream Content'].astype(float)
X = sm.add_constant(X)

# Fit OLS regression
linear_model = sm.OLS(y, X).fit()
print(linear_model.summary())

# Logistic Regression
# Create a binary target where "High Aversion" is True (1) for aversion values <= 3, False (0) otherwise
y_binary = (data_cleaned['Aversion to Mainstream Content'] <= 3).astype(int)  # Now 1-3 is High Aversion, 4-7 is not
data_cleaned['High Aversion'] = y_binary  # For visualization purposes

# Fit logistic regression
logit_model = LogisticRegression()
logit_model.fit(X, y_binary)
logit_coefficients = logit_model.coef_
print(f'Logistic Regression Coefficients: {logit_coefficients}')

# AUC-ROC Calculation and Visualization
# Get probability estimates for the positive class (High Aversion = 1)
y_prob = logit_model.predict_proba(X)[:, 1]

# Calculate AUC score
auc = roc_auc_score(y_binary, y_prob)
print(f'AUC: {auc}')

# Generate ROC curve data
fpr, tpr, thresholds = roc_curve(y_binary, y_prob)



# Correlation Tests
# Spearman Rank Correlation
spearman_corr, spearman_p = spearmanr(data_cleaned['Education Level'], data_cleaned['Aversion to Mainstream Content'])
print(f'1. Spearman Correlation: {spearman_corr}, p-value: {spearman_p}')

# Kendall Tau Correlation
kendall_corr, kendall_p = kendalltau(data_cleaned['Education Level'], data_cleaned['Aversion to Mainstream Content'])
print(f'2. Kendall Correlation: {kendall_corr}, p-value: {kendall_p}')

# Point-biserial correlation (for binary variable High Aversion vs Education Level)
point_biserial_corr, point_biserial_p = pointbiserialr(data_cleaned['High Aversion'], data_cleaned['Education Level'])
print(f'3. Point-biserial Correlation: {point_biserial_corr}, p-value: {point_biserial_p}')

# ========== Visualization for Correlation Tests ==========
# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}', color='darkorange', lw=2)
plt.plot([0, 1], [0, 1], 'k--', lw=2)  # Diagonal line representing random guess
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve for Logistic Regression Model')
plt.legend(loc='lower right')
plt.grid(True)
plt.savefig(r'C:\Users\debuf\Desktop\china\project\src\roc_curve.png')
plt.show()

# 1. Spearman Correlation Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Education Level', y='Aversion to Mainstream Content', data=data_cleaned)
plt.title(f'Spearman Correlation: {spearman_corr:.2f}, p-value: {spearman_p:.2e}')
plt.xlabel('Education Level')
plt.ylabel('Aversion to Mainstream Content')
plt.grid(True)
# Adding trend line
sns.regplot(x='Education Level', y='Aversion to Mainstream Content', data=data_cleaned, scatter=False, color='red', ci=None)
plt.savefig(r'C:\Users\debuf\Desktop\china\project\src\spearman_correlation_visualization.png')
plt.show()

# 2. Kendall Tau Correlation Visualization
plt.figure(figsize=(8, 6))
sns.scatterplot(x='Education Level', y='Aversion to Mainstream Content', data=data_cleaned)
plt.title(f'Kendall Correlation: {kendall_corr:.2f}, p-value: {kendall_p:.2e}')
plt.xlabel('Education Level')
plt.ylabel('Aversion to Mainstream Content')
plt.grid(True)
# Adding trend line
sns.regplot(x='Education Level', y='Aversion to Mainstream Content', data=data_cleaned, scatter=False, color='blue', ci=None)
plt.savefig(r'C:\Users\debuf\Desktop\china\project\src\kendall_correlation_visualization.png')
plt.show()

# 3. Point-biserial Correlation Visualization
plt.figure(figsize=(8, 6))
sns.stripplot(x='High Aversion', y='Education Level', data=data_cleaned, jitter=True)
plt.title(f'Point-biserial Correlation: {point_biserial_corr:.2f}, p-value: {point_biserial_p:.2e}')
plt.xlabel('High Aversion (1 = Yes, 0 = No)')
plt.ylabel('Education Level')
plt.grid(True)
plt.savefig(r'C:\Users\debuf\Desktop\china\project\src\point_biserial_correlation_visualization.png')
plt.show()

# ========== Logistic Regression Coefficients Visualization ==========

logit_coefficients_df = pd.DataFrame({
    'Variable': ['Const'] + list(X.columns[1:]),  # List of independent variables
    'Coefficient': logit_model.coef_[0]
})

plt.figure(figsize=(8, 6))
sns.barplot(x='Coefficient', y='Variable', data=logit_coefficients_df, palette='viridis')
plt.title('Logistic Regression Coefficients')
plt.xlabel('Coefficient Value')
plt.ylabel('Variable')
plt.grid(True)
plt.savefig(r'C:\Users\debuf\Desktop\china\project\src\logistic_regression_coefficients.png')
plt.show()

# ========== Other Visualizations ==========

# Education Level Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Education Level Label', data=data_cleaned, order=data_cleaned['Education Level Label'].value_counts().index)
plt.title('Education Level Distribution')
plt.savefig(r'C:\Users\debuf\Desktop\china\project\src\education_distribution.png')
plt.show()

# Education Level Distribution by Gender
plt.figure(figsize=(8, 6))
sns.countplot(x='Education Level Label', hue='Gender Label', data=data_cleaned, order=data_cleaned['Education Level Label'].value_counts().index)
plt.title('Education Level Distribution by Gender')
plt.savefig(r'C:\Users\debuf\Desktop\china\project\src\gender_education_distribution.png')
plt.show()

# Aversion to Mainstream Content by Gender
plt.figure(figsize=(8, 6))
sns.boxplot(x='Gender Label', y='Aversion to Mainstream Content', data=data_cleaned)
plt.title('Aversion to Mainstream Content by Gender')
plt.savefig(r'C:\Users\debuf\Desktop\china\project\src\gender_aversion_distribution.png')
plt.show()

# Aversion to Mainstream Content for All Users
plt.figure(figsize=(8, 6))
sns.boxplot(y='Aversion to Mainstream Content', data=data_cleaned)
plt.title('Aversion to Mainstream Content for All Users')
plt.ylabel('Aversion to Mainstream Content (1=Highly Offended, 7=Not Offended)')
plt.savefig(r'C:\Users\debuf\Desktop\china\project\src\aversion_boxplot_all.png')
plt.show()

# Filter the group that feels highly offended (Aversion scores of 1-3)
offended_group = data_cleaned[data_cleaned['Aversion to Mainstream Content'] <= 3]

# Education level distribution of the highly offended group
plt.figure(figsize=(8, 6))
sns.countplot(x='Education Level Label', data=offended_group, order=offended_group['Education Level Label'].value_counts().index)
plt.title('Education Level Distribution for Highly Offended Group (Aversion 1-3)')
plt.xlabel('Education Level')
plt.ylabel('Count')
plt.savefig(r'C:\Users\debuf\Desktop\china\project\src\offended_group_education_distribution.png')
plt.show()

# Pie Chart for Aversion Levels
plt.figure(figsize=(8, 8))
content_counts = data_cleaned['Aversion to Mainstream Content'].value_counts()
labels = [f'{i} (Rating {i})' for i in content_counts.index]
plt.pie(content_counts, labels=labels, autopct='%1.1f%%', startangle=90, counterclock=False)
plt.title('Distribution of Aversion to Mainstream Content (1=Highly Offended, 7=Not Offended)')
plt.savefig(r'C:\Users\debuf\Desktop\china\project\src\aversion_pie_chart.png')
plt.show()

# Generate Pie Chart for Education Background for Each Aversion Level
for aversion_level in range(1, 8):  # Iterating through aversion levels 1-7
    plt.figure(figsize=(8, 8))

    # Filter data for each aversion level
    aversion_group = data_cleaned[data_cleaned['Aversion to Mainstream Content'] == aversion_level]

    # Count the occurrences of each education level in this aversion group
    education_counts = aversion_group['Education Level Label'].value_counts()

    # Custom autopct to show both percentage and count
    def autopct_format(values):
        def my_format(pct):
            total = sum(values)
            count = int(round(pct * total / 100.0))  # Calculate the count based on the percentage
            return f'{pct:.1f}% ({count:d})'
        return my_format

    # Generate the pie chart
    plt.pie(education_counts, labels=education_counts.index, autopct=autopct_format(education_counts), startangle=90, counterclock=False)
    plt.title(f'Education Background Distribution for Aversion Level {aversion_level}')

    # Save the pie chart to a file
    plt.savefig(
        f'C:\\Users\\debuf\\Desktop\\china\\project\\src\\aversion_level_{aversion_level}_education_pie_chart.png')
    plt.show()
