import pandas as pd
import numpy as np
from sksurv.linear_model import CoxnetSurvivalAnalysis
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, accuracy_score
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.utils import concordance_index
from lifelines.plotting import add_at_risk_counts

# Load the data
data = pd.read_csv('C:\\code\\Prognosis_prediction\\retro_external_pro_features_PFS.csv')

# Split the data into training and test sets based on the patient ID
train_data = data[data['patient'].str.contains('RETRO')]
test_data = data[data['patient'].str.contains('EXTERNAL')]  # Similarly, here can be replaced with prospective test set

# Extract features and targets
X_train = train_data.drop(columns=['patient', 'PFS', 'event'])
X_test = test_data.drop(columns=['patient', 'PFS', 'event'])
y_train = train_data[['event', 'PFS']].copy()
y_test = test_data[['event', 'PFS']].copy()

# Convert the 'event' column to boolean
y_train['event'] = y_train['event'].astype(bool)
y_test['event'] = y_test['event'].astype(bool)

# Convert the dataframes to structured arrays
y_train_structured = y_train.to_records(index=False)
y_test_structured = y_test.to_records(index=False)

# Initialize and train the CoxnetSurvivalAnalysis model
coxnet_model = CoxnetSurvivalAnalysis(l1_ratio=0.1, alpha_min_ratio=0.01)
coxnet_model.fit(X_train, y_train_structured)

# Predict the risk scores for both the training and test sets
risk_scores_train = coxnet_model.predict(X_train)
risk_scores_test = coxnet_model.predict(X_test)


def bootstrap_cindex(X, y, y_event, model, n_iterations=1000):
    c_indexes = []
    for _ in range(n_iterations):
        # Resample the data
        indices = np.random.randint(0, len(X), len(X))
        X_resampled = X.iloc[indices]
        y_resampled = y.iloc[indices]  # Use .iloc for correct indexing
        y_event_resampled = y_event.iloc[indices]  # Use .iloc for correct indexing

        # Check if we have enough events to calculate C-index
        if y_event_resampled.sum() < 2:  # At least two events required for comparison
            continue

        # Predict the risk scores
        risk_scores_resampled = model.predict(X_resampled)

        # Calculate C-index
        c_index = concordance_index(y_resampled, -risk_scores_resampled, y_event_resampled)
        c_indexes.append(c_index)

    if len(c_indexes) == 0:  # Check if no valid c_indexes were calculated
        raise ValueError("No valid C-index could be calculated with the provided samples.")

    return np.percentile(c_indexes, [2.5, 97.5]), np.std(c_indexes)


# Calculate the C-index and confidence intervals for the training and testing sets
c_index_train = concordance_index(y_train['PFS'], -risk_scores_train, y_train['event'])
c_index_test = concordance_index(y_test['PFS'], -risk_scores_test, y_test['event'])
c_index_train_ci, c_index_train_std = bootstrap_cindex(X_train, y_train['PFS'], y_train['event'], coxnet_model)
c_index_test_ci, c_index_test_std = bootstrap_cindex(X_test, y_test['PFS'], y_test['event'], coxnet_model)

# Print C-index values, confidence intervals, and standard deviations
print(f"Training C-index with fixed parameters (RETRO): {c_index_train}, 95% CI: {c_index_train_ci}, Std: {c_index_train_std}")
print(f"Validation C-index with fixed parameters (EXTERNAL): {c_index_test}, 95% CI: {c_index_test_ci}, Std: {c_index_test_std}")

# Calculate the AUC for the training and testing sets
train_event_observed = y_train_structured['event']
test_event_observed = y_test_structured['event']

train_auc = roc_auc_score(train_event_observed, risk_scores_train)
test_auc = roc_auc_score(test_event_observed, risk_scores_test)

# Print AUC values
print(f"Training AUC with fixed parameters (RETRO): {train_auc}")
print(f"Validation AUC with fixed parameters (EXTERNAL): {test_auc}")

fpr_train, tpr_train, thresholds_train = roc_curve(train_event_observed, risk_scores_train)
fpr_test, tpr_test, thresholds_test = roc_curve(test_event_observed, risk_scores_test)

# Determine the optimal threshold
optimal_idx_train = np.argmax(tpr_train - fpr_train)
optimal_threshold_train = thresholds_train[optimal_idx_train]

# Binarize predictions based on the optimal threshold
train_predictions = risk_scores_train >= optimal_threshold_train
test_predictions = risk_scores_test >= optimal_threshold_train

# Calculate confusion matrix for the train set
tn1, fp1, fn1, tp1 = confusion_matrix(train_event_observed, train_predictions).ravel()

# Calculate sensitivity, specificity, PPV, NPV
sensitivity_train = tp1 / (tp1 + fn1)
specificity_train = tn1 / (tn1 + fp1)
ppv_train = tp1 / (tp1 + fp1)
npv_train = tn1 / (tn1 + fn1)
accuracy_train = accuracy_score(train_event_observed, train_predictions)

# Print the metrics
print(f"Sensitivity_train: {sensitivity_train}")
print(f"Specificity_train: {specificity_train}")
print(f"PPV_train: {ppv_train}")
print(f"NPV_train: {npv_train}")
print(f"Accuracy_train: {accuracy_train}")

# Calculate confusion matrix for the test set
tn2, fp2, fn2, tp2 = confusion_matrix(test_event_observed, test_predictions).ravel()

# Calculate sensitivity, specificity, PPV, NPV
sensitivity_test = tp2 / (tp2 + fn2)
specificity_test = tn2 / (tn2 + fp2)
ppv_test = tp2 / (tp2 + fp2)
npv_test = tn2 / (tn2 + fn2)
accuracy_test = accuracy_score(test_event_observed, test_predictions)

# Print the metrics
print(f"Sensitivity_test: {sensitivity_test}")
print(f"Specificity_test: {specificity_test}")
print(f"PPV_test: {ppv_test}")
print(f"NPV_test: {npv_test}")
print(f"Accuracy_test: {accuracy_test}")

# Plot the ROC curves
plt.figure()
plt.plot(fpr_train, tpr_train, label=f'ROC curve (area = {train_auc:.2f}) of training set')
plt.plot(fpr_test, tpr_test, label=f'ROC curve (area = {test_auc:.2f}) of validation set (EXTERNAL)')

plt.plot([0, 1], [0, 1], 'k--')  # Dashed diagonal
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Training and Validation Sets')
plt.legend(loc="lower right")
plt.show()

# Extract and print feature coefficients
feature_names = X_train.columns
coefficients = coxnet_model.coef_.ravel()
feature_coefficients = {feature: coef for feature, coef in zip(feature_names, coefficients)}
for feature, coef in feature_coefficients.items():
    print(f"{feature}: {coef}")

# Add risk scores to the original dataframe
data.loc[data['patient'].str.contains('RETRO'), 'risk_score'] = risk_scores_train
data.loc[data['patient'].str.contains('EXTERNAL'), 'risk_score'] = risk_scores_test

# Save the updated dataframe to a new CSV file
data.to_csv('C:\\path\\to\\retro_external_features_PFS_riskscore.csv', index=False)

# Load the updated data
data = pd.read_csv('C:\\path\\to\\retro_external_features_PFS_riskscore.csv')

# Filter RETRO and EXTERNAL groups
retro_data = data[data['patient'].str.contains('RETRO')]
external_data = data[data['patient'].str.contains('EXTERNAL')]

# Calculate the median risk score for the RETRO group
median_risk_score = retro_data['risk_score'].quantile(0.50)
print(f"The median risk score for RETRO group is: {median_risk_score}")


retro_data.loc[:, 'Group'] = np.where(retro_data['risk_score'] >= median_risk_score, 'High', 'Low')
external_data.loc[:, 'Group'] = np.where(external_data['risk_score'] >= median_risk_score, 'High', 'Low')


kmf_high = KaplanMeierFitter()
kmf_low = KaplanMeierFitter()
cph = CoxPHFitter()

# Calculate the maximum PFS (peritoneal recurrence-free survival) value and convert it to years
max_pfs_years = max(retro_data['PFS'].max(), external_data['PFS'].max()) / 365.25

# Set time points for Kaplan-Meier curves
time_points = np.arange(0, int(max_pfs_years) + 1, 1)  # From zero to the maximum PFS value, one time point per year

for data_set, label in zip([retro_data, external_data], ['Training', 'Validation (EXTERNAL)']):
    print(f"Processing {label} data:")
    plt.figure(figsize=(5, 5.5))
    ax = plt.subplot(111)

    high_group_data = data_set[data_set['Group'] == 'High']
    low_group_data = data_set[data_set['Group'] == 'Low']

    # Examine and draw High Risk groups
    if not high_group_data.empty:
        durations_high = high_group_data['PFS'] / 365.25
        kmf_high.fit(durations=durations_high, event_observed=high_group_data['event'], label='High Risk')
        kmf_high.plot_survival_function(ax=ax)

    # Examine and draw Low Risk groups
    if not low_group_data.empty:
        durations_low = low_group_data['PFS'] / 365.25
        kmf_low.fit(durations=durations_low, event_observed=low_group_data['event'], label='Low Risk')
        kmf_low.plot_survival_function(ax=ax)

    # Add risk table, one time point per year
    add_at_risk_counts(kmf_high, kmf_low, ax=ax, labels=['High Risk', 'Low Risk'], xticks=time_points)

    plt.title(f'Kaplan-Meier Survival Curves for {label} Set Based on Risk Score')
    plt.xlabel('Time (years)')
    plt.ylabel('Peritoneal-free survival')
    plt.legend(title='Group')
    plt.xticks(time_points)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Binary indicator for high risk group
    if not data_set.empty:
        data_set['High_Risk'] = (data_set['risk_score'] >= median_risk_score).astype(int)
        cph.fit(data_set[['PFS', 'event', 'High_Risk']], duration_col='PFS', event_col='event')
        print(f"Cox Proportional Hazards Model Summary for {label} Group:")
        cph.print_summary()  # Prints the model summary
    else:
        print(f"Warning: No data available for Cox Model fitting in {label}")