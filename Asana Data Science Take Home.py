import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# Load data
users_data = pd.read_csv("https://s3.amazonaws.com/asana-data-interview/takehome_users-intern.csv")
user_engagement_data = pd.read_csv("https://s3.amazonaws.com/asana-data-interview/takehome_user_engagement-intern.csv")

# Convert time_stamp to datetime
user_engagement_data['login_timestamp'] = pd.to_datetime(user_engagement_data['time_stamp'])

# Calculate the number of days between logins
user_engagement_data['days_between_logins'] = user_engagement_data.groupby('user_id')['login_timestamp'].diff().dt.days

# Flag users as adopted if they have logged in at least 3 times within 7 days
adopted_user_flags = user_engagement_data.groupby('user_id')['days_between_logins'].apply(
    lambda x: (x <= 7).sum() >= 3
).reset_index()
adopted_user_flags.columns = ['user_id', 'is_adopted']
adopted_user_ids = adopted_user_flags[adopted_user_flags['is_adopted']]['user_id']

# Merge with the user data
users_data['is_adopted_user'] = users_data['object_id'].isin(adopted_user_ids)

# Calculate adoption rate
user_adoption_rate = users_data['is_adopted_user'].mean()
print(f"User Adoption Rate: {user_adoption_rate:.2%}")

# Feature Engineering
# Convert creation_time and last_session_creation_time to datetime
# Replace the format string as needed
users_data['account_creation_time'] = pd.to_datetime(users_data['creation_time'], format='%Y-%m-%d', errors='coerce')
users_data['last_login_time'] = pd.to_datetime(users_data['last_session_creation_time'], unit='s', errors='coerce')

# Calculate days since last login
users_data['days_since_last_login'] = (users_data['last_login_time'] - users_data['account_creation_time']).dt.days.fillna(0).astype(int)

# Simplify email domain
users_data['simplified_email_domain'] = users_data['email_domain'].apply(
    lambda domain: 'other' if domain not in ['gmail.com', 'yahoo.com', 'outlook.com'] else domain
)

# One-hot encoding for categorical variables
encoder = OneHotEncoder(sparse_output=False)
categorical_features = ['creation_source', 'simplified_email_domain']
encoded_features = encoder.fit_transform(users_data[categorical_features])
encoded_features_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))

# Prepare the data for modeling
features = pd.concat([
    users_data[['opted_in_to_mailing_list', 'enabled_for_marketing_drip', 'days_since_last_login']],
    encoded_features_df
], axis=1)
target = users_data['is_adopted_user']

# Handle missing values by imputing
imputer = SimpleImputer(strategy='mean')
features_imputed = imputer.fit_transform(features)

# Split the data into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(features_imputed, target, test_size=0.2, random_state=42)

# Random Forest Classifier Model with class weights
random_forest_model = RandomForestClassifier(class_weight='balanced', random_state=42)
random_forest_model.fit(features_train, target_train)
predicted_target = random_forest_model.predict(features_test)

# Model Evaluation
print("Confusion Matrix:")
print(confusion_matrix(target_test, predicted_target))

print("\nClassification Report:")
print(classification_report(target_test, predicted_target))
