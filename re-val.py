import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow_decision_forests as tfdf

# Load the dataset
data = pd.read_csv('cleaned_realtor_data_new.csv')

# Splitting data into features and target
X = data.drop(columns='price')
y = data['price']

# Splitting data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining preprocessing steps
numeric_features = ['bed', 'bath', 'acre_lot', 'house_size', 'zip_code']
categorical_features = ['status', 'city', 'state']

# Creating transformers
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Applying Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Initializing tree-based models
models = {
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=50, random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=50, random_state=42)
}

mae_scores = {}

for name, model in models.items():
    # Creating and evaluating the pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    mae_scores[name] = mean_absolute_error(y_test, predictions)
print(mae_scores) // print mean absolute error for descion tree, random forrest and extra trees

def load_and_preprocess_data(filepath):
    # Load the dataset
    data = pd.read_csv(filepath)
    
    # Drop the 'city' and 'state' columns and keep only the 'zip_code'
    data.drop(columns=['city', 'state'], inplace=True)
    
    # Convert the zip_code column to categorical
    data['zip_code'] = data['zip_code'].astype('category')

    # Split the dataset into training and testing sets
    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)
    return train_df, test_df

def train_random_forest(train_df, test_df):
    # Convert the Pandas dataframes into TensorFlow datasets
    train_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, task=tfdf.keras.Task.REGRESSION)
    test_dataset = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, task=tfdf.keras.Task.REGRESSION)
    
    # Create and train the random forest model
    model = tfdf.keras.RandomForestModel(task=tfdf.keras.Task.REGRESSION)
    model.fit(train_dataset)

    # Evaluate the model
    evaluation = model.evaluate(test_dataset)
    print(f"Mean Absolute Error: {evaluation[1]}")
    
    # Return the trained model
    return model

if __name__ == "__main__":
    filepath = "cleaned_realtor_data_new.csv"
    train_df, test_df = load_and_preprocess_data(filepath)
    model = train_random_forest(train_df, test_df)
    
    # Optionally: Plot the first tree of the forest (useful for visual inspection)
    # tfdf.model_plotter.plot_model_in_colab(model, tree_idx=0, max_depth=3)

# Loading the provided dataset again
data = pd.read_csv('/mnt/data/cleaned_realtor_data_new.csv')

# Splitting data into features and target
X = data.drop(columns='price')
y = data['price']

# Splitting data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining preprocessing steps
numeric_features = ['bed', 'bath', 'acre_lot', 'house_size', 'zip_code']
categorical_features = ['status', 'city', 'state']

# Creating transformers
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Applying Column Transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Starting with hyperparameter tuning for the ExtraTrees model on 10% of the data
# Subsampling 10% of the training data
X_train_sample, _, y_train_sample, _ = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Defining parameter grid for tuning
param_dist = {
    'model__n_estimators': np.arange(10, 200, 10),
    'model__max_features': ['auto', 'sqrt', 'log2'],
    'model__max_depth': np.arange(10, 100, 10),
    'model__min_samples_split': np.arange(2, 20, 2),
    'model__min_samples_leaf': np.arange(1, 20, 2),
    'model__bootstrap': [True, False]
}

# Creating pipeline with ExtraTrees
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', ExtraTreesRegressor(random_state=42))])

# Running randomized search on 10% of data
search = RandomizedSearchCV(pipeline, param_distributions=param_dist, n_iter=20, cv=3, verbose=1, n_jobs=-1, random_state=42)
search.fit(X_train_sample, y_train_sample)

# Best parameters from the search
best_params = search.best_params_

print(best_params)
