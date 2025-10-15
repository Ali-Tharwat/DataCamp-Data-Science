"""
Machine Learning Model Training Script
Goal: Achieve 80%+ accuracy on classification task
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_and_prepare_data():
    """
    Load and prepare the Titanic dataset for modeling
    """
    # For demonstration, we'll create a synthetic dataset similar to Titanic
    # In practice, you would load from: pd.read_csv('titanic.csv')
    
    # Create synthetic data (1000 samples)
    n_samples = 1000
    
    data = pd.DataFrame({
        'Pclass': np.random.choice([1, 2, 3], n_samples, p=[0.25, 0.25, 0.5]),
        'Sex': np.random.choice(['male', 'female'], n_samples),
        'Age': np.random.normal(30, 15, n_samples).clip(0, 80),
        'SibSp': np.random.choice([0, 1, 2, 3], n_samples, p=[0.6, 0.2, 0.15, 0.05]),
        'Parch': np.random.choice([0, 1, 2], n_samples, p=[0.7, 0.2, 0.1]),
        'Fare': np.random.gamma(2, 20, n_samples),
        'Embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.7, 0.2, 0.1])
    })
    
    # Create target with some correlation to features
    survival_prob = (
        (data['Pclass'] == 1) * 0.3 +
        (data['Sex'] == 'female') * 0.4 +
        (data['Age'] < 18) * 0.2 +
        (data['Fare'] > 50) * 0.1 +
        np.random.random(n_samples) * 0.3
    )
    data['Survived'] = (survival_prob > 0.5).astype(int)
    
    return data

def preprocess_data(df):
    """
    Preprocess the dataset
    """
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Handle missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
    df['Fare'].fillna(df['Fare'].median(), inplace=True)
    
    # Encode categorical variables
    le_sex = LabelEncoder()
    df['Sex'] = le_sex.fit_transform(df['Sex'])
    
    le_embarked = LabelEncoder()
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])
    
    # Feature engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Age_Group'] = pd.cut(df['Age'], bins=[0, 12, 18, 35, 60, 100], labels=[0, 1, 2, 3, 4]).astype(float)
    df['Fare_Group'] = pd.qcut(df['Fare'], q=4, labels=[0, 1, 2, 3], duplicates='drop').astype(float)
    
    # Fill any remaining NaNs
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    return df

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models and evaluate their performance
    """
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42)
    }
    
    results = {}
    
    print("=" * 70)
    print("MODEL EVALUATION RESULTS")
    print("=" * 70)
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        cv_mean = cv_scores.mean()
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'cv_score': cv_mean,
            'predictions': y_pred
        }
        
        print(f"\n{name}:")
        print(f"  Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  CV Score (mean): {cv_mean:.4f} ({cv_mean*100:.2f}%)")
        
    return results

def optimize_best_model(X_train, X_test, y_train, y_test):
    """
    Optimize the best performing model using GridSearchCV
    """
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING - RANDOM FOREST")
    print("=" * 70)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    # Create base model
    rf = RandomForestClassifier(random_state=42)
    
    # Perform grid search
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='accuracy',
        verbose=1
    )
    
    print("\nPerforming Grid Search...")
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate on test set
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best CV Score: {grid_search.best_score_:.4f} ({grid_search.best_score_*100:.2f}%)")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    return best_model, accuracy

def main():
    """
    Main execution function
    """
    print("=" * 70)
    print("MACHINE LEARNING MODEL TRAINING")
    print("Target: Achieve 80%+ Accuracy")
    print("=" * 70)
    
    # Load and prepare data
    print("\n1. Loading and preparing data...")
    df = load_and_prepare_data()
    print(f"   Dataset shape: {df.shape}")
    print(f"   Target distribution:\n{df['Survived'].value_counts()}")
    
    # Preprocess data
    print("\n2. Preprocessing data...")
    df_processed = preprocess_data(df)
    
    # Split features and target
    X = df_processed.drop('Survived', axis=1)
    y = df_processed['Survived']
    
    print(f"   Features: {list(X.columns)}")
    print(f"   Number of features: {X.shape[1]}")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"   Training set size: {X_train.shape[0]}")
    print(f"   Test set size: {X_test.shape[0]}")
    
    # Scale features
    print("\n3. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train and evaluate models
    print("\n4. Training and evaluating models...")
    results = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Find best performing model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_accuracy = results[best_model_name]['accuracy']
    
    print("\n" + "=" * 70)
    print(f"Best Model: {best_model_name}")
    print(f"Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
    print("=" * 70)
    
    # Optimize if accuracy is below 80%
    if best_accuracy < 0.80:
        print("\n5. Optimizing model to achieve 80%+ accuracy...")
        best_model, final_accuracy = optimize_best_model(
            X_train_scaled, X_test_scaled, y_train, y_test
        )
    else:
        final_accuracy = best_accuracy
        best_model = results[best_model_name]['model']
        print(f"\n✓ Target achieved! Accuracy: {final_accuracy*100:.2f}%")
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Final Model Accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")
    
    if final_accuracy >= 0.80:
        print("✓ SUCCESS: Target of 80%+ accuracy achieved!")
    else:
        print("✗ Target of 80%+ accuracy not achieved. Consider:")
        print("  - More feature engineering")
        print("  - Different algorithms (XGBoost, LightGBM)")
        print("  - More data collection")
        print("  - Ensemble methods")
    
    return best_model, final_accuracy

if __name__ == "__main__":
    model, accuracy = main()
