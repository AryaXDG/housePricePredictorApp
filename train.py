import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import os

# Create directories if they don't exist
os.makedirs('models', exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

def load_data(path):
    """Load data from CSV file"""
    df = pd.read_csv(path)
    return df

def preprocess_df(df, fit=True):
    """
    Preprocess the dataframe
    If fit=True: fits scalers/encoders and returns (X, y, fitted_objects_dict)
    If fit=False: uses saved fitted objects and returns X
    """
    df = df.copy()
    
    # Drop Id column
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    
    # Create Age column
    CURRENT_YEAR = 2025
    df['Age'] = CURRENT_YEAR - df['YearBuilt']
    df = df.drop('YearBuilt', axis=1)
    
    # Handle missing values
    numeric_cols = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'Age']
    categorical_cols = ['Location', 'Condition', 'Garage']
    
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())
    
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown')
    
    # Separate features and target
    if 'Price' in df.columns:
        y = df['Price']
        X_df = df.drop('Price', axis=1)
    else:
        y = None
        X_df = df
    
    if fit:
        # Initialize preprocessing objects
        scaler = StandardScaler()
        ohe_location = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        
        # Process numeric features
        numeric_features = X_df[numeric_cols].copy()
        numeric_scaled = scaler.fit_transform(numeric_features)
        
        # Process categorical features
        # Garage: binary mapping
        garage_map = {'Yes': 1, 'No': 0}
        garage_encoded = X_df['Garage'].map(garage_map).fillna(0)
        
        # Condition: ordinal mapping
        condition_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
        condition_encoded = X_df['Condition'].map(condition_map).fillna(1)
        
        # Location: One-hot encoding
        location_encoded = ohe_location.fit_transform(X_df[['Location']])
        location_columns = [f'Location__{cat}' for cat in ohe_location.categories_[0]]
        
        # Combine all features in the specified order
        feature_columns = numeric_cols + ['Garage', 'Condition'] + sorted(location_columns)
        
        # Create final feature matrix
        X = np.column_stack([
            numeric_scaled,
            garage_encoded.values.reshape(-1, 1),
            condition_encoded.values.reshape(-1, 1),
            location_encoded
        ])
        
        # Store fitted objects
        fitted_objects = {
            'scaler': scaler,
            'ohe_location': ohe_location,
            'feature_columns': feature_columns
        }
        
        return X, y, fitted_objects
    
    else:
        # Load saved preprocessing objects
        scaler = joblib.load('models/scaler.pkl')
        ohe_location = joblib.load('models/ohe_location.pkl')
        
        with open('models/feature_columns.json', 'r') as f:
            feature_columns = json.load(f)
        
        # Apply preprocessing
        numeric_features = X_df[numeric_cols].copy()
        numeric_scaled = scaler.transform(numeric_features)
        
        garage_map = {'Yes': 1, 'No': 0}
        garage_encoded = X_df['Garage'].map(garage_map).fillna(0)
        
        condition_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
        condition_encoded = X_df['Condition'].map(condition_map).fillna(1)
        
        location_encoded = ohe_location.transform(X_df[['Location']])
        
        X = np.column_stack([
            numeric_scaled,
            garage_encoded.values.reshape(-1, 1),
            condition_encoded.values.reshape(-1, 1),
            location_encoded
        ])
        
        return X

def save_preprocessing_objects(obj_dict):
    """Save preprocessing objects"""
    joblib.dump(obj_dict['scaler'], 'models/scaler.pkl')
    joblib.dump(obj_dict['ohe_location'], 'models/ohe_location.pkl')
    
    with open('models/feature_columns.json', 'w') as f:
        json.dump(obj_dict['feature_columns'], f)

def compute_metrics(y_true, y_pred):
    """Compute R2 and RMSE metrics"""
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'r2': r2, 'rmse': rmse}

def generate_plots(df, rf_model, feature_columns):
    """Generate and save visualization plots"""
    
    # Price distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['Price'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Price (₹)')
    plt.ylabel('Frequency')
    plt.title('Price Distribution')
    plt.tight_layout()
    plt.savefig('static/plots/price_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Area vs Price scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(df['Area'], df['Price'], alpha=0.6, color='coral')
    
    # Add regression line
    z = np.polyfit(df['Area'], df['Price'], 1)
    p = np.poly1d(z)
    plt.plot(df['Area'], p(df['Area']), "r--", alpha=0.8, linewidth=2)
    
    plt.xlabel('Area (sq ft)')
    plt.ylabel('Price (₹)')
    plt.title('Area vs Price')
    plt.tight_layout()
    plt.savefig('static/plots/area_vs_price.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Feature importance from Random Forest
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': importances
    }).sort_values('importance', ascending=False).head(10)
    
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(feature_importance_df)), feature_importance_df['importance'])
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['feature'])
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importances (Random Forest)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('static/plots/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Price vs Age
    df_age_price = df.groupby('Age')['Price'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    plt.plot(df_age_price['Age'], df_age_price['Price'], marker='o', linewidth=2, markersize=4)
    plt.xlabel('Age (years)')
    plt.ylabel('Average Price (₹)')
    plt.title('Average Price vs Age')
    plt.tight_layout()
    plt.savefig('static/plots/price_vs_year.png', dpi=150, bbox_inches='tight')
    plt.close()

def main():
    """Main training function"""
    print("Loading data...")
    
    # Try different possible filenames
    data_files = ['data/House Price Prediction Dataset.csv', 'trainingDataset.csv', 'data/trainingDataset.csv']
    df = None
    
    for file_path in data_files:
        try:
            df = load_data(file_path)
            print(f"Data loaded from {file_path}")
            break
        except FileNotFoundError:
            continue
    
    if df is None:
        raise FileNotFoundError("Could not find dataset. Please ensure the CSV file is available.")
    
    print(f"Dataset shape: {df.shape}")
    
    # Data validation and type conversion
    numeric_columns = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt', 'Price']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Preprocess data
    print("Preprocessing data...")
    X, y, fitted_objects = preprocess_df(df, fit=True)
    
    # Save preprocessing objects
    save_preprocessing_objects(fitted_objects)
    print("Preprocessing objects saved.")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    print("Training models...")
    
    # Linear Regression
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    joblib.dump(linear_model, 'models/linear_model.pkl')
    
    # Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    joblib.dump(rf_model, 'models/rf_model.pkl')
    
    # Gradient Boosting
    gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    gb_model.fit(X_train, y_train)
    joblib.dump(gb_model, 'models/gb_model.pkl')
    
    # Evaluate models
    print("Evaluating models...")
    models = {
        'LinearRegression': linear_model,
        'RandomForest': rf_model,
        'GradientBoosting': gb_model
    }
    
    metrics = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        model_metrics = compute_metrics(y_test, y_pred)
        metrics[name] = model_metrics
        print(f"{name} - R²: {model_metrics['r2']:.3f}, RMSE: {model_metrics['rmse']:.0f}")
    
    # Save metrics
    with open('models/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Generate plots
    print("Generating plots...")
    # Add Age column to original df for plotting
    CURRENT_YEAR = 2025
    df['Age'] = CURRENT_YEAR - df['YearBuilt']
    generate_plots(df, rf_model, fitted_objects['feature_columns'])
    
    # Print summary
    print("\n=== TRAINING SUMMARY ===")
    print(f"Dataset shape: {df.shape}")
    print("\nModel Performance:")
    for name, model_metrics in metrics.items():
        print(f"  {name}: R² = {model_metrics['r2']:.3f}, RMSE = {model_metrics['rmse']:.0f}")
    
    print("\nTop 5 Feature Importances (Random Forest):")
    importances = rf_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': fitted_objects['feature_columns'],
        'importance': importances
    }).sort_values('importance', ascending=False).head(5)
    
    for _, row in feature_importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    main()