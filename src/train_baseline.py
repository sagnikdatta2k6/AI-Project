import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def train_and_evaluate(csv_path="data_features.csv"):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found! Run extract_features.py or generate_mock_data.py first.")
        return

    # 1. Load feature dataset
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {len(df)} samples")
    print(f"Class distribution: {df['label'].value_counts().to_dict()} (0=Spontaneous, 1=Scripted)\n")

    # 2. Separate features (X) and target label (y)
    feature_cols = [col for col in df.columns if col not in ['filename', 'label']]
    X = df[feature_cols]
    y = df['label']

    # 3. Define Baseline Model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=4)

    # 4. Stratified 5-Fold Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Run cross-validation
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)

    print("=" * 45)
    print("        5-FOLD CROSS-VALIDATION RESULTS     ")
    print("=" * 45)
    for metric in scoring:
        scores = cv_results[f'test_{metric}']
        print(f"Mean {metric.capitalize():<12}: {scores.mean():.3f} (+/- {scores.std():.3f})")
    print("=" * 45)

    # 5. Fit on entire set to inspect Feature Importances
    model.fit(X, y)
    importances = model.feature_importances_
    feature_ranking = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    print("\n--- Feature Importance Ranking ---")
    for idx, row in feature_ranking.iterrows():
        print(f"{row['Feature']:<25}: {row['Importance']:.4f}")

    # 6. Plot and save feature importances
    plt.figure(figsize=(9, 5))
    plt.barh(feature_ranking['Feature'][::-1], feature_ranking['Importance'][::-1], color='teal')
    plt.xlabel("Gini Importance")
    plt.title("Behavioral Feature Importance (Scripted vs Spontaneous)")
    plt.tight_layout()
    
    plot_file = "feature_importance.png"
    plt.savefig(plot_file, dpi=300)
    print(f"\nFeature importance plot saved to: {plot_file}")

if __name__ == "__main__":
    train_and_evaluate()