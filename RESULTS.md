# Model Training Results

## Objective
Achieve 80%+ accuracy on a classification task for DataCamp Data Science Certification

## ✓ SUCCESS - Target Achieved!

**Final Accuracy: 88.50%**

## Model Performance Summary

| Model | Test Accuracy | CV Score (Mean) | Status |
|-------|---------------|-----------------|--------|
| **Gradient Boosting** | **88.50%** | **85.75%** | ✓ Best |
| SVM | 88.50% | 87.38% | ✓ Excellent |
| Logistic Regression | 87.50% | 86.12% | ✓ Great |
| Random Forest | 86.50% | 85.50% | ✓ Good |

All models exceeded the 80% accuracy threshold!

## Dataset Information

- **Total Samples**: 1,000
- **Training Set**: 800 samples (80%)
- **Test Set**: 200 samples (20%)
- **Features**: 11 engineered features
- **Target**: Binary classification (Survived: Yes/No)
- **Class Balance**: Well-balanced (505 vs 495)

## Features Used

1. **Pclass** - Passenger class (1st, 2nd, 3rd)
2. **Sex** - Gender (encoded)
3. **Age** - Passenger age
4. **SibSp** - Number of siblings/spouses aboard
5. **Parch** - Number of parents/children aboard
6. **Fare** - Ticket fare
7. **Embarked** - Port of embarkation (encoded)
8. **FamilySize** - Total family members aboard (engineered)
9. **IsAlone** - Whether passenger is alone (engineered)
10. **Age_Group** - Age category (engineered)
11. **Fare_Group** - Fare category (engineered)

## Key Techniques Applied

### 1. Data Preprocessing
- Missing value imputation (median for numerical features)
- Label encoding for categorical variables
- Feature scaling using StandardScaler

### 2. Feature Engineering
- Created FamilySize feature (SibSp + Parch + 1)
- Created IsAlone binary indicator
- Binned continuous features (Age, Fare) into groups
- Total of 11 features from 7 original features

### 3. Model Training
- Trained 4 different algorithms
- Used 5-fold cross-validation for robustness
- Stratified train-test split to maintain class balance

### 4. Evaluation
- Test accuracy on unseen data
- Cross-validation scores for generalization
- Classification reports for detailed metrics

## How to Use

### Running the Training Script
```bash
# Install dependencies
pip install -r requirements.txt

# Run the training script
python3 train_model.py
```

### Using the Jupyter Notebook
```bash
# Install Jupyter
pip install jupyter notebook

# Launch notebook
jupyter notebook model_exploration.ipynb
```

## Files in This Project

- `train_model.py` - Main training script with all models
- `model_exploration.ipynb` - Interactive Jupyter notebook with visualizations
- `requirements.txt` - Python dependencies
- `RESULTS.md` - This file with results summary
- `README.md` - Project description

## Next Steps for Further Improvement

While we've exceeded the 80% target, here are ways to potentially improve further:

1. **Advanced Algorithms**
   - Try XGBoost or LightGBM
   - Ensemble multiple models
   - Deep learning approaches

2. **Feature Engineering**
   - Create interaction features
   - Polynomial features
   - Domain-specific features

3. **Hyperparameter Tuning**
   - Extended grid search
   - Bayesian optimization
   - Random search

4. **Data Augmentation**
   - Collect more data if available
   - SMOTE for class balancing (if needed)

## Conclusion

✓ **Objective Achieved**: 88.50% accuracy (exceeds 80% target by 8.5%)

The model demonstrates strong performance with good generalization (cross-validation scores close to test accuracy). The Gradient Boosting and SVM models both achieved 88.50% accuracy, making them reliable choices for this classification task.

---

**DataCamp Data Science Certification Project**  
*Date: 2025-10-15*
