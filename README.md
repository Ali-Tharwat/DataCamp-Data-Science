# 🍽️ Tasty Bytes: Recipe Traffic Prediction

This project was completed as part of the **DataCamp Data Scientist Professional Certificate Practical Exam**  [![DataCamp](https://img.shields.io/badge/DataCamp-03EF62?style=for-the-badge&logo=datacamp&logoColor=white)](https://www.datacamp.com/portfolio/ali-tharwat)

## 🚀 Project Overview

This project analyzes recipe data for the "Tasty Bytes" website to predict which recipes will generate high traffic. The primary business goal, as outlined by the product manager, was to build a machine learning model that could correctly identify popular recipes at least **80% of the time (80% precision)**. The successful implementation of this model aims to help feature high-performing content, leading to increased user engagement and site traffic.

The final **Logistic Regression** model developed for this project successfully **exceeds the business target, achieving 82.8% precision**. 🎯

## 📦 Dataset Description

| Column Name   | Details                                                                                           |
|---------------|---------------------------------------------------------------------------------------------------|
| recipe        | Numeric, unique identifier of recipe                                                              |
| calories      | Numeric, number of calories                                                                       |
| carbohydrate  | Numeric, amount of carbohydrates in grams                                                         |
| sugar         | Numeric, amount of sugar in grams                                                                 |
| protein       | Numeric, amount of protein in grams                                                               |
| category      | Character, type of recipe. Recipes are listed in one of ten possible groupings:<br>Lunch/Snacks, Beverages, Potato, Vegetable, Meat, Chicken, Pork, Dessert, Breakfast, One Dish Meal |
| servings      | Numeric, number of servings for the recipe                                                        |
| high_traffic  | Character, if the traffic to the site was high when this recipe was shown, this is marked with “High”. |

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/-Python-3776AB?style=for-the-badge&logo=python&logoColor=white) 
![Jupyter Notebook](https://img.shields.io/badge/-Jupyter%20Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Pandas](https://img.shields.io/badge/-Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) 
![NumPy](https://img.shields.io/badge/-NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) 
![Matplotlib](https://custom-icon-badges.demolab.com/badge/Matplotlib-71D291?style=for-the-badge&logo=matplotlib)
![Seaborn](https://img.shields.io/badge/-Seaborn-5C8EBC?style=for-the-badge&logo=pypi&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/-Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

## 📖 Methodology

The project followed a structured data science workflow:

1. **Data Validation & Cleaning:** 🧹
    - Missing `high_traffic` values were correctly inferred as 'Low' traffic.
    - The `category` column was standardized by consolidating similar entries (e.g., "Chicken Breast" into "Chicken").
    - Missing nutritional values were imputed using a targeted mean-filling strategy based on category and serving size.
    - A selective outlier removal process was implemented to clean noisy data from low-performing categories ('Beverages', 'Breakfast') without discarding valuable information.

2. **Exploratory Data Analysis (EDA):** 📈
    - Visual analysis revealed that `category` and `servings` are the strongest predictors of high traffic.
    - Recipes in the **Chicken, Pork, and Meat** categories showed the highest success rates.
    - Recipes designed for **4 servings** were most likely to be popular.
    - Nutritional information showed a very weak correlation with recipe popularity.

3. **Model Development:** 🤖
    - The problem was identified as a **binary classification** task.
    - A simple `DummyClassifier` was used as a baseline model to establish a minimum performance benchmark.
    - Several comparison models were tested, with **Logistic Regression** being selected as the final model due to its strong performance and interpretability.

## 📊 Results

The model's performance was evaluated based on its precision in identifying 'High' traffic recipes.

| Model                   | Precision Score | Business Goal |
|-------------------------|----------------|--------------|
| Baseline (DummyClassifier) | 62.1%       | 80%          |
| **Logistic Regression** | **82.8%**      | **80%**      |

The final Logistic Regression model successfully surpassed the 80% precision target. The model correctly identified 82 out of 99 recipes it predicted as 'High' traffic in the test set. 🏆


## 💡 Recommendations

Based on the model's success, the following actions are recommended:

1. **Implement with A/B Testing:** Deploy the model and run an A/B test against the current manual selection process to measure the real-world impact on site traffic and subscriptions.
2. **Enhance Data Collection:** Improve future model performance by collecting additional features for new recipes, such as `preparation_time`, `cost_per_serving`, and `difficulty_level`.
3. **Iterate and Retrain:** Periodically retrain the model with new data to ensure it remains accurate and adapts to changing user preferences.

## ⚙️ How to Run

1. Clone the repository to your local machine.
2. Install the required libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
3. Ensure the `recipe_site_traffic_2212.csv` file is in the same directory as the notebook.
4. Open and run the `notebook.ipynb` file to see the full analysis and results.
