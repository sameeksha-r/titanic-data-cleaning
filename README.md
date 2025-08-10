
# Titanic Dataset - Data Cleaning & Preprocessing

This project cleans and preprocesses the Titanic dataset for machine learning.

## Steps Performed
1. **Loaded Dataset** - Read the CSV file using Pandas.
2. **Handled Missing Values** - Filled numeric missing values with mean and categorical with mode.
3. **Converted Mixed Types** - Converted any mixed-type columns to strings before encoding.
4. **Encoded Categorical Variables** - Converted text data into numbers using Label Encoding.
5. **Standardized Numerical Features** - Scaled all numbers to have similar ranges.
6. **Removed Outliers** - Used the IQR method to drop extreme values.
7. **Saved Cleaned Data** - Output saved as `titanic_cleaned.csv`.

## Files
- `Titanic-Dataset.csv` → Original dataset
- `titanic_cleaned.csv` → Cleaned dataset
- `titanic_cleaning.py` → Python script used for cleaning

## Tools Used
- Python
- Pandas
- NumPy
- Scikit-learn

## How to Run
```bash
pip install pandas numpy scikit-learn
python titanic_cleaning.py
```

## Author
Submitted as part of AI & ML Internship Task 1: Data Cleaning & Preprocessing.
