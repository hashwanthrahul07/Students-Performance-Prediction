# Students-Performance-Prediction
"A machine learning project to predict student academic performance using Python"

## Overview
Trains a machine learning model to predict student pass/fail outcomes based on previous test scores and academic history. The project uses Linear Regression to analyze the relationship between historical performance metrics and final results.

⚠️ **Important - dataset not yet included**

The script (student_performance_prediction.py) is complete and fully tested, but needs your actual dataset to produce real results. Download or prepare your student dataset, save it as `data/student_data.csv`, and run the script.

The charts and CSVs currently in `outputs/` were generated from a synthetic placeholder dataset only to verify the code runs end-to-end — they are **NOT real data** and must not be submitted as-is. Re-run the script after adding the real CSV to regenerate genuine results.

---

## Expected Dataset

The script auto-detects these columns (case/whitespace tolerant):

| Column Name | Type | Description |
|---|---|---|
| `Student_ID` / `ID` | Integer | Unique student identifier (optional - dropped before modeling) |
| `Previous_Test_Score` / `Test_Score` / `Score` | Numeric | Student's previous test/exam score (0-100) |
| `Pass_Fail` / `Result` / `Outcome` | Categorical | Target variable: Pass or Fail |
| `Attendance` | Numeric | Optional - attendance percentage |
| `Study_Hours` | Numeric | Optional - average weekly study hours |
| `Grade_Level` / `Class` | Categorical | Optional - student's current grade/class |

**If your CSV's headers differ**, edit the `COLUMN MAPPING` section near the top of the script.

---

## What It Does

### 1. **Data Cleaning**
   - Removes null values and duplicate records
   - Converts Pass/Fail text to binary (0 and 1)
   - Validates data types and ranges

### 2. **Exploratory Data Analysis (EDA)**
   - Distribution of Pass/Fail outcomes
   - Score distribution and statistics
   - Correlation between previous scores and pass/fail results
   - Heatmap visualizations

### 3. **Data Preprocessing**
   - Normalization of numeric features (StandardScaler)
   - Encoding of categorical features (if any)
   - Train-Test split (80-20 ratio)

### 4. **Model Training & Evaluation**
   - **Linear Regression** model training
   - Performance metrics: Accuracy, Precision, Recall, F1-Score
   - Classification report and confusion matrix
   - Prediction vs Actual visualizations

### 5. **Output Generation**
   - Cleaned dataset (CSV)
   - EDA charts and plots
   - Model performance report
   - Confusion matrix visualization

---

## How to Run

### Prerequisites
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Steps

1. **Download/Prepare Dataset**
   - Get your student performance dataset
   - Save it as `data/student_data.csv`

2. **Run the Script**
   ```bash
   python student_performance_prediction.py
   ```

3. **Check Results**
   - Outputs are generated in `outputs/` folder:
     - `cleaned_student_data.csv` - processed dataset
     - `eda_plots/` - visualization charts
     - `model_report.txt` - performance metrics
     - `confusion_matrix.png` - prediction accuracy visualization

---

## Project Structure

```
students-performance-prediction/
│
├── student_performance_prediction.py    # Main script
├── README.md                            # This file
│
├── data/
│   └── student_data.csv                 # Your dataset (not included)
│
├── outputs/
│   ├── cleaned_student_data.csv
│   ├── model_report.txt
│   ├── confusion_matrix.png
│   └── eda_plots/
│       ├── score_distribution.png
│       ├── pass_fail_distribution.png
│       └── score_vs_outcome.png
│
└── requirements.txt                     # Python dependencies
```

---

## Tech Stack

- **Python** 3.7+
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning models
- **Matplotlib** - Data visualization
- **Seaborn** - Statistical plotting

---

## Model Performance

The script outputs:
- **Accuracy** - Overall prediction correctness
- **Precision** - Accuracy of Pass predictions
- **Recall** - Detection rate of Pass cases
- **F1-Score** - Harmonic mean of Precision and Recall
- **Confusion Matrix** - True/False positives and negatives

---

## Important Notes

✅ The Python script is **complete and tested**

❌ The provided CSV outputs use **synthetic placeholder data only**

✅ After adding real `data/student_data.csv`, re-run to generate **actual results**

⚠️ Do **NOT** submit placeholder outputs as final results

---

## Future Enhancements

- [ ] Add more features (study hours, attendance, sleep hours)
- [ ] Implement additional models (Logistic Regression, Random Forest, SVM)
- [ ] Cross-validation for better accuracy estimation
- [ ] Hyperparameter tuning
- [ ] Feature importance analysis
- [ ] Web API for real-time predictions

---

## Author
Hashwanth Rahul S B

## License
MIT License - See LICENSE file for details
