# CU5 Project Proposal: Crime Rate Prediction Model

Oct 12, 2026 · @r

## Overview

This project builds machine-learning models that predict assault and property crimes per 1,000 people for 131 districts in Malaysia, from each district's population, education, income and household data.

- **Problem:** crime rates differ widely between districts, and larger, higher-income districts record more crime per person. A district's rate alone does not show whether it has more crime than districts with a similar profile.
- **Why it matters:** crime-prevention planners, such as state police planning units or safe-city programme teams, need to decide which districts to look at first.
- **Objective:** predict each district's crime rate from its characteristics, and compare the actual rate with the predicted rate.
- **Expected outcome:** two regression models and a Streamlit app that flags districts with higher-than-expected or rising crime rates.

## Dataset

The project uses eight district-level datasets from the [OpenDOSM data catalogue](https://open.dosm.gov.my/data-catalogue).

| Dataset | Rows | Years | Used for |
| --- | --- | --- | --- |
| crime\_district | 19,152 | 2016–2023 | Assault and property crime counts (target) |
| population\_district | 383,040 | 2020–2025 | Population, age and sex shares |
| hh\_income\_district | 480 | 2019, 2022, 2024 | Mean and median household income |
| hh\_poverty\_district | 480 | 2019, 2022, 2024 | Absolute and relative poverty |
| hh\_inequality\_district | 480 | 2019, 2022, 2024 | Gini coefficient |
| hh\_access\_amenities | 688 | 2016, 2019, 2022, 2024 | Piped water access |
| schools\_district | 3,977 | 2017–2025 | Schools per 1,000 people |
| enrolment\_school\_district | 12,756 | 2017–2025 | Students per 1,000 people |

- **Cleaned dataset:** one row per district and year, 131 districts from 2016 to 2025 (1,310 rows).
- **Targets:** assault crimes per 1,000 people and property crimes per 1,000 people, with a separate model for each.
- **Features, time-based train/test model:** log population, students per 1,000 people (primary, secondary and post-secondary), schools per 1,000 people and median household income.
- **Features, cross-district model:** the four features above, plus mean income, absolute and relative poverty, Gini, piped water access, share of males aged 15–29, share of people aged 65 and over, share of males, and state (one-hot encoded).

## Workflow

The project followed seven steps, from importing the data to a Streamlit app for crime-prevention planners.

1. Import the eight OpenDOSM datasets with pandas.
2. Clean and preprocess the data:
   - Checked each table for missing values and duplicate rows. No duplicate rows were found; the amenities table had 75 missing values.
   - Removed summary rows ("All", "All Districts", "Malaysia") so crimes and people are not counted twice.
   - Corrected Serdang's 2020 and 2021 property crime totals, where "other theft" was counted twice.
   - Mapped police, education and administrative district names to 131 common districts. Counts were added up; rates were combined with a population-weighted mean.
   - Joined all tables by state, district and year, then calculated crime rates, students and schools per 1,000 people, and log population.
   - Filled missing values: 2023 median income as the geometric mean of the 2022 and 2024 surveys, the 2019–2022 growth rate for 4 districts with no 2024 survey, and Kampar's 2022 student numbers for 2023 and 2024.
   - Scaled features with StandardScaler for Linear Regression, KNN, SVR and K-Means.
3. Perform EDA on 2022, the only year with values in every column for all 131 districts:
   - Descriptive statistics, histograms of both crime rates, a correlation heatmap, a line chart of total crimes per year and a bar chart of rates by state.
   - K-Means clustering of districts, with k = 3 chosen using the elbow method and silhouette score (0.27), and a PCA plot to show the clusters.
4. Train two regression models. Each compares Linear Regression, Decision Tree, Random Forest, KNN and SVR, tuned with GridSearchCV (5-fold cross-validation):
   - Time-based train/test model: trained on 2022 and tested on 2023 for the same districts, then retrained on 2022–2023 to predict 2024.
   - Cross-district model: predicts the 2020–2023 average rate of districts not in the training set, from their 2022 profile. Tested on 10 random 80/20 splits (104 training and 27 test districts).
5. Evaluate the models with R², MAE, MSE and RMSE:
   - Time-based model: 2023 test R² of 0.27 for assault (SVR) and 0.33 for property (Random Forest). Using each district's 2022 rate as the prediction scored higher (0.48 and 0.79).
   - Cross-district model: mean test R² over the 10 splits of 0.37 for assault and 0.31 for property. SVR was saved for both crime types.
6. Demonstrate the final models in a Streamlit app: crime by year and state, a list of districts to review, a district profile page, and a page where the user enters a district profile and sees the predicted rate.
7. Apply the models for crime-prevention planners: the app compares each district's actual 2020–2023 rate with the cross-district model's expected rate, and flags districts that are above expected, rising, or both.

## Deliverables

- Jupyter Notebook (.ipynb + PDF) with data cleaning, EDA, clustering, modelling and evaluation (`district_crime_rate_prediction.ipynb`)
- Datasets: the eight raw OpenDOSM CSV files and the cleaned tables used by the app (`app_districts.csv`, `app_district_years.csv`)
- Saved models (`crime_model.joblib`)
- Streamlit app demo (`app.py`)
- GitHub repository link
