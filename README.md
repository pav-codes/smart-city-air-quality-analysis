# Smart City Air Quality Analysis and Prediction

## Project overview
This beginner-friendly data analytics project explores air pollution measurements from multiple monitoring stations and trains a machine-learning model to estimate PM2.5 concentration.

## Objectives
- Load and combine air-quality data from multiple stations.
- Clean missing or invalid records.
- Summarize pollutant measurements.
- Visualize monthly PM2.5 averages, pollutant averages, and PM2.5 distribution.
- Train and evaluate a Random Forest regression model.
- Export metrics, predictions, and descriptive statistics.

## Dataset
**Beijing Multi-Site Air-Quality Data** from the UCI Machine Learning Repository.

Dataset page: https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data

The script downloads the public dataset automatically on its first run. An internet connection is required for the first run. The data describes measurements from Beijing monitoring stations; it is a sample smart-city air-quality use case, not live measurements from your city.

## Tools and technologies
- Python
- pandas and NumPy
- Matplotlib
- scikit-learn
- Random Forest Regressor

## Requirements
Python 3.10 or newer is recommended.

Install dependencies:
```bash
python -m pip install -r requirements.txt
```

Run:
```bash
python Smart_City_Air_Quality.py
```

On some Windows installations, use `py` instead of `python`.

## Outputs
The script creates a `smart_city_outputs` folder containing:
- `monthly_pm25.png`
- `average_pollutants.png`
- `pm25_distribution.png`
- `model_metrics.csv`
- `actual_vs_predicted.csv`
- `pollutant_summary.csv`

It also downloads the dataset archive into a `data` folder.

## Understanding the metrics
- **MAE (Mean Absolute Error):** the average absolute difference between predictions and observed values. Lower is better.
- **RMSE (Root Mean Squared Error):** penalizes large errors more strongly. Lower is better.
- **R-squared:** describes how much variation the model explains on the test set. Values closer to 1 are generally better; a negative value means the model performs worse than predicting the test-set mean.

## Limitations
- This is historical data, not a real-time AQI feed.
- The model estimates PM2.5 concentration, not the official AQI category.
- Random train/test splitting can make performance look more optimistic for time-dependent data. A future improvement would be to use a chronological split and validate on a later period.
- Model outputs should not be used for health or emergency decisions.
5. Follow the internship's rules for attribution and AI assistance.

