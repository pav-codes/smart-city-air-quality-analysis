        column for column in ["PM2.5", "PM10", "SO2", "NO2", "CO", "O3"]
        if column in data.columns
    ]
    for column in pollutant_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    # Chart 1: average PM2.5 by month.
    monthly = data.groupby(data["datetime"].dt.month)["PM2.5"].mean()
    plt.figure(figsize=(9, 5))
    monthly.plot(kind="bar")
    plt.title("Average PM2.5 Concentration by Month")
    plt.xlabel("Month")
    plt.ylabel("PM2.5 (dataset units)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "monthly_pm25.png", dpi=160)
    plt.close()

    # Chart 2: pollutant average comparison.
    pollutant_means = data[pollutant_columns].mean(numeric_only=True).dropna()
    plt.figure(figsize=(9, 5))
    pollutant_means.plot(kind="bar")
    plt.title("Average Concentration of Selected Pollutants")
    plt.xlabel("Pollutant")
    plt.ylabel("Average concentration (dataset units)")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "average_pollutants.png", dpi=160)
    plt.close()

    # Chart 3: PM2.5 distribution.
    plt.figure(figsize=(9, 5))
    data["PM2.5"].plot(kind="hist", bins=40)
    plt.title("Distribution of PM2.5 Measurements")
    plt.xlabel("PM2.5 (dataset units)")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "pm25_distribution.png", dpi=160)
    plt.close()

    # Feature selection: predictors available in this dataset.
    possible_features = [
        "PM10", "SO2", "NO2", "CO", "O3",
        "TEMP", "PRES", "DEWP", "RAIN", "WSPM",
        "year", "month", "day", "hour"
    ]
    features = [column for column in possible_features if column in data.columns]
    if len(features) < 3:
        raise ValueError("Not enough predictor columns were found to train the model.")

    model_data = data[features + ["PM2.5"]].copy()
    for column in model_data.columns:
        model_data[column] = pd.to_numeric(model_data[column], errors="coerce")
    model_data = model_data.replace([np.inf, -np.inf], np.nan)
    model_data = model_data.dropna(subset=["PM2.5"])
    # Fill missing predictor values with their column medians.
    model_data[features] = model_data[features].fillna(model_data[features].median())
    model_data = model_data.dropna(subset=features)

    X = model_data[features]
    y = model_data["PM2.5"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=100, random_state=42, n_jobs=-1, min_samples_leaf=2
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("\nDATA SUMMARY")
    print("-" * 30)
    print(f"Usable records for analysis: {len(data):,}")
    print(f"Stations represented: {data['station'].nunique() if 'station' in data.columns else 'not available'}")
    print("\nAverage pollutant values:")
    print(pollutant_means.round(2).to_string())

    print("\nMODEL: Random Forest Regressor")
    print("-" * 30)
    print(f"Target: PM2.5 concentration")
    print(f"Features used: {', '.join(features)}")
    print(f"Training records: {len(X_train):,}")
    print(f"Testing records: {len(X_test):,}")
    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R-squared: {r2:.3f}")

    metrics = pd.DataFrame({
        "Metric": ["MAE", "RMSE", "R-squared"],
        "Value": [mae, rmse, r2],
        "Meaning": [
            "Average absolute prediction error; lower is better",
            "Penalizes larger prediction errors; lower is better",
            "Explained variation score; closer to 1 is generally better"
        ]
    })
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)
    pd.DataFrame({"Actual_PM2.5": y_test.to_numpy(),
                  "Predicted_PM2.5": predictions}).to_csv(
        OUTPUT_DIR / "actual_vs_predicted.csv", index=False
    )
    data[pollutant_columns].describe().round(2).to_csv(
        OUTPUT_DIR / "pollutant_summary.csv"
    )

    print(f"\nCharts and result files saved in:\n{OUTPUT_DIR}")
    print("Created: monthly_pm25.png, average_pollutants.png, pm25_distribution.png")
    print("Created: model_metrics.csv, actual_vs_predicted.csv, pollutant_summary.csv")
    print("\nNote: This is an educational model, not an official air-quality warning system.")


if __name__ == "__main__":
    main()
