from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
RAW_DATA_PATH = DATA_DIR / "imports-85.data"
TARGET_COLUMN = "price"
RANDOM_STATE = 42

COLUMNS = [
    "symboling",
    "normalized-losses",
    "make",
    "fuel-type",
    "aspiration",
    "num-of-doors",
    "body-style",
    "drive-wheels",
    "engine-location",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-type",
    "num-of-cylinders",
    "engine-size",
    "fuel-system",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price",
]

NUMERIC_COLUMNS = [
    "symboling",
    "normalized-losses",
    "wheel-base",
    "length",
    "width",
    "height",
    "curb-weight",
    "engine-size",
    "bore",
    "stroke",
    "compression-ratio",
    "horsepower",
    "peak-rpm",
    "city-mpg",
    "highway-mpg",
    "price",
]


def ensure_directories_exist() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset() -> Path:
    """Download the dataset only when it is not already saved locally."""
    ensure_directories_exist()

    if RAW_DATA_PATH.exists():
        print(f"Dane juz istnieja: {RAW_DATA_PATH}")
        return RAW_DATA_PATH

    print("Pobieram dane z UCI Machine Learning Repository...")
    urlretrieve(DATA_URL, RAW_DATA_PATH)
    print(f"Zapisano dane w: {RAW_DATA_PATH}")
    return RAW_DATA_PATH


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, names=COLUMNS, na_values="?")
    print(f"Wczytano dane. Liczba wierszy: {df.shape[0]}, liczba kolumn: {df.shape[1]}")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    before_drop = len(cleaned)
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])
    removed_rows = before_drop - len(cleaned)
    print(f"Usunieto wiersze bez ceny: {removed_rows}")

    cleaned["car-volume"] = cleaned["length"] * cleaned["width"] * cleaned["height"]
    cleaned["horsepower-per-weight"] = cleaned["horsepower"] / cleaned["curb-weight"]
    cleaned["average-mpg"] = (cleaned["city-mpg"] + cleaned["highway-mpg"]) / 2

    return cleaned


def split_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ]
    )


def get_models() -> dict[str, object]:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=10.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=RANDOM_STATE),
        "KNN Regression": KNeighborsRegressor(n_neighbors=5),
    }


def evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, Pipeline], str, np.ndarray]:
    results = []
    trained_models: dict[str, Pipeline] = {}
    predictions_by_model: dict[str, np.ndarray] = {}

    for model_name, model in get_models().items():
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X_train)),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        results.append(
            {
                "model": model_name,
                "MAE": mae,
                "RMSE": rmse,
                "R2": r2,
            }
        )
        trained_models[model_name] = pipeline
        predictions_by_model[model_name] = predictions

    results_df = pd.DataFrame(results).sort_values(by="MAE", ascending=True)
    best_model_name = results_df.iloc[0]["model"]
    best_predictions = predictions_by_model[best_model_name]

    return results_df, trained_models, best_model_name, best_predictions


def save_price_distribution(df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.histplot(df[TARGET_COLUMN], kde=True, bins=25)
    plt.title("Rozklad cen samochodow")
    plt.xlabel("Cena")
    plt.ylabel("Liczba samochodow")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "price_distribution.png", dpi=150)
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    numeric_df = df.select_dtypes(include=["number"])
    correlation = numeric_df.corr()

    plt.figure(figsize=(14, 10))
    sns.heatmap(correlation, cmap="coolwarm", center=0, linewidths=0.5)
    plt.title("Korelacje miedzy cechami liczbowymi")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()


def save_model_comparison(results_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x="MAE", y="model")
    plt.title("Porownanie modeli wedlug MAE")
    plt.xlabel("MAE - sredni blad predykcji")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_mae.png", dpi=150)
    plt.close()


def save_actual_vs_predicted(y_test: pd.Series, predictions: np.ndarray, best_model_name: str) -> None:
    min_value = min(y_test.min(), predictions.min())
    max_value = max(y_test.max(), predictions.max())

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, predictions, alpha=0.75)
    plt.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--")
    plt.title(f"Ceny prawdziwe vs przewidziane - {best_model_name}")
    plt.xlabel("Prawdziwa cena")
    plt.ylabel("Przewidziana cena")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()


def clean_feature_name(name: str) -> str:
    return (
        name.replace("numeric__", "")
        .replace("categorical__", "")
        .replace("_", " ")
        .replace("-", " ")
    )


def save_feature_importance(random_forest_pipeline: Pipeline) -> None:
    model = random_forest_pipeline.named_steps["model"]
    preprocessor = random_forest_pipeline.named_steps["preprocess"]

    if not hasattr(model, "feature_importances_"):
        return

    feature_names = preprocessor.get_feature_names_out()
    importance_df = pd.DataFrame(
        {
            "feature": [clean_feature_name(name) for name in feature_names],
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    top_features = importance_df.head(15)
    plt.figure(figsize=(10, 7))
    sns.barplot(data=top_features, x="importance", y="feature")
    plt.title("Najwazniejsze cechy wedlug modelu Random Forest")
    plt.xlabel("Waznosc cechy")
    plt.ylabel("Cecha")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "top_feature_importance.png", dpi=150)
    plt.close()


def save_results(
    results_df: pd.DataFrame,
    trained_models: dict[str, Pipeline],
    best_model_name: str,
) -> None:
    results_path = OUTPUT_DIR / "model_results.csv"
    model_path = MODELS_DIR / "best_car_price_model.joblib"

    results_df.to_csv(results_path, index=False)
    joblib.dump(trained_models[best_model_name], model_path)

    if "Random Forest" in trained_models:
        save_feature_importance(trained_models["Random Forest"])

    print("\nWyniki modeli:")
    print(results_df.to_string(index=False))
    print(f"\nNajlepszy model wedlug MAE: {best_model_name}")
    print(f"Zapisano wyniki w: {results_path}")
    print(f"Zapisano najlepszy model w: {model_path}")


def main() -> None:
    raw_data_path = download_dataset()
    raw_df = load_dataset(raw_data_path)
    cleaned_df = clean_dataset(raw_df)

    save_price_distribution(cleaned_df)
    save_correlation_heatmap(cleaned_df)

    X, y = split_features_and_target(cleaned_df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    results_df, trained_models, best_model_name, best_predictions = evaluate_models(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    save_model_comparison(results_df)
    save_actual_vs_predicted(y_test, best_predictions, best_model_name)
    save_results(results_df, trained_models, best_model_name)

    print("\nGotowe. Wykresy znajdziesz w folderze outputs/figures.")


if __name__ == "__main__":
    main()
