from __future__ import annotations

from pathlib import Path

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
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cardekho.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

TARGET_COLUMN = "selling_price"
RANDOM_STATE = 42
REFERENCE_YEAR = 2020
INR_TO_PLN = 0.04

NUMERIC_COLUMNS = [
    "year",
    "selling_price",
    "km_driven",
    "mileage(km/ltr/kg)",
    "engine",
    "max_power",
    "seats",
]

FEATURE_LABELS = {
    "car_age": "wiek samochodu",
    "km_driven": "przebieg",
    "mileage(km/ltr/kg)": "wydajnosc paliwowa",
    "engine": "pojemnosc silnika",
    "max_power": "moc maksymalna",
    "seats": "liczba miejsc",
    "brand": "marka",
    "fuel": "rodzaj paliwa",
    "seller_type": "typ sprzedawcy",
    "transmission": "skrzynia biegow",
    "owner": "liczba wlascicieli",
}


def ensure_directories_exist() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "Nie znaleziono data/raw/cardekho.csv. "
            "Najpierw uruchom: python src/download_data.py"
        )

    df = pd.read_csv(DATA_PATH)
    print(f"Wczytano dane. Liczba wierszy: {df.shape[0]}, liczba kolumn: {df.shape[1]}")
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()

    for column in NUMERIC_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    before_duplicates = len(cleaned)
    cleaned = cleaned.drop_duplicates()
    print(f"Usunieto duplikaty: {before_duplicates - len(cleaned)}")

    cleaned["brand"] = cleaned["name"].str.split().str[0].str.title()
    cleaned["car_age"] = REFERENCE_YEAR - cleaned["year"]
    cleaned = cleaned.drop(columns=["name", "year"])

    cleaned.loc[cleaned["mileage(km/ltr/kg)"] <= 0, "mileage(km/ltr/kg)"] = np.nan
    cleaned.loc[cleaned["max_power"] <= 0, "max_power"] = np.nan
    cleaned = cleaned.dropna(subset=[TARGET_COLUMN])

    lower_price = cleaned[TARGET_COLUMN].quantile(0.01)
    upper_price = cleaned[TARGET_COLUMN].quantile(0.99)
    before_outliers = len(cleaned)
    cleaned = cleaned[
        cleaned[TARGET_COLUMN].between(lower_price, upper_price)
        & (cleaned["km_driven"] <= 1_000_000)
        & (cleaned["car_age"] >= 0)
    ]
    print(f"Usunieto skrajne lub bledne rekordy: {before_outliers - len(cleaned)}")

    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN] * INR_TO_PLN
    print(f"Przeliczono ceny wedlug stalego kursu: 1 INR = {INR_TO_PLN:.2f} PLN")

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
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
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
        "Decision Tree": DecisionTreeRegressor(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
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
        print(f"Trenuje model: {model_name}")
        pipeline = Pipeline(
            steps=[
                ("preprocess", build_preprocessor(X_train)),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        results.append(
            {
                "model": model_name,
                "MAE": mean_absolute_error(y_test, predictions),
                "RMSE": np.sqrt(mean_squared_error(y_test, predictions)),
                "R2": r2_score(y_test, predictions),
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
    sns.histplot(df[TARGET_COLUMN], kde=True, bins=30)
    plt.title("Rozklad cen samochodow")
    plt.xlabel("Cena sprzedazy (PLN)")
    plt.ylabel("Liczba samochodow")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "price_distribution.png", dpi=150)
    plt.close()


def save_correlation_heatmap(df: pd.DataFrame) -> None:
    correlation = df.select_dtypes(include=["number"]).corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title("Korelacje miedzy cechami liczbowymi")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()


def save_model_comparison(results_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results_df, x="MAE", y="model")
    plt.title("Porownanie modeli wedlug MAE")
    plt.xlabel("MAE - sredni blad predykcji (PLN)")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison_mae.png", dpi=150)
    plt.close()


def save_actual_vs_predicted(
    y_test: pd.Series,
    predictions: np.ndarray,
    best_model_name: str,
) -> None:
    min_value = min(y_test.min(), predictions.min())
    max_value = max(y_test.max(), predictions.max())

    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, predictions, alpha=0.6)
    plt.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--")
    plt.title(f"Ceny prawdziwe vs przewidziane - {best_model_name}")
    plt.xlabel("Prawdziwa cena (PLN)")
    plt.ylabel("Przewidziana cena (PLN)")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "actual_vs_predicted.png", dpi=150)
    plt.close()


def clean_feature_name(name: str) -> str:
    cleaned_name = name.replace("numeric__", "").replace("categorical__", "")

    for original, polish in FEATURE_LABELS.items():
        if cleaned_name == original:
            return polish
        if cleaned_name.startswith(f"{original}_"):
            value = cleaned_name.removeprefix(f"{original}_")
            return f"{polish}: {value}"

    return cleaned_name.replace("_", " ")


def save_feature_importance(random_forest_pipeline: Pipeline) -> None:
    model = random_forest_pipeline.named_steps["model"]
    preprocessor = random_forest_pipeline.named_steps["preprocess"]
    feature_names = preprocessor.get_feature_names_out()

    importance_df = pd.DataFrame(
        {
            "feature": [clean_feature_name(name) for name in feature_names],
            "importance": model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    importance_df.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    plt.figure(figsize=(10, 7))
    sns.barplot(data=importance_df.head(15), x="importance", y="feature")
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
    save_feature_importance(trained_models["Random Forest"])

    print("\nWyniki modeli:")
    print(results_df.to_string(index=False))
    print(f"\nNajlepszy model wedlug MAE: {best_model_name}")
    print(f"Zapisano wyniki w: {results_path}")
    print(f"Zapisano najlepszy model w: {model_path}")


def main() -> None:
    ensure_directories_exist()
    raw_df = load_dataset()
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
