import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


# Ustawienia projektu
INR_TO_PLN = 0.04
REFERENCE_YEAR = 2020

project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(project_folder, "data", "raw", "cardekho.csv")
outputs_folder = os.path.join(project_folder, "outputs")
figures_folder = os.path.join(outputs_folder, "figures")

os.makedirs(outputs_folder, exist_ok=True)
os.makedirs(figures_folder, exist_ok=True)


# Wczytanie danych
if not os.path.exists(file_path):
    raise FileNotFoundError(
        "Nie znaleziono data/raw/cardekho.csv. "
        "Najpierw uruchom: python src/download_data.py"
    )

df = pd.read_csv(file_path)

print("Pierwsze wiersze danych:")
print(df.head())
print("\nLiczba wierszy i kolumn:", df.shape)
print("\nBrakujace wartosci:")
print(df.isnull().sum())


# Czyszczenie danych
rows_before_cleaning = len(df)

df = df.drop_duplicates()

numeric_columns = [
    "year",
    "selling_price",
    "km_driven",
    "mileage(km/ltr/kg)",
    "engine",
    "max_power",
    "seats",
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Z nazwy samochodu pobieramy pierwszy wyraz, czyli marke.
df["brand"] = df["name"].str.split().str[0]

# Zamiast rocznika tworzymy wiek samochodu.
df["car_age"] = REFERENCE_YEAR - df["year"]
df = df.drop(["name", "year"], axis=1)

# Wartosc 0 dla spalania lub mocy oznacza blad danych.
df.loc[df["mileage(km/ltr/kg)"] <= 0, "mileage(km/ltr/kg)"] = np.nan
df.loc[df["max_power"] <= 0, "max_power"] = np.nan

# Brakujace liczby uzupelniamy mediana.
numeric_columns = df.select_dtypes(include="number").columns
for column in numeric_columns:
    df[column] = df[column].fillna(df[column].median())

# Brakujace teksty uzupelniamy najczestsza wartoscia.
text_columns = df.select_dtypes(exclude="number").columns
for column in text_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

# Usuwamy skrajne ceny i bledne rekordy.
lower_price = df["selling_price"].quantile(0.01)
upper_price = df["selling_price"].quantile(0.99)

df = df[
    (df["selling_price"] >= lower_price)
    & (df["selling_price"] <= upper_price)
    & (df["km_driven"] <= 1_000_000)
    & (df["car_age"] >= 0)
]

# Oryginalna cena jest w rupiach indyjskich. Przeliczamy ja na PLN.
df["selling_price"] = df["selling_price"] * INR_TO_PLN

print("\nUsuniete rekordy:", rows_before_cleaning - len(df))
print("Liczba rekordow po czyszczeniu:", len(df))
print("Kurs waluty: 1 INR =", INR_TO_PLN, "PLN")


# Podstawowa analiza i wykresy
print("\nPodstawowe statystyki:")
print(df.describe())

sns.histplot(data=df, x="selling_price", bins=30, kde=True)
plt.title("Rozklad cen samochodow")
plt.xlabel("Cena sprzedazy (PLN)")
plt.ylabel("Liczba samochodow")
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "price_distribution.png"))
plt.close()

average_price_by_fuel = df.groupby("fuel")["selling_price"].mean().reset_index()
sns.barplot(data=average_price_by_fuel, x="fuel", y="selling_price")
plt.title("Srednia cena wedlug rodzaju paliwa")
plt.xlabel("Rodzaj paliwa")
plt.ylabel("Srednia cena (PLN)")
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "average_price_by_fuel.png"))
plt.close()

correlation = df.select_dtypes(include="number").corr()
sns.heatmap(correlation, cmap="coolwarm", center=0, annot=True, fmt=".2f")
plt.title("Korelacje miedzy danymi liczbowymi")
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "correlation_heatmap.png"))
plt.close()


# Zamieniamy teksty na liczby, tak jak na zajeciach.
label_encoder = LabelEncoder()
for column in text_columns:
    df[column] = label_encoder.fit_transform(df[column])


# Podzial na cechy i wynik
x = df.drop("selling_price", axis=1)
y = df["selling_price"]

# Podzial na 80% danych treningowych i 20% danych testowych.
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
)


# Model 1: regresja liniowa
model_lr = LinearRegression()
model_lr.fit(x_train, y_train)
lr_prediction = model_lr.predict(x_test)


# Model 2: KNN
# KNN wymaga skalowania, aby duze liczby nie byly automatycznie wazniejsze.
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

model_knn = KNeighborsRegressor(n_neighbors=5)
model_knn.fit(x_train_scaled, y_train)
knn_prediction = model_knn.predict(x_test_scaled)


# Model 3: drzewo decyzyjne
model_dt = DecisionTreeRegressor(max_depth=10, random_state=42)
model_dt.fit(x_train, y_train)
dt_prediction = model_dt.predict(x_test)


# Model 4: Random Forest, czyli wiele drzew decyzyjnych
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(x_train, y_train)
rf_prediction = model_rf.predict(x_test)


# Porownanie wynikow modeli
models = ["Linear Regression", "KNN", "Decision Tree", "Random Forest"]
predictions = [lr_prediction, knn_prediction, dt_prediction, rf_prediction]
results = []

for model_name, prediction in zip(models, predictions):
    mae = mean_absolute_error(y_test, prediction)
    rmse = np.sqrt(mean_squared_error(y_test, prediction))
    r2 = r2_score(y_test, prediction)

    results.append(
        {
            "model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        }
    )

results_df = pd.DataFrame(results).sort_values("MAE")
results_df.to_csv(os.path.join(outputs_folder, "model_results.csv"), index=False)

print("\nWyniki modeli:")
print(results_df)
print("\nNajlepszy model wedlug MAE:", results_df.iloc[0]["model"])


# Wykres porownujacy modele
sns.barplot(data=results_df, x="MAE", y="model")
plt.title("Porownanie modeli wedlug MAE")
plt.xlabel("Sredni blad predykcji (PLN)")
plt.ylabel("Model")
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "model_comparison_mae.png"))
plt.close()


# Najwazniejsze cechy wedlug Random Forest
feature_importance = pd.DataFrame(
    {
        "feature": x.columns,
        "importance": model_rf.feature_importances_,
    }
).sort_values("importance", ascending=False)

feature_importance.to_csv(
    os.path.join(outputs_folder, "feature_importance.csv"),
    index=False,
)

sns.barplot(data=feature_importance.head(10), x="importance", y="feature")
plt.title("Najwazniejsze cechy wedlug Random Forest")
plt.xlabel("Waznosc cechy")
plt.ylabel("Cecha")
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "top_feature_importance.png"))
plt.close()


# Porownanie cen prawdziwych i przewidzianych przez Random Forest
plt.scatter(y_test, rf_prediction, alpha=0.5)
plt.title("Ceny prawdziwe i przewidziane - Random Forest")
plt.xlabel("Prawdziwa cena (PLN)")
plt.ylabel("Przewidziana cena (PLN)")
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "actual_vs_predicted.png"))
plt.close()

print("\nGotowe. Wyniki zapisano w folderze outputs.")
