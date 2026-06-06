import os

import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree


# Ustawienia projektu
INR_TO_PLN = 0.04

project_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
outputs_folder = os.path.join(project_folder, "outputs")
figures_folder = os.path.join(outputs_folder, "figures")

os.makedirs(outputs_folder, exist_ok=True)
os.makedirs(figures_folder, exist_ok=True)


# Pobranie i wczytanie danych
path = kagglehub.dataset_download("sukhmandeepsinghbrar/car-price-prediction-dataset")
file_path = os.path.join(path, "cardekho.csv")
df = pd.read_csv(file_path)

print("Pierwsze wiersze danych:")
print(df.head())
print("\nLiczba wierszy i kolumn:", df.shape)
print("\nBrakujace wartosci:")
print(df.isnull().sum())


# Czyszczenie danych
rows_before_cleaning = len(df)

df = df.drop_duplicates()
df = df.dropna()

# Usuwamy ewidentnie bledne przebiegi powyzej miliona kilometrow
df = df[df["km_driven"] <= 1_000_000]

# Z pelnej nazwy samochodu zostawiamy tylko marke
df["brand"] = df["name"].str.split().str[0]
df = df.drop("name", axis=1)

# Oryginalna cena jest w rupiach indyjskich przeliczamy ja na PLN
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


# Zamiana tekstow na liczby
text_columns = df.select_dtypes(exclude="number").columns
label_encoder = LabelEncoder()
for column in text_columns:
    df[column] = label_encoder.fit_transform(df[column])


# Podzial na cechy i wynik
x = df.drop("selling_price", axis=1)
y = df["selling_price"]

# Podzial na 80% danych treningowych i 20% danych testowych
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
# KNN wymaga skalowania, aby duze liczby nie byly automatycznie wazniejsze
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


# Porownanie wynikow modeli
results_df = pd.DataFrame(
    {
        "model": ["Linear Regression", "KNN", "Decision Tree"],
        "MSE": [
            mean_squared_error(y_test, lr_prediction),
            mean_squared_error(y_test, knn_prediction),
            mean_squared_error(y_test, dt_prediction),
        ],
        "R2": [
            r2_score(y_test, lr_prediction),
            r2_score(y_test, knn_prediction),
            r2_score(y_test, dt_prediction),
        ],
    }
).sort_values("R2", ascending=False)

results_df["MSE"] = results_df["MSE"].round().astype(int)
results_df["R2"] = results_df["R2"].round(3)

results_df.to_csv(os.path.join(outputs_folder, "model_results.csv"), index=False)

print("\nWyniki modeli:")
print(results_df)
print("\nNajlepszy model wedlug R2:", results_df.iloc[0]["model"])


# Wykres porownujacy modele
sns.barplot(data=results_df, x="R2", y="model")
plt.title("Porownanie modeli wedlug R2")
plt.xlabel("R2")
plt.ylabel("Model")
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "model_comparison_r2.png"))
plt.close()


# Wizualizacja pierwszych poziomow drzewa decyzyjnego
plt.figure(figsize=(20, 10))
plot_tree(model_dt, feature_names=x.columns, filled=True, max_depth=2)
plt.title("Pierwsze poziomy drzewa decyzyjnego")
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "decision_tree.png"))
plt.close()


# Porownanie cen prawdziwych i przewidzianych przez drzewo decyzyjne
plt.scatter(y_test, dt_prediction, alpha=0.5)
plt.title("Ceny prawdziwe i przewidziane - drzewo decyzyjne")
plt.xlabel("Prawdziwa cena (PLN)")
plt.ylabel("Przewidziana cena (PLN)")
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "actual_vs_predicted.png"))
plt.close()

print("\nGotowe. Wyniki zapisano w folderze outputs.")
