import os

import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
df = df[df["km_driven"] <= 1_000_000]

# Usuwamy pelna nazwe samochodu, poniewaz zawiera bardzo wiele roznych modeli
df = df.drop("name", axis=1)

# Oryginalna cena jest w rupiach indyjskich, dlatego przeliczamy ja na PLN
df["selling_price"] = df["selling_price"] * INR_TO_PLN

print("\nUsuniete rekordy:", rows_before_cleaning - len(df))
print("Liczba rekordow po czyszczeniu:", len(df))
print("Kurs waluty: 1 INR =", INR_TO_PLN, "PLN")
print("\nPodstawowe statystyki:")
print(df.describe())


# Proste wizualizacje danych
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


# Zamiana tekstow na liczby
text_columns = df.select_dtypes(exclude="number").columns
label_encoder = LabelEncoder()

for column in text_columns:
    df[column] = label_encoder.fit_transform(df[column])


# Podzial danych na cechy oraz wynik
x = df.drop("selling_price", axis=1)
y = df["selling_price"]

# Podzial na 80% danych treningowych i 20% danych testowych
x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    train_size=0.8,
    random_state=25,
)


# Utworzenie i wytrenowanie modelu drzewa decyzyjnego
model = DecisionTreeRegressor(max_depth=10, random_state=42)
model.fit(x_train, y_train)

# Przewidywanie cen dla danych testowych
prediction = model.predict(x_test)

# Ocena modelu
r2 = r2_score(y_test, prediction)
print("\nR2 modelu:", round(r2, 3))


# Porownanie kilku cen prawdziwych i przewidzianych
results = pd.DataFrame(
    {
        "Prawdziwa cena": y_test,
        "Przewidziana cena": prediction,
    }
)

results = results.round(2)
results.to_csv(os.path.join(outputs_folder, "predictions.csv"), index=False)

print("\nPrzykladowe wyniki:")
print(results.head(10))


# Wykres cen prawdziwych i przewidzianych
plt.scatter(y_test, prediction, alpha=0.5)
plt.title("Ceny prawdziwe i przewidziane - drzewo decyzyjne")
plt.xlabel("Prawdziwa cena (PLN)")
plt.ylabel("Przewidziana cena (PLN)")
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "actual_vs_predicted.png"))
plt.close()

# Wykres pokazujacy pierwsze poziomy drzewa
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=x.columns, filled=True, max_depth=2)
plt.title("Pierwsze poziomy drzewa decyzyjnego")
plt.tight_layout()
plt.savefig(os.path.join(figures_folder, "decision_tree.png"))
plt.close()

print("\nGotowe. Wyniki zapisano w folderze outputs.")
