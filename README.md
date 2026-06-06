# Car Price Prediction ML

Projekt indywidualny z przedmiotu Uczenie Maszynowe.

## Autor

Marcel Szulc

## Cel projektu

Celem projektu jest przewidywanie ceny sprzedaży używanego samochodu na podstawie jego rocznika, przebiegu, marki, parametrów technicznych oraz informacji o ogłoszeniu.

Jest to problem regresji, ponieważ model przewiduje konkretną wartość liczbową, czyli cenę samochodu.

## Dane

Projekt korzysta ze zbioru **Car Price Prediction Dataset** dostępnego w serwisie Kaggle:

https://www.kaggle.com/datasets/sukhmandeepsinghbrar/car-price-prediction-dataset

Zbiór zawiera ponad 8000 ogłoszeń samochodów z indyjskiego rynku. Oryginalna cena sprzedaży jest podana w rupiach indyjskich (INR). Dla czytelności program przelicza ceny na złotówki przy użyciu stałego kursu:

```text
1 INR = 0,04 PLN
```

Stały kurs zapewnia powtarzalność wyników projektu.

Najważniejsze kolumny:

- `selling_price` - cena sprzedaży, przeliczana z INR na PLN,
- `year` - rocznik,
- `km_driven` - przebieg,
- `name` - nazwa samochodu,
- `fuel` - rodzaj paliwa,
- `transmission` - skrzynia biegów,
- `owner` - liczba poprzednich właścicieli,
- `engine` - pojemność silnika,
- `max_power` - moc maksymalna.

## Przygotowanie danych

Program:

- usuwa duplikaty,
- usuwa rekordy zawierające brakujące wartości,
- usuwa ewidentnie błędne przebiegi powyżej miliona kilometrów,
- pobiera markę z pełnej nazwy samochodu,
- przelicza ceny na złotówki,
- zamienia teksty na liczby za pomocą `LabelEncoder`.

## Porównywane modele

- Linear Regression,
- KNN Regression,
- Decision Tree Regression.

Modele są porównywane za pomocą MSE oraz R². Najlepszy model jest wybierany według najwyższego R².

## Wyniki

Najlepszym modelem okazało się **Decision Tree**:

- MSE: około 35 338 160,
- R²: około 0,892.

R² na poziomie 0,892 oznacza dobre dopasowanie modelu do danych testowych.

## Wnioski

Drzewo decyzyjne osiągnęło najlepszy wynik spośród porównywanych modeli. Regresja liniowa uzyskała najsłabszy wynik, ponieważ zależności między cechami samochodu i jego ceną nie są wyłącznie liniowe.

Rocznik okazał się jedną z najważniejszych cech. Potwierdza to, że cena używanego samochodu mocno zależy od roku jego produkcji. Model może służyć do orientacyjnego szacowania ceny, ale nie zastępuje profesjonalnej wyceny.

## Wizualizacje

Folder `outputs/figures` zawiera wykresy przedstawiające:

- rozkład cen samochodów,
- średnią cenę według rodzaju paliwa,
- korelacje między danymi liczbowymi,
- porównanie modeli według R²,
- pierwsze poziomy drzewa decyzyjnego,
- porównanie cen prawdziwych i przewidzianych.

## Uruchomienie

Instalacja bibliotek:

```powershell
pip install -r requirements.txt
```

Uruchomienie projektu:

```powershell
python src/train_car_price_model.py
```

Program automatycznie pobiera dane z Kaggle. Wyniki i wykresy zostaną zapisane w folderze `outputs`.
