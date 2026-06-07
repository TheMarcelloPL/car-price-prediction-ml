# Car Price Prediction ML

Projekt indywidualny z przedmiotu Uczenie Maszynowe.

## Autor

Marcel Szulc

## Cel projektu

Celem projektu jest przewidywanie ceny sprzedaży używanego samochodu na podstawie jego rocznika, przebiegu oraz parametrów technicznych.

Jest to problem regresji, ponieważ model przewiduje wartość liczbową, czyli cenę samochodu.

## Dane

Projekt korzysta ze zbioru **Car Price Prediction Dataset** z serwisu Kaggle:

https://www.kaggle.com/datasets/sukhmandeepsinghbrar/car-price-prediction-dataset

Zbiór zawiera ponad 8000 ogłoszeń samochodów z indyjskiego rynku. Oryginalne ceny są podane w rupiach indyjskich. Program przelicza je na złotówki przy użyciu stałego kursu:

```text
1 INR = 0,04 PLN
```

## Przygotowanie danych

Program:

- usuwa powtarzające się rekordy,
- usuwa rekordy zawierające brakujące wartości,
- usuwa błędne przebiegi powyżej miliona kilometrów,
- usuwa pełną nazwę modelu samochodu,
- przelicza ceny na złotówki,
- zamienia pozostałe teksty na liczby za pomocą `LabelEncoder`.

## Model

Projekt wykorzystuje model **Decision Tree Regressor**. Jest to regresyjny wariant drzewa decyzyjnego. Wariant regresyjny przewiduje liczbę, czyli cenę samochodu.

Dane są dzielone na:

- 80% danych treningowych,
- 20% danych testowych.

Model jest oceniany za pomocą R². Wynik bliższy `1` oznacza lepsze dopasowanie modelu.

## Wynik

Model osiągnął R² około **0,787**. Oznacza to dobre dopasowanie modelu do danych testowych.

W folderze `outputs` znajduje się tabela przykładowych cen prawdziwych i przewidzianych.

## Wnioski

Drzewo decyzyjne dobrze poradziło sobie z przewidywaniem cen, ponieważ może tworzyć różne reguły dla różnych samochodów. Model nadal nie zastępuje profesjonalnej wyceny, ponieważ w zbiorze nie ma informacji takich jak stan techniczny, historia wypadków i wyposażenie.

## Wizualizacje

Projekt tworzy cztery proste wykresy:

- rozkład cen samochodów,
- średnia cena według rodzaju paliwa,
- porównanie cen prawdziwych i przewidzianych.
- pierwsze poziomy drzewa decyzyjnego.

## Uruchomienie

Instalacja bibliotek:

```powershell
pip install -r requirements.txt
```

Uruchomienie projektu:

```powershell
python src/train_car_price_model.py
```

Program automatycznie pobiera dane z Kaggle.
