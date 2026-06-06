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
- zamienia błędne wartości liczbowe na braki danych,
- uzupełnia brakujące liczby medianą,
- uzupełnia brakujące teksty najczęstszą wartością,
- tworzy cechę wieku samochodu,
- usuwa skrajne lub błędne rekordy,
- przelicza ceny na złotówki,
- zamienia teksty na liczby za pomocą One-Hot Encoding.

## Porównywane modele

- Linear Regression,
- Ridge Regression,
- Decision Tree,
- Random Forest,
- Gradient Boosting,
- KNN Regression.

Modele są porównywane za pomocą MAE, RMSE oraz R². Najlepszy model jest wybierany według najniższego MAE.

## Wyniki

Najlepszym modelem okazał się **Random Forest**:

- MAE: około 2 812 PLN,
- RMSE: około 4 178 PLN,
- R²: około 0,897.

Oznacza to, że model myli się średnio o około 2 812 zł. Najważniejszymi cechami według modelu Random Forest były moc maksymalna, wiek samochodu, pojemność silnika i przebieg.

## Uruchomienie

Instalacja bibliotek:

```powershell
pip install -r requirements.txt
```

Pobranie danych z Kaggle:

```powershell
python src/download_data.py
```

Trenowanie i porównanie modeli:

```powershell
python src/train_car_price_model.py
```

Wyniki i wykresy zostaną zapisane w folderze `outputs`, a najlepszy model w folderze `models`.
