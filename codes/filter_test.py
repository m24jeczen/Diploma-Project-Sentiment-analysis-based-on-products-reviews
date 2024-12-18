import pandas as pd
from codes.filter import filter
from codes.parameters import categories
category = categories[32]

# Funkcja testująca różne przypadki użycia filter()
def test_filter():
    print("Test 1: Category Video_games")
    result = filter(category)
    if result is not None:
        print(result.head(2))
    print()  # Pusta linia

    # 2. Test: Niepoprawny parametr category (np. brak danych w mocku)
    print("Test 2: Wrong category")
    result = filter(category="nonexistent_category")
    if result is not None:
        print(result.head(2))
    print()  # Pusta linia

    # 3. Test: Sprawdzenie min_reviews_per_product
    print("Test 3: min_reviews_per_product = 50")
    result = filter(category, min_reviews_per_product=50)
    if result is not None:
        print(result.head(2))
    print()  # Pusta linia

    # 4. Test: Błędny parametr min_reviews_per_product (np. typ string)
    print("Test 4: min_reviews_per_product = 'abc'")
    result = filter(category, min_reviews_per_product="abc")
    if result is not None:
        print(result.head(2))
    print()  # Pusta linia

    # 5. Test: Parametr search_value
    print("Test 5: search_value ='Nintendo' ")
    result = filter(category, search_value="Nintendo")
    if result is not None:
        print(result.head(2))
    print()  # Pusta linia

    # 6. Test: Błędny parametr search_value (np. typ liczbowy)
    print("Test 6: Wrong search_value = 123")
    result = filter(category, search_value=123)
    if result is not None:
        print(result.head(2))
    print()  # Pusta linia

    # 7. Test: Parametr store (istniejący sklep)
    print("Test 7: store = 'Aerosoft'")
    result = filter(category, store="Aerosoft")
    if result is not None:
        print(result.head(2))
    print()  # Pusta linia

    # 8. Test: Parametr store (nieistniejący sklep)
    print("Test 8: Not existing store")
    result = filter(category, store="Wrong Store")
    if result is not None:
        print(result.head(2))
    print()  # Pusta linia

    # 9. Test: Parametr min_average_rating
    print("Test 9: min_average_rating = 3.5")
    result = filter(category, min_average_rating=3.5)
    if result is not None:
        print(result.head(2))
    print()  # Pusta linia

    # 10. Test: Błędny parametr min_average_rating
    print("Test 10: min_average_rating=6")
    result = filter(category, min_average_rating=6)
    if result is not None:
        print(result.head(2))
    print()  # Pusta linia

    # 11. Test: Parametr start_date i end_date
    print("Test 11: start_date='2023-01-01' i end_date='2023-06-01' ")
    result = filter(category, start_date="2023-01-01", end_date="2023-06-01")
    if result is not None:
        print(result.head(2))
    print()  # Pusta linia

    # 12. Test: Błędny format daty
    print("Test 12: Wrong data (13 month)")
    result = filter(category, start_date="2023-13-01", end_date="2023-06-01")
    if result is not None:
        print(result.head(2))
    print()  # Pusta linia

    # 13. Test: Parametr min_text_length
    print("Test 13: min_text_length = 2")
    result = filter(category, min_text_length=2)
    if result is not None:
        print(result.head(2))
    print()  # Pusta linia

    # 14. Test: Błędny parametr min_text_length
    print("Test 14: min_text_length = -2")
    result = filter(category, min_text_length=-2)
    if result is not None:
        print(result.head(2))
    print()  # Pusta linia

import sys
import contextlib
with open("test_output.txt", "w") as file:
    with contextlib.redirect_stdout(file):
        # Teraz wszystkie printy w test_filter() będą zapisywane do pliku
        test_filter()
