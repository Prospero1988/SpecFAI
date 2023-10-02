# -*- coding: cp1250 -*-

import csv

def remove_rows_with_empty_cells(csv_file):
    # Otwieranie pliku CSV do odczytu
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Sprawdzanie każdego wiersza i usuwanie tych, które zawierają puste komórki
    cleaned_rows = [row for row in rows if all(cell.strip() for cell in row)]

    # Otwieranie pliku CSV do zapisu i zapisywanie oczyszczonych wierszy
    with open(csv_file, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(cleaned_rows)

    print("Wiersze z pustymi komórkami zostały usunięte z pliku CSV.")

# Przykładowe użycie
plik_csv = input(" Podaj nazwe pliku csv: ")
remove_rows_with_empty_cells(plik_csv)
