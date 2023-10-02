import csv

def sprawdz_pusty_rekord(nazwa_pliku):
    ilosc_pustych_rekordow = 0
    wiersze_z_pustymi_rekordami = []

    with open(nazwa_pliku, 'r') as plik_csv:
        czytnik_csv = csv.reader(plik_csv)

        for wiersz_idx, wiersz in enumerate(czytnik_csv, start=1):
            puste_rekordy_w_wierszu = sum(not rekord.strip() for rekord in wiersz)
            ilosc_pustych_rekordow += puste_rekordy_w_wierszu

            if puste_rekordy_w_wierszu > 0:
                wiersze_z_pustymi_rekordami.append((wiersz_idx, puste_rekordy_w_wierszu))

    return ilosc_pustych_rekordow, wiersze_z_pustymi_rekordami

print()
print()
# Pobierz nazwę pliku od użytkownika
nazwa_pliku = input("Podaj nazwę pliku CSV: ")
print()
# Wywołaj funkcję sprawdzającą
ilosc, wiersze = sprawdz_pusty_rekord(nazwa_pliku)

# Wyświetl liczbę pustych rekordów
print(f"Liczba pustych rekordów: {ilosc}")
print()

if ilosc == 0:
    exit()
else:
    print("Wiersze z pustymi rekordami:")
    for wiersz, ilosc_pustych in wiersze:
        print(f"Wiersz: {wiersz}, Ilość pustych rekordów: {ilosc_pustych}")
print()
print()