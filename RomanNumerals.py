def roman_numeral_to_decimal(roman_numeral:str):
    roman_numeral = list(roman_numeral)

    letter_to_decimal = {
        'I': 1,
        'V': 5,
        'X': 10,
        'L': 50,
        'C': 100,
        'D': 500,
        'M': 1000
    }

    decimal = [letter_to_decimal[letter] for letter in roman_numeral]
                
    for i in range(len(decimal) - 1):
        if decimal[i] < decimal[i + 1]:
            decimal[i] *= -1

    return sum(decimal)

def decimal_to_roman_numeral(decimal:int):
    roman_numeral = []

    while decimal > 0:
        if decimal >= 1000:
            roman_numeral.append('M')
            decimal -= 1000
            
        elif decimal >= 900:
            roman_numeral.append('C')
            roman_numeral.append('M')
            decimal -= 900

        elif decimal >= 500:
            roman_numeral.append('D')
            decimal -= 500

        elif decimal >= 400:
            roman_numeral.append('C')
            roman_numeral.append('D')
            decimal -= 400

        elif decimal >= 100:
            roman_numeral.append('C')
            decimal -= 100

        elif decimal >= 90:
            roman_numeral.append('X')
            roman_numeral.append('C')
            decimal -= 90

        elif decimal >= 50:
            roman_numeral.append('L')
            decimal -= 50

        elif decimal >= 40:
            roman_numeral.append('X')
            roman_numeral.append('L')
            decimal -= 40

        elif decimal >= 10:
            roman_numeral.append('X')
            decimal -= 10

        elif decimal >= 9:
            roman_numeral.append('I')
            roman_numeral.append('X')
            decimal -= 9

        elif decimal >= 5:
            roman_numeral.append('V')
            decimal -= 5

        elif decimal >= 4:
            roman_numeral.append('I')
            roman_numeral.append('V')
            decimal -= 4

        elif decimal >= 1:
            roman_numeral.append('I')
            decimal -= 1

    return "".join(roman_numeral)

print(decimal_to_roman_numeral(10), decimal_to_roman_numeral(3), decimal_to_roman_numeral(2025))