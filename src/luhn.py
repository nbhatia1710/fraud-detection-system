def luhn_check(card_number):
    # Convert to string and check if numeric
    if not str(card_number).isdigit():
        return False

    digits = [int(d) for d in str(card_number)]
    digits.reverse()

    total = 0

    for i in range(len(digits)):
        if i % 2 == 1:
            doubled = digits[i] * 2
            if doubled > 9:
                doubled -= 9
            total += doubled
        else:
            total += digits[i]

    return total % 10 == 0