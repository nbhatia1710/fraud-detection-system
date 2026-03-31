from luhn import luhn_check

# Test card numbers
cards = [
    "4532015112830366",  # Valid
    "5555555555554444",  # Valid
    "1234567890123456",  # Invalid
    "1111222233334444"   # Invalid
]

for card in cards:
    if luhn_check(card):
        print(f"Card: {card} → Valid")
    else:
        print(f"Card: {card} → Invalid")