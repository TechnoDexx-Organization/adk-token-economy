try:
    result = 10 / 0
    print(result)
except ZeroDivisionError:
    print("Ошибка: деление на ноль")
