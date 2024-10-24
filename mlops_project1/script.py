def add_numbers(a: int, b: int) -> int:
    return a + b


if __name__ == "__main__":
    print(add_numbers(5, 3))
    # print(add_numbers(2, "3"))  # Ошибка типов для проверки mypy

# mlops_project1\script.py:7: error: Argument 2 to "add_numbers" has
# incompatible type "str"; expected "int"  [arg-type]
# Found 1 error in 1 file (checked 1 source file)
