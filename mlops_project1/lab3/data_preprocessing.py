import argparse
from typing import List
import pandas as pd


# Настройка аргументов командной строки
def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process CSV file.")
    parser.add_argument(
        "-i", "--input", required=True, help="Path to the input CSV file"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to save the cleaned data"
    )
    return parser


# Функция для загрузки CSV-файла
def read_csv_file(file_path: str) -> pd.DataFrame:
    if not file_path.endswith(".csv"):
        raise ValueError(f"Invalid file type: {file_path}. Expected .csv")
    try:
        dataframe = pd.read_csv(file_path)
        dataframe.columns = (
            dataframe.columns.str.replace("_", " ").str.title().str.replace(" ", "")
        )
        return dataframe
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        raise
    except Exception as error:
        print(f"Error reading file: {error}")
        raise


# Функция для сохранения DataFrame в CSV
def write_csv_file(dataframe: pd.DataFrame, file_path: str):
    try:
        dataframe.to_csv(file_path, index=False, encoding="utf-8")
    except Exception as error:
        print(f"Error writing file: {error}")
        raise


# Функция для очистки DataFrame от ненужных столбцов
def clean_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    filtered_df = dataframe[dataframe.NumberVmailMessages != 0].reset_index(drop=True)
    columns_to_drop = [
        "TotalDayMinutes",
        "TotalEveMinutes",
        "TotalNightMinutes",
        "TotalIntlMinutes",
    ]
    filtered_df.drop(columns=columns_to_drop, inplace=True, errors="ignore")
    return filtered_df


# Функция для получения числовых столбцов
def get_numeric_columns(dataframe: pd.DataFrame) -> List[str]:
    return [
        col
        for col in dataframe.columns
        if pd.api.types.is_numeric_dtype(dataframe[col])
    ]


# Функция для удаления выбросов
def eliminate_outliers(
    dataframe: pd.DataFrame, column_labels: List[str]
) -> pd.DataFrame:
    for label in column_labels:
        q1 = dataframe[label].quantile(0.25)
        q3 = dataframe[label].quantile(0.75)
        iqr = q3 - q1
        upper_limit = q3 + 1.5 * iqr
        lower_limit = q1 - 1.5 * iqr
        dataframe[label] = dataframe[label].mask(
            dataframe[label] < lower_limit, dataframe[label].median()
        )
        dataframe[label] = dataframe[label].mask(
            dataframe[label] > upper_limit, dataframe[label].median()
        )
    return dataframe


# Основная функция обработки данных
def process_csv_data(input_file: str, output_file: str):
    df = read_csv_file(input_file)
    if df is not None:
        df = clean_dataframe(df)
        numeric_cols = get_numeric_columns(df)
        cleaned_df = eliminate_outliers(df, numeric_cols)
        write_csv_file(cleaned_df, output_file)


# Запуск программы
if __name__ == "__main__":
    argument_parser = create_arg_parser()
    args = argument_parser.parse_args()
    process_csv_data(args.input, args.output)
