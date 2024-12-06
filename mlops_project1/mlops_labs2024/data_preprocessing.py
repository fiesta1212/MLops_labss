import argparse
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder,
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def parse_arguments():
    """Парсит аргументы командной строки."""
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-i", "--input_path", required=True, help="Путь к входному файлу"
    )
    argparser.add_argument(
        "-o",
        "--output_path",
        required=True,
        help="Путь для сохранения обработанных данных",
    )
    return argparser.parse_args()


def load_csv(file_path: str) -> pd.DataFrame:
    """Загружает CSV-файл в DataFrame."""
    if not file_path.endswith(".csv"):
        raise ValueError(f"Неправильный тип файла: {file_path}. Ожидался .csv")
    try:
        df = pd.read_csv(file_path)
        df.columns = df.columns.str.replace("_", " ").str.title().str.replace(" ", "")
        return df
    except FileNotFoundError:
        print(f"Файл не найден: {file_path}")
        raise
    except Exception as e:
        print(f"Ошибка при чтении файла: {e}")
        raise


def save_csv(df: pd.DataFrame, file_path: str):
    """Сохраняет DataFrame в CSV-файл."""
    try:
        df.to_csv(file_path, index=False, encoding="utf-8")
    except Exception as e:
        print(f"Ошибка при записи файла: {e}")
        raise


def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет ненужные столбцы и строки."""
    df = df[df.NumberVmailMessages != 0].reset_index(drop=True)
    return df.drop(
        columns=[
            "TotalDayMinutes",
            "TotalEveMinutes",
            "TotalNightMinutes",
            "TotalIntlMinutes",
        ]
    )


def collect_numerical_columns(df: pd.DataFrame) -> List[str]:
    """Собирает названия числовых столбцов."""
    return [col for col in df.columns if df[col].dtype in [np.float64, np.int64]]


def remove_outliers(data: pd.DataFrame, labels: List[str]) -> pd.DataFrame:
    """Удаляет выбросы из указанных столбцов."""
    for label in labels:
        q1 = data[label].quantile(0.25)
        q3 = data[label].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = q1 - 1.5 * iqr
        data[label] = np.where(
            data[label] < lower_bound, data[label].median(), data[label]
        )
        data[label] = np.where(
            data[label] > upper_bound, data[label].median(), data[label]
        )
    return data


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Предварительная обработка данных с использованием scikit-learn."""
    num_standard = [
        "AccountLength",
        "TotalDayCalls",
        "TotalDayCharge",
        "TotalEveCalls",
        "TotalEveCharge",
        "TotalNightCalls",
        "TotalNightCharge",
        "TotalIntlCalls",
        "TotalIntlCharge",
    ]

    num_norm = ["NumberVmailMessages", "TotalIntlCalls", "NumberCustomerServiceCalls"]
    cat_style_ohe = ["AreaCode"]
    cat_ord = ["State", "InternationalPlan", "VoiceMailPlan", "Churn"]

    preprocessors = ColumnTransformer(
        transformers=[
            ("num_standard", Pipeline([("scaler", StandardScaler())]), num_standard),
            ("num_norm", Pipeline([("norm", MinMaxScaler())]), num_norm),
            (
                "cat_style_ohe",
                Pipeline(
                    [
                        (
                            "encoder",
                            OneHotEncoder(
                                drop="if_binary",
                                handle_unknown="ignore",
                                sparse_output=False,
                            ),
                        )
                    ]
                ),
                cat_style_ohe,
            ),
            ("cat_ord", Pipeline([("encoder", OrdinalEncoder())]), cat_ord),
        ]
    )

    df_transformed = preprocessors.fit_transform(df)
    cat_ohe_names = preprocessors.transformers_[2][1]["encoder"].get_feature_names_out(
        cat_style_ohe
    )

    columns = np.hstack([num_standard, num_norm, cat_ohe_names, cat_ord])
    return pd.DataFrame(df_transformed, columns=columns)


def process_data(input_path: str, output_path: str):
    """Основная функция обработки данных."""
    df = load_csv(input_path)
    df = drop_unnecessary_columns(df=remove_outliers(df, collect_numerical_columns(df)))
    df = preprocess_data(df)
    save_csv(df, output_path)


if __name__ == "__main__":
    args = parse_arguments()
    process_data(args.input_path, args.output_path)
