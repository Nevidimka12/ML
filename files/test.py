import os
import pytest
import pickle
import pandas as pd
from click.testing import CliRunner
from main import cli


@pytest.fixture
def runner():
    return CliRunner()


# Проверка, что при корректном вводе все работает нормально и модель сохраняется
def test_train_correct_data(runner):

    if os.path.exists("model_testing.pkl"):
        os.remove("model_testing.pkl")

    result = runner.invoke(cli, [
        'train',
        '--data', 'singapore_airlines_reviews.csv',
        '--model', 'model_testing.pkl'
    ])

    assert result.exit_code == 0
    assert os.path.exists("model_testing.pkl")


# Проверка, что все сплитится в заданных пропорциях
def test_train_with_split(runner):

    if os.path.exists("model_testing.pkl"):
        os.remove("model_testing.pkl")

    split_ratio = 0.5
    result = runner.invoke(cli, [
        'train',
        '--data', 'singapore_airlines_reviews.csv',
        '--split', str(split_ratio),
        '--model', 'model_testing.pkl'
    ])

    line = result.output
    parts = line.split(',')
    test_len_str = parts[1].split('=')[1].strip()
    test_len = int(test_len_str)

    train_df = pd.read_csv('singapore_airlines_reviews.csv')
    train_size = int(len(train_df) * (1 - split_ratio))
    test_size = len(train_df) - train_size

    assert abs(test_size - test_len) < 2


# Проверка, что при некорректном вводе все работает как и задумано
def test_train_incorrect_data(runner):

    if os.path.exists("model_testing.pkl"):
        os.remove("model_testing.pkl")

    result = runner.invoke(cli, [
        'train',
        '--data', 'unexists.csv',
        '--model', 'model_testing.pkl'
    ])

    assert result.exit_code == 1
    assert result.output == 'Ошибка. Файла для обучения не существует\n'


def test_train_incorrect_data2(runner):

    if os.path.exists("model_testing.pkl"):
        os.remove("model_testing.pkl")

    data = {
        'a': ['pampam'],
        'b': ['pupupu']
    }

    df = pd.DataFrame(data)
    filename = 'tests.csv'
    df.to_csv(filename, index=False)

    result = runner.invoke(cli, [
        'train',
        '--data', 'tests.csv',
        '--model', 'model_testing.pkl'
    ])

    os.remove("tests.csv")

    assert result.exit_code == 1
    assert result.output == 'Ошибка. Файл для обучения не соответствует требуемому формату\n'


# Проверка, что ничего лишнего не появляется
def test_same_files_check(runner):

    if os.path.exists("model_testing.pkl"):
        os.remove("model_testing.pkl")

    current_directory = os.getcwd()

    start_files_in_directory = os.listdir(current_directory)

    result = runner.invoke(cli, [
        'train',
        '--data', 'singapore_airlines_reviews.csv',
        '--model', 'model_testing.pkl'
    ])

    if os.path.exists("model_testing.pkl"):
        os.remove("model_testing.pkl")

    end_files_in_directory = os.listdir(current_directory)

    assert result.exit_code == 0
    assert len(start_files_in_directory) == len(end_files_in_directory)
