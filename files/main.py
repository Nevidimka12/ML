import click
import pickle
import pandas as pd
import re
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from nltk.stem.porter import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


def preprocess_text(text):
    text = re.sub(r"[^\w\s]", '', text.lower())
    return text


def preprocess_sentence(text):
    stemmer = PorterStemmer()
    return ' '.join([stemmer.stem(word) for word in preprocess_text(text).split()])


@click.group()
def cli():
    pass


@click.command()
@click.option('--data', required=True, type=click.Path(), help='Path to the training data file.')
@click.option('--test', type=click.Path(), help='Path to the test data file.')
@click.option('--split', type=float, help='Fraction of data to be used for testing.')
@click.option('--model', required=True, type=click.Path(), help='Path where the trained model will be saved.')
def train(data, test, split, model):
    try:
        df = pd.read_csv(data)
    except FileNotFoundError:
        click.echo("Ошибка. Файла для обучения не существует")
        sys.exit(1)

    try:
        df['comment'] = df['title'] + ' ' + df['text']
        df = df.drop(columns=['published_date', 'published_platform', 'type', 'title', 'text', 'helpful_votes'])
        df['mark'] = df['rating'].apply(lambda x: 0 if x <= 3 else 1)
        df = df.drop(columns='rating')
        df['comment'] = df['comment'].apply(preprocess_text)
        df['comment'] = df['comment'].apply(preprocess_sentence)
    except:
        click.echo("Ошибка. Файл для обучения не соответствует требуемому формату")
        sys.exit(1)

    bow = CountVectorizer()

    if test:
        try:
            test_df = pd.read_csv(test)
        except FileNotFoundError:
            click.echo("Ошибка. Файла для тестов не существует")
            sys.exit(1)

        try:
            test_df['comment'] = test_df['title'] + ' ' + test_df['text']
            test_df = test_df.drop(columns=['published_date', 'published_platform', 'type', 'title', 'text', 'helpful_votes'])
            test_df['mark'] = test_df['rating'].apply(lambda x: 0 if x <= 3 else 1)
            test_df = test_df.drop(columns='rating')
            test_df['comment'] = test_df['comment'].apply(preprocess_text)
            test_df['comment'] = test_df['comment'].apply(preprocess_sentence)

            train_x = bow.fit_transform(df['comment']).toarray()
            test_x = bow.transform(test_df['comment']).toarray()
            train_y = df['mark']
            test_y = test_df['mark']
        except:
            click.echo("Ошибка. Файл для тестов не соответствует требуемому формату")
            sys.exit(1)

    elif split:
        if split == 1:
            click.echo("Ошибка. Доля данных для обучения равна 0")
            sys.exit(1)

        train_df, test_df = train_test_split(df, test_size=split)

        train_x = bow.fit_transform(train_df['comment']).toarray()
        test_x = bow.transform(test_df['comment']).toarray()
        train_y = train_df['mark']
        test_y = test_df['mark']

    else:
        click.echo("Данные для теста не предоставлены. Модель обучается без тестовых данных")
        train_x = bow.fit_transform(df['comment']).toarray()
        train_y = df['mark']
        test_x, test_y = None, None

    model_inst = LogisticRegression()
    model_inst.fit(train_x, train_y)

    with open(model, 'wb') as f:
        pickle.dump((model_inst, bow), f)

    if test or split:
        test_pred = model_inst.predict(test_x)
        f1_met = f1_score(test_pred, test_y)
        click.echo(f"f1_score = {f1_met}, test_len = {len(test_x)}")


@click.command()
@click.option('--model', required=True, type=click.Path(exists=True), help='Path to the trained model file.')
@click.option('--data', required=True, help='Data for prediction, either a file path or a text string.')
def predict(model, data):
    with open(model, 'rb') as f:
        model_inst, bow = pickle.load(f)

    try:
        # Проверяем, является ли data путем к файлу
        df = pd.read_csv(data)
        df['comment'] = df['title'] + ' ' + df['text']
        df = df.drop(columns=['published_date', 'published_platform', 'type', 'title', 'text', 'helpful_votes'])
        df['mark'] = df['rating'].apply(lambda x: 0 if x <= 3 else 1)
        df = df.drop(columns='rating')
        df['comment'] = df['comment'].apply(preprocess_text)
        df['comment'] = df['comment'].apply(preprocess_sentence)

        x_pred = bow.transform(df['comment']).toarray()

        y_pred = model_inst.predict(x_pred)

        for pred in y_pred:
            click.echo(pred)

    except FileNotFoundError:
        input_data = preprocess_sentence(data)
        x_pred = bow.transform([input_data]).toarray()
        prediction = model_inst.predict(x_pred)
        click.echo(prediction[0])
    except:
        click.echo('Ошибка. Файл для предсказаний не соответствует требуемому формату')
        sys.exit(1)


cli.add_command(train)
cli.add_command(predict)

if __name__ == '__main__':
    cli()
