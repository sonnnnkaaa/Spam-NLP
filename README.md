# NLP Spam Detection Model

Data: https://www.kaggle.com/datasets/boiniabhiram/spam-mail-prediction

## Project Description
A model that predicts whether a message is spam.

## Key Results
ROC-AUC: 0.94 | F1: 0.94 — the model effectively distinguishes spam from regular messages

## Technologies
- Python (pandas, numpy, scikit-learn, keras)
- Visualization: matplotlib, seaborn
- Fitting and prediction: Keras (LSTM + Embedding)

## How to run
```bash
git clone https://github.com/sonnnnkaaa/Spam-NLP
cd Spam-NLP
python -m venv .venv
pip install -r requirements.txt
```
if you want to see how the model was trained
```bash
jupyter notebook code/main.ipynb
```
if you want to make a prediction based on your data, place your csv file next to predict.py
```bash
python code/predict.py
```
The prediction result will be created next to your file.
File `main.ipynb` can be opened in Google Colab or VSCode.

---

# NLP модель определения спама

Данные: https://www.kaggle.com/datasets/boiniabhiram/spam-mail-prediction

## Описание проекта
Модель, предсказывающая, является ли сообщение спамом.

## Главные результаты
ROC-AUC: 0.94 | F1: 0.94 — модель эффективно отличает спам от обычных сообщений.

## Технологии
- Python (pandas, numpy, scikit-learn, keras)
- Визуализация: matplotlib, seaborn
- Классификация: Keras (LSTM + Embedding)

## Как запустить
```bash
git clone https://github.com/sonnnnkaaa/Spam-NLP
cd Spam-NLP
python -m venv .venv
pip install -r requirements.txt
```
Если вы хотите ознакомиться с тем, как обучается модель
```bash
jupyter notebook code/main.ipynb
```
Если вы хотите сделать предсказание над своими данными, поместите ваш csv-файл рядом с файлом predict.py
```bash
python code/predict.py
```
Результат предсказания появится рядом с вашим файлом
Файл `main.ipynb` открывается в Google Colab и VSCode.
