# Анализ служебных переговоров
### Краткое описание задачи

Требуется выбрать/разработать и обучить модель для проведения сплошного прослушивания регистратор переговоров и фиксации нарушений требований регламента.

### Импорт библиотек

Скачиваем все нужные для нас библиотеки.

```
pip install flask pandas scikit-learn numpy soundfile librosa speech_recognition noisereduce
```

Импортируем библиотеки.

```python
from flask import Flask, request, jsonify, send_file, send_from_directory
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import soundfile as sf
import os
import librosa
import speech_recognition as sr
import noisereduce as nr
```

### Загрузка датасета и подготовка данных

Загружаем датасет из CSV-файла в DataFrame.

```python
df = pd.read_csv('augmented_dataset.csv', sep=';', names=['Correct', 'Dialogue'])
```

Разделяем данные на тестовые и тренировочные, а также производим токенизацию и векторизацию текста.

```python
# Разделение на тестовые и тренировочные данные
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Токенизация и векторизация текста
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_df['Dialogue'])
y_train = train_df['Correct']
X_test = vectorizer.transform(test_df['Dialogue'])
y_test = test_df['Correct']
```

### Создание и обучение модели

Мы будем использовать модель логистической регрессии. **Модель логистической регрессии** - модель для бинарной классификации.

```python
# Создание и обучение модели
model = LogisticRegression()
model.fit(X_train, y_train)
```

Оцениваем производительность модели с помощью метрики classification_report.

```python
# Оценка модели
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Методы для работы с аудиофайлами 

Создаем путь для сохранения временных аудиофайорв.

```python
# Путь для сохранения временного аудиофайла
AUDIO_FOLDER = 'temp_audio'
if not os.path.exists(AUDIO_FOLDER):
    os.makedirs(AUDIO_FOLDER)
```

Метод для транскрибизации аудио (получение текста из аудио).

```python
def transcribe_audio(audio_file):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data, language="ru-RU")
            return text
        except sr.UnknownValueError:
            return "Не удалось распознать речь"
        except sr.RequestError as e:
            return f"Ошибка запроса к сервису распознавания речи: {e}"
```

Метод для конвертации аудиофайлов в WAV-формат.

```python
def convert_to_wav(input_file):
    base_name, ext = os.path.splitext(input_file)
    if ext != '.wav':
        output_file = os.path.join(AUDIO_FOLDER, base_name + '.wav')
        data, samplerate = sf.read(input_file)
        sf.write(output_file, data, samplerate)
        return output_file
    else:
        return input_file
```

Метод для удаления шума ищ аудиофайла.

```python
def remove_noise(audio_path):
    base_name, ext = os.path.splitext(audio_path)
    output_path = base_name + '.wav'

    # Загрузка аудиофайла с помощью librosa
    audio_data, sr = librosa.load(audio_path, sr=None)

    # Применение метода noisereduce для удаления шума
    reduced_noise = nr.reduce_noise(y=audio_data, sr=sr)

    # Сохранение очищенного аудиофайла
    sf.write(output_path, reduced_noise, sr)

    return output_path
```

### Создание и запуск приложения

Создаем Flask-приложение

```python
app = Flask(__name__)
```

Определяем путь, из которого будет загружаться HTML-файл (главная страница приложения).

```python
# Определение пути для загрузки HTML-файла
@app.route('/')
def index():
    return send_file('index.html')
```

Команда для предсказания на наличие нарушения и вывода текста. Если текст содержит нарушение, то звездочками будут выделяться слова, которые и вызвали нарушение, а если текст не содержит нарушение, то просто будет выведен текст.

```python
# Предсказание на наличие нарушения и вывод текста
@app.route('/predict', methods=['POST'])
def predict_audio():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        audio_file_path = convert_to_wav(file.filename)
        denoised_audio_file_path = remove_noise(audio_file_path)
        transcribed_text = transcribe_audio(denoised_audio_file_path)
        new_text_vectorized = vectorizer.transform([transcribed_text])
        prediction = model.predict(new_text_vectorized)
        if prediction[0] == 1:
            result = "Нарушений не обнаружено: " + new_text_vectorized.toarray().tolist() 
        else:
            # Получаем список признаков и коэффициенты модели
            feature_names = vectorizer.get_feature_names_out()
            coefs = model.coef_[0]

            # Находим слова, которые имеют негативное влияние
            negative_influence_words = [feature_names[i] for i in range(len(feature_names)) if new_text_vectorized[0, i] != 0 and coefs[i] < 0]

            # Разделяем транскрибированный текст на слова
            words = transcribed_text.split()

            # Создаем новый список для выделенного текста
            highlighted_words = []

            # Проходимся по каждому слову в тексте
            for word in words:
                # Если слово является негативным, выделяем его звездочками
                if word in negative_influence_words:
                    highlighted_words.append(f"*{word}*")
                else:
                    highlighted_words.append(word)

            # Объединяем слова обратно в строку
            highlighted_text = ' '.join(highlighted_words)

            result = "Имеется нарушение: " + highlighted_text
        return jsonify({'result': result})
```

Запуск сервера на локальном хосте. Для подключения к сайту требуется перейти по одной из ссылок, которые будут выведены в терминал.

```python
# Запуск сервера на локальном хосте
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```
