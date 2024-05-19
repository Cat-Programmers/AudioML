# Анализ служебных переговоров
### Краткое описание задачи

Требуется выбрать/разработать и обучить модель для проведения сплошного прослушивания регистратор переговоров и фиксации нарушений требований регламента.

### Импорт библиотек

Скачиваем все нужные для нас библиотеки.

```
pip install flask pandas scikit-learn numpy soundfile librosa nltk scipy
```

```
pip install -U openai-whisper
```

Импортируем библиотеки и загружаем модель распознавания голоса.

```python
from flask import Flask, request, jsonify, send_file
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import numpy as np
import soundfile as sf
import os
import librosa
import librosa.display
import scipy.io.wavfile as wavfile
import nltk
from nltk.corpus import stopwords
import webbrowser
from threading import Timer
import whisper

model_name = 'large'
model_whisper = whisper.load_model(model_name)
```

### Загрузка датасета и подготовка данных

Загружаем датасет из CSV-файла в DataFrame. Поля в датасете: корректность (0 или 1), текст переговора.

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
    result = model_whisper.transcribe(audio_file)
    cleaned_text = result["text"].replace("Продолжение следует...", "")
    return cleaned_text
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

    sr, y = wavfile.read(audio_path)
    y = y.astype(np.float32) / 32767

    # Параметры анализа
    win_len_sec = 0.02  # Длина окна в секундах
    hop_frac = 0.5      # Доля перекрытия окон
    win_len = int(win_len_sec * sr)
    hop_size = int(win_len * hop_frac)

    # Определение участка шума
    window_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=window_length, hop_length=hop_length)
    rms_threshold = np.percentile(rms, 10)  # Используем 10-й перцентиль для определения шума
    noise_indices = np.where(rms[0] < rms_threshold)[0]

    # Извлечение шума
    noise_start = noise_indices[0] * hop_length
    noise_end = (noise_indices[-1] + 1) * hop_length
    noise = y[noise_start:noise_end]

    # Вычисление спектрограммы
    S_full, phase = librosa.magphase(librosa.stft(y))

    # Определение среднего шума по частотам
    noise_stft = librosa.stft(noise)
    noise_mag = np.abs(noise_stft)
    mean_noise_mag = np.mean(noise_mag, axis=1, keepdims=True)

    # Удаление шума (метод спектрального вычитания)
    S_denoised = S_full - mean_noise_mag
    S_denoised = np.maximum(S_denoised, 0)  # Убираем отрицательные значения

    # Обратное преобразование в звуковой сигнал
    clean_signal = librosa.istft(S_denoised * phase)

    # Нормализация и сохранение нового аудиофайла
    clean_signal = np.int16(clean_signal / np.max(np.abs(clean_signal)) * 32767)

    # Сохранение очищенного аудиофайла
    sf.write(output_path, clean_signal, sr)

    return output_path
```

Метод для увеличения громкости голоса в аудиофайле.

```python
def voice_amplification(audio_path):
    base_name, ext = os.path.splitext(audio_path)
    output_path = base_name + '.wav'
    
    sr, y = wavfile.read(audio_path)
    y = y.astype(np.float32) / 32767  # Нормализация сигнала

    gain = 1.7
    y_amplified = y * gain

    # Обрезка значений, чтобы избежать клиппинга
    y_amplified = np.clip(y_amplified, -1.0, 1.0)

    # Обратная нормализация и сохранение аудиофайла
    y_amplified = np.int16(y_amplified * 32767)

    # Сохранение очищенного аудиофайла
    sf.write(output_path, y_amplified, sr)

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
        amplification_audio_file_path =voice_amplification(denoised_audio_file_path)
        transcribed_text = transcribe_audio(amplification_audio_file_path)
        new_text_vectorized = vectorizer.transform([transcribed_text])
        prediction = model.predict(new_text_vectorized)
        if prediction[0] == 1:
            result = "Нарушений не обнаружено: " + new_text_vectorized.toarray().tolist() 
        else:
            # Загружаем стоп-слова
            nltk.download('stopwords')
            stop_words = set(stopwords.words('russian'))

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
                if word in negative_influence_words and word.lower() not in stop_words:
                    highlighted_words.append(f"<span class='highlight'>{word}</span>")
                else:
                    highlighted_words.append(word)

            # Объединяем слова обратно в строку
            highlighted_text = ' '.join(highlighted_words)

            result = "Имеется нарушение: " + highlighted_text
        return jsonify({'result': result})
```

Открытие главной страницы (index.html).

```python
def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')
```

Запуск сервера на локальном хосте. Для подключения к сайту требуется перейти по одной из ссылок, которые будут выведены в терминал.

```python
# Запуск сервера на локальном хосте
if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(host='0.0.0.0', port=5000)
```
