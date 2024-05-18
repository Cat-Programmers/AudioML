# Анализ служебных переговоров
### Импорт библиотек

```
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
