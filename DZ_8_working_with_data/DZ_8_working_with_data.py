import pandas as pd # импорт библиотеки для обработки и анализа данных
import numpy as np # библиотека для работы с массивами данных
import matplotlib.pyplot as plt # matplotlib.pyplot модуль для построения графиков
import seaborn as sns # библиотека для визуализации данных, основана на matplotlib
from sklearn.preprocessing import LabelEncoder # инструмент, для кодирования категориальных переменных
from scipy import stats # библиотека для научных и математических вычислений

# Загрузка датасет в pandas DataFrame под названием df.
# установка стиля и цветовой палитры для графиков
sns.set(style='whitegrid')

# загрузка данных
file_path = "train.csv"
df = pd.read_csv(file_path)

# вывод датасета
print('Первые строки датасета: ')
print(df.head())

print ("\n статистика: ")
print (df.describe())

print ("\n описательная статистика: ")
#print (df.stats())