import pandas as pd # импорт библиотеки для обработки и анализа данных
import numpy as np # библиотека для работы с массивами данных
import matplotlib.pyplot as plt # matplotlib.pyplot модуль для построения графиков
import seaborn as sns # библиотека для визуализации данных, основана на matplotlib
from sklearn.preprocessing import LabelEncoder # инструмент, для кодирования категориальных переменных
from scipy import stats # библиотека для научных и математических вычислений

# установка стиля и цветовой палитры для графиков
sns.set(style='whitegrid')

# загрузка данных
file_path = "googleplaystore.csv"
df = pd.read_csv(file_path)

# вывод датасета
print('Первые строки датасета: ')
print(df.head())

print ("\n статистика: ")
print (df.describe())

print ("\n описательная статистика: ")
#print (df.stats())

# обработка отсутствующих значений. Все пропущенные цифровые данным присвоили значения
numeric_cols = df.select_dtypes(include=[np.number]) #выбор числовых колонок
df[numeric_cols.columns] = numeric_cols.fillna(numeric_cols.mean()) #замена пропущенных значений на среднее значение
print(df[numeric_cols.columns])

categorical_cols = df.select_dtypes(include=['object']) #выбор категориальных колонок
df[categorical_cols.columns] = categorical_cols.fillna(categorical_cols.mode().iloc[0]) # замена пропущенных значений на моду
print(df[categorical_cols.columns])

#удаление дублирующих строк
df.drop_duplicates(inplace=True)

#гистограмма распределения рейтингов
#plt.figure(figsize=(10, 6))
#sns.histplot(df['Rating'], kde=True, color='skyblue') # построение гистограммы и кривой плотности распределения
#plt.title("Distr of App rating")
#plt.show() #отображение графика

#распределение приложений по категориям
#plt.figure(figsize=(12, 8))
#sns.countplot(y='Category', data=df, order=df['Category'].value_counts().index, palette='viridis')
#plt.title("App Distr across Categories")
#plt.show() #отображение графика

# распределение платных  и бесплатных  приложений
#plt.figure(figsize=(7, 5))
#sns.countplot(x="Type", data=df) 
#plt.title("Free vs Paid")
#plt.show() #отображение графика

# обнаружение и обработка выбросов
z_scores = np.abs(stats.zscore(df.select_dtypes(include=np.number))) # z оценки для числовых переменных
df = df[(z_scores < 3).all(axis=1)] # удаление строк с выбросами
# Эта строка выполняет фильтрацию строк в DataFrame df на основе условия, связанного с Z-оценками (z-scores)

# стандартизация данных 
df_standardized = df.copy() # копия датафрейма
df_standardized[numeric_cols.columns] = (df_standardized[numeric_cols.columns] - \
                                         df_standardized[numeric_cols.columns].mean()) / df_standardized[numeric_cols.columns].std()

# Создание дополнительного столбца
label_encoder = LabelEncoder()
df['Type_Encoder'] = label_encoder.fit_transform(df['Type']) # преобразование категорийной переменной в числовую

#df = pd.get_dummies(df, columns=["Content Rating"], prefix='ContentRating', drop_first=True)
#df = pd.get_dummies(df, columns=["Category"], prefix='Category_type', drop_first=True)

# Создание сводной таблицы
#pivot_table = df.pivot_table(index='Category', columns='ContentRating_Teen', values="Rating", aggfunc='mean') # создание сводной таблицы средними значениями
#print('\сводная таблица:')
#print(pivot_table)

#сохранение 
output_file_path = 'clear_gapps_lable_encoding.csv'
df.to_csv(output_file_path, index=False)