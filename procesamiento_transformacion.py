import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

#1. CARGAR DATOS
#Importar el dataset
data = pd.read_csv('train.csv')

#Imprimir las primeras 5 lineas del dataset
print(data.head())

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#2. EXPLORACION INICIAL
#Mostrar número de filas y columnas
print(f"Filas: {data.shape[0]}, Columnas: {data.shape[1]}")

# Ver el tipo de dato de cada columna
print("")
print(data.dtypes)

# Ver descripción de las variables numéricas
print("")
print(data.describe())

# Ver descripción de las variables categóricas
print("")
print(data.describe(include=['object']))

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#3. LIMPIEZA DE DATOS
# Identificación de valores faltantes
data_faltante = data.isnull().sum()
print(data_faltante)

# Filtrar solo las variables con valores faltantes
print("")
variables_data_faltante= data_faltante[data_faltante > 0]
print(variables_data_faltante)

# Calcular la proporción de valores faltantes
porcentaje_faltante = (data_faltante / len(data)) * 100

# Mostrar la proporción de valores faltantes para cada variable
print("")
data_faltante_df = pd.DataFrame({'Cantidad Faltante': data_faltante, 'Proporción (%)': porcentaje_faltante})
print(data_faltante_df.sort_values(by='Proporción (%)', ascending=False))

#Imputar los valores faltantes utilizando diferentes estrategias
# Imputar LotFrontage con la mediana
data['LotFrontage'].fillna(data['LotFrontage'].median(), inplace=True)

# Imputar Alley como 'No Alley'
data['Alley'].fillna('No Alley', inplace=True)

# Imputar MasVnrType y MasVnrArea
data['MasVnrType'].fillna('None', inplace=True)
data['MasVnrArea'].fillna(0, inplace=True)

# Imputar características del sótano
bsmt_columns = ['BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2']
for col in bsmt_columns:
    data[col].fillna('No Basement', inplace=True)

# Imputar Electrical con la moda
data['Electrical'].fillna(data['Electrical'].mode()[0], inplace=True)

# Imputar FireplaceQu como 'No Fireplace'
data['FireplaceQu'].fillna('No Fireplace', inplace=True)

# Imputar características del garaje
garage_columns = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']
for col in garage_columns:
    data[col].fillna('No Garage', inplace=True)
data['GarageYrBlt'].fillna(0)

# Imputar PoolQC, Fence y MiscFeature
data['PoolQC'].fillna('No Pool', inplace=True)
data['Fence'].fillna('No Fence', inplace=True)
data['MiscFeature'].fillna('No Feature', inplace=True)

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#4. MANEJO DE OUTLIERS
# Variables numéricas a analizar
columnas_numericas = ['SalePrice', 'GrLivArea', 'TotalBsmtSF', 'LotArea', 'GarageCars', 'GarageYrBlt']

# Generar boxplots para visualizar los outliers
plt.figure(figsize=(12, 8))
for i, col in enumerate(columnas_numericas, 1):
    plt.subplot(2, 3, i)
    sns.boxplot(x=data[col])
    plt.title(col)

plt.tight_layout()
plt.show()

# Calcular el Z-score
z_scores = stats.zscore(data[columnas_numericas])

# Mostrar filas donde el valor absoluto del Z-score es mayor que 3 (potenciales outliers)
outliers = (z_scores > 3) | (z_scores < -3)
print(data[outliers.any(axis=1)][columnas_numericas])

# Cálculo del IQR para identificar outliers
Q1 = data[columnas_numericas].quantile(0.25)
Q3 = data[columnas_numericas].quantile(0.75)
IQR = Q3 - Q1

# Filtrar valores fuera de 1.5 * IQR
outlier_mask = (data[columnas_numericas] < (Q1 - 1.5 * IQR)) | (data[columnas_numericas] > (Q3 + 1.5 * IQR))
outliers_iqr = data[outlier_mask.any(axis=1)]
print(outliers_iqr[columnas_numericas])

#Tratamiento de outliers
# Eliminar outliers de GrLivArea mayores a 4000
data = data[data['GrLivArea'] < 4000]

# Aplicar transformación logarítmica para LotArea
data['LotArea'] = np.log1p(data['LotArea'])

# Imputar valores anómalos en GarageYrBlt (si algún valor es superior al año actual)
data['GarageYrBlt'] = data['GarageYrBlt'].apply(lambda x: data['GarageYrBlt'].median() if x > 2024 else x)

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#5. TRANSFORMACION DE VARIABLES

#Normalización
# Crear un objeto MinMaxScaler
scaler = MinMaxScaler()

# Aplicar la normalización a las variables numéricas
columnas_numericas = ['SalePrice', 'GrLivArea', 'TotalBsmtSF', 'LotArea', 'GarageCars', 'GarageYrBlt']
data_normalized = data.copy()
data_normalized[columnas_numericas] = scaler.fit_transform(data[columnas_numericas])

# Mostrar los resultados de la normalización
print(data_normalized[columnas_numericas].head())

#Estandarización
# Crear un objeto StandardScaler
scaler = StandardScaler()

# Aplicar la estandarización a las variables numéricas
data_standardized = data.copy()
data_standardized[columnas_numericas] = scaler.fit_transform(data[columnas_numericas])

# Mostrar los resultados de la estandarización
print( data_standardized[columnas_numericas].head())

#Transformación Logarítmica
# Histograma de SalePrice antes de la transformación
sns.histplot(data['SalePrice'], kde=True)
plt.title('Distribución de SalePrice antes de la transformación')
plt.show()

# Calcular el sesgo de la variable SalePrice
print("")
print("Sesgo de SalePrice:", data['SalePrice'].skew())

# Aplicar la transformación logarítmica a SalePrice
data['LogSalePrice'] = np.log1p(data['SalePrice'])

# Histograma de SalePrice después de la transformación
sns.histplot(data['LogSalePrice'], kde=True)
plt.title('Distribución de SalePrice después de la transformación logarítmica')
plt.show()

# Calcular el sesgo de la variable transformada
print("")
print("Sesgo de LogSalePrice:", data['LogSalePrice'].skew())

#:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

#6. INGENIERIA DE CARACTERISTICAS
#Creación de nuevas variables
# Crear la nueva variable TotalAreaRatio
data['TotalAreaRatio'] = data['GrLivArea'] / data['LotArea']

# Verificar la nueva columna
print("")
print(data[['TotalAreaRatio']].head())

# Crear la nueva variable HouseAgeAtSale
data['HouseAgeAtSale'] = data['YrSold'] - data['YearBuilt']

# Verificar la nueva columna
print("")
print(data[['HouseAgeAtSale']].head())

# Aplicar One-Hot Encoding a las variables categóricas seleccionadas
data_encoded = pd.get_dummies(data, columns=['Neighborhood', 'SaleCondition'])

# Verificar el resultado de One-Hot Encoding
print("")
print(data_encoded.head())

# Aplicar Label Encoding a una variable categórica con relación ordinal
le = LabelEncoder()
data['ExterQual_encoded'] = le.fit_transform(data['ExterQual'])

# Verificar el resultado de Label Encoding
print(data[['ExterQual', 'ExterQual_encoded']].head())

# Exportar el dataset reorganizado
data.to_csv('train_procesado.csv', index=False)