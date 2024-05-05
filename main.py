import numpy as np
from keras import Sequential
import pandas as pd
from keras.src.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

df = pd.read_csv('test_data.csv', sep=',')
data = df.to_numpy()  # Konwersja do tablicy numpy

# Wyodrębnienie cech i etykiet kategorii do osobnych kolumn
X = data[:, 0:-1].astype('float')  # Cechy - wszystkie kolumny oprócz ostatniej
Y = data[:, -1]  # Etykiety - ostatnia kolumna

label_encoder = LabelEncoder()  # Zamiana etykiet nominalnych na typ int
integer_encoder = label_encoder.fit_transform(Y)
# Kodowanie 1 z n
onehot_encoder = OneHotEncoder(sparse_output=False)
integer_encoded = integer_encoder.reshape(len(integer_encoder), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

# Podział zestawu danych na dane treningowe i testowe
X_train, X_test, Y_train, Y_test = train_test_split(X, onehot_encoded, test_size=0.3)

model = Sequential()
# Liczba neuronów, rozmiar wektora wejściowego, funkcja aktywacji
model.add(Dense(10, input_dim=72, activation='sigmoid'))
# Warstwa wyjściowa z 3 neuronami
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.summary()

# Podanie zbiorów treningowych, liczby iteracji, wymieszanie wektorów danych
model.fit(X_train, Y_train, epochs=100, batch_size=10, shuffle=True)

Y_pred = model.predict(X_test)
Y_pred_int = np.argmax(Y_pred, axis=1)
Y_test_int = np.argmax(Y_test, axis=1)
cm = confusion_matrix(Y_test_int, Y_pred_int)
print(cm)
