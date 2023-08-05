import tensorflow as tf
import numpy as np

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit = np.array([-40, 14, 32, 46.4, 59, 71.6, 100.4], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

#                Entrenamiento y almacenado del  modelo
######################################################################
#print("comienza el entrenamiento")
#historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose=0)
#print("Modelo entrenado")

#import matplotlib.pyplot as plt
#plt.xlabel("# Epoca")
#plt.ylabel("Magnitud de pérdida")
#plt.plot(historial.history["loss"])
#plt.show()

#modelo.save("CelToFah.keras")
######################################################################

modelo = tf.keras.models.load_model("CelToFah.keras")

cel = float(input("Introduce un valor en grados Celsius: "))
fah = modelo.predict([cel], verbose=0)
print(f"El valor equivalente en grados Fahrenheit es: {fah}")
