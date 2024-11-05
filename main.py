import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, F, H, Q, R, P, x):
        self.F = F  # Матриця переходу стану
        self.H = H  # Матриця вимірювання
        self.Q = Q  # Коваріація шуму процесу
        self.R = R  # Коваріація шуму вимірювання
        self.P = P  # Початкова коваріація похибки оцінки
        self.x = x  # Початковий стан

    def predict(self):
        # Передбачення стану та похибки оцінки
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        # Розрахунок коефіцієнта Калмана
        K = np.dot(self.P, self.H.T) / (np.dot(self.H, np.dot(self.P, self.H.T)) + self.R)

        # Оновлення оцінки за допомогою вимірювання z
        self.x = self.x + K * (z - np.dot(self.H, self.x))

        # Оновлення коваріації похибки
        self.P = (np.eye(len(self.P)) - K * self.H) @ self.P

        return self.x

# === Параметри сигналу ===
frequency = 1  # Частота синусоїди в Гц
amplitude = 5  # Амплітуда синусоїди
offset = 10  # Зсув синусоїди
sampling_interval = 0.001  # Інтервал вибірки у секундах (1 мс)
total_time = 1  # Загальна тривалість у секундах (1 секунда)

# === Параметри шуму ===
noise_variance = 16  # Дисперсія нормального шуму
noise_std_dev = np.sqrt(noise_variance)  # Розрахунок стандартного відхилення з дисперсії

# === Параметри фільтра ===
F = np.array([[1]])  # Матриця переходу стану
H = np.array([[1]])  # Матриця вимірювання

Q = np.array([[1]])  # Коваріація шуму процесу
R = np.array([[10]])  # Коваріація шуму вимірювання

P = np.array([[1]])  # Початкова коваріація похибки оцінки
x = np.array([[0]])  # Початкова оцінка стану

# Створення екземпляра фільтра Калмана
kf = KalmanFilter(F, H, Q, R, P, x)

# === Генерація сигналу ===
time_steps = np.arange(0, total_time, sampling_interval)  # Генерація часових кроків від 0 до total_time з кроком sampling_interval
true_signal = offset + amplitude * np.sin(2 * np.pi * frequency * time_steps)  # Генерація синусоїди на основі параметрів
noisy_signal = [val + np.random.normal(0, noise_std_dev) for val in true_signal]  # Додавання нормального шуму зі стандартним відхиленням

# === Застосування фільтра Калмана ===
kalman_estimates = []

for measurement in noisy_signal:
    kf.predict()  # Передбачення наступного стану
    estimate = kf.update(measurement)  # Оновлення за допомогою зашумленого вимірювання
    kalman_estimates.append(estimate[0][0])  # Збереження відфільтрованого результату

# === Розрахунок дисперсії до та після фільтрації ===
noise_variance_before = np.var(noisy_signal - true_signal)  # Дисперсія шуму у початковому сигналі
noise_variance_after = np.var(kalman_estimates - true_signal)  # Дисперсія шуму після фільтрації Калмана

# Виведення дисперсій
print(f"Дисперсія шуму до фільтрації: {noise_variance_before:.2f}")
print(f"Дисперсія шуму після фільтрації: {noise_variance_after:.2f}")

# === Побудова графіка результатів ===
plt.figure(figsize=(12, 6))
plt.plot(time_steps, noisy_signal, label='Зашумлений сигнал', color='orange', linestyle='-', alpha=0.6)
plt.plot(time_steps, true_signal, label='Справжній сигнал (синусоїда)', linestyle='--', color='blue')
plt.plot(time_steps, kalman_estimates, label='Оцінка фільтром Калмана', color='green')
plt.xlabel('Час (с)')
plt.ylabel('Значення')
plt.title('Фільтр Калмана, застосований до зашумленої синусоїди')
plt.legend()
plt.grid()
plt.show()
