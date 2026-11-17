import os
from pathlib import Path

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
TRAIN_FILE = BASE_DIR / 'mnist_train.csv'
TEST_FILE = BASE_DIR / 'mnist_test.csv'

IMAGE_SIZE = 28
NUM_PIXELS = IMAGE_SIZE * IMAGE_SIZE
NUM_CLASSES = 10
PIXEL_MAX = 255.0

VALIDATION_FRACTION = 0.1
EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
PATIENCE = 5
RANDOM_STATE = 42

NUM_SAMPLES_SHOWN = 24
NUM_ERRORS_SHOWN = 24


def load_data(path):
    data = pd.read_csv(path).to_numpy()
    y = data[:, 0].astype(int)
    X = data[:, 1:].astype('float32') / PIXEL_MAX
    print(f'Завантажено {path.name}: {len(X)} зображень, {X.shape[1]} пікселів')
    return X, y


def split_train_validation(X, y, fraction):
    rng = np.random.default_rng(RANDOM_STATE)
    indices = rng.permutation(len(X))
    num_val = int(len(X) * fraction)
    val_idx, train_idx = indices[:num_val], indices[num_val:]
    print(f'Навчальна вибірка: {len(train_idx)}, валідаційна: {len(val_idx)}')
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]


def to_images(X):
    return X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)


def build_mlp():
    return keras.Sequential([
        keras.Input(shape=(NUM_PIXELS,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(NUM_CLASSES, activation='softmax'),
    ], name='MLP')


def build_cnn():
    return keras.Sequential([
        keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(NUM_CLASSES, activation='softmax'),
    ], name='CNN')


def compile_model(model):
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, X_train, y_train, X_val, y_val):
    print(f'\n=== Навчання моделі {model.name} ===')
    model.summary()
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE,
                                                   restore_best_weights=True)
    history = model.fit(X_train, keras.utils.to_categorical(y_train, NUM_CLASSES),
                        validation_data=(X_val, keras.utils.to_categorical(y_val, NUM_CLASSES)),
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=[early_stopping], verbose=2)
    return history.history


def winner_takes_all(outputs):
    return np.argmax(outputs, axis=1)


def evaluate_model(model, X_test, y_test):
    outputs = model.predict(X_test, verbose=0)
    y_pred = winner_takes_all(outputs)
    accuracy = np.mean(y_pred == y_test)
    print(f'\n=== Тестування моделі {model.name} ===')
    print(f'Точність на тестових даних: {accuracy:.4f} '
          f'(помилок: {np.sum(y_pred != y_test)} з {len(y_test)})')
    return outputs, y_pred, accuracy


def print_wta_example(model_name, outputs, y_true, index=0):
    print(f'\n=== Принцип "переможець забирає все" ({model_name}, зображення #{index}) ===')
    for digit, value in enumerate(outputs[index]):
        print(f'  Вихід {digit}: {value:.6f}')
    print(f'Переможець: {winner_takes_all(outputs[index:index + 1])[0]}, '
          f'реальна цифра: {y_true[index]}')


def confusion_matrix(y_true, y_pred):
    matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
    np.add.at(matrix, (y_true, y_pred), 1)
    return matrix


def print_class_accuracy(matrix):
    print('\nТочність по кожній цифрі:')
    for digit in range(NUM_CLASSES):
        total = matrix[digit].sum()
        print(f'  Цифра {digit}: {matrix[digit, digit] / total:.4f} ({total} зображень)')


def save_figure(fig, filename):
    fig.savefig(BASE_DIR / filename, dpi=120, bbox_inches='tight')


def show_digits(axes, X, labels, colors):
    for ax, image, label, color in zip(axes.ravel(), X, labels, colors):
        ax.imshow(image.reshape(IMAGE_SIZE, IMAGE_SIZE), cmap='gray_r')
        ax.set_title(label, color=color, fontsize=9)
    for ax in axes.ravel():
        ax.axis('off')


def plot_samples(X, y):
    fig, axes = plt.subplots(3, 8, figsize=(10, 4.5))
    show_digits(axes, X[:NUM_SAMPLES_SHOWN], [f'Цифра {d}' for d in y[:NUM_SAMPLES_SHOWN]],
                ['black'] * NUM_SAMPLES_SHOWN)
    fig.suptitle('Приклади зображень з навчальної вибірки')
    save_figure(fig, 'fig1_samples.png')


def plot_loss(history, model_name, filename):
    epochs = np.arange(1, len(history['loss']) + 1)
    best_epoch = epochs[np.argmin(history['val_loss'])]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, history['loss'], 'o-', label='Навчальні дані')
    ax.plot(epochs, history['val_loss'], 's-', label='Валідаційні дані')
    ax.axvline(best_epoch, color='gray', linestyle='--',
               label=f'Найкраща епоха ({best_epoch})')
    ax.set_title(f'Функція втрат від епохи ({model_name})')
    ax.set_xlabel('Епоха')
    ax.set_ylabel('Функція втрат (categorical crossentropy)')
    ax.set_xticks(epochs)
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, filename)


def plot_accuracy_compare(histories, test_accuracies):
    fig, ax = plt.subplots(figsize=(8, 5))
    for (name, history), color in zip(histories.items(), ['tab:blue', 'tab:orange']):
        epochs = np.arange(1, len(history['accuracy']) + 1)
        ax.plot(epochs, history['accuracy'], '--', color=color,
                label=f'{name}: навчальні дані')
        ax.plot(epochs, history['val_accuracy'], 'o-', color=color,
                label=f'{name}: валідаційні (тест: {test_accuracies[name]:.4f})')
    ax.set_title('Точність від епохи: MLP і CNN')
    ax.set_xlabel('Епоха')
    ax.set_ylabel('Точність')
    ax.legend()
    ax.grid(alpha=0.3)
    save_figure(fig, 'fig4_accuracy_compare.png')


def plot_confusion(matrix, model_name):
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(matrix, cmap='Blues')
    threshold = matrix.max() / 2
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax.text(j, i, matrix[i, j], ha='center', va='center', fontsize=8,
                    color='white' if matrix[i, j] > threshold else 'black')
    fig.colorbar(image, ax=ax)
    ax.set_title(f'Матриця помилок ({model_name}, тестові дані)')
    ax.set_xlabel('Розпізнана цифра')
    ax.set_ylabel('Реальна цифра')
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    save_figure(fig, 'fig5_confusion.png')


def plot_predictions(X, y_true, y_pred, model_name):
    labels = [f'{p} (реал. {t})' for p, t in zip(y_pred, y_true)]
    colors = ['green' if p == t else 'red' for p, t in zip(y_pred, y_true)]
    fig, axes = plt.subplots(3, 8, figsize=(12, 5))
    show_digits(axes, X[:NUM_SAMPLES_SHOWN], labels, colors)
    fig.suptitle(f'Розпізнавання тестових зображень ({model_name})')
    save_figure(fig, 'fig6_predictions.png')


def plot_errors(X, y_true, y_pred, model_name):
    error_idx = np.flatnonzero(y_pred != y_true)[:NUM_ERRORS_SHOWN]
    labels = [f'{y_pred[i]} (реал. {y_true[i]})' for i in error_idx]
    fig, axes = plt.subplots(3, 8, figsize=(12, 5))
    show_digits(axes, X[error_idx], labels, ['red'] * len(error_idx))
    fig.suptitle(f'Приклади помилок розпізнавання ({model_name})')
    save_figure(fig, 'fig7_errors.png')


def main():
    keras.utils.set_random_seed(RANDOM_STATE)

    X_full, y_full = load_data(TRAIN_FILE)
    X_test, y_test = load_data(TEST_FILE)
    X_train, y_train, X_val, y_val = split_train_validation(X_full, y_full, VALIDATION_FRACTION)
    plot_samples(X_train, y_train)

    models = {
        'MLP': (compile_model(build_mlp()), lambda X: X),
        'CNN': (compile_model(build_cnn()), to_images),
    }
    loss_files = {'MLP': 'fig2_mlp_loss.png', 'CNN': 'fig3_cnn_loss.png'}

    histories, test_accuracies, predictions = {}, {}, {}
    for name, (model, prepare) in models.items():
        histories[name] = train_model(model, prepare(X_train), y_train, prepare(X_val), y_val)
        outputs, y_pred, accuracy = evaluate_model(model, prepare(X_test), y_test)
        print_wta_example(name, outputs, y_test)
        test_accuracies[name] = accuracy
        predictions[name] = y_pred
        plot_loss(histories[name], name, loss_files[name])

    best_name = max(test_accuracies, key=test_accuracies.get)
    best_pred = predictions[best_name]
    matrix = confusion_matrix(y_test, best_pred)

    print('\n=== Підсумок ===')
    for name, accuracy in test_accuracies.items():
        print(f'  {name}: точність на тестових даних = {accuracy:.4f}')
    print(f'Найкраща модель: {best_name}')
    print_class_accuracy(matrix)

    plot_accuracy_compare(histories, test_accuracies)
    plot_confusion(matrix, best_name)
    plot_predictions(X_test, y_test, best_pred, best_name)
    plot_errors(X_test, y_test, best_pred, best_name)

    plt.show()


if __name__ == '__main__':
    main()
