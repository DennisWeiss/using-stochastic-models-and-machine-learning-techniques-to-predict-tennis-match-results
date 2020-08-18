import math
import random
import pandas as pd
import numpy as np
import re
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt


def parse_score(score_string):
    if type(score_string) == float or 'Played' in score_string:
        return ['1', '1']
    try:
        score = re.split('[-,\\s]', re.sub('\\([0-9]*\\)|RET|DEF', '', score_string))
        if len(score) == 0:
            return ['1', '0']
        if score[0] == '' or score[0] == 'W/O':
            return ['1', '0']
        return score
    except:
        return ['1', '0']


def result_value(score):
    sets = int(len(score) / 2)
    sets_won = 0
    rounds = 0
    rounds_won = 0
    for i in range(sets):
        try:
            score1 = int(score[2 * i])
            score2 = int(score[2 * i + 1])
        except:
            return 0.5
        rounds += score1 + score2
        rounds_won += score1
        if score1 > score2:
            sets_won += 1
    won = int(sets_won > sets - sets_won)
    try:
        return 1 / 3 * (rounds_won / rounds + sets_won / sets + won)
    except:
        return 0.5


def logistic(x):
    return 1 / (1 + 10 ** -x)


def load_data(number_of_past_games, features='without_age'):
    alpha = 0.083

    size_per_game = 2

    if features == 'with_age':
        size_per_game = 4
    if features == 'with_age_and_surface':
        size_per_game = 8

    user_to_elo_rating = {}
    user_to_past_games = {}

    def get_mean_age(data):
        summed = 0
        count = 0
        for i in range(data.shape[0]):
            match = data.iloc[i]
            if match['loser_age'] is not None and not math.isnan(match['loser_age']):
                summed += match['loser_age']
                count += 1
            if match['winner_age'] is not None and not math.isnan(match['winner_age']):
                summed += match['winner_age']
                count += 1
        return 0 if count == 0 else summed / count

    def get_sample_std(data):
        mean = get_mean_age(data)
        summed = 0
        count = 0
        for i in range(data.shape[0]):
            match = data.iloc[i]
            if match['loser_age'] is not None and not math.isnan(match['loser_age']):
                summed += (match['loser_age'] - mean) ** 2
                count += 1
            if match['winner_age'] is not None and not math.isnan(match['winner_age']):
                summed += (match['winner_age'] - mean) ** 2
                count += 1
        return math.sqrt(summed / (count - 1))

    def rating(player):
        return user_to_elo_rating[player] if player in user_to_elo_rating else 0

    def prediction(player1, player2):
        return logistic(rating(player1) - rating(player2))

    def get_average_result(X, i, player1, number_past_games):
        if number_past_games == 0:
            return 0
        summed = 0
        for j in range(number_past_games):
            summed += X[i, size_per_game * j + (0 if player1 else size_per_game * number_of_past_games)]
        return summed / number_past_games

    def get_average_opponent_rating(X, i, player1, number_past_games):
        if number_past_games == 0:
            return 0
        summed = 0
        for j in range(number_past_games):
            summed += X[i, size_per_game * j + 1 + (0 if player1 else size_per_game * number_of_past_games)]
        return summed / number_past_games

    def fill_missing_values(X, i, player1, number_past_games):
        average_result = get_average_result(X, i, player1, number_past_games)
        average_opponent_rating = get_average_opponent_rating(X, i, player1, number_past_games)
        for j in range(number_past_games, number_of_past_games):
            X[i, size_per_game * j + (0 if player1 else size_per_game * number_of_past_games)] = average_result
            X[i, size_per_game * j + 1 + (0 if player1 else size_per_game * number_of_past_games)] = average_opponent_rating

    def get_surface_index(surface):
        if surface == 'Grass':
            return 0
        if surface == 'Clay':
            return 1
        if surface == 'Hard':
            return 2
        if surface == 'Carpet':
            return 3
        return 0

    atp = pd.read_csv('data/ATP.csv')
    # atp = atp.iloc[atp.shape[0] - 10000:atp.shape[0]]

    X = np.zeros((atp.shape[0], 2 * size_per_game * number_of_past_games + (4 if features == 'with_age_and_surface' else 0)))
    y = np.zeros(atp.shape[0])

    mean_age = get_mean_age(atp)
    std_age = get_sample_std(atp)

    for i in range(atp.shape[0]):
        match = atp.iloc[i]
        if str(match['winner_id']) not in user_to_past_games:
            user_to_past_games[str(match['winner_id'])] = []
        if str(match['winner_id']) not in user_to_elo_rating:
            user_to_elo_rating[str(match['winner_id'])] = 0
        if str(match['loser_id']) not in user_to_past_games:
            user_to_past_games[str(match['loser_id'])] = []
        if str(match['loser_id']) not in user_to_elo_rating:
            user_to_elo_rating[str(match['loser_id'])] = 0

        winner_first = random.random() < 0.5

        user_past_games1 = user_to_past_games[str(match['winner_id'])]
        number_last_games1 = min(len(user_past_games1), number_of_past_games)
        for j in range(number_last_games1):
            past_game = user_past_games1[len(user_past_games1) - 1 - j]
            X[i, size_per_game * j + (size_per_game * number_of_past_games if winner_first else 0)] = past_game['result']
            X[i, size_per_game * j + 1 + (size_per_game * number_of_past_games if winner_first else 0)] = past_game['opponent_rating']
            if features == 'with_age' or features == 'with_age_and_surface':
                X[i, size_per_game * j + 2 + (size_per_game * number_of_past_games if winner_first else 0)] = (past_game['player_age'] - mean_age) / std_age
                X[i, size_per_game * j + 3 + (size_per_game * number_of_past_games if winner_first else 0)] = (past_game['opponent_age'] - mean_age) / std_age
            if features == 'with_age_and_surface':
                X[i, size_per_game * j + 4 + get_surface_index(past_game['surface']) + (size_per_game * number_of_past_games if winner_first else 0)] = 1

        user_past_games2 = user_to_past_games[str(match['loser_id'])]
        number_last_games2 = min(len(user_past_games2), number_of_past_games)
        for j in range(number_last_games2):
            past_game = user_past_games2[len(user_past_games2) - 1 - j]
            X[i, size_per_game * j + (0 if winner_first else size_per_game * number_of_past_games)] = past_game['result']
            X[i, size_per_game * j + 1 + (0 if winner_first else size_per_game * number_of_past_games)] = past_game['opponent_rating']
            if features == 'with_age' or features == 'with_age_and_surface':
                X[i, size_per_game * j + 2 + (0 if winner_first else size_per_game * number_of_past_games)] = (past_game['player_age'] - mean_age) / std_age
                X[i, size_per_game * j + 3 + (0 if winner_first else size_per_game * number_of_past_games)] = (past_game['opponent_age'] - mean_age) / std_age
            if features == 'with_age_and_surface':
                X[i, size_per_game * j + 4 + get_surface_index(past_game['surface']) + (0 if winner_first else size_per_game * number_of_past_games)] = 1

        if features == 'with_age_and_surface':
            X[i, 2 * size_per_game * number_of_past_games + get_surface_index(match['surface'])] = 1

        result = result_value(parse_score(match['score']))
        y[i] = result if winner_first else 1 - result

        user_past_games1.append({
            'result': result,
            'opponent_rating': rating(str(match['loser_id'])),
            'player_age': mean_age if match['winner_age'] is None or math.isnan(match['winner_age']) else match['winner_age'],
            'opponent_age': mean_age if match['loser_age'] is None or math.isnan(match['loser_age']) else match['loser_age'],
            'surface': match['surface']
        })
        user_past_games2.append({
            'result': 1 - result,
            'opponent_rating': rating(str(match['winner_id'])),
            'player_age': mean_age if match['loser_age'] is None or math.isnan(match['loser_age']) else match['loser_age'],
            'opponent_age': mean_age if match['winner_age'] is None or math.isnan(match['winner_age']) else match['winner_age'],
            'surface': match['surface']
        })

        predicted = prediction(str(match['winner_id']), str(match['loser_id']))
        rating_exchange = alpha * (result - predicted)
        user_to_elo_rating[str(match['winner_id'])] += rating_exchange
        user_to_elo_rating[str(match['loser_id'])] -= rating_exchange

        fill_missing_values(X, i, winner_first, number_last_games1)
        fill_missing_values(X, i, not winner_first, number_last_games2)

    return X, y


def winner_prediction_accuracy(y_true, y_pred):
    return tf.subtract(tf.constant(1, dtype=tf.float32), tf.abs(tf.subtract(tf.round(y_true), tf.round(y_pred))))


def mean_absolute_error(y_true, y_pred):
    return tf.multiply(tf.constant(2, dtype=tf.float32), tf.abs(tf.subtract(y_true, y_pred)))


def ffnn_model_with_features(features='without_age'):
    def ffnn_model(number_of_past_games, learning_rate, number_of_hidden_layers, perceptron_count_factor, layer_shrink_factor, batch_size_exponent, activation_function):
        def run_model(X, y, best_history=None):
            size_per_match = 2

            if features == 'with_age':
                size_per_match = 4
            if features == 'with_age_and_surface':
                size_per_match = 8

            max_neuron_count = size_per_match * 128

            data_size = len(y)
            training_data_size = int(0.8 * data_size)
            cross_validation_data_size = int(0.1 * data_size)

            model = keras.models.Sequential()

            prev_layer_size = 2 * size_per_match * number_of_past_games + (4 if features == 'with_age_and_surface' else 0)

            for i in range(number_of_hidden_layers):
                this_layer_size = math.ceil(pow(layer_shrink_factor, i) * perceptron_count_factor * max_neuron_count)
                model.add(keras.layers.Dense(this_layer_size, input_dim=prev_layer_size, activation=activation_function))
                prev_layer_size = this_layer_size

            model.add(keras.layers.Dense(1, activation='sigmoid'))

            model.compile(loss='mean_squared_error', metrics=[winner_prediction_accuracy, mean_absolute_error],
                          optimizer=keras.optimizers.Adam(learning_rate))

            model.summary()

            history = []
            training_history = []

            for i in range(10):
                model_history = model.fit(
                    X[0:training_data_size, :],
                    y[0:training_data_size],
                    epochs=1,
                    batch_size=2 ** batch_size_exponent,
                    validation_data=(
                        X[training_data_size:(training_data_size+cross_validation_data_size), :],
                        y[training_data_size:(training_data_size+cross_validation_data_size)]
                    )
                )
                current_val_loss = model_history.history['val_loss'][0]
                history.append(current_val_loss)
                training_history.append(model_history.history['loss'][0])
                if best_history is not None and current_val_loss > 1.1 * best_history[i]:
                    break

            return model, history, training_history
        return run_model
    return ffnn_model


def get_random_hyper_parameter(hyper_parameter):
    if hyper_parameter['type'] == 'float':
        return random.uniform(hyper_parameter['from'], hyper_parameter['to'])
    if hyper_parameter['type'] == 'int':
        return random.randint(hyper_parameter['from'], hyper_parameter['to'])
    if hyper_parameter['type'] == 'set':
        return random.choice(hyper_parameter['values'])


def get_random_hyper_parameters(hyper_parameters):
    random_hyper_parameters = [0] * len(hyper_parameters)
    for i in range(len(hyper_parameters)):
        random_hyper_parameters[i] = get_random_hyper_parameter(hyper_parameters[i])
    return tuple(random_hyper_parameters)


def random_search(model, hyper_parameters, features='without_age'):
    best_hyper_parameters = None
    best_cross_validation_loss = float('inf')
    best_history = None
    for i in range(10):
        current_hyper_parameters = get_random_hyper_parameters(hyper_parameters)
        print('hyperparameters: ' + str(current_hyper_parameters))
        X, y = load_data(current_hyper_parameters[0], features=features)
        keras_model, history, training_history = model(*current_hyper_parameters)(X, y, best_history)
        current_cross_validation_loss = history[len(history) - 1]
        print('current cross validation loss: ' + str(current_cross_validation_loss))
        if current_cross_validation_loss < best_cross_validation_loss:
            best_cross_validation_loss = current_cross_validation_loss
            best_hyper_parameters = current_hyper_parameters
            best_history = history
    return best_hyper_parameters, best_cross_validation_loss


def main(print_training_history=False):
    features = 'with_age'

    n_past_games = 32

    X, y = load_data(n_past_games, features=features)
    model, history, training_history = ffnn_model_with_features(features=features)(n_past_games, 0.008544899405772726, 2, 0.046727832302539, 0.7270628586602064, 4, 'relu')(X, y)

    if print_training_history:
        plot_x = list(range(1, 10 + 1))

        plt.plot(plot_x, training_history, label='training loss')
        plt.plot(plot_x, history, label='test loss')

        plt.xlabel('epochs')
        plt.ylabel('MSE')

        plt.legend()
        plt.show()

    data_size = len(y)
    training_data_size = int(0.8 * data_size)
    cross_validation_data_size = int(0.1 * data_size)

    model.evaluate(
        X[(training_data_size+cross_validation_data_size):data_size, :],
        y[(training_data_size+cross_validation_data_size):data_size]
    )


def do_hyper_parameter_optimization():
    features = 'with_age_and_surface'
    print(random_search(ffnn_model_with_features(features=features), [
        {'type': 'int', 'from': 1, 'to': 40},
        {'type': 'float', 'from': 0.0001, 'to': 0.01},
        {'type': 'int', 'from': 1, 'to': 6},
        {'type': 'float', 'from': 0.01, 'to': 1},
        {'type': 'float', 'from': 0.01, 'to': 1},
        {'type': 'int', 'from': 0, 'to': 15},
        {'type': 'set', 'values': ['sigmoid', 'tanh', 'relu']}
    ], features=features))


main(True)
# do_hyper_parameter_optimization()

