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

    user_to_elo_rating = {}
    user_to_past_games = {}

    size_per_match = 2

    if features == 'with_age':
        size_per_match = 4
    if features == 'with_age_and_surface':
        size_per_match = 8

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

    def get_average_result(X, i, number_past_games):
        if number_past_games == 0:
            return 0
        summed = 0
        for j in range(number_of_past_games - number_past_games, number_of_past_games):
            summed += X[i, j, 0]
        return summed / number_past_games

    def get_average_opponent_rating(X, i, number_past_games):
        if number_past_games == 0:
            return 0
        summed = 0
        for j in range(number_of_past_games - number_past_games, number_of_past_games):
            summed += X[i, j, 1]
        return summed / number_past_games

    def fill_missing_values(X, i, number_past_games):
        average_result = get_average_result(X, i, number_past_games)
        average_opponent_rating = get_average_opponent_rating(X, i, number_past_games)
        for j in range(number_of_past_games - number_past_games):
            X[i, j, 0] = average_result
            X[i, j, 1] = average_opponent_rating

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

    mean_age = get_mean_age(atp)
    std_age = get_sample_std(atp)

    X1 = np.zeros((atp.shape[0], number_of_past_games, size_per_match))
    X2 = np.zeros((atp.shape[0], number_of_past_games, size_per_match))
    X3 = np.zeros((atp.shape[0], 4)) if features == 'with_age_and_surface' else None
    y = np.zeros(atp.shape[0])

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
            past_game = user_past_games1[j]
            X = X1 if winner_first else X2
            X[i, number_of_past_games - number_last_games1 + j, 0] = past_game['result']
            X[i, number_of_past_games - number_last_games1 + j, 1] = past_game['opponent_rating']
            if features == 'with_age' or features == 'with_age_and_surface':
                X[i, number_of_past_games - number_last_games1 + j, 2] = (past_game['player_age'] - mean_age) / std_age
                X[i, number_of_past_games - number_last_games1 + j, 3] = (past_game['opponent_age'] - mean_age) / std_age
            if features == 'with_age_and_surface':
                X[i, number_of_past_games - number_last_games1 + j, get_surface_index(past_game['surface']) + 4] = 1

        user_past_games2 = user_to_past_games[str(match['loser_id'])]
        number_last_games2 = min(len(user_past_games2), number_of_past_games)
        for j in range(number_last_games2):
            past_game = user_past_games2[j]
            X = X2 if winner_first else X1
            X[i, number_of_past_games - number_last_games2 + j, 0] = past_game['result']
            X[i, number_of_past_games - number_last_games2 + j, 1] = past_game['opponent_rating']
            if features == 'with_age' or features == 'with_age_and_surface':
                X[i, number_of_past_games - number_last_games2 + j, 2] = (past_game['player_age'] - mean_age) / std_age
                X[i, number_of_past_games - number_last_games2 + j, 3] = (past_game['opponent_age'] - mean_age) / std_age
            if features == 'with_age_and_surface':
                X[i, number_of_past_games - number_last_games2 + j, get_surface_index(past_game['surface']) + 4] = 1

        result = result_value(parse_score(match['score']))
        y[i] = result if winner_first else 1 - result

        if features == 'with_age_and_surface':
            X3[i, get_surface_index(match['surface'])] = 1

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

        fill_missing_values(X1 if winner_first else X2, i, number_last_games1)
        fill_missing_values(X2 if winner_first else X1, i, number_last_games2)

    return X1, X2, X3, y


def winner_prediction_accuracy(y_true, y_pred):
    return tf.subtract(tf.constant(1, dtype=tf.float32), tf.abs(tf.subtract(tf.round(y_true), tf.round(y_pred))))


def mean_absolute_error(y_true, y_pred):
    return tf.multiply(tf.constant(2, dtype=tf.float32), tf.abs(tf.subtract(y_true, y_pred)))


def split_X_data(X1, X2, X3, features, start, end):
    if features == 'with_age_and_surface':
        return [X1[start:end, :, :], X2[start:end, :, :], X3[start:end, :]]
    return [X1[start:end, :, :], X2[start:end, :, :]]


def split_y_data(y, start, end):
    return y[start:end]


def rnn_model_with_features(features='without_age'):
    def rnn_model(n_past_games, learning_rate, embedding_dimension, n_hidden_layers, perceptron_count_factor,
              hidden_layer_shrink_factor, batch_size_exponent, activation_function_rnn, activation_function_ffnn):
        MAX_NEURON_COUNT = 128

        size_per_match = 2

        if features == 'with_age':
            size_per_match = 4
        if features == 'with_age_and_surface':
            size_per_match = 8

        def run_model(X1, X2, X3, y, best_history=None):
            data_size = X1.shape[0]
            training_data_size = int(0.8 * data_size)
            cross_validation_data_size = int(0.1 * data_size)

            player_1_history = keras.layers.Input(shape=(n_past_games, size_per_match), dtype='float32')
            player_1_embedding = keras.layers.LSTM(embedding_dimension, activation=activation_function_rnn)(player_1_history)
            player_2_history = keras.layers.Input(shape=(n_past_games, size_per_match), dtype='float32')
            player_2_embedding = keras.layers.LSTM(embedding_dimension, activation=activation_function_rnn)(player_2_history)

            ffnn_input = keras.layers.concatenate([player_1_embedding, player_2_embedding])

            if features == 'with_age_and_surface':
                surface_input = keras.layers.Input(shape=4, dtype='float32')
                ffnn_input = keras.layers.concatenate([player_1_embedding, player_2_embedding, surface_input])

            prev_layer = ffnn_input
            prev_layer_size = 2 * embedding_dimension
            for i in range(n_hidden_layers):
                this_layer_size = math.ceil(math.pow(hidden_layer_shrink_factor, i) * perceptron_count_factor * MAX_NEURON_COUNT)
                prev_layer = keras.layers.Dense(this_layer_size, input_dim=prev_layer_size, activation=activation_function_ffnn)(prev_layer)
                prev_layer_size = this_layer_size

            output = keras.layers.Dense(1, activation='sigmoid')(prev_layer)

            model = keras.models.Model([player_1_history, player_2_history, surface_input] if features == 'with_age_and_surface' else [player_1_history, player_2_history], output)

            model.compile(loss='mean_squared_error', metrics=[winner_prediction_accuracy, mean_absolute_error], optimizer=keras.optimizers.Adam(learning_rate))

            model.summary()

            history = []
            training_history = []

            for i in range(20):
                model_history = model.fit(
                    split_X_data(X1, X2, X3, features, 0, training_data_size),
                    split_y_data(y, 0, training_data_size),
                    epochs=1,
                    validation_data=(
                        split_X_data(X1, X2, X3, features, training_data_size, training_data_size + cross_validation_data_size),
                        split_y_data(y, training_data_size, training_data_size + cross_validation_data_size)
                    ),
                    batch_size=2 ** batch_size_exponent
                )

                current_val_loss = model_history.history['val_loss'][0]
                history.append(current_val_loss)
                training_history.append(model_history.history['loss'][0])
                if best_history is not None and current_val_loss > 1.1 * best_history[i]:
                    break

            return model, history, training_history
        return run_model
    return rnn_model


def main(print_training_history=False):
    features = 'with_age'

    n_past_games = 34

    X1, X2, X3, y = load_data(n_past_games, features=features)

    data_size = X1.shape[0]
    training_data_size = int(0.8 * data_size)
    cross_validation_data_size = int(0.1 * data_size)

    model, history, training_history = rnn_model_with_features(features=features)(n_past_games, 0.007231477705881956, 11, 1, 0.5668506172536623, 0.2445231254351173, 11, 'relu', 'relu')(X1, X2, X3, y)

    if print_training_history:
        plot_x = list(range(1, 20 + 1))

        plt.plot(plot_x, training_history, label='training loss')
        plt.plot(plot_x, history, label='test loss')

        plt.xlabel('epochs')
        plt.ylabel('MSE')

        plt.legend()
        plt.show()

    model.evaluate(
        split_X_data(X1, X2, X3, features, training_data_size + cross_validation_data_size, data_size),
        split_y_data(y, training_data_size + cross_validation_data_size, data_size)
    )


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
        X1, X2, X3, y = load_data(current_hyper_parameters[0], features=features)
        keras_model, history, training_history = model(*current_hyper_parameters)(X1, X2, X3, y, best_history)
        current_cross_validation_loss = history[len(history) - 1]
        print('current cross validation loss: ' + str(current_cross_validation_loss))
        if current_cross_validation_loss < best_cross_validation_loss:
            best_cross_validation_loss = current_cross_validation_loss
            best_hyper_parameters = current_hyper_parameters
            best_history = history
    return best_hyper_parameters, best_cross_validation_loss


def do_hyper_parameter_optimization():
    features = 'with_age'
    print(random_search(rnn_model_with_features(features=features), [
        {'type': 'int', 'from': 1, 'to': 50},
        {'type': 'float', 'from': 0.0001, 'to': 0.01},
        {'type': 'int', 'from': 1, 'to': 30},
        {'type': 'int', 'from': 1, 'to': 3},
        {'type': 'float', 'from': 0.01, 'to': 1},
        {'type': 'float', 'from': 0.01, 'to': 1},
        {'type': 'int', 'from': 4, 'to': 15},
        {'type': 'set', 'values': ['sigmoid', 'tanh', 'relu']},
        {'type': 'set', 'values': ['sigmoid', 'tanh', 'relu']}
    ], features=features))


main(True)
# do_hyper_parameter_optimization()
