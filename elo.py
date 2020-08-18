import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import re
import math
import itertools


np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


t = 400
alpha = 0.083

learning_rate = 1


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


def user_index_maps(users):
    user_to_index = {}
    index_to_user = {}
    index_to_username = {}

    for i in range(len(users)):
        user_to_index[users[i]['id']] = i
        index_to_user[i] = users[i]['id']
        index_to_username[i] = users[i]['name']

    return user_to_index, index_to_user, index_to_username


def construct_A_S(users, games):
    A = np.zeros((len(games), len(users)))
    S = np.zeros(A.shape)

    user_to_index, index_to_user, index_to_username = user_index_maps(users)

    for i in range(len(games)):
        player1 = user_to_index[games[i]['players1'][0]]
        player2 = user_to_index[games[i]['players2'][0]]
        result = result_value(games[i]['score'])
        A[i, player1] = 1
        A[i, player2] = 1
        S[i, player1] = result
        S[i, player2] = 1 - result

    return A, S


def logistic(x):
    return 1 / (1 + 10 ** -x)


def logistic_derivative(x):
    return 10 ** (-x) * math.log(10) / (1 + 10 ** (-x)) ** 2


def compute_P_R(A, S, t, alpha):
    print(alpha)
    P = np.zeros(A.shape)
    R = np.zeros(A.shape)

    def rating(i, j):
        if i == -1:
            return 0
        return R[i, j]

    def prediction(i, player1, player2):
        return logistic((rating(i - 1, player1) - rating(i - 1, player2)) / t)

    for i in range(A.shape[0]):
        player1 = None
        player2 = None
        for j in range(A.shape[1]):
            if A[i, j] == 1:
                if player1 is None:
                    player1 = j
                else:
                    player2 = j
                    break
        if player1 is not None and player2 is not None:
            p1 = prediction(i, player1, player2)
            P[i, player1] = p1
            p2 = prediction(i, player2, player1)
            P[i, player2] = p2
            for j in range(A.shape[1]):
                if A[i, j] == 1:
                    R[i, j] = rating(i-1, j) + (S[i, j] - P[i, j]) * alpha * t
                else:
                    R[i, j] = rating(i-1, j)

    return P, R


def cost(A, S, P):
    summed = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] == 1:
                summed += (S[i, j] - P[i, j]) ** 2
    return summed / (2 * A.shape[0])


def mean_absolute_error(A, S, P):
    summed = 0
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j] == 1:
                summed += abs(S[i, j] - P[i, j])
    return summed / A.shape[0]


def win_prediction_accuracy(A, S, P):
    correct_win_prediction = 0
    for i in range(A.shape[0]):
        indices = []
        for j in range(A.shape[1]):
            if A[i, j] == 1:
                indices.append(j)
        if (S[i, indices[0]] >= S[i, indices[1]] and P[i, indices[0]] > P[i, indices[1]]) or (S[i, indices[0]] <= S[i, indices[1]] and P[i, indices[0]] < P[i, indices[1]]):
            correct_win_prediction += 1
    return correct_win_prediction / A.shape[0]


def filter_single_player_games(games):
    single_player_games = []
    for game in games:
        if len(game['players1']) == 1 and len(game['players2']) == 1:
            single_player_games.append(game)
    return single_player_games


def load_A_S_of_atp_data():
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

    atp = pd.read_csv('data/ATP.csv')
    atp = atp.iloc[atp.shape[0] - 20000:atp.shape[0]]
    users_set = set()
    games = []
    for i, match in atp.iterrows():
        users_set.add((str(match['loser_id']), match['loser_name']))
        users_set.add((str(match['winner_id']), match['winner_name']))
    users = list(map(lambda user: {'id': user[0], 'name': user[1]}, list(users_set)))
    for i, match in atp.iterrows():
        games.append({
            'players1': [str(match['winner_id'])],
            'players2': [str(match['loser_id'])],
            'score': parse_score(match['score'])
        })
    A, S = construct_A_S(users, filter_single_player_games(games))
    return A, S, users, games


def compute_gradient_cost_alpha(A, S, t, alpha):
    P, R = compute_P_R(A, S, t, alpha)

    def r(i, j):
        if i < 0:
            return 0
        return R[i, j]

    def d_R(i, j):
        if i < 0:
            return 0
        return dR[i, j]

    dP = np.zeros(A.shape)
    dR = np.zeros(A.shape)

    sum = 0
    for i in range(A.shape[0]):
        player1 = None
        player2 = None
        for j in range(A.shape[1]):
            dR[i, j] = d_R(i-1, j)
            if A[i, j] != 0:
                if player1 is None:
                    player1 = j
                else:
                    player2 = j
        dP[i, player1] = logistic_derivative((r(i - 1, player1) - r(i - 1, player2)) / t) / t * (
                    d_R(i - 1, player1) - d_R(i - 1, player2))
        dP[i, player2] = logistic_derivative((r(i-1, player2) - r(i-1, player1)) / t) / t * (d_R(i-1, player2) - d_R(i-1, player1))
        dR[i, player1] = (S[i, player1] - P[i, player1]) * t + d_R(i-1, player1) - alpha * t * dP[i, player1]
        dR[i, player2] = (S[i, player2] - P[i, player2]) * t + d_R(i-1, player2) - alpha * t * dP[i, player2]
        sum += (P[i, player1] - S[i, player1]) * dP[i, player1]
        sum += (P[i, player2] - S[i, player2]) * dP[i, player2]
    return sum / A.shape[0], cost(A, S, P)


def gradient_descent_optimization(A, S, t):
    alpha = 0.4
    iterations = list(range(0, 15))
    costs = []
    for i in iterations:
        gradient, _cost = compute_gradient_cost_alpha(A, S, t, alpha)
        print('gradient: ' + str(gradient))
        alpha -= learning_rate * gradient
        print('alpha: ' + str(alpha))
        print('cost: ' + str(_cost))
        costs.append(_cost)
    plt.plot(iterations, costs)
    plt.xlabel('epochs')
    plt.ylabel('MSE')
    plt.show()
    return alpha


def print_final_rankings(R, index_to_username, file_name):
    ranking = {'name': [], 'rating': []}
    for i in range(R.shape[1]):
        ranking['name'].append(index_to_username[i])
        ranking['rating'].append(int(round(R[R.shape[0] - 1, i])))

    ranking_df = pd.DataFrame(ranking)
    ranking_df = ranking_df.sort_values(by='rating', ascending=False)
    ranking_df = ranking_df.reset_index(drop=True)

    ranking_df.to_csv(file_name)


def get_costs(A, S, P_list):
    return list(map(lambda P: cost(A, S, P), P_list))


def get_win_prediction_accuracies(A, S, P_list):
    return list(map(lambda P: win_prediction_accuracy(A, S, P), P_list))


def show_cost_and_wpa_by_alpha(A, S):
    alphas = np.arange(0.01, 0.4, 0.01)

    P_list = []

    for alpha in alphas:
       P_list.append(compute_P_R(A, S, t, alpha)[0])

    cost_list = get_costs(A, S, P_list)

    plt.plot(alphas, cost_list)

    plt.xlabel('α')
    plt.ylabel('MSE')

    win_prediction_accuracy_fig = plt.figure(2)
    plt.plot(alphas, get_win_prediction_accuracies(A, S, P_list))

    plt.show()


def show_optimal_alpha(A, S, t):
    optimal_alpha = gradient_descent_optimization(A, S, t)


def generate_final_rankings(A, S, t, alpha):
    P, R = compute_P_R(A, S, t, alpha)
    print_final_rankings(R, index_to_username, 'atp_ranking.csv')


A, S, users, games = load_A_S_of_atp_data()

user_to_index, index_to_user, index_to_username = user_index_maps(users)

# show_cost_and_wpa_by_alpha(A, S)
#
# show_optimal_alpha(A, S, t)

generate_final_rankings(A, S, t, alpha)


