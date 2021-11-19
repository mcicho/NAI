"""
Małgorzata Cichowlas s16512
Recommendation movies app:
Input: User ('name and surname')
Output: 5 recommended movies and 5 not recommended movies
Problem: Chceck dataset of all users, then match similar user to input user. Chceck movies of similar user and recommend best movies. 
Code based on program from lecture NAI and the "collaborative_filtering.py" file.
"""

import argparse
import json
import numpy as np
import csv

from compute_scores import euclidean_score

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find users who are similar to the input user')
    parser.add_argument('--user', dest='user', required=True,
            help='Input user')
    return parser

# Finds users in the dataset that are similar to the input user 
def find_similar_users(dataset, user, num_users):
    if user not in dataset:
        raise TypeError('Cannot find ' + user + ' in the dataset')

    # Compute Pearson score between input user 
    # and all the users in the dataset
    scores = np.array([[x, euclidean_score(dataset, user, 
            x)] for x in dataset if x != user])

    # Sort the scores in decreasing order
    scores_sorted = np.argsort(scores[:, 1])[::-1]

    # Extract the top 'num_users' scores
    top_users = scores_sorted[:num_users] 

    return scores[top_users] 



if __name__=='__main__':
    args = build_arg_parser().parse_args()
    user = args.user

    # Read dataset with users and movies
    ratings_file = 'newratings.json'
    with open(ratings_file, 'r', encoding="utf8") as f:
        data = json.loads(f.read())

    # Find 5 similar users
    similar_users = find_similar_users(data, user, 5)
    movies = data[similar_users[0][0]]
    user_movies = data[user]
    dictionary = {}

    for title in movies.keys():
        if title not in user_movies.keys():
            dictionary[title] = movies[title]

    # Sort dictionary by values
    dictionary = {k: v for k, v in sorted(dictionary.items(), key=lambda item: item[1], reverse=True)}


    print('For user: ' + user)
    recommended_movies = list(dictionary.keys())[:5]
    print('\nRecommended movies: ')
    for i in recommended_movies:
        print(i)


    not_recommended_movies = list(reversed(dictionary.keys()))[:5]
    print('\nNot recommended movies: ')
    for j in not_recommended_movies:
        print(j)

       
