#!/usr/bin/env python3

# =======================================
# =         VALENTINES DAY CARDS        =
# =   https://twitter.com/telepathics   =
# =======================================

import json
import numpy as np
import pandas as pd

# from twitter import Twitter, OAuth
# from t import ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET

# import tensorflow as tf
# from tensorflow.python.data import Dataset

STARTC='\033[90m'
ENDC='\033[0m'

class Generator(object):
	def __init__(self):
		self.poems = pd.read_json('./poems.json')
		self.randomize_data()

		self.feature = self.poems[["poem"]]
		# self.feature_columns = [tf.feature_column.category_column("poem")]


		print(STARTC)
		print("\n-----------\n")
		print('Loaded poem data.')
		print(self.poems.describe())
		print(self.feature)
		print("\n-----------\n")
		print(ENDC)
		return

	def randomize_data(self):
		self.poems = self.poems.reindex(np.random.permutation(self.poems.index))

		return

	def train(self, features, targets, batch_size=1, shuffle=True, num_epochs=None):

		return



class Valentines(object):
	def __init__(self):
		# self.t = Twitter(auth=OAuth(ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET))
		# print(self.t.auth)
		return

	def fetch_recent_tweets(self, screen_name):
		# TODO
		# fetch tweets
		return

	def generate_card(self, screen_name):
		self.fetch_recent_tweets(screen_name)
		# TODO
		# create card ???
		return

	def prepare_valentines(self):
		with open('users.json', 'r') as users:
			userslist = json.load(users)
			for user in userslist:
				self.generate_card(user)
				# TODO
				# save card to folder
			print('cards have been generated.')
		return

	def send_valentines(self):
		# TODO send DM with image attached
		with open('users.json', 'r') as users:
			userslist = json.load(users)
			for user in userslist:
				print('sending to ' + user)
				# t.direct_messages.events.new(
				# 	_json={
				# 		"event": {
				# 			"type": "message_create",
				# 			"message_create": {
				# 				"target": {
				# 						"recipient_id": t.users.show(screen_name=uscreen_name)["id"]},
				# 				"message_data": {
				# 						"text": DM_MSG}
				# 			}
				# 		}
				# 	}
				# )
		return

def main():
	running = True

	while running:
		gen = Generator()

		# _tw = Valentines()
		# _tw.prepare_valentines()

		answ = input('ready to send? (Y/N): ')
		if answ.lower() in ('no', 'n', 'exit', 'e', 'quit', 'q'):
			running = False
		else:
			_tw.send_valentines()

if __name__ == '__main__':
	main()
