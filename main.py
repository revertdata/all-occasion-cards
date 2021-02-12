#!/usr/bin/env python3

# =======================================
# =         VALENTINES DAY CARDS        =
# =   https://twitter.com/telepathics   =
# =======================================

import os
import json

import numpy as np
import pandas as pd

# from twitter import Twitter, OAuth
# from t import ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET

import tensorflow as tf
from tensorflow.python.data import Dataset
from tensorflow.keras.layers.experimental import preprocessing

STARTC='\033[90m'
ENDC='\033[0m'

def print_maryn(text1, text2=''):
	print(STARTC)
	print("\n-----------\n")
	print(ENDC)
	print("maryn: ", text1)
	print(STARTC)
	print(text2)
	print("\n-----------\n")
	print(ENDC)

	return

class Generator(object):
	def __init__(self):
		print(ENDC)
		self.EPOCHS = int(input('\nEPOCH: ') or 0)
		self.steps = int(input('steps: ') or 130)
		print(STARTC)

		self.poems = open('./poems.txt', 'rb').read().decode(encoding="utf-8")
		self.vocab = sorted(set(self.poems))

		self.ids_from_chars = []
		self.chars_from_ids = []
		self.dataset = []
		self.process_text()

		model = TrainingModel(vocab_size=len(self.ids_from_chars.get_vocabulary()))
		self.train(model=model)
		self.write(model=model)

		return

	def randomize_data(self):
		self.poems = self.poems.reindex(np.random.permutation(self.poems.index))

		return

	def text_from_ids(self, ids):
		return tf.strings.reduce_join(self.chars_from_ids(ids), axis=-1)

	def split_input_target(self, sequence):
		input_text = sequence[:-1]
		target_text = sequence[1:]

		return input_text, target_text

	def process_text(self):
		example_texts = ['abcdefg', 'xyz']
		chars = tf.strings.unicode_split(example_texts, input_encoding="UTF-8")

		self.ids_from_chars = preprocessing.StringLookup(vocabulary=list(self.vocab))
		ids = self.ids_from_chars(chars)

		self.chars_from_ids = preprocessing.StringLookup(vocabulary=self.ids_from_chars.get_vocabulary(), invert=True)
		chars = self.chars_from_ids(ids)

		seq_length = 100
		all_ids = self.ids_from_chars(tf.strings.unicode_split(self.poems, 'UTF-8'))

		ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
		sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)
		dataset = sequences.map(self.split_input_target)

		# Batch size
		BATCH_SIZE = 64

		# Buffer size to shuffle the dataset
		# (TF data is designed to work with possibly infinite sequences,
		# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
		# it maintains a buffer in which it shuffles elements).
		BUFFER_SIZE = 10000

		self.dataset = (
				dataset
				.shuffle(BUFFER_SIZE)
				.batch(BATCH_SIZE, drop_remainder=True)
				.prefetch(tf.data.experimental.AUTOTUNE))

		return

	def train(self, model):
		print()

		checkpoint_count = 0
		try:
			checkpoint_dir = './training_checkpoints'
			checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
			files = [os.path.join(checkpoint_dir, file) for file in os.listdir("./training_checkpoints") if (file.lower().endswith('.index'))]
			files = sorted(files,key=os.path.getmtime)
			checkpoint_count = int(''.join(filter(str.isdigit, files[len(files)-1])))

			model.load_weights(checkpoint_prefix.format(epoch=checkpoint_count))
		except:
			pass

		loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
		model.compile(optimizer='adam', loss=loss)
		checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
		model.fit(self.dataset, epochs=self.EPOCHS, initial_epoch=checkpoint_count, callbacks=[checkpoint_callback])

		return

	def write(self, model):
		one_step_model = OneStep(model, self.chars_from_ids, self.ids_from_chars)
		states = None
		next_char = tf.constant(['valentines'])
		result = [next_char]

		for _ in range(self.steps):
			next_char, states = one_step_model.generate_one_step(next_char, states=states)
			result.append(next_char)

		result = tf.strings.join(result)

		print(ENDC)
		print(result[0].numpy().decode('utf-8'), '\n\n' + '_'*80)
		print(STARTC)

class TrainingModel(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim=256, rnn_units=1024):
		super().__init__(self)
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = tf.keras.layers.GRU(rnn_units, return_sequences=True, return_state=True)
		self.dense = tf.keras.layers.Dense(vocab_size)

	def call(self, inputs, states=None, return_state=False, training=False):
		x = inputs
		x = self.embedding(x, training=training)
		if states is None:
			states = self.gru.get_initial_state(x)
		x, states = self.gru(x, initial_state=states, training=training) # breaks here
		x = self.dense(x, training=training)

		if return_state:
			return x, states
		else:
			return x

class OneStep(tf.keras.Model):
	def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
		super().__init__()
		self.temperature=temperature
		self.model = model
		self.chars_from_ids = chars_from_ids
		self.ids_from_chars = ids_from_chars

		# Create a mask to prevent "" or "[UNK]" from being generated.
		skip_ids = self.ids_from_chars(['','[UNK]'])[:, None]
		sparse_mask = tf.SparseTensor(
				# Put a -inf at each bad index.
				values=[-float('inf')]*len(skip_ids),
				indices = skip_ids,
				# Match the shape to the vocabulary
				dense_shape=[len(ids_from_chars.get_vocabulary())])
		self.prediction_mask = tf.sparse.to_dense(sparse_mask)

	@tf.function
	def generate_one_step(self, inputs, states=None):
		# Convert strings to token IDs.
		input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
		input_ids = self.ids_from_chars(input_chars).to_tensor()

		# Run the model.
		# predicted_logits.shape is [batch, char, next_char_logits]
		predicted_logits, states =  self.model(inputs=input_ids, states=states, return_state=True)
		# Only use the last prediction.
		predicted_logits = predicted_logits[:, -1, :]
		predicted_logits = predicted_logits/self.temperature
		# Apply the prediction mask: prevent "" or "[UNK]" from being generated.
		predicted_logits = predicted_logits + self.prediction_mask

		# Sample the output logits to generate token IDs.
		predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
		predicted_ids = tf.squeeze(predicted_ids, axis=-1)

		# Convert from token ids to characters
		predicted_chars = self.chars_from_ids(predicted_ids)

		# Return the characters and model state.
		return predicted_chars, states

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
	print(STARTC)
	Generator()

	return

	# running = True
	# while running:
		# _tw = Valentines()
		# _tw.prepare_valentines()

		# answ = input('\nready to send? (Y/N): ')
		# if answ.lower() in ('no', 'n', 'exit', 'e', 'quit', 'q'):
		# 	running = False
		# else:
		# 	# _tw.send_valentines()
		# 	pass

if __name__ == '__main__':
	main()
