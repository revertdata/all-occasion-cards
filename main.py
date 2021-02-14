#!/usr/bin/env python3

# =======================================
# =           CARD GENERATORS           =
# =   https://twitter.com/telepathics   =
# =======================================

import os
import json

import numpy as np
import pandas as pd
import urllib.request
import enchant
import random

from twitter import Twitter, OAuth
from t import ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET

import tensorflow as tf
from tensorflow.python.data import Dataset
from tensorflow.keras.layers.experimental import preprocessing

from PIL import Image, ImageDraw, ImageFont, ImageFilter

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

def answ_yes(answ):
	if answ.lower() in ('no', 'n', 'exit', 'e', 'quit', 'q'):
		return False

	return True

class Generator(object):
	def __init__(self):
		print(ENDC)
		self.EPOCHS = int(input('\nEPOCH: ') or 0)
		self.steps = int(input('steps: ') or 130)
		print(STARTC)

		self.poems = open('./assets/poems.txt', 'rb').read().decode(encoding="utf-8")

		# add insta poems
		with open('./assets/insta_poems.txt', 'r') as instapoems:
			for line in instapoems:
				if '--@ ' not in line:
					self.poems += line

		self.vocab = sorted(set(self.poems))

		self.ids_from_chars = []
		self.chars_from_ids = []
		self.dataset = []
		self.process_text()

		self.model = TrainingModel(vocab_size=len(self.ids_from_chars.get_vocabulary()))
		self.train()

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
		self.ids_from_chars = preprocessing.StringLookup(vocabulary=list(self.vocab))
		self.chars_from_ids = preprocessing.StringLookup(vocabulary=self.ids_from_chars.get_vocabulary(), invert=True)

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

	def train(self):
		print()

		checkpoint_count = 0
		try:
			checkpoint_dir = './assets/util/training_checkpoints'
			checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
			files = [os.path.join(checkpoint_dir, file) for file in os.listdir("./assets/util/training_checkpoints") if (file.lower().endswith('.index'))]
			files = sorted(files,key=os.path.getmtime)
			checkpoint_count = int(''.join(filter(str.isdigit, files[len(files)-1])))

			self.model.load_weights(checkpoint_prefix.format(epoch=checkpoint_count)).expect_partial()
		except:
			pass

		loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
		adam = tf.keras.optimizers.Adam(
			learning_rate=tf.Variable(0.001),
			beta_1=tf.Variable(0.9),
			beta_2=tf.Variable(0.999),
			epsilon=tf.Variable(1e-7),
		)
		adam.iterations
		adam.decay = tf.Variable(0.0)  # Adam.__init__ assumes ``decay`` is a float object, so this needs to be converted to tf.Variable **after** __init__ method.
		self.model.compile(optimizer=adam, loss=loss)
		checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)
		self.model.fit(self.dataset, epochs=self.EPOCHS, initial_epoch=checkpoint_count, callbacks=[checkpoint_callback])

		return

	def write(self, constant='friend'):
		one_step_model = OneStep(self.model, self.chars_from_ids, self.ids_from_chars)
		states = None
		next_char = tf.constant([constant])
		result = [next_char]

		for _ in range(self.steps):
			next_char, states = one_step_model.generate_one_step(next_char, states=states)
			result.append(next_char)

		result = tf.strings.join(result)
		text = result[0].numpy().decode('utf-8')

		arr = text.split()
		ench_dict = enchant.Dict("en_US")
		for i in range(len(arr) - 1):
			if arr[i] not in constant:
				try:
					word = ''.join(e for e in arr[i] if e.isalnum())
					if not ench_dict.check(word):
						suggestions = ench_dict.suggest(word)
						closest = suggestions[random.randint(0, len(suggestions) - 1)].lower()
						text = text.replace(arr[i], closest)
				except:
					pass

		return text

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

class CardDesigns(object):
	def __init__(self, _gen, occasion):
		self._gen = _gen
		self.occasion = ''
		return

	def draw(self, screen_name):
		if self.occasion == 'vday':
			self.valentines(screen_name)
		else:
			self.valentines(screen_name) # default

		return

	def valentines(self, screen_name):
		print('drawing a valentine for {screen_name}...'.format(screen_name=screen_name))
		text = self._gen.write(screen_name + ',\n\n')
		bg_pink = (255, 158, 167, 255)

		img = Image.new('RGBA', (500, 500), color = bg_pink)
		d = ImageDraw.Draw(img)

		telebotics = Image.open('./assets/maryn.png', 'r')
		telebotics.convert("RGBA")
		img.paste(telebotics, (130, 50), telebotics)

		hearts = Image.open('./assets/hearts.png', 'r')
		hearts.convert("RGBA")
		img.paste(hearts, (220, 50), hearts)

		valentine = Image.open('./assets/twitter/{screen_name}-pfp.png'.format(screen_name=screen_name), 'r')
		img.paste(valentine, (300, 50))

		fnt = ImageFont.truetype('./assets/VT323/VT323-Regular.ttf', 22)
		d.multiline_text((30,167), str(text) + '\n\n\nlove,\ntelebotics', font=fnt, fill=(10,10,10))

		img.save('./assets/cards/{screen_name}.png'.format(screen_name=screen_name))

		return


class Cards(object):
	def __init__(self, _gen, occasion):
		self.t = Twitter(auth=OAuth(ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET))
		self.t_upload = Twitter(domain="upload.twitter.com", auth=OAuth(ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET, CONSUMER_KEY, CONSUMER_SECRET))
		self._gen = _gen
		self.design = CardDesigns(_gen, occasion)

		return

	def fetch_twitter_data(self, screen_name):
		user_pfp_png = "./assets/twitter/{screen_name}-pfp.png".format(screen_name=screen_name)
		if not os.path.exists(user_pfp_png):
			user = self.t.users.show(screen_name=screen_name)
			urllib.request.urlretrieve(user['profile_image_url_https'].replace('normal','bigger'), user_pfp_png)

		return

	def generate_card(self, screen_name):
		self.fetch_twitter_data(screen_name)
		self.design.draw(screen_name)

		return

	def prepare_cards(self):
		print()
		with open('./assets/twitter/users.json', 'r') as screen_names:
			userslist = json.load(screen_names)
			for user in userslist:
				self.generate_card(user.replace('@', ''))
			print('\ncards have been generated.')

		return

	def upload_image(self, file_loc):
		print('\nuploading {image} to twitter...'.format(image=file_loc).replace('./assets/cards/', ''))

		with open(file_loc, 'rb') as imagefile:
			imagedata = imagefile.read()
		media_id = self.t_upload.media.upload(media = imagedata)['media_id_string']

		return media_id

	def send_card_dm(self, screen_name, media_id):
		print('sending to {screen_name}'.format(screen_name=screen_name))
		self.t.direct_messages.events.new(
			_json={
				"event": {
					"type": "message_create",
					"message_create": {
						"target": {
								"recipient_id": self.t.users.show(screen_name=screen_name)["id"]},
						"message_data": {
								"text": "happy valentines day :3",
								"attachment": {
									"type": "media",
									"media": {
										"id": media_id
									}
								}
						}
					}
				}
			}
		)

		return

	def send_cards(self):
		with open('./assets/twitter/users.json', 'r') as screen_names:
			userslist = json.load(screen_names)
			for user in userslist:
				screen_name = user.replace('@', '')
				file_loc = './assets/cards/{screen_name}.png'.format(screen_name=screen_name)

				media_id = self.upload_image(file_loc)
				self.send_card_dm(screen_name, media_id)

		return

def main():
	print(STARTC)
	_gen = Generator()

	print(ENDC)
	occasion = input('whats the occasion? (vday): ' or 'vday')
	answ = input('generate new cards? (Y/N): ')
	print(STARTC)
	if answ_yes(answ):
		_cards = Cards(_gen, occasion)
		_cards.prepare_cards()

	print(ENDC)
	answ = input('ready to send? (Y/N): ')
	print(STARTC)
	if answ_yes(answ):
		try:
			_cards.send_cards()
		except:
			_cards = Cards(_gen, occasion)
			_cards.send_cards()

	return

if __name__ == '__main__':
	main()
