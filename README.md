# All-Occasion Cards <!-- omit in toc -->

> Generating all-occasion cards for my twitter followers with tensorflow.

🗣 Say hello! [@telepathics](https://twitter.com/telepathics)

Currently, you can only generate Valentines day cards, but I will be adding/updating designs as the year progresses and I come up with new ideas :)

## Installation

Personally, I prefer to use a [Conda](https://formulae.brew.sh/cask/anaconda) env for keeping my python packages tidy, so feel free to check that out.  This is good to keep track of what packages are actually necessary.

`conda env create -f environment.yml`

Otherwise, feel free to install the requirements via pip.

`pip install -r requirements.txt`

You will also need to install enchant's C library for the dictionary, [libenchant](https://pyenchant.github.io/pyenchant/install.html).

*And finally, I would highly advise not using a new macbook with M1 with this project.  Just trust me.*

`python ./main.py`

## Contributing
Please feel free to fork + make pull requests with your own card designs, bug fixes, features, and whatever you think would improve this project!

If you want to support myself or the project, consider [sponsoring my GitHub](https://github.com/sponsors/revertdata)! <3

## Credits

Tutorials referenced:
* [tensorflow - text generation](https://www.tensorflow.org/tutorials/text/text_generation#build_the_model)
* [tensorflow - save and load](https://www.tensorflow.org/tutorials/keras/save_and_load#save_checkpoints_during_training)
* [PIL/pillow basic starting guide](https://code-maven.com/create-images-with-python-pil-pillow)

Valentines Poems scraped from:
* [inspirational-quotes-short-funny-stuff](https://www.inspirational-quotes-short-funny-stuff.com/)
* [poemsource](https://www.poemsource.com/)
* [crafts u print](https://www.craftsuprint.com/)
* [familyfriendpoems](https://www.familyfriendpoems.com/poems/valentine/short/)
* [holidappy](https://holidappy.com)
* [allbestmessages](https://www.allbestmessages.co)
* some user-submitted ones on instagram lmao
* some generated by another ai (thanks for the GPT3 access, eric)