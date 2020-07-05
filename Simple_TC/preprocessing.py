import re

def clean_text(text):
	"""
	Steps:
	1. Remove html tags
	2. Remove punctuation
	3. Lower case everything
	"""

	#remove html tags
	text = re.sub( r'<.*?>','',text)

	#remove punctutations [\], ['] and [""]
	text = re.sub(r'\'', '', text)
	text = re.sub(r'\\', '', text)
	text = re.sub(r'\"', '', text)

	#lower case and strip of additinal spaces at the beginning and end
	text = text.strip().lower()

	#replace punctuation characters with spaces
	filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
	translate_dict = dict((c, " ") for c in filters)
	translate_map = str.maketrans(translate_dict)
	text =  text.translate(translate_map)

	return text


def test(text):
	print(clean_text(text))

#test_text = "<div>This is not a sentence.<\div>"
#test(test_text)