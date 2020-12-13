import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='model_path', type=str, default=os.path.join('pretrained', 'model-29'),
                    help='(optional) DL model to use')
parser.add_argument('--text', dest='text', type=str, help='Text to write')
parser.add_argument('--text-file', dest='file', type=str, default=None, help='Path to the input text file')
parser.add_argument('--style', dest='style', type=int, default=0, help='Style of handwriting (1 to 7)')
parser.add_argument('--bias', dest='bias', type=float, default=0.9,
                    help='Bias in handwriting. More bias is more unclear handwriting (0.00 to 1.00)')
parser.add_argument('--force', dest='force', action='store_true', default=False)
parser.add_argument('--color', dest='color_text', type=str, default='0,0,150',
                    help='Color of handwriting in RGB format')
parser.add_argument('--output', dest='output', type=str, default='./handwritten.pdf',
                    help='Output PDF file path and name')
args = parser.parse_args()

if args.file:
    text = open(args.file, 'r').read()
else:
    text = args.text

if text is not None:
    if len(text) > 50:
        pass
    else:
        print("Text too short!")
        exit()
else:
    print("Please provide either --text or --text-file in arguments")
    exit()

import pickle

import matplotlib
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


import generate

matplotlib.use('agg')


def main():
    with open(os.path.join('data', 'translation.pkl'), 'rb') as file:
        translation = pickle.load(file)
    rev_translation = {v: k for k, v in translation.items()}
    charset = [rev_translation[i] for i in range(len(rev_translation))]
    charset[0] = ''

    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )

    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph(args.model_path + '.meta')
        saver.restore(sess, args.model_path)

        print("\n\nInitialization Complete!\n\n\n\n")

        color = [int(i) for i in args.color_text.replace(' ', '').split(',')]
        pdf = generate.generate(text.replace('1', 'I'), args, sess, translation, color[:3])


if __name__ == '__main__':
    main()
