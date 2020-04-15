import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='model_path', type=str, default=os.path.join('pretrained', 'model-29'))
parser.add_argument('--text', dest='text', type=str, default=None)
parser.add_argument('--text-file', dest='file', type=str, default=None)
parser.add_argument('--style', dest='style', type=int, default=None)
parser.add_argument('--bias', dest='bias', type=float, default=1.)
parser.add_argument('--force', dest='force', action='store_true', default=False)
parser.add_argument('--animation', dest='animation', action='store_true', default=False)
parser.add_argument('--noinfo', dest='info', action='store_false', default=True)
parser.add_argument('--save', dest='save', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default='./handwritten.pdf')
args = parser.parse_args()

import pickle

import matplotlib
import tensorflow as tf

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

        if args.file:
            text = open(args.file, 'r').read()
        else:
            text = args.text

        if text is not None:
            if len(text) > 50:
                pdf = generate.generate(text.replace('1', 'I'), args, sess, translation, [0, 0, 150])
            else:
                print("Text too short! Atleast write that much by yourself!")
        else:
            print("Please provide either --text or --text-file in arguments")


if __name__ == '__main__':
    main()
