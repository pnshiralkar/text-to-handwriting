import os
import pickle

import bottle
import matplotlib
import tensorflow as tf
import argparse

matplotlib.use('agg')
import generate

parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='model_path', type=str, default=os.path.join('pretrained', 'model-29'))
parser.add_argument('--text', dest='text', type=str, default=None)
parser.add_argument('--style', dest='style', type=int, default=None)
parser.add_argument('--bias', dest='bias', type=float, default=1.)
parser.add_argument('--force', dest='force', action='store_true', default=False)
parser.add_argument('--animation', dest='animation', action='store_true', default=False)
parser.add_argument('--noinfo', dest='info', action='store_false', default=True)
parser.add_argument('--save', dest='save', type=str, default=None)
parser.add_argument('--output', dest='output', type=str, default='./handwritten.pdf')
args = parser.parse_args()

def main():
    with open(os.path.join('data', 'translation.pkl'), 'rb') as file:
        translation = pickle.load(file)
    rev_translation = {v: k for k, v in translation.items()}
    charset = [rev_translation[i] for i in range(len(rev_translation))]
    charset[0] = ''

    config = tf.ConfigProto(
        device_count={'GPU': 0}
    )
    app = bottle.Bottle()

    @app.post("/")
    def home():
        return '''https://github.com/theSage21/handwriting-generation'''

    with tf.Session(config=config) as sess:
        saver = tf.train.import_meta_graph(args.model_path + '.meta')
        saver.restore(sess, args.model_path)

        @app.post("/write")
        def write_post():
            args_text = bottle.request.json['text']
            args.style = bottle.request.json['style']
            args.bias = bottle.request.json['bias']

            pdf = generate(args_text, args, sess, translation, [7, 7, 82])
            return bottle.static_file(pdf, root='./')

        port = os.environ.get("PORT")
        port = port if port else 8000
        app.run(port=port, host='0.0.0.0')


if __name__ == '__main__':
    main()
