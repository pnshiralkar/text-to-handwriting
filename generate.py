import argparse
import os
import pickle
from collections import namedtuple
from io import BytesIO

import matplotlib
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


matplotlib.use('agg')
import matplotlib.pyplot as plt

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


def sample(e, mu1, mu2, std1, std2, rho):
    cov = np.array([[std1 * std1, std1 * std2 * rho],
                    [std1 * std2 * rho, std2 * std2]])
    mean = np.array([mu1, mu2])

    x, y = np.random.multivariate_normal(mean, cov)
    end = np.random.binomial(1, e)
    # print(np.array([x, y, end]))
    return np.array([x, y, end])


def split_strokes(points):
    points = np.array(points)
    strokes = []
    b = 0
    for e in range(len(points)):
        if points[e, 2] == 1.:
            strokes += [points[b:
                               e + 1, :2].copy()]
            b = e + 1
    return strokes


def cumsum(points):
    sums = np.cumsum(points[:, :2], axis=0)
    return np.concatenate([sums, points[:, 2:]], axis=1)


def sample_text(sess, args_text, translation, bias, style=None):
    fields = ['coordinates', 'sequence', 'bias', 'e', 'pi', 'mu1', 'mu2', 'std1', 'std2',
              'rho', 'window', 'kappa', 'phi', 'finish', 'zero_states']
    vs = namedtuple('Params', fields)(
        *[tf.get_collection(name)[0] for name in fields]
    )

    text = np.array([translation.get(c, 0) for c in args_text])
    coord = np.array([0., 0., 1.])
    coords = [coord]

    # Prime the model with the author style if requested
    prime_len, style_len = 0, 0
    if style is not None:
        # Priming consist of joining to a real pen-position and character sequences the synthetic sequence to generate
        #   and set the synthetic pen-position to a null vector (the positions are sampled from the MDN)
        style_coords, style_text = style
        prime_len = len(style_coords)
        style_len = len(style_text)
        prime_coords = list(style_coords)
        coord = prime_coords[0]  # Set the first pen stroke as the first element to process
        text = np.r_[style_text, text]  # concatenate on 1 axis the prime text + synthesis character sequence
        sequence_prime = np.eye(len(translation), dtype=np.float32)[style_text]
        sequence_prime = np.expand_dims(np.concatenate([sequence_prime, np.zeros((1, len(translation)))]), axis=0)

    sequence = np.eye(len(translation), dtype=np.float32)[text]
    sequence = np.expand_dims(np.concatenate([sequence, np.zeros((1, len(translation)))]), axis=0)

    phi_data, window_data, kappa_data, stroke_data = [], [], [], []
    sess.run(vs.zero_states)
    sequence_len = len(args_text) + style_len
    for s in range(1, 60 * sequence_len + 1):
        is_priming = s < prime_len

        # print('\r[{:5d}] sampling... {}'.format(s, 'priming' if is_priming else 'synthesis'), end='')

        e, pi, mu1, mu2, std1, std2, rho, \
        finish, phi, window, kappa = sess.run([vs.e, vs.pi, vs.mu1, vs.mu2,
                                               vs.std1, vs.std2, vs.rho, vs.finish,
                                               vs.phi, vs.window, vs.kappa],
                                              feed_dict={
                                                  vs.coordinates: coord[None, None, ...],
                                                  vs.sequence: sequence_prime if is_priming else sequence,
                                                  vs.bias: bias
                                              })

        if is_priming:
            # Use the real coordinate if priming
            coord = prime_coords[s]
        else:
            # Synthesis mode
            phi_data += [phi[0, :]]
            window_data += [window[0, :]]
            kappa_data += [kappa[0, :]]
            # ---
            g = np.random.choice(np.arange(pi.shape[1]), p=pi[0])
            coord = sample(e[0, 0], mu1[0, g], mu2[0, g],
                           std1[0, g], std2[0, g], rho[0, g])
            coords += [coord]
            stroke_data += [[mu1[0, g], mu2[0, g], std1[0, g], std2[0, g], rho[0, g], coord[2]]]

            if not args.force and finish[0, 0] > 0.8:
                # print('\nFinished sampling!\n')
                break

    coords = np.array(coords)
    coords[-1, 2] = 1.

    return phi_data, window_data, kappa_data, stroke_data, coords


from PIL import Image


def add_color(color, image_out):
    print("Applying color : ", color)
    img = Image.open(image_out)
    width, height = img.size
    for x in range(width):
        for y in range(height):
            old_color = list(img.getpixel((x, y)))
            if old_color != [255, 255, 255, 255]:
                new_color = [color[x]
                             for x in range(3)]
                img.putpixel((x, y), tuple(new_color))
            else:
                new_color = [255, 255, 255, 0]
                img.putpixel((x, y), tuple(new_color))
    imgout = BytesIO()
    img.save(imgout, 'PNG')
    imgout.seek(0)
    return imgout


##################################################################
#                     The Generator Function                     #
##################################################################

def generate(args_text, args, sess, translation, text_color=[0, 0, 0]):
    style = None
    if args.style is not None:
        style = None
        with open(os.path.join('data', 'styles.pkl'), 'rb') as file:
            styles = pickle.load(file)

        if args.style > len(styles[0]):
            raise ValueError('Requested style is not in style list')

        style = [styles[0][args.style], styles[1][args.style]]

    currentX = 0
    currentY = 0
    currentLen = 0
    line_length = 50
    line_height = -4
    num_lines = len(args_text) // 50
    text_remaining = len(args_text)
    lines_per_page = 28
    curr_page = 1
    cuur_line = 1

    fig, ax = plt.subplots(1, 1)
    plt.figure(num=None, figsize=(115, 5 * min(lines_per_page, text_remaining // line_length + args_text.count('\n'))),
               dpi=35,
               facecolor='w', edgecolor='k')

    print('Writing...')
    for multiline_text in args_text.split(' '):
        for text_without_spaces in multiline_text.split('\n'):
            text = " {} ".format(text_without_spaces)
            phi_data, window_data, kappa_data, stroke_data, coords = sample_text(sess, text, translation, args.bias,
                                                                                 style)

            if currentLen + len(text_without_spaces) > line_length or multiline_text.split('\n').index(
                    text_without_spaces) > 0:
                # print(currentLen)
                currentY += line_height
                currentX = 0
                currentLen = 0
                print('')
                cuur_line += 1

            strokes = np.array(stroke_data)
            epsilon = 1e-8
            strokes[:, :2] = np.cumsum(strokes[:, :2], axis=0)
            minx, maxx = np.min(strokes[:, 0]), np.max(strokes[:, 0])
            miny, maxy = np.min(strokes[:, 1]), np.max(strokes[:, 1])

            for stroke in split_strokes(cumsum(np.array(coords))):
                if np.min(stroke[:, 0]) > maxx - 2 and np.max(stroke[:, 0]) < maxx + 2:
                    continue
                plt.plot(stroke[:, 0] + currentX, -stroke[:, 1] + currentY)
            currentX += maxx - 2
            currentLen += len(text_without_spaces) + 1
            text_remaining -= (len(text_without_spaces) + 1)
            print(text, end=' ', flush=True)
            if cuur_line >= lines_per_page:
                ax.set_aspect('equal')
                plt.axis('off')
                figfile = BytesIO()
                print("\n\nProcessing page No. {}...\nCreating image...".format(curr_page), flush=True)
                plt.savefig(figfile, format='png', bbox_inches='tight')
                figfile.seek(0)  # rewind to beginning of file
                print("Colouring text...", flush=True)
                figfile1 = add_color(text_color, figfile)
                print("Saving image...", flush=True)
                image_out = 'pages/page{}.png'.format(curr_page)
                with open(image_out, 'wb') as fl:
                    for x in figfile1:
                        fl.write(x)
                from PIL import Image
                img = Image.open(image_out)
                img.load()
                img = img.resize((int(img.size[0] * 0.8), int(img.size[1] * 0.804)), Image.ANTIALIAS)
                # background = Image.new("RGB", img.size, (255, 255, 255))
                background = Image.open('blank_page.jpg')
                background.load()
                background.paste(img, mask=img.split()[3], box=(30, 220))  # 3 is the alpha channel
                background.save(image_out.replace('.png', '.jpg'), 'JPEG', quality=100)

                print("\nPage No. {} done!\n\n".format(curr_page), flush=True)

                fig, ax = plt.subplots(1, 1)
                plt.figure(num=None, figsize=(115, 5 * min(lines_per_page, text_remaining // line_length + args_text[
                                                                                                           args_text.index(
                                                                                                               text_without_spaces):].count(
                    '\n'))), dpi=40, facecolor='w',
                           edgecolor='k')
                curr_page += 1
                currentX = 0
                currentY = 0
                currentLen = 0
                cuur_line = 1

    ax.set_aspect('equal')
    plt.axis('off')
    figfile = BytesIO()
    print("\n\nProcessing page No. {}...\nCreating image...".format(curr_page), flush=True)
    plt.savefig(figfile, format='png', bbox_inches='tight')
    figfile.seek(0)  # rewind to beginning of file
    print("Colouring text...", flush=True)
    figfile1 = add_color(text_color, figfile)
    print("Saving image...", flush=True)
    image_out = 'pages/page{}.png'.format(curr_page)
    with open(image_out, 'wb') as fl:
        for x in figfile1:
            fl.write(x)
    from PIL import Image
    img = Image.open(image_out)
    img.load()
    img = img.resize((int(img.size[0] * 0.8), int(img.size[1] * 0.804)), Image.ANTIALIAS)
    # background = Image.new("RGB", img.size, (255, 255, 255))
    background = Image.open('blank_page.jpg')
    background.load()
    background.paste(img, mask=img.split()[3], box=(30, 315))  # 3 is the alpha channel
    background.save(image_out.replace('.png', '.jpg'), 'JPEG', quality=100)

    print("\nPage No. {} done!\n\n".format(curr_page), flush=True)

    # Generate PDF
    print('\nGenerating PDF...', end='')
    from PIL import Image
    img1 = Image.open('pages/page1.jpg')
    im_list = [Image.open('pages/page{}.jpg'.format(i)) for i in range(2, curr_page + 1)]
    img1.save(args.output, "PDF", resolution=100.0, save_all=True, append_images=im_list)
    print("done\n\nSuccessfully generated handwritten pdf from text at :\n{}".format(args.output))
    return args.output
