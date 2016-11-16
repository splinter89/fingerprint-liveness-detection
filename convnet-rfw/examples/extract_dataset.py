import os
from glob import glob

import optparse
import numpy as np
from scipy import misc

from cnnrandom import BatchExtractor
import cnnrandom.models as cnn_models

DEFAULT_IMG_TYPE = 'jpg'
DEFAULT_IN_SHAPE = (200, 200)
DEFAULT_MODEL = 'fg11_ht_l3_1_description'


def retrieve_fnames(dataset_path, image_type):

    dir_names = []

    for root, subFolders, files in os.walk(dataset_path):
        for file in files:
            if file[-len(image_type):] == image_type:
                dir_names += [root]
                break

    dir_names = sorted(dir_names)
    fnames = []

    for dir_name in dir_names:
        dir_fnames = sorted(glob(os.path.join(dataset_path, dir_name,
                                              '*.' + image_type)))

        fnames += dir_fnames

    return fnames


def load_imgs(fnames, out_shape):

    n_imgs = len(fnames)
    img_set = np.empty((n_imgs,) + out_shape, dtype='float32')

    for i, fname in enumerate(fnames):

        arr = misc.imread(fname, flatten=True)
        arr = misc.imresize(arr, out_shape).astype('float32')

        arr -= arr.min()
        arr /= arr.max()

        img_set[i] = arr

    return img_set


def extract_dataset(dataset_path, output_path, image_type, model_name):

    try:
        model = eval('cnn_models.' + model_name)
    except:
        print 'problem importing model!'
        return

    # -- retrieve the name of the available files in dataset_path
    fnames = retrieve_fnames(dataset_path, image_type)

    # -- initialize extractor
    extractor = BatchExtractor(in_shape=DEFAULT_IN_SHAPE, model=model)

    print 'loading images...'
    imgs = load_imgs(fnames, DEFAULT_IN_SHAPE)

    if len(imgs) > 0:
        print 'extracting features...'
        feat_set = extractor.extract(imgs)

        # -- reshape feature set to save it in 2D
        feat_set.shape = feat_set.shape[0], -1

        print 'saving extracted features and corresponding list of images...'
        np.save(os.path.join(output_path, 'cnnrandom-feat-set.npy'), feat_set)
        np.savetxt(os.path.join(output_path, 'cnnrandom-img-list.txt'),
                   fnames, fmt='%s')
    else:
        print 'no images to be extracted.'

    print 'done!'


def get_optparser():

    usage = "usage: %prog [options] <dataset_path> <output_path>"

    parser = optparse.OptionParser(usage=usage)

    parser.add_option("--image_type", "-t",
                      default=DEFAULT_IMG_TYPE,
                      type="str",
                      metavar="STR",
                      help="[DEFAULT='%default']")

    parser.add_option("--model_name", "-m",
                      default=DEFAULT_MODEL,
                      type="str",
                      metavar="STR",
                      help="[DEFAULT='%default']")

    return parser


def main():
    parser = get_optparser()
    opts, args = parser.parse_args()

    if len(args) != 2:
        parser.print_help()
    else:
        dataset_path = args[0]
        output_path = args[1]
        image_type = opts.image_type
        model_name = opts.model_name

        extract_dataset(dataset_path, output_path, image_type, model_name)

if __name__ == "__main__":
    main()
