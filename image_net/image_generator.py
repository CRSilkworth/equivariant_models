# import .label_maps as lm
import numpy as np
import hashlib
import os
import urllib2
import cv2
import random
import socket
import httplib
import glob
import pprint


class ImageGenerator:
    """
    Image generator to be used in a training/testing pipeline. Downloads images or reads them from disk.
    """
    def __init__(self, img_dir='./images/', img_label_file_name=None, allowed_extensions=['JPEG'], mean_brg=np.array([104., 117., 124.], dtype=np.float32), size=(227, 227), label_file_name='all_labels.txt'):
        self.img_label_file_name = img_label_file_name
        self.img_dir = img_dir
        self.allowed_extensions = allowed_extensions
        self.mean_brg = mean_brg
        self.size = size
        self.img_data = []
        # Create the dictionary which maps the coded label to the label number
        # and the english label.
        self.label_map = {}
        with open(label_file_name, 'r') as label_file:
            for line in label_file:
                coded_label, label_num, label_english = line.split()

                self.label_map[int(label_num)] = label_english

        # Pull out the labels from the label file.
        if img_label_file_name is not None:
            with open(img_label_file_name, 'r') as img_label_file:
                img_labels = [int(l_num) for l_num in img_label_file]

        # Gather all the image paths. It will be assumed their sorted names
        # correspond to the same order as the labels in img_label_file
        img_paths = []
        for ext in allowed_extensions:
            img_paths.extend(glob.glob(os.path.join(img_dir, '*.' + ext)))
        img_paths.sort()

        # Create the full list of image information.
        for line_num, img_path in enumerate(img_paths):
            label_num = None
            label_english = None
            if img_label_file_name is not None:
                label_num = img_labels[line_num]
                label_english = self.label_map[label_num]
            self.img_data.append((img_path, label_num, label_english))

    def __iter__(self):
        for img_path, label_num, label_english in self.img_data:
            img = cv2.imread(img_path)
            img = self._alter_img(img)

            yield img, label_num, label_english

    def _alter_img(self, img):
        """
        Resize, offset mean and then reshape the image.
        """
        img = cv2.resize(img.astype(np.float32), self.size)
        img -= self.mean_brg
        img = img.reshape((1, self.size[0], self.size[1], 3))
        return img

    def _obtain_image(self, url, img_file_name, line_num, lines_to_skip_file):
        """
        Download the image if it's not already found in the img_dir.
        """
        if not os.path.isfile(img_file_name):
            # Try and download the image. Add to lines_to_skip if url
            # doesn't exist or it you get a timeout
            try:
                img_file = urllib2.urlopen(url, timeout=5)
            except (urllib2.HTTPError, httplib.BadStatusLine):
                print url, "got 404, skipping..."
                self._add_line_to_skip(line_num, lines_to_skip_file)
                return False
            except (urllib2.URLError, socket.timeout):
                print url, "got timeout error, skipping..."
                self._add_line_to_skip(line_num, lines_to_skip_file)
                return False

            # Save the new file to disk.
            with open(img_file_name, 'w') as new_img_file:
                new_img_file.write(img_file.read())

        return True

    def shuffle(self):
        random.shuffle(self.img_data)
