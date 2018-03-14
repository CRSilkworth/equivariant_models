# import .label_maps as lm
import numpy as np
import hashlib
import os
import urllib2
import cv2
import random
import socket


class ImageGenerator:
    """
    Image generator to be used in a training/testing pipeline. Downloads images or reads them from disk.
    """
    def __init__(self, url_file_name, img_dir='./images/', allowed_extensions=['jpg'], mean_brg=np.array([104., 117., 124.], dtype=np.float32), size=(227, 227), label_file_name='all_labels.txt', lines_to_skip_file_name=None):

        self.url_file_name = url_file_name
        self.img_dir = img_dir
        self.allowed_extensions = allowed_extensions
        self.mean_brg = mean_brg
        self.size = size
        self.reverse_url_hash = {}

        # If a file of lines in the url file to skip to exist then load it up.
        # Used to avoid cycling through all the broken images.
        self.lines_to_skip_file_name = lines_to_skip_file_name
        self.lines_to_skip = set()
        if lines_to_skip_file_name is not None and os.path.isfile(self.lines_to_skip_file_name):
            with open(self.lines_to_skip_file_name) as lines_to_skip_file:
                for line in lines_to_skip_file:
                    self.lines_to_skip.add(int(line))

        # Create the dictionary which maps the coded label to the label number
        # and the english label.
        label_map = {}
        with open(label_file_name, 'r') as label_file:
            for line in label_file:
                coded_label, label_num, label_english = line.split()
                label_map[coded_label] = label_num, label_english

        with open(self.url_file_name) as url_file:
            self.urls = []
            for line_num, line in enumerate(url_file):
                if line_num in self.lines_to_skip:
                    continue

                # Get the label, the image url and the image type from each line
                # of the url file
                coded_label = line.split()[0].split('_')[0]
                url = line.split()[1]
                ext = url.split('.')[-1]

                # Only take allowed types
                if ext not in self.allowed_extensions:
                    continue

                if coded_label not in label_map:
                    continue

                label_num, label_english = label_map[coded_label]

                self.urls.append((line_num, url, label_num, label_english))

    def __iter__(self):
        lines_to_skip_file = None
        if self.lines_to_skip_file_name is not None:
            lines_to_skip_file = open(self.lines_to_skip_file_name, 'a')

        for line_num, url, label_num, label_english in self.urls:
            if line_num in self.lines_to_skip:
                continue

            # Create a hash for the url to be used as the file name. Keep
            # information about the which hash is which url.
            hash_m = hashlib.md5()
            hash_m.update(url)
            url_hash = hash_m.hexdigest()
            self.reverse_url_hash[url_hash] = url
            ext = url.split('.')[-1]

            # Download the image if it hasn't already been downloaded.
            img_file_name = self.img_dir + url_hash + '.' + ext
            if not self._obtain_image(url, img_file_name, line_num, lines_to_skip_file):
                continue

            img = cv2.imread(img_file_name)
            if img is None:
                print url, "failed to read. skipping..."
                self._add_line_to_skip(line_num, lines_to_skip_file)
                continue

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
            except urllib2.HTTPError:
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
        random.shuffle(self.urls)

    def _add_line_to_skip(self, line_num, lines_to_skip_file):
        if self.lines_to_skip_file_name is None:
            return
        lines_to_skip_file.write(str(line_num) + '\n')
