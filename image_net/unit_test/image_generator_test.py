import unittest
import tempfile
import shutil
import image_net.image_generator as ig
import numpy as np

class TestImageGenerator(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.mkdtemp()
        url_file_text = ("""n02108915_5684	http://farm4.static.flickr.com/3218/2654308303_760cd2f9f5.jpg
n02259212_63627 http://farm3.static.flickr.com/2433/3590414981_4eb91f8f8a.jpg
n00007846_64104 http://farm3.static.flickr.com/2305/2346380437_eee92d1398.jpg
n01622779_40140 http://farm3.static.flickr.com/2513/3689743433_e2d772e538.jpg
n02859443_40325 http://farm4.static.flickr.com/3109/2751578066_06f31cd57f.jpg
n02640242_40606 http://farm1.static.flickr.com/94/265820989_8415c3a4bb.jpg
n02086079_2890  http://farm4.static.flickr.com/3053/3046324514_29570a809b.jpg
n00015388_2940  http://farm4.static.flickr.com/3069/2677556348_6e1e9ab18f.jpg
n02108915_3026  http://farm4.static.flickr.com/3186/2810074087_980989fdc5.jpg
n01990800_3044  http://www.thennattravel.com/img/galeria/img-91.jpg
""")

        self.url_file_name = self.tempdir + 'url_file.txt'
        with open(self.url_file_name, 'w') as url_file:
            url_file.write(url_file_text)

    def tearDown(self):
        shutil.rmtree(self.tempdir)

    def test_init(self):
        img_gen = ig.ImageGenerator(self.url_file_name, label_file_name='../all_labels.txt')

        self.assertEquals(
            [(0, 'http://farm4.static.flickr.com/3218/2654308303_760cd2f9f5.jpg', '82', 'French_bulldog'), (1, 'http://farm3.static.flickr.com/2433/3590414981_4eb91f8f8a.jpg', '637', 'leafhopper'), (3, 'http://farm3.static.flickr.com/2513/3689743433_e2d772e538.jpg', '400', 'great_grey_owl'), (4, 'http://farm4.static.flickr.com/3109/2751578066_06f31cd57f.jpg', '689', 'boathouse'), (5, 'http://farm1.static.flickr.com/94/265820989_8415c3a4bb.jpg', '456', 'sturgeon'), (6, 'http://farm4.static.flickr.com/3053/3046324514_29570a809b.jpg', '69', 'Pekinese'), (8, 'http://farm4.static.flickr.com/3186/2810074087_980989fdc5.jpg', '82', 'French_bulldog'), (9, 'http://www.thennattravel.com/img/galeria/img-91.jpg', '612', 'isopod')],
            img_gen.urls
        )

    def test_iter(self):
        img_gen = ig.ImageGenerator(self.url_file_name, label_file_name='../all_labels.txt')

        mean = np.mean([img for img, _, __ in img_gen])

        self.assertTrue(
            abs(mean - 27.87963) < 0.001
        )


if __name__ == "__main__":
    unittest.main()
