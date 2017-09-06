import os, sys, glob
import yaml

class NYTReader:
    ROOT_PATH = "/mnt/data/workspace/nlp/content-dense-detector/data/"

    THRESHOLD = {
            "Business": [0.33, 0.61],
            "Science": [0.23, 0.47],
            "Sports": [0.287, 0.497]
            }

    def __init__(self, genre):
        self._genre = genre
        self.docs = {}
        self.train_ids = []
        self.test_ids = []

        # load docs and train_ids, test_ids
        self.load_data()

    def get_text_and_y(self, mode):
        if mode=="train":
            return self._load_text_and_y(self.train_ids)
        elif mode =="test":
            return self._load_text_and_y(self.test_ids)
        else:
            raise "Error, mode %s is not supported.", mode

    def load_data(self):
        fn_fnames = os.path.join(
                self.ROOT_PATH, "fnames/{0}.ab.fname".format(self._genre))
        fn_scores = os.path.join(
                self.ROOT_PATH, "scores/{0}.ab.score".format(self._genre))
        fn_texts = os.path.join(self.ROOT_PATH, 'texts/{0}.text'.format(self._genre))

        fnames = [ fname.strip() for fname in open(fn_fnames).readlines() ]
        scores = [ float(x.split()[1]) for x in open(fn_scores).readlines() ]
        texts = [ text.strip() for text in open(fn_texts).readlines() ]

        for fname, score, text in zip(fnames, scores, texts):
            fname = fname.strip()
            if score >= self.THRESHOLD[self._genre][1]:
                self.docs[fname] = {"label": 1}
            elif score <= self.THRESHOLD[self._genre][0]:
                self.docs[fname] = {"label": 0}
            else:
                continue
            self.docs[fname]['text'] = text


        fnames = self.docs.keys()
        fnames.sort()
        num = len(fnames)
        th = int(num*0.8)
        self.train_ids = fnames[:th]
        self.test_ids = fnames[th:]

    def _load_text_and_y(self, docids):
        texts = []
        ys = []
        for docid in docids:
            texts.append(self.docs[docid]['text'])
            ys.append([self.docs[docid]['label']])

        return texts, ys

        return texts

if __name__ == '__main__':
    nyt_reader = NYTReader(genre="Sports")
    nyt_reader.get_text_and_y('train')
