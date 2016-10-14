from sqlite3 import dbapi2 as sqlite
import pickle
from numpy.ma import sqrt
from PIL import Image
from functools import cmp_to_key
from numpy import *
from pylab import *

from n1_local_image_descriptors import sift
from n2_image_to_image_mappings import homography
import imtools
from n6_searching_images.vocabulary import Vocabulary


class Indexer(object):
    def __init__(self, db, voc):
        """ Initialize with the name of the database and a vocabulary object. """

        self.con = sqlite.connect(db)
        self.voc = voc

    def __del__(self):
        self.con.close()

    def db_commit(self):
        self.con.commit()

    def create_tables(self):
        """ Create the database tables. """

        self.con.execute('CREATE TABLE imlist(filename)')   # contains the filenames of all indexed images
        self.con.execute('CREATE TABLE imwords(imid,wordid,vocname)')
            # contains a word index of the words, which vocabulary was used, and which images the words appear in
        self.con.execute('CREATE TABLE imhistograms(imid,histogram,vocname)')
            # contains the full word histograms for each image
        self.con.execute('CREATE INDEX im_idx ON imlist(filename)')     # create index to make searching faster
        self.con.execute('CREATE INDEX wordid_idx ON imwords(wordid)')
        self.con.execute('CREATE INDEX imid_idx ON imwords(imid)')
        self.con.execute('CREATE INDEX imidhist_idx ON imhistograms(imid)')
        self.db_commit()

    def add_to_index(self, imname, descr):
        """ Take an image with feature descriptors, project on vocabulary and add to database. """

        if self.is_indexed(imname): return
        print('indexing', imname)

        # get the imid
        imid = self.get_id(imname)

        # get the words
        imwords = self.voc.project(descr)
        nbr_words = imwords.shape[0]

        # link each word to image
        for i in range(nbr_words):
            word = imwords[i]
            # wordid is the word number itself
            self.con.execute("insert into imwords(imid,wordid,vocname) values(?, ?, ?)", (imid,word,self.voc.name))

        # store word histogram for image
        # use pickle to encode NumPy arrays as strings
        self.con.execute("insert into imhistograms(imid,histogram,vocname) values(?, ?, ?)",
                         (imid,pickle.dumps(imwords),self.voc.name))

    def is_indexed(self, imname):
        """ Returns True if imname has been indexed. """

        im = self.con.execute("select rowid from imlist where filename = '%s'" % imname).fetchone()
        return im != None

    def get_id(self, imname):
        """ Get an entry id and add if not present. """

        cur = self.con.execute("select rowid from imlist where filename='%s'" % imname)
        res = cur.fetchone()
        if res == None:
            cur = self.con.execute("insert into imlist(filename) values ('%s')" % imname)
            return cur.lastrowid
        else:
            return res[0]


def index_images():
    from n1_local_image_descriptors import sift
    import imtools

    imlist = imtools.get_imlist('images')
    nbr_images = len(imlist)
    featlist = [imlist[i][:-3] + 'sift' for i in range(nbr_images)]

    # load vocabulary
    with open('vocabulary.pkl', 'rb') as f:
        voc = pickle.load(f)

    # create indexer
    indx = Indexer('test.db', voc)
    indx.create_tables()

    # go through all images, project features on vocabulary and insert
    for i in range(nbr_images):
        locs, descr = sift.read_features_from_file(featlist[i])
        indx.add_to_index(imlist[i], descr)

    # commit to database
    indx.db_commit()


class Searcher(object):
    def __init__(self, db, voc):
        """ Initialize with the name of the database. """
        self.con = sqlite.connect(db)
        self.voc = voc

    def __del__(self):
        self.con.close()

    def candidates_from_word(self, imword):
        """ Get list of images containing imword. """

        im_ids = self.con.execute("select distinct imid from imwords where wordid=%d" % imword).fetchall()
        return [i[0] for i in im_ids]

    def candidates_from_histogram(self, imwords):
        """ Get list of images with similar words. """

        # get the word ids
        words = imwords.nonzero()[0]

        # find candidates
        candidates = []
        for word in words:
            c = self.candidates_from_word(word)
            candidates += c

        # take all unique words and reverse sort on occurrence
        # so that the first candidate has most matches to the imwords
        tmp = [(w, candidates.count(w)) for w in set(candidates)]
        tmp.sort(key=cmp_to_key(lambda x, y: (x[1]>y[1])-(x[1]<y[1])))
            # the python2 version is tmp.sort(cmp=lambda x,y:cmp(x[1],y[1]))
        tmp.reverse()

        # return sorted list, best matches first
        return [w[0] for w in tmp]

    def get_imhistogram(self, imname):
        """ Return the word histogram for an image. """

        im_id = self.con.execute("select rowid from imlist where filename='%s'" % imname).fetchone()
        s = self.con.execute("select histogram from imhistograms where rowid='%d'" % im_id).fetchone()

        # use pickle to decode NumPy arrays from string
        return pickle.loads(s[0])

    def query(self, imname):
        """ Find a list of matching images for imname"""

        h = self.get_imhistogram(imname)
        candidates = self.candidates_from_histogram(h)

        matchscores = []
        for imid in candidates:
            # get the name
            cand_name = self.con.execute("select filename from imlist where rowid=%d" % imid).fetchone()
            cand_h = self.get_imhistogram(cand_name)
            cand_dist = sqrt(sum((h - cand_h) ** 2))  # use L2 distance
            matchscores.append((cand_dist, imid))

        # return a sorted list of distances and database ids
        matchscores.sort()
        return matchscores

    def get_filename(self, imid):
        """ Return the filename for an image id"""

        s = self.con.execute("select filename from imlist where rowid='%d'" % imid).fetchone()
        return s[0]


def compute_ukbench_score(src, imlist):
    """ Returns the average number of correct
        images on the top four results of queries."""

    nbr_images = len(imlist)
    pos = zeros((nbr_images, 4))
    # get first four results for each image
    for i in range(nbr_images):
        pos[i] = [w[1] - 1 for w in src.query(imlist[i])[:4]]

    # compute score and return average
    score = array([(pos[i] // 4) == (i // 4) for i in range(nbr_images)]) * 1.0
    return sum(score) / (nbr_images)


def plot_results(src,res):
    """ Show images in result list 'res'."""

    figure()
    nbr_results = len(res)
    for i in range(nbr_results):
        imname = src.get_filename(res[i])
        subplot(1, nbr_results, i + 1)
        imshow(array(Image.open(imname)))
        axis('off')


def example_search_image():
    imlist = imtools.get_imlist('images')
    nbr_images = len(imlist)
    featlist = [imlist[i][:-3] + 'sift' for i in range(nbr_images)]

    # load vocabulary
    with open('vocabulary.pkl', 'rb') as f:
        voc = pickle.load(f)
    src = Searcher('test.db', voc)

    print('try a query...')
    nbr_results = 6
    res = [w[1] for w in src.query(imlist[0])[:nbr_results]]
    plot_results(src, res)
    show()


def example_search_with_geometric_result_ranking():
    # load image list and vocabulary
    with open('ukbench_imlist.pkl', 'rb') as f:
        imlist = pickle.load(f)
        featlist = pickle.load(f)

    nbr_images = len(imlist)

    with open('vocabulary.pkl', 'rb') as f:
        voc = pickle.load(f)

    src = Searcher('test.db', voc)

    # index of query image and number of results to return
    q_ind = 1
    nbr_results = 20

    # regular query
    res_reg = [w[1] for w in src.query(imlist[q_ind])[:nbr_results]]
    print('top matches (regular):', res_reg)

    # load image features for query image
    q_locs, q_descr = sift.read_features_from_file(featlist[q_ind])
    fp = homography.make_homog(q_locs[:, :2].T)

    # RANSAC model for homography fitting
    model = homography.RansacModel()

    rank = {}
    # load image features for result
    for ndx in res_reg[1:]:
        locs, descr = sift.read_features_from_file(featlist[ndx])

        # get matches
        matches = sift.match(q_descr, descr)
        ind = matches.nonzero()[0]
        ind2 = matches[ind]
        tp = homography.make_homog(locs[:, :2].T)

        # compute homography, count inliers. if not enough matches return empty list
        try:
            H, inliers = homography.H_from_ransac(fp[:, ind], tp[:, ind2], model, match_theshold=4)
        except:
            inliers = []

        # store inlier count
        rank[ndx] = len(inliers)

    # sort dictionary to get the most inliers first
    sorted_rank = sorted(rank.items(), key=lambda t: t[1], reverse=True)
    res_geom = [res_reg[0]] + [s[0] for s in sorted_rank]
    print('top matches (homography):', res_geom)

    # plot the top results
    plot_results(src, res_reg[:8])
    plot_results(src, res_geom[:8])
    show()


# index_images()
# example_search_image()
# example_search_with_geometric_result_ranking()
