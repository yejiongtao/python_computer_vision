from sqlite3 import dbapi2 as sqlite
import pickle

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

        self.con.execute('CREATE TABLE imlist(filename)')
        self.con.execute('CREATE TABLE imwords(imid,wordid,vocname)')
        self.con.execute('CREATE TABLE imhistograms(imid,histogram,vocname)')
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
    # import sys
    # sys.path.insert(0,'..')
    # sys.path.insert(1,'../n1_local_image_descriptors')
    # import sift

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


index_images()