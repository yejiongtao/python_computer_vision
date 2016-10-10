from sqlite3 import dbapi2 as sqlite


class Indexer(object):
    def __init__(self, db, voc):
        """ Initialize with the name of the database
          and a vocabulary object. """

        self.con = sqlite.connect(db)
        self.voc = voc

    def __del__(self):
        self.con.close()

    def db_commit(self):
        self.con.commit()