from numpy import *
from numpy.ma import sqrt, cos, sin
from scipy.constants import pi
from pylab import *
import os


class KnnClassifier(object):
    def __init__(self, labels, samples):
        """ Initialize classifier with training data. """

        self.labels = labels    # one label for one sample. can be integers or strings because we use dictionary
        self.samples = samples

    def classify(self, point, k=3):
        """ Classify a point against k nearest
          in the training data, return label. """

        # compute distance to all training points
        dist = array([L2dist(point, s) for s in self.samples])  # feel free to use other distance measure

        # sort them
        ndx = dist.argsort()

        # use dictionary to store the k nearest
        votes = {}
        for i in range(k):
            label = self.labels[ndx[i]]
            votes.setdefault(label, 0)
            votes[label] += 1

        return max(votes)


def L2dist(p1,p2):
    return sqrt( sum( (p1-p2)**2) )


def create_sample_2d_points():
    from numpy.random import randn
    import pickle

    # create sample data of 2D points
    n = 200

    # two normal distributions
    class_1 = 0.6 * randn(n, 2)
    class_2 = 1.2 * randn(n, 2) + array([5, 1])
    labels = hstack((ones(n), -ones(n)))

    # save with Pickle
    with open('points_normal_test.pkl', 'wb') as f:
        pickle.dump(class_1, f)
        pickle.dump(class_2, f)
        pickle.dump(labels, f)

    # normal distribution and ring around it
    class_1 = 0.6 * randn(n, 2)
    r = 0.8 * randn(n, 1) + 5
    angle = 2 * pi * randn(n, 1)
    class_2 = hstack((r * cos(angle), r * sin(angle)))
    labels = hstack((ones(n), -ones(n)))

    # save with Pickle
    with open('points_ring_test.pkl', 'wb') as f:
        pickle.dump(class_1, f)
        pickle.dump(class_2, f)
        pickle.dump(labels, f)


def example_classify_2d():
    import pickle
    import imtools

    def classify_2d(traning_data, test_data):
        # load 2D points using Pickle
        with open(traning_data, 'rb') as f:
            class_1 = pickle.load(f)
            class_2 = pickle.load(f)
            labels = pickle.load(f)

        model = KnnClassifier(labels, vstack((class_1, class_2)))

        # load test data using Pickle
        with open(test_data, 'rb') as f:
            class_1 = pickle.load(f)
            class_2 = pickle.load(f)
            labels = pickle.load(f)

        # define function for plotting
        def classify(x, y, model=model):
            return array([model.classify([xx, yy]) for (xx, yy) in zip(x, y)])

        # plot the classification boundary
        imtools.plot_2D_boundary([-6, 6, -6, 6], [class_1, class_2], classify, [1, -1])

    classify_2d('points_normal.pkl', 'points_normal_test.pkl')
    classify_2d('points_ring.pkl', 'points_ring_test.pkl')
    show()


def process_images_to_dsift():
    from n7_classifying_image_content import dsift

    path1 = 'train'
    path2 = 'test'
    imlist = [os.path.join(path1, f) for f in os.listdir(path1) if f.endswith('.ppm')]
    imlist.extend([os.path.join(path2, f) for f in os.listdir(path2) if f.endswith('.ppm')])

    # process images at fixed size (50 ,50)
    # otherwise the images will have varying number of descriptors, and therefore varying length of feature vectors
    for filename in imlist:
        featfile = filename[:-3] + 'dsift'
        dsift.process_image_dsift(filename, featfile, 10, 5, resize=(50,50))


def example_classify_images():
    from n1_local_image_descriptors import sift

    def read_gesture_features_labels(path):
        # create list of all files ending in .dsift
        featlist = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.dsift')]

        # read the features
        features = []
        for featfile in featlist:
            l, d = sift.read_features_from_file(featfile)
            features.append(d.flatten())
        features = array(features)

        # create labels (the first char of the filename)
        labels = [featfile.split('/')[-1][0] for featfile in featlist]

        return features, array(labels)

    def print_confusion(res, labels, classnames):
        n = len(classnames)

        # confusion matrix
        class_ind = dict([(classnames[i], i) for i in range(n)])

        confuse = zeros((n, n))
        for i in range(len(test_labels)):
            confuse[class_ind[res[i]], class_ind[test_labels[i]]] += 1

        print('Confusion matrix for')
        print(classnames)
        print(confuse)
        # column 'A' contains how many times an 'A' was classified as each classes

    features, labels = read_gesture_features_labels('train/')
    test_features, test_labels = read_gesture_features_labels('test/')
    classnames = unique(labels)

    # test kNN
    k = 1
    knn_classifier = KnnClassifier(labels, features)
    res = array([knn_classifier.classify(test_features[i], k) for i in
                 range(len(test_labels))])

    # accuracy
    acc = sum(1.0 * (res == test_labels)) / len(test_labels)
    print('Accuracy:', acc)

    print_confusion(res, test_labels, classnames)


# example_classify_2d()
# example_classify_images()