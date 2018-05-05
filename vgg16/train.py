import sys, getopt
import tensorflow as tf
import numpy as np
import os
import utilities

IMG_WIDTH = 224
IMG_HEIGHT = 224
VGG_MEAN = [103.939, 116.779, 123.68]


class VGG16:
    def __init__(self, vgg16_npy_path):

        self.vgg16_npy_path = vgg16_npy_path
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()
        print("VGG16 npy file loaded")

        #self.network = self.build_network()
        #self.sess = tf.Session()


    def build_network(self, rgb):
        rgb_scaled = rgb * 255.0
        r, g, b = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert r.get_shape().as_list()[1:] == [224, 224, 1]
        assert g.get_shape().as_list()[1:] == [224, 224, 1]
        assert b.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            b - VGG_MEAN[0],
            g - VGG_MEAN[1],
            r - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]


        self.conv1_1 = self.conv_layer(bgr, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, "pool1")

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, "pool2")

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, "pool3")

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, "pool4")

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, "pool5")

        self.fc6 = self.fc_layer(self.pool5, "fc6")
        assert  self.fc6.get_shape().as_list()[1:] == [4096]
        self.relu6 = tf.nn.relu(self.fc6)

        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        self.fc8 = self.fc_layer(self.relu7, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")
        self.data_dict = None

        print("Model built successfully!")



    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

    def fc_layer(self, bottom, name):
        with tf.variable_scope(name):
            shape = bottom.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(bottom, [-1, dim])

            weights = self.get_fc_weight(name)
            biases = self.get_bias(name)

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

    def get_fc_weight(self, name):
        return tf.constant(self.data_dict[name][0], name="weights")


def get_batches(x, y, n_batches=10):
    batch_size = len(x) // n_batches

    for ii in range(0, n_batches*batch_size, batch_size):
        if ii != (n_batches-1)*batch_size:
            X, Y = x[ii:ii+batch_size], y[ii:ii+batch_size]
        else:
            X, Y = x[ii:], y[ii:]

        yield  X, Y


# System input parcer
#####################################################
def parcer(argv):
    input_folder = ''
    vgg16_npy_path = ''
    try:
        opts, args = getopt.getopt(argv, "hiv", ["help=", "input=", "vggnpy="])
    except  getopt.GetoptError:
        print ('Error 1": train.py -i <input_directory> -v <path_to_vgg16.npy>')
        sys.exit(1)

    for opt, arg in opts:
        if opt == '-h':
            print('Error 2: train.py -i <input_directory> -v <path_to_vgg16.npy>')
            sys.exit(2)
        elif opt == '-i' or opt == "--input":
            input_folder = arg
        elif opt == '-v' or opt == "--vggnpy":
            vgg16_npy_path = arg

    return input_folder, vgg16_npy_path

# Extract codes from fc6
#####################################################
def extract_codes(input_folder, vgg16_npy_path):

    contents = os.listdir(input_folder)
    classes = [each for each in contents if os.path.isdir(input_folder + each)]

    batch_size = 10
    codes_list = []
    labels = []
    batch = []
    codes = None

    with tf.Session() as sess:
        vgg = VGG16(vgg16_npy_path)
        _input = tf.placeholder(tf.float32, [None, IMG_WIDTH, IMG_HEIGHT, 3])
        with tf.name_scope("content_vgg"):
            vgg.build_network(_input)

        for cls in classes:
            print ("Processing {} images ...".format(cls))
            class_path = input_folder + cls

            files = os.listdir(class_path)
            for i, file in enumerate(files, 1):
                img = utilities.load_image(os.path.join(class_path, file))
                batch.append(img.reshape((1, IMG_WIDTH, IMG_HEIGHT, 3)))
                labels.append(cls)

                if i % batch_size == 0 or  i == len(files):
                    images = np.concatenate(batch)
                    feed_dict = {_input: images}
                    codes_batch = sess.run(vgg.relu6, feed_dict=feed_dict)

                    if codes is None:
                        codes= codes_batch
                    else:
                        codes = np.concatenate((codes, codes_batch))

                    batch = []
                    prog = float(i) * 100.0 / float(len(files))
                    if prog % 10 == 0:
                        print("{}% processed...".format(prog))

        with open("codes", "w") as f:
            codes.tofile(f)
        import csv
        with open("labels","w") as f:
            writer = csv.writer(f, delimiter='\n')
            writer.writerow(labels)


###########################################
def train(epoches=10):
    import csv
    with open("labels", "r") as f:
        reader = csv.reader(f, delimiter='\n')
        labels = np.array([x  for x in reader if len(x) > 0]).squeeze()
    with open("codes", "r") as f:
        codes = np.fromfile(f,  dtype=np.float32)
        codes = codes.reshape((len(labels), -1))


    from sklearn.preprocessing import LabelBinarizer

    lb = LabelBinarizer()
    lb.fit(labels)
    labels_vec = lb.transform(labels)


    from sklearn.model_selection import StratifiedShuffleSplit
    ss = StratifiedShuffleSplit(n_splits=1, test_size = 0.1)
    train_idx, test_idx = next(ss.split(codes, labels_vec))

    train_x, train_y = codes[train_idx], labels_vec[train_idx]
    test_x, test_y = codes[test_idx], labels_vec[test_idx]

    print("Train shapes (x, y):", train_x.shape, train_y.shape)
    print("Test shapes (x, y):", test_x.shape, test_y.shape)


    inputs_ = tf.placeholder(tf.float32, shape=[None, codes.shape[1]])
    labels_ = tf.placeholder(tf.int64, shape=[None, labels_vec.shape[1]])

    fc = tf.contrib.layers.fully_connected(inputs_, 256)

    logits = tf.contrib.layers.fully_connected(fc,  labels_vec.shape[1], activation_fn=None)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels_, logits=logits)
    cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    predict = tf.nn.softmax(logits)
    correct_pred = tf.equal(tf.argmax(predict, 1), tf.argmax(labels_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    iteration  = 0
    save = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for e in range(epoches):
            for x, y, in get_batches(train_x, train_y):
                feed = {inputs_: x, labels_: y}
                loss, _ = sess.run([cost, optimizer], feed_dict=feed)
                print("Epoch: {}/{}".format(e+1, epoches),
                      "Iteration: {}".format(iteration),
                      "Training loss: {:.5f}".format(loss))
                iteration += 1

                if iteration % 10 == 0:
                    feed = {inputs_: test_x, labels_: test_y}
                    val_acc = sess.run(accuracy, feed_dict=feed)
                    print("=========================")
                    print("Epoch: {}/{}".format(e, epoches),
                          "Iteration: {}".format(iteration),
                          "Validation Acc: {:.4f}".format(val_acc))
                    print("=========================")
        save.save(sess, "checkpoints/flowers.ckpt")



class FlowerClassifier:
    def __init__(self, vgg16_npy_path):
        import csv
        with open("labels", "r") as f:
            reader = csv.reader(f, delimiter='\n')
            labels = np.array([x for x in reader if len(x) > 0]).squeeze()

        from sklearn.preprocessing import LabelBinarizer
        self.lb = LabelBinarizer()
        self.lb.fit(labels)

        self.input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
        with tf.Session() as sess:
            self.vgg = VGG16(vgg16_npy_path)
            self.vgg.build_network(self.input_)


        self.inputs_ = tf.placeholder(tf.float32, shape=[None, 4096])
        self.labels_ = tf.placeholder(tf.int64, shape=[None, 5])

        self.fc = tf.contrib.layers.fully_connected(self.inputs_, 256)

        self.logits = tf.contrib.layers.fully_connected(self.fc, 5, activation_fn=None)
        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.labels_, logits=self.logits)
        self.cost = tf.reduce_mean(self.cross_entropy)

        self.optimizer = tf.train.AdamOptimizer().minimize(self.cost)
        self.predicted = tf.nn.softmax(self.logits)
        self.correct_pred = tf.equal(tf.argmax(self.predicted, 1), tf.argmax(self.labels_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.saver.restore(self.sess, tf.train.latest_checkpoint("checkpoints"))


    def predict(self, image):
        img = utilities.load_image(image)
        img = img.reshape((1, IMG_HEIGHT, IMG_WIDTH, 3))

        feed_dict = {self.input_: img}
        with tf.Session() as sess:
            code = sess.run(self.vgg.relu6, feed_dict=feed_dict)


        feed = {self.inputs_: code}
        prediction = self.sess.run(self.predicted, feed_dict=feed).squeeze()

        import matplotlib.pyplot as plt
        from scipy.ndimage import imread
        test_img = imread(image)

        plt.subplot(121)
        plt.imshow(test_img)
        plt.subplot(122)
        plt.barh(np.arange(5), prediction)
        _ = plt.yticks(np.arange(5), self.lb.classes_)
        plt.show()








def test(image, vgg16_npy_path):
    import matplotlib.pyplot as plt
    from scipy.ndimage import imread
    import csv
    with open("labels", "r") as f:
        reader = csv.reader(f, delimiter='\n')
        labels = np.array([x  for x in reader if len(x) > 0]).squeeze()

    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    lb.fit(labels)
    #labels_vec = lb.transform(labels)



    test_img = imread(image)
    input_ = tf.placeholder(tf.float32, [None, 224, 224, 3])
    with tf.Session() as sess:
        vgg = VGG16(vgg16_npy_path)
        vgg.build_network(input_)

        img = utilities.load_image(image)
        img = img.reshape((1, IMG_HEIGHT, IMG_WIDTH, 3))

        feed_dict = {input_: img}
        code = sess.run(vgg.relu6, feed_dict=feed_dict)



    inputs_ = tf.placeholder(tf.float32, shape=[None, code.shape[1]])
    labels_ = tf.placeholder(tf.int64, shape=[None, 5])

    fc = tf.contrib.layers.fully_connected(inputs_, 256)

    logits = tf.contrib.layers.fully_connected(fc, 5, activation_fn=None)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels_, logits=logits)
    cost = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    predicted = tf.nn.softmax(logits)
    correct_pred = tf.equal(tf.argmax(predicted, 1), tf.argmax(labels_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint("checkpoints"))
        feed = {inputs_: code}
        prediction = sess.run(predicted, feed_dict=feed).squeeze()

    plt.subplot(121)
    plt.imshow(test_img)
    plt.subplot(122)
    plt.barh(np.arange(5), prediction)
    _ = plt.yticks(np.arange(5), lb.classes_)
    plt.show()







if __name__ == '__main__':
    input_folder, vgg16_npy_path = parcer(sys.argv[1:])

    #extract_codes(input_folder, vgg16_npy_path)
    #train(epoches=400)

    classifier = FlowerClassifier(vgg16_npy_path)


    img1 = "/home/masoud/projects/data/flower_photos/roses/9159362388_c6f4cf3812_n.jpg"
    img2 = "/home/masoud/projects/data/flower_photos/daisy/8887005939_b19e8305ee.jpg"
    img3 = "/home/masoud/projects/data/flower_photos/tupils/8762189906_8223cef62f.jpg"
    img4 = "/home/masoud/projects/data/flower_photos/dandelion/9152356642_06ae73113f.jpg"

    images = [img1, img2, img3, img4]

    for img in images:
        classifier.predict(img)

