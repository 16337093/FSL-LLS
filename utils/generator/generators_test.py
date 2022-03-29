import numpy as np
import random
import pickle as pkl

class miniImageNetGenerator(object):

    def __init__(self, data_file, nb_classes=5, nb_samples_per_class=15,
                  max_iter=None, extra=30, distractor=0):
        super(miniImageNetGenerator, self).__init__()
        self.data_file = data_file
        self.nb_classes = nb_classes
        self.nb_samples_per_class = nb_samples_per_class
        self.max_iter = max_iter
        self.num_iter = 0
        self.data = self._load_data(self.data_file)
        self.extra = extra
        self.distractor = distractor

    def _load_data(self, data_file):
        dataset = self.load_data(data_file)
        data = dataset['data']
        labels = dataset['labels']
        label2ind = self.buildLabelIndex(labels)

        return {key: np.array(data[val]) for (key, val) in label2ind.items()}

    def load_data(self, data_file):
        with open(data_file, 'rb') as f:
            u = pkl._Unpickler(f)
            u.encoding = 'latin1'
            data = u.load()
        return data

    def buildLabelIndex(self, labels):
        label2inds = {}
        for idx, label in enumerate(labels):
            if label not in label2inds:
                label2inds[label] = []
            label2inds[label].append(idx)

        return label2inds


    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if (self.max_iter is None) or (self.num_iter < self.max_iter):
            self.num_iter += 1
            images, labels, raw_images = self.sample(self.nb_classes, self.nb_samples_per_class)

            return (self.num_iter - 1), (images, labels, raw_images)
        else:
            raise StopIteration()

    def sample(self, nb_classes, nb_samples_per_class):
        sampled_characters = random.sample(self.data.keys(), nb_classes+self.distractor)
        labels_and_images = []
        extra_images = []
        for (k, char) in enumerate(sampled_characters[:nb_classes]):
            # print("ind", char)
            _imgs = self.data[char]
            _ind = random.sample(range(len(_imgs)), nb_samples_per_class)
            labels_and_images.extend([(k, np.array(_imgs[i]/np.float32(255).flatten())) for i in _ind])
            
        for (k, char) in enumerate(sampled_characters):
            _imgs = self.data[char]
            _ind = random.sample(range(len(_imgs)), self.extra)
            extra_images.extend([np.array(_imgs[i]/np.float32(255)) for i in _ind])


        arg_labels_and_images = []
        for i in range(self.nb_samples_per_class):
            for j in range(self.nb_classes):
                arg_labels_and_images.extend([labels_and_images[i+j*self.nb_samples_per_class]])

        labels, images = zip(*arg_labels_and_images)
        return images, labels, extra_images

def make_input(images):
    images = np.stack(images)
    # print(images.shape)
    images = torch.Tensor(images)
    images = images.view(images.size(0), 84, 84, 3)
    images = images.permute(0, 3, 1, 2)

    return images

if __name__ == "__main__":
    import torch, cv2
    data_path = '/data/zhangjunyi/few-shot'
    train_path = data_path + '/miniImageNet_category_split_train_phase_train.pickle'
    val_path = data_path + '/miniImageNet_category_split_val.pickle'
    test_path = data_path + '/miniImageNet_category_split_test.pickle'
    train_generator = miniImageNetGenerator(data_file=test_path, nb_classes=15,
                                    nb_samples_per_class=10, max_iter=10000)
    # exit()

    for t, (images, labels, _) in train_generator:
        images = make_input(images)
        labels_DC = torch.tensor(labels, dtype=torch.long).cuda()

        images = images.permute(0, 2, 3, 1)
        for i in range(len(images)):
            im1 = images[i, :, :, [2,1,0]].data.cpu().numpy()
            cv2.imwrite("image/{:2d}.jpg".format(i), im1*255)
        exit()

