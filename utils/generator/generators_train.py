"""
This code based on codes from https://github.com/tristandeleu/ntm-one-shot \
                              and https://github.com/kjunelee/MetaOptNet
"""
import numpy as np
import random
import pickle as pkl

class miniImageNetGenerator(object):

    def __init__(self, data_file, nb_classes=5, nb_samples_per_class=10,
                  max_iter=None, xp=np, batch_size=160):
        super(miniImageNetGenerator, self).__init__()
        self.data_file = data_file
        self.batch_size = batch_size
        self.nb_classes = nb_classes
        self.nb_samples_per_class = nb_samples_per_class
        self.max_iter = max_iter
        self.xp = xp
        self.num_iter = 0
        self.data = self._load_data(self.data_file)

    def _load_data(self, data_file):
        dataset = self.load_data(data_file)
        data = dataset['data']
        labels = dataset['labels']
        label2ind = self.buildLabelIndex(labels)
        self.index = [[key, ind] for (key, val) in label2ind.items() for ind in range(len(val))]

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
            images, labels = self.sample_task(self.nb_classes, self.nb_samples_per_class)

            return (self.num_iter - 1), (images, labels)
        else:
            raise StopIteration()


    def augment(self, img):

        # random cropping
        npad = ((8, 8), (8, 8), (0, 0))
        img = np.pad(img, npad, 'constant', constant_values=(0.0))
        x = random.randint(0, 16)
        y = random.randint(0, 16)
        img = img[y:y + 84, x:x + 84]

        # random flipping
        flip_sign = random.randint(1,2)
        if flip_sign == 1:
            img = np.flip(img, 1)

        return img

    def sample_task(self, nb_classes, nb_samples_per_class):
        
        picture_list = sorted(set(self.data.keys()))
        pic_to_idx = {pic: i for i, pic in enumerate(picture_list)}
        sampled_characters = random.sample(self.data.keys(), nb_classes)

        labels_and_images = []
        temp = []
        for (k, char) in enumerate(sampled_characters):
            label = pic_to_idx[char]
            _imgs = self.data[char]
            _ind = random.sample(range(len(_imgs)), nb_samples_per_class)
            labels_and_images.extend([(label, self.xp.array(self.augment(_imgs[i])/np.float32(255))) for i in _ind])
            temp.extend([_imgs[i] for i in _ind])

        arg_labels_and_images = []
        for i in range(self.nb_samples_per_class):
            for j in range(self.nb_classes):
                arg_labels_and_images.extend([labels_and_images[i+j*self.nb_samples_per_class]])

        labels, images = zip(*arg_labels_and_images)
        return images, labels


    def sample(self):
        idx = random.sample(self.index, self.batch_size)
        images, labels = [], []
        for key, ind in idx:
            labels.append(key)
            images.append(self.augment(self.data[key][ind]/np.float32(255)))
        return images, labels


def make_input(images):
    images = np.stack(images)
    images = torch.Tensor(images).cuda()
    images = images.view(images.size(0), 84, 84, 3)
    images = images.permute(0, 3, 1, 2)

    return images

if __name__ == "__main__":
    import torch
    import cv2
    data_path = '/data/zhangjunyi/few-shot'
    train_path = data_path + '/miniImageNet_category_split_train_phase_train.pickle'
    val_path = data_path + '/miniImageNet_category_split_val.pickle'
    test_path = data_path + '/miniImageNet_category_split_test.pickle'
    train_generator = miniImageNetGenerator(data_file=train_path, nb_classes=15,
                                    nb_samples_per_class=10, max_iter=10000)

    for t, (images, labels, _) in train_generator:
        images = make_input(images)
        labels_DC = torch.tensor(labels, dtype=torch.long).cuda()

        images = images.permute(0, 2, 3, 1)
        for i in range(len(images)):
            im1 = images[i, :, :, [0, 1, 2]].data.cpu().numpy()
            cv2.imwrite("image/{:2d}.jpg".format(i), im1*255)
        exit()

