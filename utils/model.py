import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 

from utils.backbone.resnet12 import Resnet12
from fvcore.nn import smooth_l1_loss

class Runner(object):
    def __init__(self, nb_class_train, nb_class_test, n_shot, n_query, is_train=True,
                iteration=10, norm_shift=True, tau1=0.25, tau2=0.45, tau3=1.0, transfer=False, beta=0.8):

        self.nb_class_train = nb_class_train
        self.nb_class_test = nb_class_test
        self.n_shot = n_shot
        self.n_query = n_query
        self.is_train = is_train
        self.iteration, self.norm_shift, self.tau1, self.tau2, self.tau3, self.transfer, self.beta = \
            iteration, norm_shift, tau1, tau2, tau3, transfer, beta

        self.model = Resnet12().cuda()
        self.loss = nn.CrossEntropyLoss()
        self.loss_reg = nn.MSELoss()

    def set_optimizer(self, learning_rate, weight_decay_rate):
        self.optimizer = optim.SGD([{'params': self.model.parameters(), 'weight_decay': weight_decay_rate}],
                                   lr = learning_rate, momentum=0.9, nesterov=True)

    def make_protomap(self, support_set, nb_class, n_shot=None):
        if n_shot is None:
            n_shot = self.n_shot
        B, C, W, H = support_set.shape
        protomap = support_set.reshape(n_shot, nb_class, C, W, H)
        protomap = protomap.mean(dim=0)
        return protomap

    def data_norm(self, images):
        mean = torch.FloatTensor([x / 255.0 for x in [125.3, 123.0, 113.9]]).cuda()
        std  = torch.FloatTensor([x / 255.0 for x in [63.0, 62.1, 66.7]]).cuda()
        return (images - mean) / std

    def make_input(self, images):
        images = np.stack(images)
        images = torch.Tensor(images).cuda()
        if self.norm_shift: images = self.data_norm(images)
        images = images.view(images.size(0), 84, 84, 3)
        images = images.permute(0, 3, 1, 2)
        return images

    def add_query(self, support_set, query_set, prob, nb_class, n_shot=None):
        if n_shot is None:
            n_shot = self.n_shot
        B, C, W, H = support_set.shape
        per_class = support_set.reshape(n_shot, nb_class, C, W, H)

        for i in range(nb_class):
            ith_prob = prob[:,i].reshape(prob.size(0), 1, 1, 1)
            ith_map = torch.cat((per_class[:,i], query_set*ith_prob), dim=0)
            ith_map = torch.sum(ith_map, dim=0, keepdim=True)/(ith_prob.sum()+n_shot)
            if i == 0: protomap = ith_map
            else: protomap = torch.cat((protomap, ith_map), dim=0)

        return protomap

    def norm_flatten(self, key, flatten=True):
        if flatten:
            key = torch.flatten(key, start_dim=1)
        else:
            key = torch.mean(key.reshape(key.size(0), key.size(1), -1), 2)
        key = F.normalize(key, dim=1)
        return key

    def norm_flatten_chamfer(self, key):                            
        key = key.reshape(key.size(0), key.size(1), -1)
        key = key.permute(0, 2, 1)
        b, n, c = key.size()
        key = F.normalize(key.reshape(key.size(0), -1), dim=1).reshape(b, n, c)
        return key

    def proto_distance(self, all_key, nb_class):
        _, C, W, H = all_key.shape
        protomap = all_key.reshape(-1, nb_class, C, W, H)
        protomap = protomap.mean(dim=0)
        scaled_proto = self.norm_flatten(protomap) / 0.25
        scaled_query = self.norm_flatten(all_key) / 0.25
        distance = scaled_query.unsqueeze(1) - scaled_proto
        distance = distance.pow(2).sum(dim=2)
        return distance

    def train(self, images, labels):
        nb_class = self.nb_class_train
        images = self.make_input(images)
        labels_DC = torch.tensor(labels, dtype=torch.long).cuda()

        self.model.train()
        original_key = self.model(images)
        flipped_key = self.model(torch.flip(images, dims=[3]))

        key1 = original_key[0]
        pred1 = self.model.weight(key1)
        pred1 = torch.flatten(pred1, start_dim=2)
        pred1 = pred1.permute(0, 2, 1)

        loss_dense = 0
        for i in range(pred1.size(1)):
            loss_dense += self.loss(pred1[:, i], labels_DC)/pred1.size(1)

        labels_IC = tuple([i for i in range(nb_class)]) * (self.n_query + self.n_shot) * 2
        labels_IC = torch.tensor(labels_IC, dtype=torch.long).cuda()

        key1 = torch.cat([original_key[0], flipped_key[0]], dim=0)
        key2 = torch.cat([original_key[1], flipped_key[1]], dim=0)
        distance1 = self.proto_distance(key1, nb_class)
        distance2 = self.proto_distance(key2, nb_class)
        loss_global = self.loss(-distance1, labels_IC) + self.loss(-distance2, labels_IC)*0.1
        
        # B*(W*H)
        loss_reg1 = 0
        bias1 = key1.pow(2).sum(1).view(key1.size()[0], -1).sqrt()

        loss_reg1 = self.loss_reg(bias1 - bias1.mean(1, keepdim=True), torch.zeros(bias1.size()).cuda())

        loss = 0
        loss += 1 * loss_dense
        loss += 0.2 * loss_global
        loss += 0.0001 * loss_reg1

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.data

    def distChamfer(self, x, y):
        xx, yy, zz = torch.sum(x**2, 2), torch.sum(y**2, 2), x.unsqueeze(1)@y.transpose(1,2)
        P = xx.unsqueeze(1).unsqueeze(3) + yy.unsqueeze(0).unsqueeze(2) - 2*zz
        min_dis, _ = torch.min(P, 3)
        distance = torch.sum(min_dis, 2)
        return distance

    def distattention(self, x, y, tau):
        xx, yy, zz = torch.sum(x**2, 2), torch.sum(y**2, 2), x.unsqueeze(1)@y.transpose(1,2)
        P = xx.unsqueeze(1).unsqueeze(3) + yy.unsqueeze(0).unsqueeze(2) - 2*zz
        atten = F.softmax(-P*tau, dim=3) @ y
        que = torch.flatten(x, start_dim=1)
        sup = torch.flatten(atten, start_dim=2)
        sup = F.normalize(sup, dim=2) / self.tau2
        return (que.unsqueeze(1) - sup).pow(2).sum(2)

    def vis_sample(self, images):
        images = self.make_input(images)
        self.model.eval()
        with torch.no_grad():
            flipped_key = self.model(torch.flip(images, dims=[3]))
            original_key = self.model(images)
            flipped_key_fine = self.model.transfer(torch.flip(images, dims=[3]), self.beta)
            original_key_fine = self.model.transfer(images, self.beta)
            result = original_key + flipped_key
            result_fine = flipped_key_fine+original_key_fine
        from sklearn import manifold, datasets

        def distance(x, y):
            dis_local = x[:, np.newaxis] - y
            dis_local = np.sum(dis_local**2, 2)
            return dis_local   
            
        X = result[0].data.cpu().numpy()
        X = X.reshape(X.shape[0], -1)
        X = X/np.sum(X**2, 1).reshape(-1, 1)
        dis_glob = distance(X, X)
        tsne = manifold.TSNE(n_components=2, random_state=501, metric='precomputed')
        X_glob = tsne.fit_transform(dis_glob) 

        X = result_fine[0].data.cpu().numpy()
        X = X.reshape(X.shape[0], -1)
        X = X/np.sum(X**2, 1).reshape(-1, 1)
        dis_glob = distance(X, X)
        tsne = manifold.TSNE(n_components=2, random_state=501, metric='precomputed')
        X_fine = tsne.fit_transform(dis_glob) 



        return [X_glob, X_fine]

    def evaluate(self, images, labels):
        nb_class = self.nb_class_test
        images = self.make_input(images)
        labels = torch.tensor(labels, dtype=torch.long).cuda()
        self.model.eval()
        with torch.no_grad():
            if self.transfer:
                flipped_key = self.model.transfer(torch.flip(images, dims=[3]), self.beta)
                original_key = self.model.transfer(images, self.beta)
            else:
                flipped_key = self.model(torch.flip(images, dims=[3]))
                original_key = self.model(images)
            nb_key = 2
            prob_list = []
            acc_list = []
            for iter in range(self.iteration+1):
                prob_sum = 0
                aa = 1.0
                weight = [aa, 2-aa]
                for idx in range(nb_key):
                    key1, key2 = original_key[idx], flipped_key[idx]
                    support_set = torch.cat([key1[:nb_class*self.n_shot], key2[:nb_class*self.n_shot]], dim=0)
                    query_set = torch.cat([key1[nb_class*self.n_shot:], key2[nb_class*self.n_shot:]], dim=0)
                    # support_set = original_key[idx][:nb_class*self.n_shot]
                    # query_set = original_key[idx][nb_class*self.n_shot:]
                    # Make Protomap
                    if iter == 0: protomap = self.make_protomap(support_set, nb_class, self.n_shot*2)
                    # if iter == 0: protomap = self.make_protomap(support_set, nb_class, self.n_shot)
                    else: protomap = self.add_query(support_set, query_set, prob_list[iter-1], nb_class, self.n_shot*2) 
                    
                    query_NF = self.norm_flatten(query_set) / self.tau1
                    proto_NF = self.norm_flatten(protomap) / self.tau1

                    # if self.train:
                    if False:
                        distance = query_NF.unsqueeze(1) - proto_NF
                        distance = distance.pow(2).sum(dim=2)
                        prob = F.softmax(-distance, dim=1)
                    else:
                        dis_local = query_NF.unsqueeze(1) - proto_NF
                        dis_local = dis_local.pow(2).sum(dim=2)

                        query_NF = self.norm_flatten_chamfer(query_set) / self.tau2
                        proto_NF = self.norm_flatten_chamfer(protomap) / self.tau2
                        
                        dis_chamf = self.distChamfer(query_NF, proto_NF)
                        # dis_chamf = self.distattention(query_NF, proto_NF, self.tau3)

                        prob = F.softmax(-(dis_chamf+dis_local)/2, dim=1)

                    prob_sum += prob * weight[idx]

                prob_list.append(prob_sum/nb_key)
                prob = prob_list[-1]
                nq = len(labels[nb_class * self.n_shot:])
                pred = torch.argmax((prob[:nq]+prob[nq:])/2, dim=1)
                # pred = torch.argmax(prob, dim=1)
                acc = np.mean((pred==labels[nb_class * self.n_shot:]).data.cpu().numpy())
                acc_list.append(acc)
            return acc_list

    def evaluate_semi(self, images, labels, extra_img):
        nb_class = self.nb_class_test
        images = self.make_input(images)
        extra_img = self.make_input(extra_img)
        labels = torch.tensor(labels, dtype=torch.long).cuda()
        self.model.eval()
        with torch.no_grad():
            flipped_key = self.model(torch.flip(images, dims=[3]))
            original_key = self.model(images)
            flipped_key_extra = self.model(torch.flip(extra_img, dims=[3]))
            original_key_extra = self.model(extra_img)
            nb_key = 2
            prob_list = []
            acc_list = []
            for iter in range(self.iteration+1):
                prob_sum = 0
                pred_sum = 0
                for idx in range(nb_key):
                    key1, key2 = original_key[idx], flipped_key[idx]
                    key3, key4 = original_key_extra[idx], flipped_key_extra[idx]
                    support_set = torch.cat([key1[:nb_class*self.n_shot], key2[:nb_class*self.n_shot]], dim=0)
                    query_set = torch.cat([key1[nb_class*self.n_shot:], key2[nb_class*self.n_shot:]], dim=0)
                    extra_set = torch.cat([key3, key4], dim=0)

                    # Make Protomap
                    if iter == 0: protomap = self.make_protomap(support_set, nb_class, self.n_shot*2)
                    else: protomap = self.add_query(support_set, extra_set, prob_list[-1], nb_class, self.n_shot*2) 

                    extra_NF = self.norm_flatten_chamfer(extra_set) / self.tau2
                    proto_NF = self.norm_flatten_chamfer(protomap) / self.tau2 
                    dis_chamf = self.distChamfer(extra_NF, proto_NF)
                    
                    extra_NF = self.norm_flatten(extra_set) / self.tau1
                    proto_NF = self.norm_flatten(protomap) / self.tau1
                    dis_local = extra_NF.unsqueeze(1) - proto_NF
                    dis_local = dis_local.pow(2).sum(dim=2)
                    prob = F.softmax(-(dis_chamf+dis_local)/2, dim=1)
                    
                    query_NF = self.norm_flatten_chamfer(query_set) / self.tau2
                    proto_NF = self.norm_flatten_chamfer(protomap) / self.tau2 
                    dis_chamf = self.distChamfer(query_NF, proto_NF)
                    
                    query_NF = self.norm_flatten(query_set) / self.tau1
                    proto_NF = self.norm_flatten(protomap) / self.tau1
                    dis_local = query_NF.unsqueeze(1) - proto_NF
                    dis_local = dis_local.pow(2).sum(dim=2)
                    pred = F.softmax(-(dis_chamf+dis_local)/2, dim=1)

                    prob_sum += prob
                    pred_sum += pred

                prob_list.append(prob_sum/nb_key)
                pred = pred_sum/nb_key
                nq = len(labels[nb_class * self.n_shot:])
                pred = torch.argmax((pred[:nq]+pred[nq:])/2, dim=1)
                
                acc = np.mean((pred==labels[nb_class * self.n_shot:]).data.cpu().numpy())
                acc_list.append(acc)
            return acc_list

    def sample(self, sup, ext, nb_class, max_c=11):
        sup = self.make_protomap(sup, nb_class)
        key1, key2 = self.norm_flatten_chamfer(sup), self.norm_flatten_chamfer(ext)
        sup_idx = []
        distance = self.distChamfer(key2, key1)

        for i in range(max_c):
            idx = torch.max(distance.mean(1), dim=0)[1]
            new_dis = self.distChamfer(key2, key2[idx].unsqueeze(0))
            distance = torch.cat([distance, new_dis], dim=1)
            sup_idx.append(idx)
        qur_idx = [i for i in range(ext.size()[0]) if not i in sup_idx]
        return torch.tensor(sup_idx, dtype=torch.long).cuda(), torch.tensor(qur_idx, dtype=torch.long).cuda()

    def evaluate_semi_distract(self, images, labels, extra_img, max_c=11):
        nb_class = self.nb_class_test
        images = self.make_input(images)
        extra_img = self.make_input(extra_img)
        labels = torch.tensor(labels, dtype=torch.long).cuda()
        self.model.eval()
        with torch.no_grad():
            flipped_key = self.model(torch.flip(images, dims=[3]))
            original_key = self.model(images)
            flipped_key_extra = self.model(torch.flip(extra_img, dims=[3]))
            original_key_extra = self.model(extra_img)
            sup_idx, qur_idx = self.sample((original_key[0]+flipped_key[0])[:nb_class*self.n_shot]/2, 
                                    (original_key_extra[0]+flipped_key_extra[0])/2, nb_class, max_c=max_c)
            nb_key = 2
            prob_list = []
            acc_list = []
            for iter in range(self.iteration+1):
                prob_sum = 0
                pred_sum = 0
                for idx in range(nb_key):
                    key1, key2 = original_key[idx], flipped_key[idx]
                    key3, key4 = original_key_extra[idx], flipped_key_extra[idx]
                    support_set = torch.cat([key1[:nb_class*self.n_shot], key3[sup_idx], key2[:nb_class*self.n_shot], key4[sup_idx]], dim=0)
                    query_set = torch.cat([key1[nb_class*self.n_shot:], key2[nb_class*self.n_shot:]], dim=0)
                    extra_set = torch.cat([key3[qur_idx], key4[qur_idx]], dim=0)
                    # Make Protomap
                    if iter == 0: protomap = self.make_protomap(support_set, nb_class+max_c, self.n_shot*2)
                    else: protomap = self.add_query(support_set, extra_set, prob_list[-1], nb_class+max_c, self.n_shot*2)

                    extra_NF = self.norm_flatten_chamfer(extra_set) / self.tau2
                    proto_NF = self.norm_flatten_chamfer(protomap) / self.tau2 
                    dis_chamf = self.distChamfer(extra_NF, proto_NF)
                    
                    extra_NF = self.norm_flatten(extra_set) / self.tau1
                    proto_NF = self.norm_flatten(protomap) / self.tau1
                    dis_local = extra_NF.unsqueeze(1) - proto_NF
                    dis_local = dis_local.pow(2).sum(dim=2)
                    prob = F.softmax(-(dis_chamf+dis_local)/2, dim=1)
                    
                    query_NF = self.norm_flatten_chamfer(query_set) / self.tau2
                    proto_NF = self.norm_flatten_chamfer(protomap[:nb_class]) / self.tau2 
                    dis_chamf = self.distChamfer(query_NF, proto_NF)
                    
                    query_NF = self.norm_flatten(query_set) / self.tau1
                    proto_NF = self.norm_flatten(protomap[:nb_class]) / self.tau1
                    dis_local = query_NF.unsqueeze(1) - proto_NF
                    dis_local = dis_local.pow(2).sum(dim=2)
                    pred = F.softmax(-(dis_chamf+dis_local)/2, dim=1)

                    prob_sum += prob
                    pred_sum += pred

                prob_list.append(prob_sum/nb_key)
                pred = pred_sum/nb_key
                nq = len(labels[nb_class * self.n_shot:])
                pred = torch.argmax((pred[:nq]+pred[nq:])/2, dim=1)
                
                acc = np.mean((pred==labels[nb_class * self.n_shot:]).data.cpu().numpy())
                acc_list.append(acc)
            return acc_list


