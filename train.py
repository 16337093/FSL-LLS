import os
import sys
sys.path.append('../')
import argparse
import numpy as np
import shutil
from tqdm import tqdm

import torch
from utils.generator.generators_train import miniImageNetGenerator as train_loader
from utils.generator.generators_test import miniImageNetGenerator as test_loader

from utils.model import Runner
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # training setting
    parser.add_argument('--is_train', type=str2bool, default=True,
                        help='Choice train or test.')
    parser.add_argument('--n_folder', type=str, default="0",
                        help='Number of folder.')
    parser.add_argument('--model_name', type=str, default='/model',
                        help='Number of folder.')
    parser.add_argument('--initial_lr', type=float, default=1e-1,
                        help='Initial learning rate.')
    parser.add_argument('--first_decay', type=int, default=25000,
                        help='First decay step.')
    parser.add_argument('--second_decay', type=int, default=35000,
                        help='Second decay step.')
    parser.add_argument('--max_iter', type=int, default=40000,
                        help='Maxium iterator')
    parser.add_argument('--n_train_class', type=int, default=15,
                        help='Number of way for training episode.')
    parser.add_argument('--n_query', type=int, default=8,
                        help='Number of queries per class in train.')

    # hyperparameter
    parser.add_argument('--transfer', type=str2bool, default=False,
                        help='Whether to transfer the feature map')
    parser.add_argument('--tau1', type=float, default=0.25,
                        help='Global Distance scale factor.')
    parser.add_argument('--tau2', type=float, default=0.45,
                        help='Matching Distance scale factor.')
    parser.add_argument('--tau3', type=float, default=0.45,
                        help='Matching Distance scale factor.')
    parser.add_argument('--beta', type=float, default=0.80,
                        help='transfer balance factor.')
    parser.add_argument('--iter', type=int, default=10,
                        help='Iteration in Soft-KMeans.')
    parser.add_argument('--norm_shift', type=str2bool, default=True,
                        help='Whether Normalize Shift.')

    # inference setting
    parser.add_argument('--n_shot', type=int, default=5,
                        help='Number of support set per class in train.')
    parser.add_argument('--n_test_query', type=int, default=15,
                        help='Number of queries per class in test.')
    parser.add_argument('--n_test_class', type=int, default=5,
                        help='Number of way for test episode.')
    parser.add_argument('--semi_super', type=str2bool, default=False,
                        help='Semi-supervised inference')
    parser.add_argument('--n_extra', type=int, default=30,
                        help='Number of unlabeled samples per class in SSFSL.')
    parser.add_argument('--n_distractor', type=int, default=0,
                        help='Number of distractor class in distan.')

    args = parser.parse_args()

    #######################
    folder_num = args.n_folder

    # optimizer setting
    max_iter = args.max_iter
    lrstep2 = args.second_decay
    lrstep1 = args.first_decay
    initial_lr = args.initial_lr

    # train episode setting
    n_shot=args.n_shot
    n_query=args.n_query
    nb_class_train = args.n_train_class

    # test episode setting
    n_query_test = args.n_test_query
    nb_class_test=args.n_test_class

    #data path
    data_path = '/data/huangjy/meta-learning/miniImageNet'
    train_path = data_path + '/miniImageNet_category_split_train_phase_train.pickle'
    val_path = data_path + '/miniImageNet_category_split_val.pickle'
    test_path = data_path + '/miniImageNet_category_split_test.pickle'

    #save_path
    save_path = '/data/huangjy/model/NIFM/miniImage/model_' + folder_num
    filename_5shot = save_path + args.model_name
    filename_5shot_last = save_path + '/model_last'

    # set up training
    # ------------------
    model = Runner(nb_class_train, nb_class_test, n_shot, n_query, 
                    is_train=args.is_train, iteration=args.iter, norm_shift=args.norm_shift, 
                    tau1=args.tau1, tau2=args.tau2, tau3=args.tau3, transfer=args.transfer, beta=args.beta)
    model.set_optimizer(learning_rate=initial_lr, weight_decay_rate=5e-4)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    loss_h=[]
    accuracy_h_val=[]
    accuracy_h_test=[]

    acc_best=0
    epoch_best=0
    if_DC = True
    # start training
    # ----------------
    if args.is_train:
        train_generator = train_loader(data_file=train_path, nb_classes=nb_class_train,
                                       nb_samples_per_class=n_shot + n_query, max_iter=max_iter)
        for t, (images_train, labels_train) in train_generator:
            if t == lrstep1 :
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] *= 0.06
                    print('-------Decay Learning Rate to ', param_group['lr'], '------')
            if t == lrstep2:
                for param_group in model.optimizer.param_groups:
                    param_group['lr'] *= 0.2
                    print('-------Decay Learning Rate to ', param_group['lr'], '------')
            # train
            loss = model.train(images_train, labels_train)
            loss_h.extend([loss.tolist()])
            if (t % 100 == 0):
                print("Episode: %d, Train Loss: %f "%(t, loss))
                torch.save(model.model.state_dict(), filename_5shot_last)

            if (t != 0) and (t % 1000 == 0):
                print('Evaluation in Validation data')
                test_generator = test_loader(data_file=val_path, nb_classes=nb_class_test,
                                             nb_samples_per_class=n_shot+n_query_test, max_iter=600)
                scores = []
                for i, (images, labels, _) in tqdm(test_generator):
                    acc_list = model.evaluate(images, labels)
                    scores.append(acc_list)
                scores = np.vstack(scores) 
                scores = np.mean(scores, 0)

                print(('Accuracy {}-shot =%').format(n_shot), 100*scores, folder_num)
                accuracy_t=100*np.mean(scores[-1])

                if acc_best < accuracy_t:
                    acc_best = accuracy_t
                    epoch_best = t
                    torch.save(model.model.state_dict(),filename_5shot)
                accuracy_h_val.extend([accuracy_t.tolist()])
                del(test_generator)
                del(accuracy_t)

                print('Evaluation in Test data')
                iter_num = 10000 if t>=lrstep1 else 1000
                test_generator = test_loader(data_file=test_path, nb_classes=nb_class_test,
                                             nb_samples_per_class=n_shot+n_query_test, max_iter=1000)
                scores = []
                for i, (images, labels, _) in tqdm(test_generator):
                    acc_list = model.evaluate(images, labels)
                    scores.append(acc_list)
                scores = np.vstack(scores) 
                scores = np.mean(scores, 0)

                print(('Accuracy {}-shot =%').format(n_shot), 100*scores)
                accuracy_t=100*np.mean(scores[-1])
                
                if t>=lrstep1: torch.save(model.model.state_dict(),filename_5shot+"_"+str(scores))

                accuracy_h_test.extend([accuracy_t.tolist()])
                del(test_generator)
                del(accuracy_t)
                print('***Average accuracy on past 10 test acc***')
                print('Best epoch =',epoch_best,'Best {}-shot acc='.format(n_shot), acc_best)

    else:
        accuracy_h5=[]
        total_acc = []
        load_state = torch.load(filename_5shot)
        state = model.model.state_dict()
        for x in state:
            state[x] = load_state[x]
        model.model.load_state_dict(state)
        print('Evaluating the best {}-shot model... whih'.format(n_shot))
        for tt in range(10):
            test_generator = test_loader(data_file=test_path, nb_classes=nb_class_test,
                    nb_samples_per_class=n_shot+n_query_test, max_iter=1000, extra=args.n_extra, distractor=args.n_distractor)
            scores = []
            for i, (images, labels, extra_img) in tqdm(test_generator):
                if args.semi_super and args.n_distractor>0:
                    acc_list = model.evaluate_semi_distract(images, labels, extra_img)
                elif args.semi_super:
                    acc_list = model.evaluate_semi(images, labels, extra_img)
                else:
                    acc_list = model.evaluate(images, labels, )
                scores.append(acc_list)
                total_acc.append(acc_list[-1] * 100)

            scores = np.vstack(scores) 
            stds = np.std(scores[:, -1]*100, axis=0)
            ci95 = 1.96 * stds / np.sqrt(scores.shape[0])
            scores = np.mean(scores, 0)
            print(('Accuracy {}-shot =%').format(n_shot), 100*scores[-1], ci95, folder_num)
            
            stds = np.std(total_acc, axis=0)
            ci95 = 1.96 * stds / np.sqrt(len(total_acc))
            print(sum(total_acc)/len(total_acc), "confidence all", ci95, model_name)
            del(test_generator)