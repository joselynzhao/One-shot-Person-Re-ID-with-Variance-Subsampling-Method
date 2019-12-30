import torch
from torch import nn
from reid import models
from reid.trainers import Trainer
from reid.evaluators import extract_features, Evaluator
from reid.dist_metric import DistanceMetric
import numpy as np
from collections import OrderedDict
import os.path as osp
import pickle
from reid.utils.serialization import load_checkpoint
from reid.utils.data import transforms as T
from torch.utils.data import DataLoader
from reid.utils.data.preprocessor import Preprocessor
import random
from sklearn.metrics.pairwise import cosine_similarity
# from run import outf
# import run
import  math
import codecs


import ssl
ssl._create_default_https_context = ssl._create_unverified_context


class EUG():
    def __init__(self, model_name, batch_size, mode, num_classes, data_dir, l_data, u_data, save_path, dropout=0.5, max_frames=900):

        self.model_name = model_name
        self.num_classes = num_classes
        self.mode = mode
        self.data_dir = data_dir
        self.save_path = save_path

        self.l_data = l_data
        self.u_data = u_data
        self.l_label = np.array([label for _,label,_,_ in l_data])
        self.u_label = np.array([label for _,label,_,_ in u_data])


        self.dataloader_params = {}
        self.dataloader_params['height'] = 256
        self.dataloader_params['width'] = 128
        self.dataloader_params['batch_size'] = batch_size
        self.dataloader_params['workers'] = 6


        self.batch_size = batch_size
        self.data_height = 256
        self.data_width = 128
        self.data_workers = 6

        # batch size for eval mode. Default is 1.
        self.eval_bs = 1
        self.dropout = dropout
        self.max_frames = max_frames


    def get_dataloader(self, dataset, training=False) :
        normalizer = T.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

        if training:
            transformer = T.Compose([
                T.RandomSizedRectCrop(self.data_height, self.data_width),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalizer,
            ])
            batch_size = self.batch_size

        else:
            transformer = T.Compose([
                T.RectScale(self.data_height, self.data_width),
                T.ToTensor(),
                normalizer,
            ])
            batch_size = self.eval_bs

        data_loader = DataLoader(
            Preprocessor(dataset, root=self.data_dir,
                         transform=transformer, is_training=training, max_frames=self.max_frames),
            batch_size=batch_size, num_workers=self.data_workers,
            shuffle=training, pin_memory=True, drop_last=training)

        current_status = "Training" if training else "Test"
        print("create dataloader for {} with batch_size {}".format(current_status, batch_size))
        return data_loader

    def train(self, train_data, step, epochs=10, step_size=55, init_lr=0.1, dropout=0.5):

        """ create model and dataloader """
        model = models.create(self.model_name, dropout=self.dropout, num_classes=self.num_classes, mode=self.mode)
        model = nn.DataParallel(model).cuda()
        # model = nn.DataParallel(model, device_ids=[3,4]).cuda()
        # model.to(device)
        dataloader = self.get_dataloader(train_data, training=True)

        # the base parameters for the backbone (e.g. ResNet50)
        base_param_ids = set(map(id, model.module.CNN.base.parameters()))

        # we fixed the first three blocks to save GPU memory
        base_params_need_for_grad = filter(lambda p: p.requires_grad, model.module.CNN.parameters())

        # params of the new layers
        new_params = [p for p in model.parameters() if id(p) not in base_param_ids]

        # set the learning rate for backbone to be 0.1 times
        param_groups = [
            {'params': base_params_need_for_grad, 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]

        criterion = nn.CrossEntropyLoss().cuda()  # 标准
        optimizer = torch.optim.SGD(param_groups, lr=init_lr, momentum=0.5, weight_decay = 5e-4, nesterov=True)

        # change the learning rate by step
        def adjust_lr(epoch, step_size):     #学习率的衰减也可以做调整
            lr = init_lr / (10 ** (epoch // step_size))
            for g in optimizer.param_groups:
                g['lr'] = lr * g.get('lr_mult', 1)

            if epoch % step_size == 0:
                print("Epoch {}, current lr {}".format(epoch, lr))

        """ main training process """
        trainer = Trainer(model, criterion)
        for epoch in range(epochs):
            adjust_lr(epoch, step_size)
            trainer.train(epoch, dataloader, optimizer)
            # trainer.train(epoch, dataloader, optimizer, print_freq=len(dataloader)//30 * 10)

        torch.save(model.state_dict(), osp.join(self.save_path,  "{}_step_{}.ckpt".format(self.mode, step)))
        self.model = model


    def get_feature(self, dataset):
        dataloader = self.get_dataloader(dataset, training=False)
        features,_ = extract_features(self.model, dataloader)
        features = np.array([logit.numpy() for logit in features.values()])
        return features

    # def get_Classification_result(self):  #是指的用CNN的分类结果贴标签
    #     logits = self.get_feature(self.u_data)
    #     exp_logits = np.exp(logits)
    #     predict_prob = exp_logits / np.sum(exp_logits,axis=1).reshape((-1,1))
    #     assert len(logits) == len(predict_prob)
    #     assert predict_prob.shape[1] == self.num_classes
    #
    #     pred_label = np.argmax(predict_prob, axis=1)
    #     pred_score = predict_prob.max(axis=1)
    #     print("get_Classification_result", predict_prob.shape)
    #
    #
    #     num_correct_pred = 0
    #     for idx, p_label in enumerate(pred_label):
    #         if self.u_label[idx] == p_label:
    #             num_correct_pred +=1
    #
    #
    #     print("{} predictions on all the unlabeled data: {} of {} is correct, accuracy = {:0.3f}".format(
    #         self.mode, num_correct_pred, pred_label.shape[0], num_correct_pred/pred_label.shape[0]))
    #
    #     return pred_label, pred_score



    def get_Dissimilarity_result(self):

        # extract feature
        u_feas = self.get_feature(self.u_data)
        l_feas = self.get_feature(self.l_data)
        print("u_features", u_feas.shape, "l_features", l_feas.shape)

        scores = np.zeros((u_feas.shape[0]))
        labels = np.zeros((u_feas.shape[0]))
        # 分别用来存 _ufeas的分数和标签

        id_num = {}  #以标签名称作为字典
        # a = 1
        num_correct_pred = 0
        dists = []
        for idx, u_fea in enumerate(u_feas):
            diffs = l_feas - u_fea
            dist = np.linalg.norm(diffs,axis=1)
            index_min = np.argmin(dist)
            dists.append(dist)
            scores[idx] = - dist[index_min]  # "- dist" : more dist means less score
            labels[idx] = self.l_label[index_min] # take the nearest labled neighbor as the prediction label
            # if a:
            #     print("labels :-------------------------------------------", labels[idx])
            #     a = 0
            #     输出的结果是0.0
            # count the correct number of Nearest Neighbor prediction
            if self.u_label[idx] == labels[idx]:
                num_correct_pred +=1
            # 统计各个id的数量
            # if str(labels[idx]) in id_num.keys():
            #     id_num[str(labels[idx])]=id_num[str(labels[idx])]+1 #值加1
            # else:
            #     id_num[str(labels[idx])]=1


        print("{} predictions on all the unlabeled data: {} of {} is correct, accuracy = {:0.3f}".format(
            self.mode, num_correct_pred, u_feas.shape[0], num_correct_pred/u_feas.shape[0]))

        # sorted(id_num.items(),key = lambda item:item[1])
        # print("id_num:--------------------------------------------id_num----------------- ")
        # print(id_num)
        dists = np.vstack(dists)
        return labels, scores,num_correct_pred/u_feas.shape[0],dists

    def get_Dissimilarity_result2(self):
        # l_feas_file = codecs.open("logs/l_feas/test1.txt",'a')
        # extract feature
        u_feas = self.get_feature(self.u_data)
        l_feas = self.get_feature(self.l_data)
        l_mean = l_feas.mean(axis=0)
        l_std = l_feas.std(axis=0)
        l_feas -= l_mean
        l_feas /= l_std
        # np.save("logs/l_feas/test1.npy",l_feas)
        print("u_features", u_feas.shape, "l_features", l_feas.shape)
        scores = np.zeros((u_feas.shape[0]))
        labels = np.zeros((u_feas.shape[0]))
        # 分别用来存 _ufeas的分数和标签
        id_num = {}  #以标签名称作为字典
        num_correct_pred = 0
        for idx, u_fea in enumerate(u_feas):
            # dist = []
            # for l_fea in l_feas:
            #     d = np.dot(l_fea,u_fea)/(np.linalg.norm(l_fea)*(np.linalg.norm(u_fea)))
            #     dist.appen(d)
            # dist = cosine_similarity(l_feas,u_fea) #维度不对
            # dist = cosine_similarity(l_feas,np.array([u_fea])).reshape(-1)
            # print("----------------------------------------------------dist:{}".format(dist))
            u_fea -=l_mean
            u_fea /=l_std
            diffs = l_feas - u_fea
            dist = np.linalg.norm(diffs,axis=1)
            index_min = np.argmin(dist)
            scores[idx] = - dist[index_min]  # "- dist" : more dist means less score
            labels[idx] = self.l_label[index_min] # take the nearest labled neighbor as the prediction label
            # if a:
            #     print("labels :-------------------------------------------", labels[idx])
            #     a = 0
            #     输出的结果是0.0
            # count the correct number of Nearest Neighbor prediction
            if self.u_label[idx] == labels[idx]:
                num_correct_pred +=1
            # 统计各个id的数量
            if str(labels[idx]) in id_num.keys():
                id_num[str(labels[idx])]=id_num[str(labels[idx])]+1 #值加1
            else:
                id_num[str(labels[idx])] =1


        print("{} predictions on all the unlabeled data: {} of {} is correct, accuracy = {:0.3f}".format(
            self.mode, num_correct_pred, u_feas.shape[0], num_correct_pred/u_feas.shape[0]))

        sorted(id_num.items(),key = lambda item:item[1])
        # print("id_num:--------------------------------------------id_num----------------- ")
        # print(id_num)
        return labels, scores,num_correct_pred/u_feas.shape[0],id_num


    def estimate_label(self):

        print("label estimation by {} mode.".format(self.mode))

        if self.mode == "Dissimilarity":
            # predict label by dissimilarity cost
            [pred_label, pred_score,label_pre,id_num] = self.get_Dissimilarity_result()

        elif self.mode == "Classification":
            # predict label by classification
            [pred_label, pred_score] = self.get_Classification_result()
        else:
            raise ValueError

        return pred_label, pred_score,label_pre,id_num


    def select_top_data(self, pred_score, nums_to_select):
        v = np.zeros(len(pred_score))
        index = np.argsort(-pred_score)
        for i in range(nums_to_select):  #排序,求最前面的n个
            v[index[i]] = 1
        return v.astype('bool')

    def select_top_data_nlvm_b1(self,pred_score,dists,new_expend_nums_to_select,new_nums_to_select):
        # pred_score = pred_score.T # if necessary
        # 方案2, 求最近的P%样本的方差
        N_u,N_l = dists.shape
        stds = np.zeros(N_u)
        selection1=selection2= np.zeros(N_u, 'bool')
        index = np.argsort(-pred_score)
        selection1[index[:new_expend_nums_to_select]] = True  # selection1 就是选出 距离在一定范围内的数量
        # 求最近的P%样本的方差
        for i in index[:new_expend_nums_to_select]:
            score = - dists[i]
            # 求k近邻
            # topk = int(N_l * percent_P)
            topk = 2
            topk_idxs = np.argpartition(score, topk)[:topk]
            stds[i] = score[topk_idxs].std()   # 这里要求pre_score 要是二维的才行
        # 根据方差排序
        idxs = np.argsort(-stds)  # 这里该取负还是取正呢?
        # print(stds[idxs[:nums_to_select]])
        selection2[idxs[:new_nums_to_select]] = True
        return selection2

    def select_top_data_nlvm_b5(self,pred_score,dists,new_nums_to_select,percent_vari):
        # pred_score = pred_score.T # if necessary
        # 方案2, 求最近的P%样本的方差
        N_u,N_l = dists.shape
        vari_num = round(N_u * percent_vari)  # 取前面vari_num个
        if vari_num <= new_nums_to_select: # 就不需要在判断方差了
            return self.select_top_data(pred_score,new_nums_to_select)
        stds = np.zeros(N_u)
        selection = np.zeros(N_u,'bool')
        for i in range(N_u):  #对所有的值求方差
            score = -dists[i]
            # 求k近邻
            topk = 2
            topk_idxs = np.argpartition(score, topk)[:topk]
            stds[i] = score[topk_idxs].std()
        idxs = np.argsort(-stds)
        index = np.argsort(-pred_score[idxs[:vari_num]])  # 分数从大到小排序
        selection[index[:new_nums_to_select]] = True
        return selection

    def select_top_data_nlvm_b6(self,pred_score,dists,new_nums_to_select,percent_vari):
        # pred_score = pred_score.T # if necessary
        # 方案2, 求最近的P%样本的方差
        N_u,N_l = dists.shape
        vari_num = round(N_u * percent_vari ) # 取前面vari_num个
        if vari_num <= new_nums_to_select: # 就不需要在判断方差了
            return self.select_top_data(pred_score,new_nums_to_select)
        stds = np.zeros(N_u)
        selection1=selection2 = np.zeros(N_u,'bool')
        for i in range(N_u):  #对所有的值求方差
            score = -dists[i]
            # 求k近邻
            topk = 2
            topk_idxs = np.argpartition(score, topk)[:topk]
            stds[i] = score[topk_idxs].std()
        idxs = np.argsort(-stds)
        selection1[idxs[:vari_num]] = True
        index = np.argsort(-pred_score)  # 分数从大到小排序
        selection2[index[:new_nums_to_select]] = True
        selection = selection1 & selection2
        return selection,np.count_nonzero(selection)

    def select_top_data_NLVM(self, pred_score, nums_to_select, percent_P = 0.1, percent_N = 0.1):
        # pred_score = pred_score.T # if necessary
        N_u,N_l = pred_score.shape
        diam = pred_score.max()
        # 标记距离
        masks = np.zeros_like(pred_score, dtype='int32')
        masks[pred_score < diam * percent_P] = 1
        masks[pred_score > diam * (1-percent_N)] = -1
        stds = np.zeros(N_u)
        selection = np.zeros(N_u,'bool')
        # 计算P样本方差
        for i in range(N_u):
            score = pred_score[i] # 这个是距离
            mask = masks[i] == 1
            # print(score.std(),score[mask].std())
            if sum(mask) > 1:
                stds[i] = score[mask].std()
        # 根据方差排序
        idxs = np.argsort(-stds)
        # print(stds[idxs[:nums_to_select]])
        selection[idxs[:nums_to_select]] = True
        return selection

    def select_top_data_NLVM_2(self, pred_score, nums_to_select, percent_P = 0.1, percent_N = 0.1):
        # pred_score = pred_score.T # if necessary
        # 方案2, 求最近的P%样本的方差
        N_u,N_l = pred_score.shape
        stds = np.zeros(N_u)
        selection = np.zeros(N_u,'bool')
        # 求最近的P%样本的方差
        for i in range(N_u):
            score = pred_score[i]
            # 求k近邻
            topk = int(N_l * percent_P)
            topk_idxs = np.argpartition(score,topk)[:topk]
            stds[i] = score[topk_idxs].std()
        # 根据方差排序
        idxs = np.argsort(-stds)
        # print(stds[idxs[:nums_to_select]])
        selection[idxs[:nums_to_select]] = True
        return selection

    def select_top_data3(self, pred_score, nums_to_select,id_num,pred_y,u_data):
        total_number = 0
        for item in id_num:
            id_num[item] = round(id_num[item] * nums_to_select / len(u_data))  #向下取整/ 四舍五入
            total_number = total_number+id_num[item]

        print("nums_to_select vs total_number = {} vs {}".format(nums_to_select,total_number))
        v = np.zeros(len(pred_score))
        index = np.argsort(-pred_score)
        count = 0
        for i in range(len(pred_score)):
            if count == total_number:
                break
            if round(id_num[str(pred_y[i])]):
                v[index[i]] = 1
                count  = count+1
                id_num[str(pred_y[i])] = id_num[str(pred_y[i])]-1
        return v.astype('bool')




    def generate_new_train_data(self, sel_idx, pred_y):
        """ generate the next training data """

        seletcted_data = []
        correct, total = 0, 0
        for i, flag in enumerate(sel_idx):
            if flag: # if selected
                seletcted_data.append([self.u_data[i][0], int(pred_y[i]), self.u_data[i][2], self.u_data[i][3]])
                total += 1
                if self.u_label[i] == int(pred_y[i]):
                    correct += 1
        acc = correct / total

        new_train_data = self.l_data + seletcted_data
        print("selected pseudo-labeled data: {} of {} is correct, accuracy: {:0.4f}  new train data: {}".format(
                correct, len(seletcted_data), acc, len(new_train_data)))

        return new_train_data,acc

    def resume(self, ckpt_file, step):
        print("continued from step", step)
        model = models.create(self.model_name, dropout=self.dropout, num_classes=self.num_classes, mode=self.mode)
        self.model = nn.DataParallel(model).cuda()
        self.model.load_state_dict(load_checkpoint(ckpt_file))

    def evaluate(self, query, gallery):
        test_loader = self.get_dataloader(list(set(query) | set(gallery)), training = False)
        evaluator = Evaluator(self.model)
        return  evaluator.evaluate(test_loader, query, gallery)



"""
    Get one-shot split for the input dataset.
"""
def get_one_shot_in_cam1(dataset, load_path, seed=0):

    np.random.seed(seed)
    random.seed(seed)

    # if previous split exists, load it and return
    if osp.exists(load_path):
        with open(load_path, "rb") as fp:
            dataset = pickle.load(fp)
            label_dataset = dataset["label set"]
            unlabel_dataset = dataset["unlabel set"]

        print("  labeled  |   N/A | {:8d}".format(len(label_dataset)))
        print("  unlabel  |   N/A | {:8d}".format(len(unlabel_dataset)))
        print("\nLoad one-shot split from", load_path)
        return label_dataset, unlabel_dataset



    #print("random create new one-shot split and save it to", load_path)

    label_dataset = []
    unlabel_dataset = []

    # dataset indexed by [pid, cam]
    dataset_in_pid_cam = [[[] for _ in range(dataset.num_cams)] for _ in range(dataset.num_train_ids) ]
    for index, (images, pid, camid, videoid) in enumerate(dataset.train):
        dataset_in_pid_cam[pid][camid].append([images, pid, camid, videoid])


    # generate the labeled dataset by randomly selecting a tracklet from the first camera for each identity
    for pid, cams_data  in enumerate(dataset_in_pid_cam):
        for camid, videos in enumerate(cams_data):
            if len(videos) != 0:
                selected_video = random.choice(videos)
                break
        label_dataset.append(selected_video)
    assert len(label_dataset) == dataset.num_train_ids
    labeled_videoIDs =[vid for _, (_,_,_, vid) in enumerate(label_dataset)]

    # generate unlabeled set
    for (imgs, pid, camid, videoid) in dataset.train:
        if videoid not in labeled_videoIDs:
            unlabel_dataset.append([imgs, pid, camid, videoid])


    with open(load_path, "wb") as fp:
        pickle.dump({"label set":label_dataset, "unlabel set":unlabel_dataset}, fp)


    print("  labeled    | N/A | {:8d}".format(len(label_dataset)))
    print("  unlabeled  | N/A | {:8d}".format(len(unlabel_dataset)))
    print("\nCreate new one-shot split, and save it to", load_path)
    return label_dataset, unlabel_dataset

def get_one_shot_in_cam2(dataset, load_path, seed=0):

    np.random.seed(seed)
    random.seed(seed)

    # if previous split exists, load it and return
    if osp.exists(load_path):
        with open(load_path, "rb") as fp:
            dataset = pickle.load(fp)
            label_dataset = dataset["label set"]
            unlabel_dataset = dataset["unlabel set"]

        print("  labeled  |   N/A | {:8d}".format(len(label_dataset)))
        print("  unlabel  |   N/A | {:8d}".format(len(unlabel_dataset)))
        print("\nLoad one-shot split from", load_path)
        return label_dataset+unlabel_dataset,[]



    #print("random create new one-shot split and save it to", load_path)

    label_dataset = []
    unlabel_dataset = []

    # dataset indexed by [pid, cam]
    dataset_in_pid_cam = [[[] for _ in range(dataset.num_cams)] for _ in range(dataset.num_train_ids) ]
    for index, (images, pid, camid, videoid) in enumerate(dataset.train):
        dataset_in_pid_cam[pid][camid].append([images, pid, camid, videoid])


    # generate the labeled dataset by randomly selecting a tracklet from the first camera for each identity
    for pid, cams_data  in enumerate(dataset_in_pid_cam):
        for camid, videos in enumerate(cams_data):
            if len(videos) != 0:
                selected_video = random.choice(videos)
                break
        label_dataset.append(selected_video)
    assert len(label_dataset) == dataset.num_train_ids
    labeled_videoIDs =[vid for _, (_,_,_, vid) in enumerate(label_dataset)]

    # generate unlabeled set
    for (imgs, pid, camid, videoid) in dataset.train:
        if videoid not in labeled_videoIDs:
            unlabel_dataset.append([imgs, pid, camid, videoid])


    with open(load_path, "wb") as fp:
        pickle.dump({"label set":label_dataset, "unlabel set":unlabel_dataset}, fp)


    print("  labeled    | N/A | {:8d}".format(len(label_dataset)))
    print("  unlabeled  | N/A | {:8d}".format(len(unlabel_dataset)))
    print("\nCreate new one-shot split, and save it to", load_path)
    return label_dataset+unlabel_dataset,[]
