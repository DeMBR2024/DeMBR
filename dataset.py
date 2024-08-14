import torch
import numpy as np
import os
import random
import scipy.sparse as sp
from torch.utils.data import Dataset

class TrainDataset(Dataset):
    '''
    Here relation include all relations
    '''
    def __init__(self,flags_obj):
        self.path = flags_obj.path
        self.name = flags_obj.dataset_name
        self.__decode_relation(flags_obj)
        self.__load_size()
        self.__create_relation_matrix()
        self.__calculate_user_behaviour_r()
        self.__calculate_user_behaviour_v()
        self.__generate_ground_truth()
        self.__generate_train_matrix()
        self.cnt = 0
        self.__read_train_data(self.cnt)

    def __decode_relation(self, flags_obj):
        self.relation = flags_obj.relation


    def __load_size(self):
        with open(os.path.join(self.path,self.name,'data_size.txt')) as f:
            data = f.readline()
            user_num, item_num = data.strip().split()
            self.user_num = int(user_num)
            self.item_num = int(item_num)

    def __load_item_graph(self):
        self.item_graph = {}
        self.item_graph_degree = {} 
        for tmp_relation in self.relation:
            self.item_graph[tmp_relation] = torch.load(os.path.join(self.path,self.name,'item\\'+'item_'+tmp_relation+'.pth'))
            self.item_graph_degree[tmp_relation] = self.item_graph[tmp_relation].sum(dim=1).float().unsqueeze(-1)

    def __create_relation_matrix(self):
        '''
        create a matrix for every relation
        '''
        self.relation_dict = {}

        for i in range(len(self.relation)):

            index = []
            with open(os.path.join(self.path,self.name,self.relation[i]+'.txt')) as f:
                data = f.readlines()
                for row in data:
                    user, item = row.strip().split()
                    user, item = int(user), int(item)
                    index.append([user,item])
            index_tensor = torch.LongTensor(index)
            lens, _ = index_tensor.shape
            self.relation_dict[self.relation[i]]=torch.sparse.FloatTensor(index_tensor.t(),torch.ones(lens,dtype=torch.float),torch.Size([self.user_num,self.item_num]))


    def __calculate_user_behaviour_r(self):
        self.user_behaviour_degree_r = {}
        self.item_behaviour_degree_r = {}
        for i in range(len(self.relation)):
            self.user_behaviour_degree_r[self.relation[i]]= self.relation_dict[self.relation[i]].to_dense().sum(dim=1).unsqueeze(-1)

            self.item_behaviour_degree_r[self.relation[i]] = self.relation_dict[self.relation[i]].to_dense().t().sum(dim=1).unsqueeze(-1)

    def __calculate_user_behaviour_v(self):
        self.user_behaviour_degree_v = {}
        self.item_behaviour_degree_v = {}
        for i in range(len(self.relation)):
            self.user_behaviour_degree_v[self.relation[i]] = torch.full((13358,1),13262.0,dtype=torch.float32)
            self.item_behaviour_degree_v[self.relation[i]] = torch.full((13262,1),13358.0,dtype=torch.float32)

    def __generate_ground_truth(self):
        '''
        use train data to build the ground truth matrix
        '''
        row_data = []
        col = []
        with open(os.path.join(self.path,self.name,'train.txt')) as f:
            data = f.readlines()
            for row in data:
                user, item = row.strip().split()
                user, item = int(user), int(item)
                row_data.append(user)
                col.append(item)
        row_data = np.array(row_data)
        col = np.array(col)
        values = np.ones(len(row_data),dtype=float)
        self.ground_truth = sp.csr_matrix((values,(row_data,col)),shape=(self.user_num,self.item_num))
        self.checkins = np.concatenate((row_data[:,None],col[:,None]),axis=1)

    def virtual_matrix(self):
        pr = torch.load('./pre_score.pt')
        index,value = torch.topk(pr,1000)
        self.vir_matrix = value


    def __generate_train_matrix(self):
        '''
        bring all relation together to a big matrix for GCN
        '''
        index = []
        with open(os.path.join(self.path,self.name,'train.txt')) as f:
            data = f.readlines()
            for row in data:
                user, item = row.strip().split()
                user, item = int(user), int(item)
                index.append([user,item])
        index_tensor = torch.LongTensor(index)
        lens, _ = index_tensor.shape
        self.train_matrix = torch.sparse.FloatTensor(index_tensor.t(),torch.ones(lens,dtype=torch.float),torch.Size([self.user_num,self.item_num]))


    def __read_train_data(self, i):
        tmp_array = []
        with open(os.path.join(self.path,self.name,'sample_file','sample_'+str(i)+'.txt')) as f:
            data = f.readlines()
            for row in data:
                user, pid, nid = row.strip().split()
                user, pid, nid = int(user), int(pid), int(nid)
                tmp_array.append([user, pid, nid])
        self.train_tmp=torch.LongTensor(tmp_array)

        print('Read Epoch{} Train Data over!'.format(i))

    def newit(self):
        self.cnt += 1
        self.__read_train_data(self.cnt)

    def __getitem__(self,index):
        return self.train_tmp[index,0].unsqueeze(-1), self.train_tmp[index,1:]

    def __len__(self):
        return len(self.checkins)

class TestDataset(Dataset):
    def __init__(self, flags_obj, trainset,task='test'):
        self.path = flags_obj.path
        self.name = flags_obj.dataset_name
        self.train_mask = trainset.ground_truth
        self.user_num, self.item_num = trainset.user_num, trainset.item_num
        self.task = task
        self.__read_testset()

    def __read_testset(self):
        row = []
        col = []
        with open(os.path.join(self.path, self.name, self.task+'.txt')) as f:
            data = f.readlines()
            for line in data:
                user, item = line.strip().split()
                user, item = int(user), int(item)
                row.append(user)
                col.append(item)
        row = np.array(row)
        col = np.array(col)
        values = np.ones(len(row),dtype=float)
        self.checkins = np.concatenate((row[:,None],col[:,None]),axis=1)
        self.ground_truth = sp.csr_matrix((values,(row,col)),shape=(self.user_num,self.item_num))

    def __getitem__(self, index):
        return index, torch.from_numpy(self.ground_truth[index].toarray()).squeeze(), torch.from_numpy(self.train_mask[index].toarray()).float().squeeze()

    def __len__(self):
        return self.user_num
