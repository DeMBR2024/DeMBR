import os
import torch
import torch.nn as nn
import random
import numpy as np
import scipy.sparse as sparse
import torch.nn.functional as F
from tqdm import tqdm
import pandas as pd




class model_base(nn.Module):
    def __init__(self, flags_obj, trainset, device):
        super().__init__()
        self.embed_size = flags_obj.embedding_size
        self.L2_norm = flags_obj.L2_norm
        self.device = device
        self.user_num = trainset.user_num
        self.item_num = trainset.item_num
        if flags_obj.create_embeddings=='True':
            self.item_embedding = nn.Parameter(torch.FloatTensor(self.item_num,self.embed_size))
            nn.init.xavier_normal_(self.item_embedding)
            self.user_embedding = nn.Parameter(torch.FloatTensor(self.user_num,self.embed_size))
            nn.init.xavier_normal_(self.user_embedding)
            self.user_embedding_click_add = nn.Parameter(torch.FloatTensor(self.user_num,self.embed_size))
            nn.init.xavier_normal_(self.user_embedding_click_add)
            self.user_embedding_collect_add = nn.Parameter(torch.FloatTensor(self.user_num, self.embed_size))
            nn.init.xavier_normal_(self.user_embedding_collect_add)
            self.user_embedding_cart_add = nn.Parameter(torch.FloatTensor(self.user_num, self.embed_size))
            nn.init.xavier_normal_(self.user_embedding_cart_add)

    def propagate(self,*args,**kwargs):
        '''
        raw embeddings -> embeddings for predicting
        return (user's,POI's)
        '''
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        '''
        embeddings of targets for predicting -> scores
        return scores
        '''
        raise NotImplementedError

    def regularize(self, user_embeddings, item_embeddings):
        '''
        embeddings of targets for predicting -> extra loss(default: L2 loss...)
        '''
        return self.L2_norm*((user_embeddings**2).sum()+(item_embeddings**2).sum())

    def forward(self, users, items):
        users_feature, item_feature = self.propagate()
        item_embeddings = item_feature[items]
        user_embeddings = users_feature[users].expand(-1,items.shape[1],-1)
        pred = self.predict(user_embeddings, item_embeddings)
        L2_loss = self.regularize(user_embeddings, item_embeddings)
        return pred, L2_loss

    def evaluate(self, users):
        '''
        just for testing, compute scores of all POIs for `users` by `propagate_result`
        '''
        raise NotImplementedError

class MF(model_base):

    def __init__(self, flags_obj, trainset, device):
        super().__init__(flags_obj, trainset, device)

    def propagate(self, task='train'):
        return self.user_embedding, self.item_embedding

    def predict(self, user_embedding, item_embedding):

        return torch.sum(user_embedding*item_embedding,dim=2)

    def evaluate(self, propagate_result, users,i):

        users_feature, item_feature = propagate_result
        if i==0:
            scores_a = torch.mm(users_feature,item_feature.t())
            torch.save(scores_a, './pre_score.pt')
        user_feature = users_feature[users]
        scores = torch.mm(user_feature, item_feature.t())


        return scores

class M_LightGCN(nn.Module):
    def __init__(self, flags_obj, trainset, device, matrix, flag):
        super(M_LightGCN,self).__init__()
        self.device = device
        self.num_users = trainset.user_num
        self.num_items = trainset.item_num
        self.behavior = flags_obj.relation
        self.flag = flag
        self.mat = matrix
        if self.flag == 'real':
            self.layers = flags_obj.Layers
            self.relation_dict = trainset.relation_dict
            self.item_behaviour_degree = trainset.item_behaviour_degree_r
            self.user_behaviour_degree = trainset.user_behaviour_degree_r
        else:
            self.layers = flags_obj.Layers
            self.relation_dict = self.mat
            self.item_behaviour_degree = trainset.item_behaviour_degree_v
            self.user_behaviour_degree = trainset.user_behaviour_degree_v
    def updata_degree(self,mat):

        if self.flag == 'real':
            for i in range(0,len(mat)):
                self.user_behaviour_degree[self.behavior[i]] = mat[self.behavior[i]].sum(dim=1).unsqueeze(-1)
                self.item_behaviour_degree[self.behavior[i]] = mat[self.behavior[i]].t().sum(dim=1).unsqueeze(-1)

    def compute(self, user_embedding, item_embedding):
        user_behavior_emb = {}
        item_behavior_emb = {}
        if self.flag != 'real':
            behavior = self.behavior[:-1]
        else:
            behavior = self.behavior
        for type in behavior:
            user_behavior = []
            item_behavior = []
            for i in range(self.layers):
                if i == 0:
                    user_emb = torch.mm(self.relation_dict[type].float().to(self.device), item_embedding.to(self.device)) / (
                                self.user_behaviour_degree[type].to(self.device) + 1e-8)
                    item_emb = torch.mm((self.relation_dict[type].float().t().to(self.device)), user_embedding.to(self.device)) / (
                                self.item_behaviour_degree[type].to(self.device) + 1e-8)
                    user_behavior.append(user_emb)
                    item_behavior.append(item_emb)
                else:
                    user_emb_m = user_emb
                    item_emb_m = item_emb
                    user_emb = torch.mm(self.relation_dict[type].float().to(self.device), item_emb_m) / (
                                self.user_behaviour_degree[type].to(self.device) + 1e-8)
                    item_emb = torch.mm(self.relation_dict[type].float().t().to(self.device), user_emb_m) / (
                                self.item_behaviour_degree[type].to(self.device) + 1e-8)
                    user_behavior.append(user_emb)
                    item_behavior.append(item_emb)

            user_emb = torch.mean(torch.stack(user_behavior, dim=1), dim=1)
            item_emb = torch.mean(torch.stack(item_behavior, dim=1), dim=1)
            user_behavior_emb[type] = user_emb
            item_behavior_emb[type] = item_emb
        return user_behavior_emb, item_behavior_emb

    def forward(self, user_embedding, item_embedding):
        return self.compute(user_embedding, item_embedding)




class DeMBR(model_base):
    def __init__(self, flags_obj, trainset, device):
        super().__init__(flags_obj, trainset, device)
        self.relation_dict = trainset.relation_dict
        self.relation_dict2 = trainset.relation_dict
        self.mgnn_weight = flags_obj.mgnn_weight
        self.train_matrix = trainset.train_matrix.to(self.device)
        self.relation = trainset.relation
        self.lamb = flags_obj.lamb
        self.message_drop = nn.Dropout(p=flags_obj.message_dropout)
        self.train_node_drop = nn.Dropout(p=flags_obj.node_dropout)
        self.node_drop = nn.ModuleList([nn.Dropout(p=flags_obj.node_dropout) for _ in self.relation_dict])
        self.__to_gpu()
        self.matrix = nn.ParameterDict({'click': nn.Parameter(torch.ones(self.user_num, self.item_num)),'favorite': nn.Parameter(torch.ones(self.user_num, self.item_num)),'cart': nn.Parameter(torch.ones(self.user_num, self.item_num))})
        self.mlightgcn_r = M_LightGCN(flags_obj, trainset, device,self.matrix,'real').to(device)
        self.mlightgcn_v = M_LightGCN(flags_obj, trainset, device,self.matrix,'virtually').to(device)
        self.beta =  torch.tensor(random.random(), dtype=torch.float64, requires_grad=True)
        self.gama =  torch.tensor(random.random(), dtype=torch.float64, requires_grad=True)
        self.weight_user_r_dic = nn.ParameterDict({'click': nn.Parameter(torch.tensor(1.0)),'favorite': nn.Parameter(torch.tensor(1.0)),'cart': nn.Parameter(torch.tensor(1.0)),'buy': nn.Parameter(torch.tensor(1.0))})
        self.weight_item_r_dic = nn.ParameterDict({'click': nn.Parameter(torch.tensor(1.0)),'favorite': nn.Parameter(torch.tensor(1.0)),'cart': nn.Parameter(torch.tensor(1.0)),'buy': nn.Parameter(torch.tensor(1.0))})
        self.weight_user_v_dic = nn.ParameterDict({'click': nn.Parameter(torch.tensor(1.0)),'favorite': nn.Parameter(torch.tensor(1.0)),'cart': nn.Parameter(torch.tensor(1.0))})
        self.weight_item_v_dic = nn.ParameterDict({'click': nn.Parameter(torch.tensor(1.0)),'favorite': nn.Parameter(torch.tensor(1.0)),'cart': nn.Parameter(torch.tensor(1.0))})

    def __to_gpu(self):
        for key in self.relation_dict:
            self.relation_dict[key] = self.relation_dict[key].to(self.device)

    def __decode_weight(self):
        weight = nn.softmax(self.mgnn_weight).unsqueeze(-1)
        total_weight = torch.mm(self.user_behaviour_degree, weight)
        self.user_behaviour_weight = self.user_behaviour_degree.float() / (total_weight + 1e-8)

    def __param_init(self):

        self.mgnn_weight = nn.Parameter(torch.FloatTensor(self.mgnn_weight))
        self.item_behaviour_W = nn.ParameterList([nn.Parameter(torch.FloatTensor(self.embed_size * 2, self.embed_size * 2)) for _ in self.mgnn_weight])
        for param in self.item_behaviour_W:
            nn.init.xavier_normal_(param)
        self.item_propagate_W = nn.ParameterList([nn.Parameter(torch.FloatTensor(self.embed_size, self.embed_size)) for _ in self.mgnn_weight])
        for param in self.item_propagate_W:
            nn.init.xavier_normal_(param)
        self.W = nn.Parameter(torch.FloatTensor(self.embed_size, self.embed_size))
        nn.init.xavier_normal_(self.W)

    def forward(self, user, item ,batch):
        user_dic_r , item_dic_r = self.mlightgcn_r(self.user_embedding,self.item_embedding)
        user_dic_v, item_dic_v = self.mlightgcn_v(self.user_embedding,self.item_embedding)
        v_embed_dic={}
        v_embed_dic['click'] = user_dic_v['click']+self.user_embedding_click_add
        v_embed_dic['favorite'] = user_dic_v['favorite']+self.user_embedding_collect_add
        v_embed_dic['cart'] = user_dic_v['cart']+self.user_embedding_cart_add

        if batch % 5 == 0 and batch != 0:
            for i,type in enumerate(self.relation[0:-1]):

                u_emb =F.normalize(user_dic_r[type]+user_dic_v[type],p=2,dim=1)
                it_emb = F.normalize(item_dic_r[type]+item_dic_v[type],p=2,dim=1)
                self.matrix[type] = nn.Parameter((self.matrix[type]+torch.mm(u_emb,it_emb.t()))/2)

        loss_s = 0
        for type in self.relation[0:-1]:
            user_emb_sup = user_dic_r[type]+user_dic_r['buy']
            loss_s+=self.pro_loss(v_embed_dic[type],user_emb_sup)

        veter_lis = []
        flag = 0
        for type in self.relation[0:-1]:
            flag += 1
            for key in self.relation[flag:4]:
                user_emb_sup = user_dic_r[key] - user_dic_r[type]
                veter_lis.append(user_emb_sup)

        for mater in veter_lis:
            loss_s += self.pro_loss_2(mater)


        for i,type in enumerate(self.relation):
            if i == 0:
                Finall_user_embedding_r = self.weight_user_r_dic[type]*user_dic_r[type]
                Finall_item_embedding_r = self.weight_item_r_dic[type]*item_dic_r[type]
            else:
                Finall_item_embedding_r += self.weight_user_r_dic[type]*item_dic_r[type]
                Finall_user_embedding_r += self.weight_item_r_dic[type]*user_dic_r[type]

        Finall_item_embedding_r = self.message_drop(Finall_item_embedding_r)

        for i,type in enumerate(self.relation[:-1]):
            if i == 0:
                Finall_user_embedding_v = self.weight_user_v_dic[type]*v_embed_dic[type]
                Finall_item_embedding_v = self.weight_item_v_dic[type]*item_dic_v[type]
            else:
                Finall_user_embedding_v += self.weight_user_v_dic[type]*v_embed_dic[type]
                Finall_item_embedding_v += self.weight_item_v_dic[type]*item_dic_v[type]

        Finall_item_embedding_v = self.message_drop(Finall_item_embedding_v)


        batch_finall_user_emb_v = self.message_drop(F.normalize(Finall_user_embedding_v[user].squeeze(),p=2,dim=1))
        batch_finall_item_emb_v_p = F.normalize(Finall_item_embedding_v[item[:,0]].squeeze(),p=2,dim=1)
        batch_finall_item_emb_v_n = F.normalize(Finall_item_embedding_v[item[:,1]].squeeze(),p=2,dim=1)
        batch_finall_user_emb_r = self.message_drop(F.normalize(Finall_user_embedding_r[user].squeeze(),p=2,dim=1))
        batch_finall_item_emb_r_p = F.normalize(Finall_item_embedding_r[item[:,0]].squeeze(),p=2,dim=1)
        batch_finall_item_emb_r_n =F.normalize( Finall_item_embedding_r[item[:,1]].squeeze(),p=2,dim=1)



        score_v_p = torch.sum(batch_finall_user_emb_v*batch_finall_item_emb_v_p,dim=1).unsqueeze(1)
        score_v_n = torch.sum(batch_finall_user_emb_v*batch_finall_item_emb_v_n,dim=1).unsqueeze(1)
        score_r_p = torch.sum(batch_finall_user_emb_r*batch_finall_item_emb_r_p,dim=1).unsqueeze(1)
        score_r_n = torch.sum(batch_finall_user_emb_r*batch_finall_item_emb_r_n,dim=1).unsqueeze(1)

        score_v = torch.cat((score_v_p,score_v_n),dim=1)
        score_r =torch.cat((score_r_p,score_r_n),dim=1)

        score = 0.8* score_r + 0.2* score_v

        loss =self.gama*loss_s

        return score,loss

    def pro_loss(self,uesr_emb_p,user_emb):
        user_emb_p_n = F.normalize(uesr_emb_p,p=2,dim=1)
        user_emb_n  = F.normalize( user_emb , p=2,dim=1)
        sim_matrix = torch.mm(user_emb_p_n,user_emb_n.T)
        positive = torch.sigmoid(torch.diag(sim_matrix).unsqueeze(1))
        loss = torch.sum(-torch.log(positive))
        return loss

    def pro_loss_2(self, mater):
        return torch.sum(torch.pow(mater, 2))

    def add_noise(self):
        new_dic = { }
        count = 0
        for type in self.relation[:1]:
            mat = self.relation_dict2[type].to_dense()
            for i in tqdm(range(self.user_num)):
                try:
                    index_r = list(torch.nonzero(mat[i]==1).squeeze(dim=1))
                    index_r = torch.tensor(index_r).int().tolist()
                    num = int(len(index_r)*0.05)
                    index_add = list(torch.nonzero(mat[i]==0).squeeze(dim=1))
                    index_add = torch.tensor(index_add).int().tolist()
                    add = random.sample(index_add,num)
                    count += len(add)
                    mat[i, add] = torch.tensor(1, dtype=torch.float)
                except:

                    a=0
            new_dic[type] = mat
            nonzero_indices = torch.nonzero(mat)
            nonzero_values = mat[nonzero_indices[:, 0], nonzero_indices[:, 1]]
            self.relation_dict[type] = torch.sparse.FloatTensor(nonzero_indices.t(), nonzero_values, mat.size())
        self.mlightgcn_r.updata_degree(new_dic)

    def updata_interact_matric(self,loss_f,loss_n,matrix_f):
        print('begin memory pruning')
        new_dic={ }
        threshold = 0.3
        for type in self.relation[:-1]:
            count_chan = 0
            mat = self.relation_dict[type].to_dense()
            mask = torch.ones(self.user_num,self.item_num)
            for i in tqdm(range(self.user_num)):
                try:
                    index_r = list(torch.nonzero(mat[i]==1).squeeze(dim=1))
                    index_r = torch.tensor(index_r).int().tolist()
                    index_v = torch.lt(self.matrix[type][i],threshold)
                    index_v = list(torch.nonzero(index_v).squeeze(dim=1))
                    index_v = torch.tensor(index_v).int().tolist()
                    intersection = list(set(index_v) & set(index_r))
                    count_chan+=len(intersection)
                    mask[i, intersection] = torch.tensor(0, dtype=torch.float)

                except:
                    num1 = 0
            mat = torch.mul(mat,mask)
            new_dic[type] = mat
            print(type,'del number：',count_chan)


            nonzero_indices = torch.nonzero(mat)
            nonzero_values = mat[nonzero_indices[:, 0], nonzero_indices[:, 1]]
            self.relation_dict[type] = torch.sparse.FloatTensor(nonzero_indices.t(), nonzero_values, mat.size())
        self.mlightgcn_r.updata_degree(new_dic)

    def get_interact_v(self):
        return self.matrix

    def evaluate(self, user):
        user_dic_r, item_dic_r = self.mlightgcn_r(self.user_embedding, self.item_embedding)
        user_dic_v, item_dic_v = self.mlightgcn_v(self.user_embedding, self.item_embedding)
        v_embed_dic = {}
        v_embed_dic['click'] = user_dic_v['click'] + self.user_embedding_click_add
        v_embed_dic['favorite'] = user_dic_v['favorite'] + self.user_embedding_collect_add
        v_embed_dic['cart'] = user_dic_v['cart'] + self.user_embedding_cart_add

        for i, type in enumerate(self.relation):
            if i == 0:
                Finall_user_embedding_r = self.weight_user_r_dic[type] * user_dic_r[type]
                Finall_item_embedding_r = self.weight_item_r_dic[type] * item_dic_r[type]
            else:
                Finall_item_embedding_r += self.weight_user_r_dic[type] * item_dic_r[type]
                Finall_user_embedding_r += self.weight_item_r_dic[type] * user_dic_r[type]

        for i, type in enumerate(self.relation[:-1]):
            if i == 0:
                Finall_user_embedding_v = self.weight_user_v_dic[type] * v_embed_dic[type]
                Finall_item_embedding_v = self.weight_item_v_dic[type] * item_dic_v[type]
            else:
                Finall_user_embedding_v += self.weight_user_v_dic[type] * v_embed_dic[type]
                Finall_item_embedding_v += self.weight_item_v_dic[type] * item_dic_v[type]

        batch_finall_user_emb_v =F.normalize( Finall_user_embedding_v[user].squeeze(),p=2,dim=1)
        batch_finall_item_emb_v = F.normalize( Finall_item_embedding_v.squeeze(),p=2,dim=1)
        batch_finall_user_emb_r = F.normalize( Finall_user_embedding_r[user].squeeze(),p=2,dim=1)
        batch_finall_item_emb_r =F.normalize(  Finall_item_embedding_r.squeeze(),p=2,dim=1)


        score_v = torch.mm(batch_finall_user_emb_v,batch_finall_item_emb_v.t())
        score_r = torch.mm(batch_finall_user_emb_r,batch_finall_item_emb_r.t())

        scores = 0.8 * score_r + 0.2 * score_v


        return scores