from absl import app
from absl import flags

from train import TrainManager
from utils import VisManager, ContentManager
from torch.utils.data import DataLoader

FLAGS = flags.FLAGS

flags.DEFINE_enum('model','DeMBR',['MF','DeMBR'],'Model name')
flags.DEFINE_string('name','MF-experiment','Experiment name')
flags.DEFINE_string('dataset_name','Taobao',"Dataset's name")
flags.DEFINE_float('lr',0.0001,'Learning rate')
flags.DEFINE_float('L2_norm',0.0001,'L2 norm')
flags.DEFINE_bool('gpu','False','Use GPU or not')
flags.DEFINE_integer('gpu_id',0,'GPU ID')
flags.DEFINE_integer('num_workers', 0, 'Number of processes for training and testing')
flags.DEFINE_integer('epoch', 150, 'Max epochs for training')
flags.DEFINE_string('path','D:\DeMBR\DeMBR\TheCode\DeMBR-main','The path where the data is')
flags.DEFINE_string('output','D:\DeMBR\DeMBR\TheCode\DeMBR-main\output','The path to store output message')
flags.DEFINE_integer('port',33337,'Port to show visualization results for visdom')
flags.DEFINE_integer('batch_size',4096,'Batch size')
flags.DEFINE_integer('test_batch_size',1024,'Test batch size')
flags.DEFINE_integer('Layers',3,'The layers of LightGCN')
flags.DEFINE_integer('embedding_size',512,'Embedding Size')
flags.DEFINE_integer('es_patience',10,'Early Stop Patience')
flags.DEFINE_integer('epoch_interval',5,'Early Stop Patience')
flags.DEFINE_enum('loss_mode','mean',['mean','sum'],'Loss Mode')
flags.DEFINE_multi_string('relation', ['click','favorite','cart','buy'], 'Relations')
flags.DEFINE_bool('pretrain_frozen','False','Froze the pretrain parameter or not')
flags.DEFINE_string('create_embeddings','True','Pretrain or not? If not create embedding here!')
flags.DEFINE_string('pretrain_path','D:\TheCode\DeMBR-main\output\Taobao\MF-experiment',"Path where the pretrain model is.")
flags.DEFINE_float('node_dropout',0.2,'Node dropout ratio')
flags.DEFINE_float('message_dropout',0.2,'Message dropout ratio')
flags.DEFINE_float('lamb',0.1,'Lambda for the loss for MultiGNN with item space calculation')
flags.DEFINE_multi_float('mgnn_weight',[1,1,1,1],'Weight for MGNN')

def main(argv):

    flags_obj = FLAGS
    vm = VisManager(flags_obj)
    cm = ContentManager(flags_obj)
    Train = TrainManager(flags_obj, vm, cm)

    Train.train()

if __name__=='__main__':

    app.run(main)
