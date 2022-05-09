import argparse
import numpy as np
from Model.model_multi import DiseasesPredictor
import time
import pickle


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.3)
    parser.add_argument('--bs', type=int, default=200)
    parser.add_argument('--n-epochs', type=int, default=8000)
    parser.add_argument('--f-dim', type=int, default=10000)
    parser.add_argument('--n-neighbor', type=int, default=5)
    parser.add_argument('--train-enc-num', type=int, default=1)
    parser.add_argument('--enc-dim', type=int, default=1000)
    return parser.parse_args()



file_path = "./data/sample_data/sample_garph"
node_list = pickle.load(open(file_path + ".nodes.pkl", "rb"))
adj_lists = pickle.load(open(file_path + ".adj.pkl", "rb"))
rare_patient = pickle.load(open(file_path + ".rare.label.pkl", "rb"))
labels = pickle.load(open(file_path + ".label.pkl", "rb"))
node_map = pickle.load(open(file_path + ".map.pkl", "rb"))
train = pickle.load(open(file_path + ".train.pkl", "rb"))
test = pickle.load(open(file_path + ".test.pkl", "rb"))

args = parse_args()
multi_class_num = 108
feature_dim = args.f_dim
epoch = args.n_epochs
batch_num = args.bs
lr = args.lr
feat_data = np.random.random((50000, feature_dim))
n_neighbor = args.n_neighbor
enc_dim = args.enc_dim
train_enc_num = args.trian_enc_num
train_enc_dim = [enc_dim, enc_dim, enc_dim, enc_dim]
t1 = time.time()
model = DiseasesPredictor(feat_data=feat_data,
                          b_labels=rare_patient,
                          multi_class_num=108,
                          labels=labels,
                          adj_lists=adj_lists,
                          feature_dim=feature_dim,
                          train_enc_num=train_enc_num,
                          train_enc_dim=train_enc_dim,
                          train_sample_num=[n_neighbor, n_neighbor, n_neighbor, n_neighbor],
                          train=train, test=test,
                          kernel='GCN',
                          topk=(1, 2, 3, 4, 5,))

model.run(epoch, batch_num, lr)  # epoch, batch_num, lr
print(feature_dim, train_enc_dim)
print("running time:", time.time()-t1, "s")
