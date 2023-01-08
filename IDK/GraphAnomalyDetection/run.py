import numpy as np
from scipy.sparse import csc_matrix
from utility import load_graph
import argparse
from subgraph_centralization import subgraph_embeddings
from sklearn.metrics import roc_auc_score
from iNN_IK import iNN_IK


def generate_scores(scores, M, args):
    score_weight = [np.math.pow(args.lamda, i) for i in range(6)]
    en_scores = np.zeros_like(scores)
    tot = np.zeros_like(scores)
    for i in range(len(M)):
        for key, values in M[i].items():
            en_scores[key] += score_weight[values] * scores[i]
            tot[key] += score_weight[values]
    return np.divide(en_scores, tot)


def main(args):
    attr, adj, label = load_graph(args.dataset)
    embedding, M = subgraph_embeddings(attr, adj, args.h)
    kmembeddings = iNN_IK(args.psi, 100).fit_transform(embedding)
    mean_embedding = np.mean(kmembeddings, axis=0)
    scores = kmembeddings.dot(mean_embedding.transpose())
    final_scores = generate_scores(scores, M, args)
    print(
        f'dataset : {args.dataset}, auc = {roc_auc_score(label, -final_scores)}\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--psi', default=2, type=int)
    parser.add_argument('--dataset', default='cora')
    parser.add_argument('--h', default=1, type=int)
    parser.add_argument('--lamda', default=0.0625, type=float)
    args = parser.parse_args()
    main(args)
