
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from generate_embedding import *
from utilities import read_labels
from IGK_kernel_S import *
import time


def main():
    psi = 256
    t = 100
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='ENZYMES')
    parser.add_argument('--crossvalidation', default=True, action='store_true',
                        help='Enable a 10-fold crossvalidation')
    parser.add_argument('--h', type=int, required=False, default=5, help="(Max) number of WL iterations")

    args = parser.parse_args()
    dataset = args.dataset
    h = args.h
    data_path = os.path.join('./data', dataset)
    output_path = os.path.join('output', dataset)
    results_path = os.path.join('results', dataset)

    for path in [output_path, results_path]:
        if not os.path.exists(path):
            os.makedirs(path)

    label_file = os.path.join(data_path, 'Labels.txt')
    y = np.array(read_labels(label_file))

    cv = StratifiedKFold(n_splits=10, shuffle=True)

    accuracy_scores = []
    label_sequences = compute_wl_embeddings_continuous_by_IK(data_path, h, t, psi)
    label_sequences = np.array(label_sequences)
    for train_index, test_index in cv.split(label_sequences, y):
        X_train = label_sequences[train_index]
        X_test = label_sequences[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # use liblinear
        X_train = average_graph(X_train,t)
        X_test = average_graph(X_test,t)
        gs = LinearSVC().fit(X_train, y_train)
        y_pred = gs.predict(X_test)

        accuracy_scores.append(accuracy_score(y_test, y_pred))
        print(accuracy_scores)
        if not args.crossvalidation:
            break

    if args.crossvalidation:
        print('Mean 10-fold accuracy: {:2.2f} +- {:2.2f} %'.format(
            np.mean(accuracy_scores) * 100,
            np.std(accuracy_scores) * 100))
    else:
        print('Final accuracy: {:2.3f} %'.format(np.mean(accuracy_scores) * 100))


if __name__ == '__main__':
    main()

