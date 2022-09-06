from util import *

def main():
    data_dir = "./Data/NS.txt"
    data_name = "NS"
    test_data_ratio = 0.1
    connected = True
    h = 2
    include_embedding = 1
    include_attribute = 1

    FormNet(data_dir)

    data_mat = np.load(str(data_dir)[:-4] + ".npy")
    train_mat, test_mat = train_test_split(data_mat, test_data_ratio, connected)

    train_pos, train_neg, test_pos, test_neg = sample_negatives(train_mat, test_mat, k=1, evaluate_on_all_unseen=False)
    test = {"pos": test_pos,
            "neg": test_neg}

    train_mix = {"train": train_mat, # the observed network to extract enclosing subgraphs
                 "data_name": data_name, # store the name of the current data set
                 "pos": train_pos,
                 "neg": train_neg # train pos and neg links (used by learning based methods, not by heuristic methods)
                 }
    SEAL_data_prepration(train_mix, test, h, include_embedding, include_attribute)

    print('Data is ready to feed to GNN!')
if __name__ == "__main__":
    main()