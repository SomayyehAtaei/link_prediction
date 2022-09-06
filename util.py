import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as ssp
from scipy.stats import zscore
from sklearn.metrics import roc_curve
from node2vec import Node2Vec

########################################################################################################################
def FormNet(data_dir):
    data = np.loadtxt(data_dir)
    data = np.pad(data[:,:2], (0, 1), 'constant', constant_values=(1))
    m = int(np.max(data[:, :2]))

    i, j, value = data.T
    i = i.astype(int)
    j = j.astype(int)
    data_pos = (i, j)
    data_mat = coo_matrix((value, (i, j)), shape=(m+1, m+1)).toarray()
    np.fill_diagonal(data_mat, 0)
    np.save(str(data_dir)[:-4], data_mat)

########################################################################################################################
def train_test_split(data_mat, test_data_ratio, connected):
    train_data_mat = data_mat.copy()
    train_data_pos = get_data_pos(train_data_mat)
    edges_count = len(train_data_pos[0])
    test_edges_count = int(test_data_ratio * edges_count)
    test_data_mat = np.zeros(train_data_mat.shape)

    train_data_pos = list(train_data_pos)
    while len(np.nonzero(test_data_mat)[0]) < test_edges_count:
        flag = 0
        edges_count = len(train_data_pos[0])
        random_edge = np.random.randint(edges_count)
        v1 = train_data_pos[0][random_edge]
        v2 = train_data_pos[1][random_edge]
        # #Determine whether the nodes uid1 and uid2 at both ends of the selected edge are reachable, if they are reachable,
        # # they can be put into the test set, otherwise select an edge again
        train_data_mat[v1, v2] = 0
        # Dig this edge from the network to determine whether the network is still connected after being dug
        # Take out the point that uid1 can reach in one step and construct it into a one-dimensional vector
        # Mark whether this side can be removed, sign = 0 Means not possible; sign = 1 Means yes
        v = train_data_mat[v1, :]
        # v1_v2 represents the reachable point within two steps
        v1_v2 = np.matmul(v, train_data_mat) + v

        if v1_v2[v2] > 0:
            flag = 1
            #Just two steps
        else:
            v1_v2[v1_v2.nonzero()] = 1

            while len(np.nonzero(v1_v2 - v)[0]) != 0:
                # Until the reachable point reaches a stable state, uid2 still cannot be reached, and this side cannot be deleted
                v = v1_v2
                v[v.nonzero()] = 1
                v1_v2 = np.matmul(v, train_data_mat) + v
                v1_v2[v1_v2.nonzero()] = 1

                # The v1_v2 of this step represents the reachable point in step K
                if v1_v2[v2] > 0:
                    flag = 1
                    # Reachable within a certain step
                    break

        if connected == False:
            flag = 1 # overwrite, keep all selected links in test, no matter whether the remaining net is connected

        # If this edge can be deleted, put it into the test set and remove this edge from the linklist
        if flag == 1: # This side can be deleted
            test_data_mat[train_data_pos[0][random_edge], train_data_pos[1][random_edge]] = 1
            np.delete(train_data_pos[0], random_edge)
            np.delete(train_data_pos[1], random_edge)

        else:
            train_data_mat[train_data_pos[0][random_edge], train_data_pos[1][random_edge]] = 1
            np.delete(train_data_pos[0], random_edge)
            np.delete(train_data_pos[1], random_edge)

    return train_data_mat, test_data_mat

########################################################################################################################
def sample_negatives(train_data, test_data, k=1, evaluate_on_all_unseen=False):
    ''' Usage: to sample negative links for train and test datasets.
        When sampling negative train links, assume all testing links are known and thus sample negative train links only
        from other unknown links.
        Set evaluate_on_all_unseen to true to do link prediction on all links unseen during training.
        -- Input --
        --train: half train positive adjacency matrix
        --test: half test positive adjacency matrix
        --k: how many times of negative links (w.r.t.pos links) to sample
        --evaluate_on_all_unseen: if true, will not randomly sample negative testing links, but regard all links unseen during
          training as neg testing links; train negative links are sampled in the original way
        -- Output --
        --column indices for four datasets'''

    n = train_data.shape[0]

    i, j, _ = ssp.find(train_data)
    train_pos = (i, j)
    i, j, _ = ssp.find(test_data)
    test_pos = (i, j)

    train_size = len(train_pos[0])
    test_size = len(test_pos[0])

    if test_size == 0:
        data = train_data
    else:
        data = train_data + test_data

    assert np.max(data) == 1 # ensure positive train, test not overlap
    neg_data = np.triu(-(data - 1), 1)
    neg_data_i, neg_data_j, _ = ssp.find(neg_data)
    neg_links = (i, j)

    # sample negative links
    if evaluate_on_all_unseen:
        # first let all unknown links be negative test links.
        test_neg = neg_links

        # randomly select train neg from all unknown links
        perm = np.random.randperm(len(neg_links)[0])
        train_neg = neg_links[perm[1: k * train_size], :]

        # remove train negative links from test negative links
        test_neg[perm[1: k * train_size], :] = -1
        test_neg.remove(-1)

    else:
        nlinks = len(neg_links[0])
        ind = np.random.permutation(nlinks)

        if k * (train_size + test_size) <= nlinks:
            train_ind = ind[1: k * train_size]
            test_ind = ind[k * train_size + 1: k * train_size + k * test_size]

        else: # if negative links not enough, divide them proportionally
            ratio = train_size / (train_size + test_size)
            train_ind = ind[: int(np.floor(ratio * nlinks))]
            test_ind = ind[int(np.floor(ratio * nlinks))+1: ]

        train_neg_i = [neg_links[0][i] for i in train_ind]
        train_neg_j = [neg_links[1][i] for i in train_ind]
        train_neg = (train_neg_i, train_neg_j)
        test_neg_i = [neg_links[0][i] for i in test_ind]
        test_neg_j = [neg_links[1][i] for i in test_ind]
        test_neg = (test_neg_i, test_neg_j)

    return train_pos, train_neg, test_pos, test_neg

########################################################################################################################
def get_data_pos(data_mat):
    i, j = np.nonzero(data_mat)
    data_pos = (i, j)
    return data_pos

########################################################################################################################
def SEAL_data_prepration(train_mix, test, h=1, include_embedding=0, include_attribute=0):
    ''' Usage: the main program of SEAL (learning from Subgraphs, Embeddings, and Attributes for Link prediction)
        --Input--
        --train_mix: a struct where train_mix.pos contains indices[(i1, j1); (i2, j2); ...] of positive train links,
            train_mix.neg contains indices of negative train links,
            train_mix.train is a sparse adjacency matrix of observed network (1: link, 0: otherwise),
            train_mix.data_name is dataset name
        --test: a struct where
            test.pos contains indices of positive test links, and
            test.neg contains indices of negative test links
        --h: maximum hop to extract enclosing subgraphs
        --include_embedding: 1 to include node embeddings into node information matrix, default 0
        --include_attribute: 1 to include node attributes into node information matrix, default 0
        --ith_experiment: exp index, for parallel computing, default 1
        --Output--
        --auc: the AUC score on testing links'''

    A = train_mix['train'] # the observed network
    data_name = train_mix['data_name']
    train_pos = train_mix['pos']
    train_neg = train_mix['neg'] # the indices of observed links used as training data
    test_pos = test['pos']
    test_neg = test['neg'] # the indices of unobserved links used as testing data

    train_size = len(train_pos[0]) + len(train_neg[0])
    test_size = len(test_pos[0]) + len(test_neg[0])

    # extract enclosing subgraphs
    data, max_size = graph2mat(np.hstack((train_pos, train_neg)), np.hstack((test_pos, test_neg)), A, h, data_name, include_embedding, include_attribute)
    # graph labels (classes), not to confuse with node labels
    label = np.vstack((np.ones((np.array(train_pos).shape[1], 1)),
                       np.zeros((np.array(train_neg).shape[1], 1)),
                       np.ones((np.array(test_pos).shape[1], 1)),
                       np.zeros((np.array(test_neg).shape[1], 1))))

    # permutate the train set
    perm = np.random.permutation(train_size)
    data[0: train_size] = [data[i] for i in perm]
    label[0: train_size] = [label[i] for i in perm]

    for i in range(len(data)):
        data[i]['am'] = data[i]['am'].astype(int)
        data[i]['nl'] = data[i]['nl'].astype(np.float16)

    np.save('./Data/tempdata/' + data_name, data)
    np.save('./Data/tempdata/' + data_name + '_label', label)

########################################################################################################################
def sub2ind(array_shape, rows, cols):
    return rows * array_shape[1] + cols

########################################################################################################################
def CalcAUC(train, test, sim):
    # calculate auc on given testing links for heurisitc - based methods
    # test is a struct, test.pos contains the positive testing links
    # sim is the matrix of similarity scores
    # train is legacy variable (useless)

    test_pos = test.pos
    test_neg = test.neg
    test_pos = sub2ind(sim.shape, test_pos[:, 0], test_pos[:, 1])
    test_neg = sub2ind(sim.shape, test_neg[:, 0], test_neg[:, 1])

    pos_pre = sim[test_pos]
    neg_pre = sim[test_neg]

    labels = [np.ones(1, pos_pre.shape[0]), np.zeros(1, neg_pre.shape[0])]
    scores = [np.transpose(pos_pre), np.transpose(neg_pre)]
    _, _, auc = roc_curve(labels, scores, 1)
    xs, ys, aucpr = roc_curve(labels, scores, 1, 'XCrit', 'reca', 'YCrit', 'prec')
    precision = sum(np.diff(xs) * ys[1: ]) # average precision
    return auc, precision

########################################################################################################################
def graph2mat(train_ind, test_ind, A, h, data_name, include_embedding=0, include_attribute=0):
    # Usage: to extract links' enclosing subgraphs saved as adjacency matrices, and generate their node information matrices
    # (if include_embedding == 1 or include_attribute == 1, node information matrix will contain one-hot encoding
    # of node label + node embedding vector + node attribute vector; otherwise will only contain "integer" node labels; used by SEAL
    # --Input--
    # --train: indices of training links
    # --test: indices of testing links
    # --A: the observed network's adjacency matrix from which to extract enclosing subgraphs
    # --h: the maximum hop to extract enclosing subgraphs
    # --for_graph_kernel: if 1, node adajacency lists will be stored (needed by some graph kernels)
    # --data_name: the name of the dataset
    # --include_embedding: if 1, node embeddings are included
    # --include_attribute: if 1, node attributes are included
    # --Output --
    # --data: a collection of graphs in the WL kernel format
    # --max_size: either 1) maximum node label (if no embedding or attribute are included, but only node label) or
    # 2) length of one - hot encoding of node label + node embedding + node attribute

    all = np.hstack((train_ind, test_ind))
    train_size = train_ind.shape[1]
    test_size = test_ind.shape[1]
    all_size = train_size + test_size

    data = [{} for i in range(all_size)]

    # generate node embeddings
    node_information = []
    emd_method = 'node2vec'
    if include_embedding == 1:
        # negative injection to avoid overfitting
        A1 = A
        train_neg = train_ind[:, int(train_ind.shape[1] / 2 + 1):]
        train_neg = np.hstack((train_neg, [train_neg[1, :], train_neg[0, :]]))
        A1[train_neg[0], train_neg[1]] = 1

        # generate the node embeddings from the injected network
        node_embeddings = generate_embeddings(A1, data_name, emd_method)

        # whether to include some global node centrality features (increase performance on some networks)
        # add some global node features into the embeddings
        G = nx.Graph(A1)
        deg = [i[1] for i in G.degree]
        closeness = nx.closeness_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        pr = nx.pagerank(G)
        eig = nx.eigenvector_centrality(G)
        node_centralities = np.vstack((deg, list(closeness.values()), list(betweenness.values()), list(pr.values()), list(eig.values())))
        node_centralities = zscore(node_centralities) # standardization
        node_embeddings = np.hstack((node_embeddings.vectors, np.transpose(node_centralities)))

        node_information = node_embeddings

    # load node attributes
    if include_attribute == 1:
        group = A
        # use node classes as node attributes, assume data_name.mat contains a variable 'group' that stores the attributes
        node_attributes = ssp.coo_matrix(group).toarray()
        node_information = np.hstack((node_attributes, node_information))

    # now begin enclosing subgraph extraction
    print('Subgraph Extraction Begins...')

    for i in range(all_size):
        ind = all[:, i]
        if include_embedding == 1 or include_attribute == 1:
            sample, max_nl_size = subgraph2mat(ind, A, h, node_information)
        else:
            sample, max_nl_size = subgraph2mat(ind, A, h)

        data[i]['am'] = sample['am']
        data[i]['nl'] = sample['nl']

    # convert integer node labels to one - hot embeddings
    number_of_nodes = [data[i]['am'].shape[0] for i in range(len(data))]
    max_size = max(number_of_nodes)
    if include_embedding == 1 or include_attribute == 1:
        for i in range(all_size):
            tmp = data[i]['nl']
            tmp = tmp[:, 0] # integer node labels
            tmp_onehot = np.zeros((tmp.shape[0], max_size)) # the one - hot embedding matrix
            tmp_onehot[:, sub2ind(tmp_onehot.shape, np.zeros((1, tmp.shape[0])), tmp).astype(int)] = 1

            data[i]['nl'] = np.hstack((tmp_onehot, data[i]['nl'][:, 1:]))

    return data, max_size

########################################################################################################################
def subgraph2mat(ind, A, h, node_information):
    # Usage: to extract the enclosing subgraph for a link up to h hop
    if node_information is None:
        node_information_flag = 0 # whether to include node information
    else:
        node_information_flag = 1

    fringe = [ind]
    links = [(ind[0], ind[1])]
    nodes = [ind[0], ind[1]]
    nodes_dist = np.zeros((2, 1))

    for dist in range(h):
        fringe = neighbors(fringe, A)
        fringe = list(set(fringe) - set(links))
        if len(fringe) == 0: # no more new nodes
            temp = A[nodes[0], :]
            for i in nodes[1:]:
                temp = np.vstack((temp, A[i, :]))

            subgraph_mat = temp[:, nodes[0]]
            for i in nodes[1:]:
                subgraph_mat = np.vstack((subgraph_mat, temp[:, i]))

            subgraph_mat[0, 1] = 0 # ensure subgraph patterns do not contain information about link existence
            subgraph_mat[1, 0] = 0
            break

        new_nodes = list(set(np.setdiff1d(fringe, nodes, 'rows')))
        nodes = np.hstack((nodes, new_nodes))
        nodes_dist = np.vstack((nodes_dist, np.ones((len(new_nodes), 1)) * (dist+1)))
        links = [links.append(fringe[i]) for i in range(len(fringe))]
            # np.vstack((links, fringe))

        if dist+1 == h: # nodes enough (reach h hops), extract subgraph
            A = A.astype(int)
            nodes = nodes.astype(int)
            temp = A[nodes[0], :]
            for i in nodes[1:]:
                temp = np.vstack((temp, A[i, :]))

            subgraph_mat = temp[:, int(nodes[0])]
            for i in nodes[1:]:
                subgraph_mat = np.vstack((subgraph_mat, temp[:, i]))

            subgraph_mat[0, 1] = 0 # ensure subgraph patterns do not contain information about link existence
            subgraph_mat[1, 0] = 0
            break

    sample = {}
    sample['nl'] = []

    # calculate node labels
    # labels = nodes_dist + 1;
    # use node distance as node labels
    labels = node_label(subgraph_mat, h)# node labeling method
    max_nl_size = max(labels) # determin the nl size after one - hot embedding
    sample['nl'] = [np.int(i) for i in labels]

    # whether to include node information (embeddings or attributes)
    if node_information_flag == 1:
        node_info = node_information[nodes, :].astype(int)
        tmp = np.reshape(np.array(sample['nl']), (len(sample['nl']), 1))
        sample['nl'] = np.hstack((tmp, node_info))

    sample['am'] = subgraph_mat # adjacency matrix

    return sample, max_nl_size

########################################################################################################################
def node_label(subgraph_mat, h):
    # Usage: give integer labels to subgraph nodes based on their roles
     # node labeling method of SEAL, double - radius
    subgraph_wo1 = subgraph_mat[1:, 1:] # subgraph without node 1
    subgraph_wo2 = np.vstack((subgraph_mat[0, :], subgraph_mat[2:, :])) # subgraph without node 2
    subgraph_wo2 = np.hstack((np.reshape(subgraph_wo2[:, 0], (subgraph_wo2.shape[0], 1)), subgraph_wo2[:, 2:]))

    subgraph_wo1 = nx.Graph(subgraph_wo1)
    subgraph_wo2 = nx.Graph(subgraph_wo2)

    path_to_1 = nx.shortest_path(subgraph_wo2, source=0)

    dist_to_1 = [0 for i in range(len(subgraph_wo2.nodes))]
    for i in path_to_1.keys():
        dist_to_1[i] = len(path_to_1[i]) - 1

    path_to_2 = nx.shortest_path(subgraph_wo1, source=0)

    dist_to_2 = [0 for i in range(len(subgraph_wo1.nodes))]
    for i in path_to_2.keys():
        dist_to_2[i] = len(path_to_2[i]) - 1

    dist_to_1 = dist_to_1[1:]
    dist_to_2 = dist_to_2[1:]

    d = [dist_to_1[x] + dist_to_2[x] for x in range(len(dist_to_1))]
    d_over_2 = [np.floor(x/2) for x in d]
    d_mod_2 = [np.mod(x, 2) for x in d]

    labels = [1 + min(dist_to_1[i], dist_to_2[i]) + d_over_2[i] * (d_over_2[i] + d_mod_2[i] - 1) for i in range(len(dist_to_1))]
    if np.any(np.isinf(labels)):
        labels[np.isinf(labels)] = 0

    if np.any(np.isnan(labels)):
        labels[np.isnan(labels)] = 0

    labels = np.hstack(([1, 1], labels))
    labels = [i + 1 for i in labels]
    labels = np.transpose(labels)
    return labels

########################################################################################################################
def neighbors (fringe, A):
    # Usage: from A to find the neighbor links of all nodes in fringe

    a = []
    b = []
    for no in range(len(fringe)):
        ind = fringe[no]
        i = ind[0]
        j = ind[1]
        i1, i2, _ = ssp.find(A[i, :])
        a.extend(i1)
        b.extend(i2)
        j1, j2, _ = ssp.find(A[:, j])
        a.extend(j1)
        b.extend(j2)

    fringe_neighbors = []
    for i in range(len(a)):
        t = (a[i], b[i])
        if t not in fringe_neighbors:
            fringe_neighbors.append(t)

    return fringe_neighbors

########################################################################################################################
def generate_embeddings(A_mat, data_name, emd_method='node2vec'):
    #  Usage: generate node embeddings
    #  --Input--
    #  --A: the observed network's adjacency matrix from which to generate node embeddings
    #  --data_name: the name of the dataset
    #  --Output--
    #  --node_embeddings: a matrix, ith row contains the ith node's embeddings

    i, j, _ = ssp.find(np.triu(A_mat))
    train = [i, j]

    if emd_method == 'node2vec':
        np.save('Data/embedding/' + data_name + '.edgelist', train)  # convert train to edgelist which will be read by node2vec
        p = 1.0
        q = 1.0

        G = nx.Graph(A_mat)
        node2vec = Node2Vec(G, dimensions=128, walk_length=10, num_walks=50, workers=1)
        model = node2vec.fit(window=10, min_count=1)
        node_embeddings = model.wv
    else:
        node_embeddings = []

    return node_embeddings

########################################################################################################################
def gen_primes(N):
    """Generate primes up to N"""
    primes = set()
    for n in range(2, N):
        if all(n % p > 0 for p in primes):
            primes.add(n)
            yield n