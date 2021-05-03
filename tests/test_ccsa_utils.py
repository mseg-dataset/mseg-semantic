#!/usr/bin/python3

import math
import numpy as np
import pdb
import time
import torch

from mseg_semantic.domain_generalization.ccsa_utils import (
	contrastive_loss, 
	paired_euclidean_distance,
	downsample_label_map,
	sample_pair_indices,
	find_matching_pairs,
	remove_pairs_from_same_domain,
	get_merged_pair_embeddings,
	pytorch_random_choice,
	shuffle_pytorch_tensor,
	get_pair_embedding,
	count_per_domain_statistics,
	sample_px_locations_uniformly,
	sample_crossdomain_pos_neg_pairs,
	form_pair_info_tensor
)

"""
For sake of unit tests, pretend we have the following categories:
Let 0 = Sky
    1 = Mountain
    2 = Road
    3 = Person
    4 = Vegetation
"""


def test_contrastive_loss1():
    """
    Should be no loss here (zero from pull term, and zero from push term)
    """
    # which pairs share the same semantic class label
    y_c = torch.tensor([ 1., 0., 0., 0., 1.], dtype=torch.float32)

    # distances between pairs
    pred_dists = torch.tensor([0, 1.1, 1.1, 1.1, 0], dtype=torch.float32)

    loss = contrastive_loss(y_c, pred_dists)
    gt_loss = torch.tensor([0])

    assert torch.allclose(loss, gt_loss)


def test_contrastive_loss2():
    """ 
    There should be more loss here (coming only from push term)
    """
    # which pairs share the same semantic class label
    y_c = torch.tensor([ 1., 0., 0., 0., 1.], dtype=torch.float32)

    # distances between pairs
    pred_dists = torch.tensor([0, 0.2, 0.3, 0.1, 0], dtype=torch.float32)

    loss = contrastive_loss(y_c, pred_dists)
    gt_loss = torch.tensor([0.3880])

    assert torch.allclose(loss, gt_loss, atol=1e-3)


def test_contrastive_loss3():
    """
    There should be the most loss here (some from pull term, and some from push term also)
    """
    # which pairs share the same semantic class label
    y_c = torch.tensor([ 1., 0., 0., 0., 1.], dtype=torch.float32)

    # distances between pairs
    pred_dists = torch.tensor([2.0, 0.2, 0.3, 0.1, 4.0], dtype=torch.float32)

    loss = contrastive_loss(y_c, pred_dists)
    gt_loss = torch.tensor([4.3880])

    assert torch.allclose(loss, gt_loss, atol=1e-3)


def test_paired_euclidean_distance():
    """ """
    X = torch.tensor(
        [
            [3,0],
            [4,0],
            [1,1]
        ], dtype=torch.float32)
    Y = torch.tensor(
        [
            [1,1],
            [0,3],
            [0,4]
        ], dtype=torch.float32)
    dists = paired_euclidean_distance(X, Y)
    gt_dists = torch.tensor(
        [
            [ math.sqrt(2*2 + 1) ], # (3,0) vs. (1,1)
            [ math.sqrt(3*3 + 4*4) ], # (4,0) vs. (0,3) 
            [ math.sqrt(3*3 + 1) ] #  (1,1) vs. (0,4)
        ])
    torch.allclose(gt_dists.squeeze(), dists, atol=1e-3)


def test_downsample_label_map():
    """
    Downsample two label maps "Y"
    """
    labelmap_1 = torch.tensor(
        [
            [0,0,0,0,0,0,0,0],
            [4,4,0,0,0,0,4,4],
            [4,3,2,2,2,2,3,4],
            [4,2,2,2,2,2,2,4]
        ])

    labelmap_2 = torch.tensor(
        [
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,2,2,2,4],
            [4,4,4,4,2,2,2,4],
            [4,4,4,3,2,2,2,4]
        ])
    Y = torch.stack([labelmap_1, labelmap_2])
    Y = Y.type(torch.float32)
    assert Y.shape == (2,4,8)

    dY = downsample_label_map(Y, d=2)
    assert dY.shape == (2,2,4)
    gt_dY = torch.tensor(
        [
            [[0., 0., 0., 0.],
            [4., 2., 2., 3.]],

            [[1., 1., 0., 0.],
            [4., 4., 2., 2.]]
        ])

    dY = downsample_label_map(Y, d=4)
    gt_dY = torch.tensor(
        [
            [[0., 0.]],
            [[1., 0.]]
        ])
    assert dY.shape == (2,1,2)



def test_sample_pair_indices1():
    """
    Given labels for 3 images, sample corresponding pixels that
    are known positives and that are known negatives.
    Suppose images 0 and 2 come from Domain-0, and image 1 comes
    from Domain-1.
    """
    labelmap_0 = torch.tensor(
        [
            [0,0,0,0,0,0,0,0],
            [4,4,0,0,0,0,4,4],
            [4,3,2,2,2,2,3,4],
            [4,2,2,2,2,2,2,4]
        ], dtype=torch.float32)

    labelmap_1 = torch.tensor(
        [
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,2,2,2,4],
            [4,4,4,4,2,2,2,4],
            [4,4,4,3,2,2,2,4]
        ], dtype=torch.float32)
    labelmap_2 = torch.tensor(
        [
            [4,4,4,4,4,4,4,4],
            [4,4,4,4,4,4,4,4],
            [4,4,4,4,4,4,4,4],
            [4,4,4,4,4,4,4,4]
        ], dtype=torch.float32)

    Y = torch.stack([labelmap_0, labelmap_1, labelmap_2])
    assert Y.shape == (3,4,8)

    batch_domain_indices = torch.tensor([0,1,0], dtype=torch.int32)

    pos_pair_info, neg_pair_info = sample_pair_indices(Y, batch_domain_indices, num_pos_pairs=30000, neg_to_pos_ratio=3, downsample_factor=1)

    for (bi, hi, wi, bj, hj, wj) in pos_pair_info:
        assert Y[bi,hi,wi] == Y[bj,hj,wj] # is same class
        assert batch_domain_indices[bi] != batch_domain_indices[bj] # must be cross-domain

    for (bi, hi, wi, bj, hj, wj) in neg_pair_info:
        assert Y[bi,hi,wi] != Y[bj,hj,wj] # is different class
        assert batch_domain_indices[bi] != batch_domain_indices[bj] # must be cross-domain


def test_sample_pair_indices2():
    """
    Given labels for 3 images, sample corresponding pixels that
    are known positives and that are known negatives.
    Suppose images 0 and 2 come from Domain-0, and image 1 comes
    from Domain-1.
    """
    labelmap_0 = torch.tensor(
        [
            [0,0,0,0,1,1,1,1],
            [0,0,0,0,1,1,1,1],
            [2,2,2,2,4,4,4,4],
            [2,2,2,2,4,4,4,4]
        ], dtype=torch.float32)

    labelmap_1 = torch.tensor(
        [
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,0,0,0,0],
            [4,4,4,4,2,2,2,2],
            [4,4,4,4,2,2,2,2]
        ], dtype=torch.float32)
    labelmap_2 = torch.tensor(
        [
            [4,4,4,4,4,4,4,4],
            [4,4,4,4,4,4,4,4],
            [4,4,4,4,4,4,4,4],
            [4,4,4,4,4,4,4,4]
        ], dtype=torch.float32)

    Y = torch.stack([labelmap_0, labelmap_1, labelmap_2])
    assert Y.shape == (3,4,8)

    batch_domain_indices = torch.tensor([0,1,0], dtype=torch.int32)

    pos_pair_info, neg_pair_info = sample_pair_indices(Y, batch_domain_indices, num_pos_pairs=3000, neg_to_pos_ratio=3, downsample_factor=2)
    for (bi, hi, wi, bj, hj, wj) in pos_pair_info:
        assert Y[:,::2,::2][bi,hi,wi] == Y[:,::2,::2][bj,hj,wj] # is same class
        assert batch_domain_indices[bi] != batch_domain_indices[bj] # must be cross-domain

    for (bi, hi, wi, bj, hj, wj) in neg_pair_info:
        assert Y[:,::2,::2][bi,hi,wi] != Y[:,::2,::2][bj,hj,wj] # is different class
        assert batch_domain_indices[bi] != batch_domain_indices[bj] # must be cross-domain



def test_remove_pairs_from_same_domain():
    """
    Consider a minibatch of size 5 (examples). Suppose we have sampled 4 pairs
    of pixel locations.

    In training, we want only pairs from different domains. We
    enforce that their feature embeddings are similar.

    We could have 1 million sampled pairs from a minibatch of size 5.
    (Number of elements in batch (batch_domain_indices) need not
    agree with number of sampled pairs!)
    """
    # show which minibatch examples belong to which domain
    batch_domain_indices = torch.tensor([0,1,2,1,0])
    # sampled pairs (a,b) are enumerated here.
    a_info_ = torch.tensor(
        [
            [0, 1, 2], # Belongs to domain 0 (will be removed)
            [0, 1, 2], # Belongs to domain 0
            [2, 1, 2], # Belongs to domain 2
            [3, 1, 2]  # Belongs to domain 1 (will be removed)
        ])
    b_info_ = torch.tensor(
        [
            [4, 3, 4], # Belongs to domain 0 (will be removed)
            [1, 3, 4], # Belongs to domain 1
            [3, 3, 4], # Belongs to domain 1
            [1, 3, 4]  # Belongs to domain 1 (will be removed)
        ])
    a_pair_info, b_pair_info = remove_pairs_from_same_domain(batch_domain_indices, a_info_, b_info_)
    gt_a_pair_info = torch.tensor(
        [
            [0, 1, 2],
            [2, 1, 2]
        ])
    assert torch.allclose(gt_a_pair_info, a_pair_info)
    gt_b_pair_info = torch.tensor(
        [
            [1, 3, 4],
            [3, 3, 4]
        ])
    assert torch.allclose(gt_b_pair_info, b_pair_info)

def test_form_pair_info_tensor():
    """
    Ensure hstacking of 3 length-N 1d arrays into a (N,3) array
    is successful.

    Given batch_dim_idxs (representing indices of examples in a minibatch),
    and px_1d_y (representing row indices) and px_1d_x 
    (representing column indices), stack them along axis-0 (row dimension).
    """
    batch_dim_idxs = torch.tensor([5,6,7,8,9], dtype=torch.int32)
    px_1d_y = torch.tensor([4,3,2,1,0], dtype=torch.int32)
    px_1d_x = torch.tensor([0,2,4,6,8], dtype=torch.int32)

    pair_info = form_pair_info_tensor(batch_dim_idxs, px_1d_y, px_1d_x)
    gt_pair_info = torch.tensor(
        [
            [5,4,0],
            [6,3,2],
            [7,2,4],
            [8,1,6],
            [9,0,8]
        ], dtype=torch.int32)
    assert torch.allclose(pair_info, gt_pair_info)


def test_find_matching_pairs():
    """
    Given a batch of ground truth label maps, and sampled pixel
    pair locations (pairs are across label maps), identify which 
    pairs are matching vs. non-matching and return corresponding metadata
    (basically, partition them).

    Get back pos_pair_info --  Pytorch tensor containing info about each positive pair (a,b). Contains
                (a batch_idx, a row, a col, b batch_idx, b row, b col)
    Also get back neg_pair_info -- same as above, but for negative pairs.
    """
    labelmap_0 = torch.tensor(
        [
            [0,0,0,0,0,0,0,0],
            [4,4,0,0,0,0,4,4],
            [4,3,2,2,2,2,3,4],
            [4,2,2,2,2,2,2,4]
        ])

    labelmap_1 = torch.tensor(
        [
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,2,2,2,4],
            [4,4,4,4,2,2,2,4],
            [4,4,4,3,2,2,2,4]
        ])
    labelmap_2 = torch.tensor(
        [
            [4,4,4,4,4,4,4,4],
            [4,4,4,4,4,4,4,4],
            [4,4,4,4,4,4,4,4],
            [4,4,4,4,4,4,4,4]
        ])

    Y = torch.stack([labelmap_0, labelmap_1, labelmap_2])
    assert Y.shape == (3,4,8)
    
    a_pair_info = torch.tensor(
        [
            [0,1,1], # pos
            [2,1,4], # neg
            [1,1,7], # pos
            [0,2,2] # neg
        ])
    b_pair_info = torch.tensor(
        [
            [2,3,7], # pos
            [0,1,4], # neg
            [2,3,0], # pos
            [1,3,3] # neg
        ])
    pos_pair_info, neg_pair_info = find_matching_pairs(Y, a_pair_info, b_pair_info)
    gt_pos_pair_info = torch.tensor(
        [
            [0, 1, 1, 2, 3, 7], # pos pairs
            [1, 1, 7, 2, 3, 0]
        ])
    assert torch.allclose(pos_pair_info, gt_pos_pair_info)
    gt_neg_pair_info = torch.tensor(
        [
            [2, 1, 4, 0, 1, 4], # neg pairs
            [0, 2, 2, 1, 3, 3]
        ])
    assert torch.allclose(neg_pair_info, gt_neg_pair_info)


def test_sample_crossdomain_pos_neg_pairs():
    """ """
    labelmap_0 = torch.tensor(
        [
            [0,0,0,0,0,0,0,0],
            [4,4,0,0,0,0,4,4],
            [4,3,2,2,2,2,3,4],
            [4,2,2,2,2,2,2,4]
        ])

    labelmap_1 = torch.tensor(
        [
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,2,2,2,4],
            [4,4,4,4,2,2,2,4],
            [4,4,4,3,2,2,2,4]
        ])
    labelmap_2 = torch.tensor(
        [
            [4,4,4,4,4,4,4,4],
            [4,4,4,4,4,4,4,4],
            [4,4,4,4,4,4,4,4],
            [4,4,4,4,4,4,4,4]
        ])

    Y = torch.stack([labelmap_0, labelmap_1, labelmap_2])
    assert Y.shape == (3,4,8)
    
    # here, domain 1 would be sampled more than others
    batch_domain_indices = torch.tensor([0,1,0], dtype=torch.int64)

    _, unique_domain_idxs = count_per_domain_statistics(batch_domain_indices)
    b, h, w = Y.shape
    INITIAL_SAMPLE_NUM = int(1e4)

    pos_pair_info, neg_pair_info = sample_crossdomain_pos_neg_pairs(Y, batch_domain_indices, unique_domain_idxs, w, h, INITIAL_SAMPLE_NUM)
    for (bi, hi, wi, bj, hj, wj) in pos_pair_info:
        assert Y[bi,hi,wi] == Y[bj,hj,wj] # is same class
        assert batch_domain_indices[bi] != batch_domain_indices[bj] # must be cross-domain

    for (bi, hi, wi, bj, hj, wj) in neg_pair_info:
        assert Y[bi,hi,wi] != Y[bj,hj,wj] # is different class
        assert batch_domain_indices[bi] != batch_domain_indices[bj] # must be cross-domain


def test_count_per_domain_statistics():
    """
    """
    domain_idxs = torch.tensor([0,1,0,1,4])
    examples_per_domain, unique_domain_idxs = count_per_domain_statistics(domain_idxs)
    gt_examples_per_domain = np.array([2., 2., 0., 0., 1.], dtype=np.int32)
    gt_unique_domain_idxs = np.array([0, 1, 4])
    assert np.allclose(examples_per_domain, gt_examples_per_domain)
    assert np.allclose(unique_domain_idxs, gt_unique_domain_idxs)
    assert examples_per_domain.dtype == np.int64


def test_sample_px_locations_uniformly():
    """
        Let 0 = Sky
            1 = Mountain
            2 = Road
            3 = Person
            4 = Vegetation

    In expectation, minibatch examples from less common domains should be
    sampled more often, if domains sampled uniformly.
    """
    labelmap_1 = torch.tensor(
        [
            [0,0,0,0,0,0,0,0],
            [4,4,0,0,0,0,4,4],
            [4,3,2,2,2,2,3,4],
            [4,2,2,2,2,2,2,4]
        ])

    labelmap_2 = torch.tensor(
        [
            [1,1,1,1,0,0,0,0],
            [1,1,1,1,2,2,2,4],
            [4,4,4,4,2,2,2,4],
            [4,4,4,3,2,2,2,4]
        ])
    labelmap_3 = torch.tensor(
        [
            [4,4,4,4,4,4,4,4],
            [4,4,4,4,4,4,4,4],
            [4,4,4,4,4,4,4,4],
            [4,4,4,4,4,4,4,4]
        ])

    Y = torch.stack([labelmap_1, labelmap_2, labelmap_3])
    assert Y.shape == (3,4,8)
    
    # here, domain 1 would be sampled more than others (sampled twice as often)
    domain_indices = torch.tensor([0,1,0], dtype=torch.int64)

    # unique domain indices would be [0,1]
    _, unique_domain_idxs = count_per_domain_statistics(domain_indices)
    b, h, w = Y.shape
    INITIAL_SAMPLE_NUM = int(1e6)

    b_idxs, w_idxs, h_idxs = sample_px_locations_uniformly(
        domain_indices,
        unique_domain_idxs,
        w,
        h,
        INITIAL_SAMPLE_NUM
    )
    # Verify expected value vs. empirical. Allow for some margin of error.
    # Less common domain (minibatch example 1) should be sampled roughly
    # 2x as often, since it appears less often.
    assert 245000 < (b_idxs == 0).sum() and (b_idxs == 0).sum() < 255000
    assert 495000 < (b_idxs == 1).sum() and (b_idxs == 1).sum() < 505000
    assert 245000 < (b_idxs == 2).sum() and (b_idxs == 2).sum() < 255000

    # Sample minibatch indices should lie in [0,b)
    assert (b_idxs >= 0).sum() == INITIAL_SAMPLE_NUM
    assert (b_idxs < b).sum() == INITIAL_SAMPLE_NUM

    # Sampled pixel rows should lie in [0,h)
    assert (h_idxs >= 0).sum() == INITIAL_SAMPLE_NUM
    assert (h_idxs < h).sum() == INITIAL_SAMPLE_NUM

    # Sampled pixel columns should lie in [0,w)
    assert (w_idxs >= 0).sum() == INITIAL_SAMPLE_NUM
    assert (w_idxs < w).sum() == INITIAL_SAMPLE_NUM


def test_shuffle_pytorch_tensor():
    """
    Given all possible permutations, ensure that the shuffling that was
    executed corresponds to any valid permutation.
    """
    t = torch.tensor(
        [
            [1,2],
            [3,4],
            [5,6]
        ])

    shuffled = shuffle_pytorch_tensor(t)

    gt_permutations = torch.tensor(
        [
            [[1,2],
            [3,4],
            [5,6]],

            [[1,2],
            [5,6],
            [3,4]],

            [[3,4],
            [5,6],
            [1,2]],

            [[5,6],
            [3,4],
            [1,2]],

            [[3,4],
            [1,2],
            [5,6]],

            [[5,6],
            [1,2],
            [3,4]]
        ])
    assert any([torch.allclose(gt_permutations[i], shuffled) for i in range(6)])



def test_pytorch_random_choice():
    """
    Ensure that sampling with replacement returns values that are found
    in original array, and of correct shape.
    """
    x = np.array([0,2,4,5,6])
    vals = pytorch_random_choice(x, num_samples=10)
    for val in list(torch.unique(vals).cpu().numpy()):
        assert val in list(x)
    assert vals.shape == (10,)

    x = np.array([0,2,4,5,6])
    vals = pytorch_random_choice(x, num_samples=3)
    for val in list(torch.unique(vals).cpu().numpy()):
        assert val in list(x)
    assert vals.shape == (3,)

    x = np.array([0,2])
    vals = pytorch_random_choice(x, num_samples=10)
    for val in list(torch.unique(vals).cpu().numpy()):
        assert val in list(x)
    assert vals.shape == (10,)


def test_get_merged_pair_embeddings():
    """
    """
    pos_pair_info = torch.tensor(
        [
            [0,1,1,1,2,2],
            [1,3,4,2,0,0]
        ])
    neg_pair_info = torch.tensor(
        [
            [0,1,1,1,2,2],
            [1,3,4,2,0,0]
        ])
    resnet_embedding = torch.arange(2*3*4*5).reshape(3,2,4,5)

    y_c, a_embedding, b_embedding = get_merged_pair_embeddings(
        pos_pair_info,
        neg_pair_info,
        resnet_embedding
    )
    gt_y_c = torch.tensor([1,1,0,0], dtype=torch.float32)
    gt_a_embedding = torch.tensor(
        [
            [ 6, 26],
            [59, 79],
            [ 6, 26],
            [59, 79]
        ])
    gt_b_embedding = torch.tensor(
        [
            [ 52,  72],
            [ 80, 100],
            [ 52,  72],
            [ 80, 100]
        ])
    assert torch.allclose(a_embedding, gt_a_embedding)
    assert torch.allclose(b_embedding, gt_b_embedding)
    assert torch.allclose(y_c, gt_y_c)

def test_get_pair_embedding():
    """
    """
    pair_info = torch.tensor(
        [
        #   (bi,hi,wi,bj,hj,wj)
	        [0, 1, 1, 1, 2, 2],
	        [1, 3, 4, 2, 0, 0]
        ])
    embedding = torch.arange(2*3*4*5).reshape(3,2,4,5)
    a_embedding, b_embedding = get_pair_embedding(pair_info, embedding)

    gt_a_embedding = torch.tensor(
        [
            [ 6, 26],
            [59, 79]
        ])
    gt_b_embedding = torch.tensor(
        [
            [ 52,  72],
            [ 80, 100]
        ])

    assert torch.allclose(a_embedding, gt_a_embedding)
    assert torch.allclose(b_embedding, gt_b_embedding)


def time_sample_pair_indices():
    """
    Count how long it takes to sample pairs.
    Suppose we have a batch size of 128 images, and 194 possible
    classes. Suppose the 128 minibatch examples come from 7 different
    domains.

    Takes around 0.5 sec on Macbook Pro to sample pair indices each time.
    """
    for _ in range(10):
        batch_domain_idxs = torch.randint(low=0, high=7, size=(128,))
        Y = torch.randint(low=0, high=194, size=(128,201,201))

        start = time.time()
        out = sample_pair_indices(
            Y.type(torch.float32),
            batch_domain_idxs,
            num_pos_pairs=int(1e3),
            neg_to_pos_ratio=3,
            downsample_factor=8
        )
        end = time.time()
        duration = end - start
        print(f'Duration was {duration}')


if __name__ == '__main__':
    """ """
    test_contrastive_loss1()
    test_contrastive_loss2()
    test_contrastive_loss3()
    test_paired_euclidean_distance()
    test_downsample_label_map()

    test_shuffle_pytorch_tensor()
    test_pytorch_random_choice()
    test_count_per_domain_statistics()
    test_sample_px_locations_uniformly()

    test_form_pair_info_tensor()
    test_remove_pairs_from_same_domain()

    test_find_matching_pairs()
    test_sample_crossdomain_pos_neg_pairs()
    test_sample_pair_indices1()
    test_sample_pair_indices2()

    test_get_pair_embedding()
    test_get_merged_pair_embeddings()
    time_sample_pair_indices()
