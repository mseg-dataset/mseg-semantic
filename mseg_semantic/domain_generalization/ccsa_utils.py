#!/usr/bin/python3

import numpy as np
import time
import torch
import torch.nn.functional as F

from typing import Tuple

"""
Set of utilities for metric learning. We use extensive sampling
techniques and also a contrastive loss to learn the metric space.
"""

def contrastive_loss(
    y_c: torch.Tensor, 
    pred_dists: torch.Tensor, 
    margin: int = 1
    ) -> torch.Tensor:
    """
        Compute the similarities in the separation loss (4) by 
        computing average pairwise similarities between points
        in the embedding space.

		element-wise square, element-wise maximum of two tensors.

		Contrastive loss also defined in:
		-	"Dimensionality Reduction by Learning an Invariant Mapping" 
				by Raia Hadsell, Sumit Chopra, Yann LeCun

        Args:
        -   y_c: Indicates if pairs share the same semantic class label or not
        -   pred_dists: Distances in the embeddding space between pairs. 

        Returns:
        -   tensor representing contrastive loss values.
    """
    N = pred_dists.shape[0]

    # corresponds to "d" in the paper. If same class, pull together.
    # Zero loss if all same-class examples have zero distance between them.
    pull_losses = y_c * torch.pow(pred_dists, 2)
    # corresponds to "k" in the paper. If different class, push apart more than margin
    # if semantically different examples have distances are in [0,margin], then there WILL be loss
    zero = torch.zeros(N)
    device = y_c.device
    zero = zero.to(device)
    # if pred_dists for non-similar classes are <1, then incur loss >0.
    clamped_dists = torch.max(margin - pred_dists, zero )
    push_losses = (1 - y_c) * torch.pow(clamped_dists, 2)
    return torch.mean(pull_losses + push_losses)


def paired_euclidean_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
        Compute the distance in the semantic alignment loss (3) by 
        computing average pairwise distances between *already paired*
        points in the embedding space.

        Note this is NOT computed between all possible pairs. Rather, we
        compare i'th vector of X vs. i'th vector of Y (i == j always).

        Args:
        -   X: Pytorch tensor of shape (N,D) representing N embeddings of dim D
        -   Y: Pytorch tensor of shape (N,D) representing N embeddings of dim D

        Returns:
        -   dists: Pytorch tensor of shape (N,) representing distances between 
                fixed pairs
    """
    device = X.device
    N, D = X.shape
    assert Y.shape == X.shape
    eps = 1e-08 * torch.ones((N,1))
    eps = eps.to(device) # make sure in same memory (CPU or CUDA)
    # compare i'th vector of x vs. i'th vector of y (i == j always)
    diff = torch.pow(X - Y, 2)

    affinities = torch.sum(diff, dim=1, keepdim=True)
    # clamp the affinities to be > 1e-8 ?? Unclear why the authors do this...
    affinities = torch.max(affinities, eps)
    return torch.sqrt(affinities)


def downsample_label_map(y: torch.Tensor, d: int = 2):
    """
        size (int or Tuple[int] or Tuple[int, int] or Tuple[int, int, int]) – 
            output spatial size.

        scale_factor (float or Tuple[float]) – multiplier for spatial size. 
        Has to match input size if it is a tuple.

        mode (str) – algorithm used for upsampling: 
        'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area'. Default: 'nearest'

        align_corners (bool, optional) – Geometrically, we consider the pixels of the input 
        and output as squares rather than points. If set to True, the input and output 
        tensors are aligned by the center points of their corner pixels, preserving the 
        values at the corner pixels. If set to False, the input and output tensors are 
        aligned by the corner points of their corner pixels, and the interpolation uses 
        edge value padding for out-of-boundary values, making this operation independent 
        of input size when scale_factor is kept the same. This only has an effect when 
        mode is 'linear', 'bilinear', 'bicubic' or 'trilinear'. Default: False
    
        Args:
        -   Y: Pytorch tensor of shape (batch size, height, width)
        -   d: downsample factor

        Returns:
        -   dY: Pytorch tensor of shape (batch_size, height/d, width/d)
    """
    b, h, w = y.shape
    y = y.unsqueeze(dim=1) # add num_channels = 1
    # Size must be 2 numbers -- for height and width, only
    dY = F.interpolate(y, size=(h//d, w//d), mode='nearest')
    dY = torch.squeeze(dY, dim=1)
    assert dY.shape == (b, h//d, w//d)
    return dY


def sample_pair_indices(
    Y: torch.Tensor,
    batch_domain_idxs: torch.Tensor, 
    num_pos_pairs: int = 100, 
    neg_to_pos_ratio: int = 3,
    downsample_factor: int = 2
):
    """
    In our case, positive/negative pairs can be found in almost any two images
    (as long as ground truth label maps are not identical). Thus, we sample negative
    positive pairs not on an *image* level, but rather on a pixel-level, as long
    as both images come from different domains.

    when i get resnet embedding E1 of 
    shape (C,H,W) of image1 from domain 1, 
    and resnet embedding E2 of shape (C,H,W) of 
    image 2 from domain 2, my contrastive loss will 
    be between random feature map locations E1[:,x,y] and E2[:,x,y]
    
        Args:
        -   Y: torch.Tensor, Pytorch tensor of shape (N,H,W) representing labels
        -   domain_idxs: torch.Tensor, 
        -   num_pos_pairs: int = 100, 
        -   neg_to_pos_ratio: int = 3,
        -   downsample_factor: int = 2: 

        Returns:
        -   all_pos_pair_info
        -   all_neg_pair_info
    """
    assert Y.dtype in [torch.float32, torch.float64] # cannot upsample dtype int
    INITIAL_SAMPLE_NUM = int(1e6)
    # downsample the class label map to the feature map resolution
    # use nearest interpolation
    dY = downsample_label_map(Y, d=downsample_factor)
    _, unique_domain_idxs = count_per_domain_statistics(batch_domain_idxs)
    batch_sz, h, w = dY.shape

    # Indices ordered as (bi,hi,wi,bj,hj,wj)
    all_pos_pair_info = torch.zeros((0,6), dtype=torch.int64)
    all_neg_pair_info = torch.zeros((0,6), dtype=torch.int64)

    # keep sampling until we get enough, append to array each time we get more
    dataprep_complete = False
    while not dataprep_complete:
        
        pos_pair_info, neg_pair_info = sample_crossdomain_pos_neg_pairs(dY, batch_domain_idxs, unique_domain_idxs, 
                                                                        w, h, INITIAL_SAMPLE_NUM)
        # add to list of positives
        all_pos_pair_info = torch.cat([pos_pair_info, all_pos_pair_info])
        # add to list of negatives
        all_neg_pair_info = torch.cat([neg_pair_info, all_neg_pair_info])

        curr_num_pos = all_pos_pair_info.shape[0]
        curr_num_neg = all_neg_pair_info.shape[0]
        sufficient_pos = (curr_num_pos > num_pos_pairs)
        sufficient_neg = (curr_num_neg > neg_to_pos_ratio * num_pos_pairs)
        dataprep_complete = sufficient_pos and sufficient_neg

    # shuffle the negatives among themselves
    all_pos_pair_info = shuffle_pytorch_tensor(all_pos_pair_info)
    # shuffle the positives among themselves
    all_neg_pair_info = shuffle_pytorch_tensor(all_neg_pair_info)

    # clip number of pos to num_pos_pairs
    all_pos_pair_info = all_pos_pair_info[:num_pos_pairs]
    # clip number of neg to 3x positive
    all_neg_pair_info = all_neg_pair_info[:neg_to_pos_ratio * num_pos_pairs]

    # we won't backprop through this function
    all_pos_pair_info.requires_grad = False
    all_neg_pair_info.requires_grad = False

    return all_pos_pair_info, all_neg_pair_info


def remove_pairs_from_same_domain(
    batch_domain_indices: torch.Tensor, 
    a_pair_info: torch.Tensor, 
    b_pair_info: torch.Tensor
    ) -> Tuple[torch.Tensor,torch.Tensor]:
    """
    In training, we want only pairs from different domains. We
    enforce that their feature embeddings are similar.

    We could have 1 million sampled pairs from a minibatch of size 5.
    (Number of elements in batch (batch_domain_indices) need not
    agree with number of sampled pairs!)

        Args:
        -   batch_domain_indices: Tensor of shape (K,), for each example
                in minibatch, which domain did it come from.
        -   a_pair_info:  (M,3) array representing (bi,hi,wi)
                where these represent (batch index, row index, column index)
                into a NCHW tensor for samples A.
        -   b_pair_info: (M,3) as above, but for samples B. (a,b) are paired

        Returns:
        -   a_pair_info: (N,3), where N <= M (discarded same domain elements)
        -   b_pair_info: (N,3), where N <= M
    """
    batch_dim_a_idxs = a_pair_info[:,0]
    batch_dim_b_idxs = b_pair_info[:,0]
    # remove locations with identical domains in pos/neg pairs
    a_domain = batch_domain_indices[batch_dim_a_idxs]
    b_domain = batch_domain_indices[batch_dim_b_idxs]

    is_valid_pair = (a_domain != b_domain).nonzero().squeeze()
    return a_pair_info[is_valid_pair], b_pair_info[is_valid_pair]


def form_pair_info_tensor(
    batch_dim_idxs: torch.Tensor, 
    px_1d_y: torch.Tensor, 
    px_1d_x: torch.Tensor
    ):
    """ Hstack 3 length-N 1d arrays into a (N,3) array

        Args:
        -   batch_dim_idxs: size (N,) array representing indices
                of examples in a minibatch
        -   px_1d_y: size (N,) array representing row indices
        -   px_1d_x: size (N,) array representing column indices

        Returns:
        -   pair_info: (N,3) array
    """
    # batch dim
    N = batch_dim_idxs.shape[0]
    assert batch_dim_idxs.shape == (N,)
    assert px_1d_y.shape == (N,)
    assert px_1d_x.shape == (N,)

    pair_info = torch.stack([batch_dim_idxs, px_1d_y, px_1d_x])
    return pair_info.t() # tranpose it now


def find_matching_pairs(
    y: torch.Tensor,
    a_pair_info: torch.Tensor,
    b_pair_info: torch.Tensor) -> Tuple[torch.Tensor,torch.Tensor]:
    """
    Given a batch of ground truth label maps, and sampled pixel
    pair locations (pairs are across label maps), identify which 
    pairs are matching vs. non-matching and return corresponding metadata
    (basically, partition them).

        Args:
        -   y: Tensor of size (B,H,W) representing 2-d label maps
                for B examples.
        -   a_pair_info:
        -   b_pair_info:

        Returns:
        -   pos_pair_info: Pytorch tensor containing info about each positive pair (a,b). Contains
                (a batch_idx, a row, a col, b batch_idx, b row, b col)
        -   neg_pair_info: Same as above, but for negative pairs.
    """
    batch_dim_a_idxs = a_pair_info[:,0]
    px_1d_a_y = a_pair_info[:,1]
    px_1d_a_x = a_pair_info[:,2]

    batch_dim_b_idxs = b_pair_info[:,0]
    px_1d_b_y = b_pair_info[:,1]
    px_1d_b_x = b_pair_info[:,2]

    # extract category indices
    cls_vals_a = y[batch_dim_a_idxs, px_1d_a_y, px_1d_a_x]
    cls_vals_b = y[batch_dim_b_idxs, px_1d_b_y, px_1d_b_x]

    # compare category indices for equality
    is_same_class = (cls_vals_a == cls_vals_b).nonzero().squeeze()
    is_diff_class = (cls_vals_a != cls_vals_b).nonzero().squeeze()

    a_pos_info = a_pair_info[is_same_class]
    a_neg_info = a_pair_info[is_diff_class]

    b_pos_info = b_pair_info[is_same_class]
    b_neg_info = b_pair_info[is_diff_class]

    pos_pair_info = torch.cat([a_pos_info, b_pos_info], dim=1)
    neg_pair_info = torch.cat([a_neg_info, b_neg_info], dim=1)

    return pos_pair_info, neg_pair_info


def sample_crossdomain_pos_neg_pairs(
    Y: torch.Tensor, 
    batch_domain_indices: torch.Tensor, 
    unique_domain_idxs: np.ndarray, 
    w: int, 
    h: int, 
    INITIAL_SAMPLE_NUM: int
    ):
    """
        Args:
        -   Y: Pytorch tensor of shape (N,H,W) with batch of ground truth label maps
        -   batch_domain_indices: which domain each example in the training batch belongs to
        -   unique_domain_idxs: unique domain IDs
        -   w: integer representing label map width
        -   h: integer representing label map height
        -   INITIAL_SAMPLE_NUM: 

        Returns:
        -   pos_pair_info: Pytorch tensor of shape (N,6)
        -   neg_pair_info: Pytorch tensor of shape (N,6)
    """
    cache_a = sample_px_locations_uniformly(batch_domain_indices, unique_domain_idxs, w, h, INITIAL_SAMPLE_NUM)
    batch_dim_a_idxs, px_1d_a_x, px_1d_a_y = cache_a
    cache_b = sample_px_locations_uniformly(batch_domain_indices, unique_domain_idxs, w, h, INITIAL_SAMPLE_NUM)
    batch_dim_b_idxs, px_1d_b_x, px_1d_b_y = cache_b

    a_pair_info = form_pair_info_tensor(batch_dim_a_idxs, px_1d_a_y, px_1d_a_x)
    b_pair_info = form_pair_info_tensor(batch_dim_b_idxs, px_1d_b_y, px_1d_b_x)

    # remove examples where they come from the same domain
    a_pair_info, b_pair_info = remove_pairs_from_same_domain(batch_domain_indices, a_pair_info, b_pair_info)
    # calculate positive and negative semantic pair assignments
    pos_pair_info, neg_pair_info = find_matching_pairs(Y, a_pair_info, b_pair_info)
    return pos_pair_info, neg_pair_info


def count_per_domain_statistics(
    domain_idxs: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
        Args:
        -   domain_idxs: Pytorch tensor of shape (N,) showing assignment 
            of each example to each particular domain

        Returns:
        -   examples_per_domain: Numpy array of shape (max_idx+1,)
                where max_idx is the largest domain index. 
                Containss number of examples per each domain.
        -   unique_domain_idxs: Numpy array containing unique domain indices.
    """
    unique_domain_idxs = torch.unique(domain_idxs).cpu().numpy()
    # get the number of examples from each domain
    examples_per_domain = np.bincount( domain_idxs.cpu().numpy() )
    return examples_per_domain, unique_domain_idxs


def sample_px_locations_uniformly(
    batch_domain_indices: torch.Tensor, 
    unique_domain_idxs: np.ndarray, 
    w: int, 
    h: int,
    initial_sample_num: int
    ):
    """
        We are given a list of which batch examples belong to which domains.
        We first sample an array of uniformly random domain assignments for samples.
        Then for each domain sample, we choose which example it could have come from
        (sampling uniformly from the corresponding items in the batch).

        After an example is chosen (sampling uniformly over domains), we sample
        uniformly random pixel locations.

        We cannot sample uniformly over classes because of severe imbalance
        in each minibatch.

        Args:
        -   batch_domain_indices: Integer tensor of shape (B) representing
                which domain each minibatch example came from,
        -   unique_domain_idxs: Integer tensor of shape (D), if D domains
                present in a minibatch (not necessarily consecutive integers)
        -   w: integer representing label map width
        -   h: integer representing label map height
        -   initial_sample_num: integer representing initial number of samples

        Returns:
        -   all_batch_dim_idxs: Tensor of shape (initial_sample_num,)
        -   px_1d_x: Tensor of shape (initial_sample_num,) representing label
                map column indices
        -   px_1d_y: Tensor of shape (initial_sample_num,) representing label
                map row indices
    """
    sampled_domain_idxs = pytorch_random_choice(unique_domain_idxs, num_samples=initial_sample_num)

    # translate the sampled domains into batchh indices!
    all_batch_dim_idxs = torch.ones(initial_sample_num, dtype=torch.int64) * -1

    # need a loop here -- have to manipulate the batch indices per domain independently
    for domain_idx in unique_domain_idxs:
        num_samples_in_domain = int( (sampled_domain_idxs == domain_idx).sum().cpu().numpy() )

        # generate random example/batch indices for each domain 
        # (drawing from those batch examples that belong to domain)
        relevant_batch_idxs = (batch_domain_indices == domain_idx).nonzero().squeeze()
        if len(relevant_batch_idxs.shape) == 0: # when just a scalar
            relevant_batch_idxs = torch.tensor([ int(relevant_batch_idxs) ])
        domain_batch_dim_idxs = pytorch_random_choice(relevant_batch_idxs.cpu().numpy(), num_samples=num_samples_in_domain)
        
        relevant_sample_idxs = (sampled_domain_idxs == domain_idx).nonzero().squeeze()
        # place the selected batch locations into the correct places for this domain.
        all_batch_dim_idxs[relevant_sample_idxs] = domain_batch_dim_idxs

    px_1d_x = pytorch_random_choice(np.arange(w), num_samples=initial_sample_num)
    px_1d_y = pytorch_random_choice(np.arange(h), num_samples=initial_sample_num)

    return all_batch_dim_idxs, px_1d_x, px_1d_y


def shuffle_pytorch_tensor(x: torch.Tensor) -> torch.Tensor:
    """ Do not set torch.manual_seed(1) here, since we want to have
        a different random result each time.

        Args:
        -   x: (N,M) tensor we wish to shuffle along dim=0

        Returns:
        -   x: (N,M) tensor represneting shuffled version of input, along dim=0
    """
    n_examples = x.shape[0]
    r = torch.randperm(n_examples)
    return x[r]


def pytorch_random_choice(x: np.ndarray, num_samples: int) -> torch.Tensor:
    """ Provide Numpy's "random.choice" functionality to Pytorch.

        Do not put a manual seed in this function, since we want a different
        result each time we call it.

        Args:
        -   x: 1d Numpy array of shape (N,) to sample elements from
                (with replacement).
        -   num_samples

        Returns:
        -   torch.Tensor of shape (num_samples,)
    """
    # valid_idx = x.nonzero().view(-1)
    # choice = torch.multinomial(valid_idx.float(), 1)
    # return x[valid_idx[choice]]

    vals = np.random.choice(x, num_samples)
    return torch.from_numpy(vals)


def get_merged_pair_embeddings(pos_pair_info, neg_pair_info, embedding):
    """
    Given indices positive pairs (a,b) and negative pairs (a,b),
    obtain paired embeddings (stacked together).

        Args:
        -   pos_pair_info: (N,6) array representing (bi,hi,wi, bj,hj,wj)
                where these represent (batch index, row index, column index)
                into a NCHW tensor for paired samples A and B.
        -   neg_pair_info: (M,6) array, as above.
        -   embedding: (N,C,H,W) array representing output of a 
                feature extractor backbone, e.g. ResNet.

        Returns:
        -   y_c: (N+M) array representing binary same-class (1) vs. 
                different class (0) samples.
        -   a_embedding: (N+M,C) array
        -   b_embedding: (N+M,C) array
    """
    device = embedding.device

    n_pos = pos_pair_info.shape[0]
    n_neg = neg_pair_info.shape[0]
    y_c = torch.zeros(n_pos + n_neg, dtype=torch.float32)
    y_c[:n_pos] = 1.0 # means belong to same semantic class

    y_c = y_c.to(device) # Make sure in same memory as embedding (CPU or GPU)

    a_pos_embedding, b_pos_embedding = get_pair_embedding(pos_pair_info, embedding)
    a_neg_embedding, b_neg_embedding = get_pair_embedding(neg_pair_info, embedding)

    a_embedding = torch.cat([a_pos_embedding, a_neg_embedding])
    b_embedding = torch.cat([b_pos_embedding, b_neg_embedding])

    return y_c, a_embedding, b_embedding


def get_pair_embedding(
    pair_info: torch.Tensor, 
    embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    We are working with N pairs, the k'th pair is (a_k,b_k).

        Args:
        -   pair_info: (N,6) array representing (bi,hi,wi, bj,hj,wj)
                where these represent (batch index, row index, column index)
                into a NCHW tensor for paired samples A and B.
        -   embedding: NCHW tensor representing a minibatch of per-pixel embeddings

        Returns:
        -   a_embedding: (N,C) array representing channels at pixel (i,j) 
                of specific minibatch examples
        -   b_embedding: As above.
    """
    bi = pair_info[:,0]
    hi = pair_info[:,1]
    wi = pair_info[:,2]

    bj = pair_info[:,3]
    hj = pair_info[:,4]
    wj = pair_info[:,5]

    a_embedding = embedding[bi,:,hi,wi]
    b_embedding = embedding[bj,:,hj,wj]
    return a_embedding, b_embedding

