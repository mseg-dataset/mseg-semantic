
## Domain Generalization (DG) Implementation

As discussed in the [MSeg paper](), we apply a state-of-the-art Domain Generalization (DG) algorithm [1] to MSeg which uses the Classification and Contrastive Semantic Alignment (CCSA) loss. We find that this DG technique seems to hurt performance significantly compared with our technique.

Suppose we have a deep network h(g(X)), where g(·) is feature extractor, and h(·) is classifier. For context, CCSA ensures sure that the embedding function g(·) maps to a domain invariant space. To do so, we consider every distinct unordered pair of source domains (u, v), and impose the semantic alignment loss as well as the separation loss.

We adapt the DG technique proposed for the image classification task in [1] to semantic segmentation as follows:
- We add no new parameters to PSPNet, but simply add a contrastive loss. 
- We feed a minibatch of 128 crops X through g(·), the ResNet backbone of a PSPNet. We then sample N positive pairs of feature map embeddings, corresponding to an 8 × 8 pixel
region per feature map location, and 3N negative pairs. In our experiments, we set N = 1000 or N = 100.
- We choose these 4N pairs by **first** sampling uniformly randomly **from domains**, and **subsequently sampling uniformly randomly from pixel locations** available in each input crop. 
- When N > 1000 with a batch size of 128, CUDA memory is insufficient to compute the Euclidean distances between embeddings, forcing us to use N = 1000. In order to determine positive or negative pairs, we downsample the ground truth label map by 8x with ‘nearest’ interpolation
and then compare the corresponding ground truth semantic class of feature map locations. In such a way, we identify N pairs of embeddings that belong to the same semantic class.

### Differences from [Original Implementation](https://github.com/samotiian/CCSA)

Our implementation differs from [1] in the following ways: 
1. We sample pairs on the fly, instead of choosing fixed pairs for each epoch in advance.
2. We sample uniformly randomly an image crop from all domains first, then sample uniformly from pixel locations in each image crop. Finding evenly-balanced pairs from each class would
require sampling a very large number of pairs (perhaps billions, since we observe 10^5 times more density in the most populous MSeg class vs. the least populous class).
3. We compute classification loss over all pixel locations and the contrastive loss only
over sampled pixel locations, whereas [1] computed classification loss only over sampled pairs. 
4. We use SGD with momentum, a standard optimization technique for PSPNet, rather than using Adadelta.
5. We use a ResNet backbone instead of a VGG backbone for the feature extractor.

### Code Structure

The implementation is found in the following files:
-	`ccsa_utils.py`: Tools for sampling pairs for a contrastive loss.
-	`ccsa_pspnet.py`: PSPNet model architecture with contrastive loss added before PPM.
-	`ccsa_data.py`: Pytorch dataloader to form minibatch with uniform sampling from each domain.

### References
[1] Saeid Motiian, Marco Piccirilli, Donald A. Adjeroh, and Gianfranco Doretto. [Unified deep supervised domain adaptation and generalization.](https://arxiv.org/abs/1709.10190) In The IEEE International Conference on Computer Vision (ICCV), Oct 2017.