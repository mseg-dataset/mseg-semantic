

import numpy as np

"""
As we loop through two dataloaders (unrelabeled and relabeled ground truth),
we feed each pair of label maps to this function.

for o_img, r_img in zip(orig_dataloader, relabeled_dataloader)
    // for every single pixel (no sounding horribly slow...) see if it matches orig label map. 
    
    mod_idxs = np.where(o_img != r_img)
    // remap the predictions at these indices only
    for mod_idx in mod_idxs:
        if prediction[mod_idx] == r_img[mod_idx]:
            # set prediction's class index to not be what network said, but what original ground truth wass
            prediction[mod_idx] = o_img[mod_idx] # substitute prediction with ground truth..

"""


def convert_label_to_pred_taxonomy(target_img):
    """ """
    if self.eval_taxonomy == 'universal':
        _, target_img = self.to_universal_transform(target_img, target_img)
        return target_img.type(torch.uint8).numpy()
    else:
        return target_img


def eval_relabeled_pair(pred: np.ndarray, target_img: np.ndarray, target_img_relabeled: np.ndarray):
    """
        Rather than unrelabeled model on the univ. relabeled data, we instead map correctness
        of relabeled model on relabeled univ. data, back to the unrelabeled univ. space. 
        Could go either way since it is a 1:1 mapping per pixel. Operate on universal taxonomy.

        If in universal space, any pixel differs, if in coco-relabeled-universal prediction is correct,
        we map it to the coco-unrelabeled-universal label and thus count it as correct.

        Look at diff on disk to see where ground truth was changed. Relabeled data is the ``oracle''.
        coco-unrel->coco-unrel-universal, then coco-unrel-universal is already in universal on disk.

        Args:
        -   pred: Numpy array of shape (H,W) of dtype np.int64 representing prediction,
                predictions are already in universal taxonomy.
        -   target_img: Numpy array of shape (H,W) of dtype np.int64 representing unrelabeled ground truth
        -   target_img_relabeled:  Numpy array of shape (H,W) of dtype np.int64 representing relabeled ground truth

        Returns:
        -   
    """
    target_img = convert_label_to_pred_taxonomy(target_img)
    # construct a "correct" target image here: if pixel A is relabeled as pixel B, and prediction is B, then map prediction B back to A

    relabeled_pixels = (target_img_relabeled != target_img)

    correct_pixels = (pred == target_img_relabeled)

    correct_relabeled_pixels = relabeled_pixels * correct_pixels

    pred_final = np.where(correct_relabeled_pixels, target_img, pred)
    accuracy_before = (pred == target_img).sum()/target_img.size
    accuracy_after = (pred_final == target_img).sum()/target_img.size
    print(np.sum(target_img_relabeled == target_img)/target_img.size, accuracy_before, accuracy_after)

    # pred[correct_pixels]

    return pred_final, target_img


def test_eval_relabeled_pair1():
    """
    Person is 0
    Motorcyclist is 1
    Sky is 2
    """
    target_img = np.array(
        [
            [2,2,2,2],
            [2,0,0,2],
            [2,0,0,2],
            [2,0,0,2]
        ])
    target_img_relabeled = np.array(
        [
            [2,2,2,2],
            [2,1,1,2],
            [2,1,1,2],
            [2,1,1,2]
        ])
    eval_relabeled_pair(target_img, target_img_relabeled)




if __name__ == '__main__':
    """ """
    test_eval_relabeled_pair1()



