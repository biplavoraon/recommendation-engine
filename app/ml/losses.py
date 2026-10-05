import torch


def bpr_loss(
    positive_scores,
    negative_scores,
):
    difference = positive_scores - negative_scores

    return -torch.mean(
        torch.log(
            torch.sigmoid(difference) + 1e-8
        )
    )
