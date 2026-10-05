import torch


def pairwise_ranking_loss(
    positive_scores,
    negative_scores,
):
    """
    Encourage:
        positive_score > negative_score
    """

    difference = (
        positive_scores - negative_scores
    )

    return -torch.mean(
        torch.log(
            torch.sigmoid(difference) + 1e-8
        )
    )
    
def listwise_ranking_loss(scores):
    """
    Listwise softmax loss.

    scores:
        [batch_size, group_size]

    The positive item must be at index 0.
    """

    log_probs = torch.log_softmax(scores, dim=1)

    return -log_probs[:, 0].mean()
