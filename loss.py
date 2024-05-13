import torch
from torch import Tensor, nn

from constants import THRESHOLD, TRADEOFF_ANGLE, TRADEOFF_SCALE


class DareGramLoss(nn.Module):
    def __init__(self):
        super(DareGramLoss, self).__init__()

    def forward(self, H1: Tensor, H2: Tensor) -> Tensor:
        # b = batch size
        # p = dimensionality of feature space
        b, p = H1.shape

        A = torch.cat((torch.ones(b, 1).to(device), H1), 1)
        B = torch.cat((torch.ones(b, 1).to(device), H2), 1)

        cov_A = A.t() @ A
        cov_B = B.t() @ B

        L_A: Tensor
        L_B: Tensor
        _, L_A, _ = torch.linalg.svd(cov_A)
        _, L_B, _ = torch.linalg.svd(cov_B)

        eigen_A = torch.cumsum(L_A.detach(), dim=0) / L_A.sum()
        eigen_B = torch.cumsum(L_B.detach(), dim=0) / L_B.sum()

        T_A: float = max(THRESHOLD, eigen_A[1].detach())
        T_B: float = max(THRESHOLD, eigen_B[1].detach())

        index_A = torch.argwhere(eigen_A.detach() <= T_A)[-1].item()
        index_B = torch.argwhere(eigen_B.detach() <= T_B)[-1].item()

        k = max(index_A, index_B)

        A = torch.linalg.pinv(cov_A, rtol=(L_A[k] / L_A[0]).detach())
        B = torch.linalg.pinv(cov_B, rtol=(L_B[k] / L_B[0]).detach())

        cos_sim = nn.CosineSimilarity(dim=0, eps=1e-6)
        cos = torch.dist(torch.ones((p + 1)).to(device), (cos_sim(A, B)), p=1) / (p + 1)

        return (
            TRADEOFF_ANGLE * cos + TRADEOFF_SCALE * torch.dist((L_A[:k]), (L_B[:k])) / k
        )
