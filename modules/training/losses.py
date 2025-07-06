import torch
import torch.nn.functional as F

from modules.training import utils

from third_party.alike_wrapper import extract_alike_kpts


def smooth_l1_loss(input, target, beta=2.0, size_average=True):
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return loss.mean() if size_average else loss.sum()

# def fine_loss(f1, f2, pts1, pts2, fine_module, ws=7):
#     '''
#         Compute Fine features and spatial loss
#     '''
#     C, H, W = f1.shape
#     N = len(pts1)

#     #Sort random offsets
#     with torch.no_grad():
#         a = -(ws//2)
#         b = (ws//2)
#         offset_gt = (a - b) * torch.rand(N, 2, device = f1.device) + b
#         pts2_random = pts2 + offset_gt

#     #pdb.set_trace()
#     patches1 = utils.crop_patches(f1.unsqueeze(0), (pts1+0.5).long(), size=ws).view(C, N, ws * ws).permute(1, 2, 0) #[N, ws*ws, C]
#     patches2 = utils.crop_patches(f2.unsqueeze(0), (pts2_random+0.5).long(), size=ws).view(C, N, ws * ws).permute(1, 2, 0)  #[N, ws*ws, C]

#     #Apply transformer
#     patches1, patches2 = fine_module(patches1, patches2)

#     features = patches1.view(N, ws, ws, C)[:, ws//2, ws//2, :].view(N, 1, 1, C) # [N, 1, 1, C]
#     patches2 = patches2.view(N, ws, ws, C) # [N, w, w, C]

#     #Dot Product
#     heatmap_match = (features * patches2).sum(-1)
#     offset_coords = utils.subpix_softmax2d(heatmap_match)

#     #Invert offset because center crop inverts it
#     offset_gt = -offset_gt 

#     #MSE
#     error = ((offset_coords - offset_gt)**2).sum(-1).mean()

#     #error = smooth_l1_loss(offset_coords, offset_gt)

#     return error

# def hard_triplet_loss(X,Y, margin = 0.5):

#     if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
#         raise RuntimeError('Error: X and Y shapes must match and be 2D matrices')

#     dist_mat = torch.cdist(X, Y, p=2.0)
#     dist_pos = torch.diag(dist_mat)
#     dist_neg = dist_mat + 100.*torch.eye(*dist_mat.size(), dtype = dist_mat.dtype, 
#             device = dist_mat.get_device() if dist_mat.is_cuda else torch.device("cpu"))

#     #filter repeated patches on negative distances to avoid weird stuff on gradients
#     dist_neg = dist_neg + dist_neg.le(0.01).float()*100.

#     #Margin Ranking Loss
#     hard_neg = torch.min(dist_neg, 1)[0]

#     loss = torch.clamp(margin + dist_pos - hard_neg, min=0.)

#     return loss.mean()

# def contrastive_loss(features1, features2, margin=1.0):
    # 计算特征之间的欧氏距离
    distance = F.pairwise_distance(features1, features2)
    # 对比损失
    loss = torch.mean((1 - distance) ** 2)  # 让相似特征的距离接近 0
    return loss