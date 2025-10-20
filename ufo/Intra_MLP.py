import torch
import numpy

# Borrowed from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/models/pointnet2_utils.py

def index_points(device, points, idx):
    """Gather point features by batch index."""
    bsz = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(bsz, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def knn_l2(device, net, k, u):
    """Batched k-NN using L2 distance."""
    inf = 1e8
    batch_size = net.size(0)
    npoint = net.size(1)

    square = torch.pow(torch.norm(net, dim=2, keepdim=True), 2)

    def u_block(batch_sz, npoint_sz, block_u):
        block = numpy.zeros([batch_sz, npoint_sz, npoint_sz])
        n = npoint_sz // block_u
        for i in range(n):
            block[:, (i * block_u):(i * block_u + block_u), (i * block_u):(i * block_u + block_u)] = (
                numpy.ones([batch_sz, block_u, block_u]) * (-inf)
            )
        return block

    minus_distance = (
        2 * torch.matmul(net, net.transpose(2, 1))
        - square
        - square.transpose(2, 1)
        + torch.tensor(u_block(batch_size, npoint, u), device=device)
    )
    _, indices = torch.topk(minus_distance, k, largest=True, sorted=False)
    return indices
