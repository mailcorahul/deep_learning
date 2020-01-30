import torch
import torchvision.models as models

import time
import sys

if __name__ == '__main__':

    net = getattr(models, sys.argv[1])(pretrained=True)
    net.cuda()
    iter = 10
    batch_size = 50
    for i in range(iter):
        batch_images = torch.ones((batch_size, 3, 224, 224)).normal_(mean=0, std=1.).cuda()
        print(batch_images.size(), batch_images.dtype, batch_images[0].dtype)
        torch.cuda.synchronize()
        start_time = time.time()
        with torch.no_grad():
            output = net(batch_images)
            torch.cuda.synchronize()
        print('[/] time taken: {:.4f}'.format(time.time() - start_time))
