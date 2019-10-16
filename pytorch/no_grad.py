import time

import torchvision.models as models
import torch

if __name__ == '__main__':

    r50 = models.resnet50(pretrained=True)
    r50.eval()
    r50.cuda()

    rnd_batch = torch.zeros((32, 3, 224, 224)).random_().cuda()

    print(rnd_batch.size())


    print('[/] forward pass...')
    start_time = time.time()

    for i in range(100):
        with torch.no_grad():
            predictions = r50(rnd_batch)

    print('[/] done: {}'.format(time.time() - start_time))
