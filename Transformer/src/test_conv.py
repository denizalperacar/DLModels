from torch import randn, cat, device
from torch.nn import Conv2d, Dropout


n = 1000
l = 256

dev = device("cuda:0")

a = Conv2d(6, 128, 4, 2).to(dev)
b = Conv2d(128, 128, 4, 3).to(dev)
c = Conv2d(128, 256, 4, 1).to(dev)


for i in range(1000):
    out = []
    x = randn(2, n, 6, 32, 32).to(dev)
    for ele in x:
        out.append(c(b(a(ele))).reshape(1, n, l))
    y = cat(out, 0)

    print(y.shape)
