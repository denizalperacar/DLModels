from torch import randn, cat, device
from torch.nn import Conv2d, Dropout, ConvTranspose2d


n = 1000
l = 256

dev = device("cuda:0")

a = ConvTranspose2d(256, 128, 4).to(dev)
b = ConvTranspose2d(128, 128, 4, 4).to(dev)
c = ConvTranspose2d(128, 6, 2, 2).to(dev)





x = randn(n, 256, 1, 1).to(dev)
y = c(b(a(x)))
print(y.shape)
