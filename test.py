import torch

if __name__ == '__main__':
    a = torch.tensor([0.])
    print(a)
    print(a.data)
    print(a.item())
    print(a.squeeze())

    b = torch.tensor(0.)
    print(b)
    print(b.data)
    print(b.item())
    print(b.squeeze())

    c = torch.tensor([0])
    d = torch.tensor(0.)

    print(c.data)
    print(c.data[0])
    print(c.data == 0)
    print(c.data[0] == 0)
    print(c.item() == 0)