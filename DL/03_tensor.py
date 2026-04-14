import torch, numpy

def demo01():
    t1 = torch.arange(0, 10, 2)
    print(f't1{t1}, type:{type(t1)}')
    print("-"*30)

    t2 = torch.linspace(1, 10, 4)
    print(f't1{t2}, type:{type(t2)}')
    print("-"*30)

def demo02():
    torch.manual_seed(0)
    t1 = torch.rand(size=(2,3))
    print(f't1{t1}, type:{type(t1)}')
    print("-"*30)

    t2 = torch.randn(2,3)
    print(f't1{t2}, type:{type(t2)}')
    print("-"*30)

    t3 = torch.randint(0, 10, (2,3))
    print(f't1{t3}, type:{type(t3)}')
    print("-"*30)

if __name__ == "__main__":
    #demo01()
    demo02()