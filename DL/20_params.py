from torch import nn

def demo01():
    linear = nn.Linear(5, 3)
    nn.init.uniform_(linear.weight)
    nn.init.uniform_(linear.bias)

    print(linear.weight.data)
    print(linear.bias.data)

def demo02():
    linear = nn.Linear(5, 3)
    nn.init.constant_(linear.weight, 3)
    nn.init.constant_(linear.bias, 3)

    print(linear.weight.data)
    print(linear.bias.data)

def demo03():
    linear = nn.Linear(5, 3)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)

    print(linear.weight.data)
    print(linear.bias.data)

def demo04():
    linear = nn.Linear(5, 3)
    nn.init.ones_(linear.weight)

    print(linear.weight.data)

def demo05():
    linear = nn.Linear(5, 3)
    nn.init.normal_(linear.weight)

    print(linear.weight.data)

def demo06():
    linear = nn.Linear(5, 3)
    nn.init.kaiming_normal_(linear.weight)

    print(linear.weight.data)

def demo07():
    linear = nn.Linear(5, 3)
    nn.init.xavier_normal_(linear.weight)

    print(linear.weight.data)

if __name__ == '__main__':
    demo01()
    print("=" * 60)
    demo02()
    print("=" * 60)
    demo03()
    print("=" * 60)
    demo04()
    print("=" * 60)
    demo05()
    print("=" * 60)
    demo06()
    print("=" * 60)
    demo07()