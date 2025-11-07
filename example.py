import torch

from my_exes import MyEx


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(8),
            torch.nn.ReLU(),
        )
        self.fc1 = torch.nn.Sequential(
            torch.nn.Linear(2048, 256),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.fc1(out.view(out.size()[0], -1))
        return out


def main():
    my_ex = MyEx("example.yaml")

    net = Net()
    my_ex.log_model_graph(net, sample_input=torch.randn(1, 1, 16, 16))

    # if my_ex.cfg.optimizer.algo == "Adam":
    epochs = my_ex.cfg.epochs
    batch_size = my_ex.cfg.batch_size
    optimizer = torch.optim.Adam(net.parameters(), lr=my_ex.cfg.lr)

    data = torch.randn(batch_size * 2, 1, 16, 16)
    num_batches = data.shape[0] // batch_size

    for e in range(epochs):
        for b in range(num_batches):
            batch_input = data[b * batch_size:(b + 1) * batch_size]
            out = net(batch_input)
            loss = torch.mean((out - batch_input.view(out.size()[0], -1)) ** 2)
            loss.backward()
            optimizer.step()

            my_ex.log("train", "loss", loss.item(), e * num_batches + b)
            print(f"Epoch {e}: Loss={loss.item()}")


if __name__ == "__main__":
    main()
