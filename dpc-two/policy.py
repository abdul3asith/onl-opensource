import torch


class TinyRobotModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.vision = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 5, stride=2),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
        )

        self.head = torch.nn.Sequential(
            torch.nn.Linear(16 + 7, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 10 * 7),
        )

    def forward(self, image, state):
        features = self.vision(image)
        x = torch.cat([features, state], dim=1)
        return self.head(x).view(-1, 10, 7)


def load_policy(device):
    torch.manual_seed(0)
    model = TinyRobotModel().to(device)
    model.eval()
    return model


def make_observation(device):
    image = torch.rand(1, 3, 96, 96, device=device)
    state = torch.rand(1, 7, device=device)
    return image, state


def act(model, image, state):
    with torch.no_grad():
        return model(image, state)
