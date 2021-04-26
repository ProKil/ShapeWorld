import tqdm
from shapeworld.datasets.referential_game.referential_game import ReferentialGamePyTorchDataset
from typing import List, Tuple
import torch
from torch import nn
from torch.nn.modules.loss import MSELoss
import torchvision.models as models

from transformers import BertModel, BertTokenizerFast

from shapeworld.datasets.referential_game.coordconv import CoordConv2d, AddCoords


class CoordConvNet(nn.Module):
    def __init__(self, x_dim=224, y_dim=224):
        super(CoordConvNet, self).__init__()
        include_r = False
        self.model = nn.Sequential(*[
            AddCoords(rank=2, with_r=include_r), # (batch, 224, 224, 4 or 5)
            nn.Conv2d(5, 8, (1,1)),
            nn.ReLU(),
            nn.Conv2d(8, 8, (1,1)),
            nn.ReLU(),
            nn.Conv2d(8, 8, (1,1)),
            nn.ReLU(),
            nn.Conv2d(8, 8, (3,3), padding=2),
            nn.ReLU(),
            nn.Conv2d(8, 2, (3,3), padding=2),
            nn.MaxPool2d(32,stride=32),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(2 * (x_dim // 32) * (y_dim // 32), 256)
            ])

    def forward(self, x):
        return self.model(x)

class ImageEncoder(nn.Module):
    def __init__(self, model: str = "coordconv"):
        super(ImageEncoder, self).__init__()
        if model == "coordconv":
            pass
        elif model == "resnet":
            resnet50 = models.resnet50(pretrained=False)
            modules = list(resnet50.children())[:-1]
            self.model = nn.Sequential(*modules)
        else:
            raise NotImplementedError

    def forward(self, x): return self.model(x)

class RGListener(nn.Module):
    def __init__(self):
        super(RGListener, self).__init__()

        # image encoder
        # resnet152 = models.resnet152(pretrained=True)
        # modules=list(resnet152.children())[:-1]
        # self.resnet152=nn.Sequential(*modules)
        # for p in self.resnet152.parameters():
        #     p.requires_grad = False
        self.resnet152 = CoordConvNet()
        

        # caption encoder
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False

        # regressor
        self.regressor = nn.Sequential(
            nn.Linear(768+256, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid())

        # trainer
        self.loss_fn = MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(self, images: torch.Tensor, text: List[str], target_coordinates: torch.Tensor, training: bool = False) -> Tuple[torch.Tensor, float]:
        tokenized_text = self.tokenizer(
            text, return_tensors="pt", padding=True).to('cuda')
        encoded_text = self.bert(**tokenized_text).pooler_output

        encoded_images = self.resnet152(images.cuda()).squeeze()

        coordinates = self.regressor(
            torch.cat([encoded_images, encoded_text], dim=-1))
        loss = self.loss_fn(coordinates, target_coordinates.float().cuda())

        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return coordinates.detach(), loss.item()


def eval(model: RGListener, dataset: ReferentialGamePyTorchDataset, batch_size=128) -> float:
    with torch.no_grad():
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, pin_memory=True)
        losses = []
        for batch in dataloader:
            _, loss = model(batch["image_tensor"],
                            batch["caption"], batch["target_coordinates"])
            losses.append(loss)
        return sum(losses) / len(losses)


def train(model: RGListener, dataset: ReferentialGamePyTorchDataset,
          eval_dataset: ReferentialGamePyTorchDataset, batch_size=2048, n_epochs=100):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, pin_memory=True)
    running_average = 0.06
    for epoch in range(n_epochs):
        pbar = tqdm.tqdm(dataloader, desc=f"running_average={running_average}")
        for batch in pbar:
            _, loss = model(batch["image_tensor"], batch["caption"],
                            batch["target_coordinates"], training=True)
            running_average = running_average * 0.99 + loss * 0.01
            pbar.set_description(f"running_average={running_average}")
        print(f"loss on dev set: {eval(model, eval_dataset)}")


if __name__ == '__main__':
    model = RGListener().cuda()
    dataset = ReferentialGamePyTorchDataset(
        directory="/data/hzhu2/referential-game/train")
    eval_dataset = ReferentialGamePyTorchDataset(
        directory="/data/hzhu2/referential-game/dev")
    train(model, dataset, eval_dataset)
    torch.save(model.state_dict(), "model.pt")
