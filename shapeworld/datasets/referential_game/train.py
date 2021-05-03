import tqdm
from shapeworld.datasets.referential_game.referential_game import ReferentialGamePyTorchDataset
from typing import List, Tuple
import torch
from torch import nn
from torch.nn.modules.loss import MSELoss, CrossEntropyLoss
import torchvision.models as models

from transformers import BertModel, BertTokenizerFast

from shapeworld.datasets.referential_game.coordconv import CoordConv2d, AddCoords


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
        self.image_encoder = nn.Sequential(
            *list(models.resnet18(pretrained=True).children())[:-2]
        )

        # position embedding
        self.pos_emb = nn.Parameter(torch.randn(64, 7 * 7))
        

        # caption encoder
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        for param in self.bert.parameters():
            param.requires_grad = False

        # regressor
        self.mlp = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 576)
        )

        # trainer
        # self.loss_fn = MSELoss()
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters())

    def forward(
        self,
        images: torch.Tensor,
        text: List[str],
        labels: torch.Tensor = None,
        training: bool = False
        ) -> Tuple[torch.Tensor, float]:

        batch_size = images.size()[0]

        tokenized_text = self.tokenizer(
            text, return_tensors="pt", padding=True).to('cuda')
        encoded_text = self.bert(**tokenized_text).pooler_output

        encoded_images = self.image_encoder(images.cuda()).view(-1, 512, 7 * 7)
        encoded_images = torch.cat([encoded_images,
                                    self.pos_emb.expand(batch_size, -1, -1)], dim=1)

        logits = torch.matmul(self.mlp(encoded_text).unsqueeze(1), encoded_images).squeeze(1)

        loss = self.loss_fn(logits, labels.cuda())
        prediction = torch.argmax(logits, dim=-1).detach()

        if training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        return prediction, loss
        
def quantize_coordinates(x: torch.Tensor):
    return (x[:, 0] * 7).long() * 7 + (x[:, 1] * 7).long()

def eval(model: RGListener, dataset: ReferentialGamePyTorchDataset, batch_size=128) -> float:
    with torch.no_grad():
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, num_workers=6, pin_memory=True)
        losses = []
        accuracies = []
        for batch in dataloader:
            labels = quantize_coordinates(batch["target_coordinates"])
            prediction, loss = model(batch["image_tensor"],
                            batch["caption"], labels)
            losses.append(loss)
            accuracies.append(torch.sum(prediction == labels.cuda()).item() / batch["image_tensor"].size()[0])
        return sum(losses) / len(losses), sum(accuracies) / len(accuracies)


def train(model: RGListener, dataset: ReferentialGamePyTorchDataset,
          eval_dataset: ReferentialGamePyTorchDataset, batch_size=256,
          n_epochs=2, eval_interval=500):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)
    running_average = 0.06
    iteration = 0
    eval_loss = 0
    eval_acc = 0
    max_eval_acc = 0
    best_model = None
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        pbar = tqdm.tqdm(
            dataloader,
            desc=f"train_loss={running_average} eval_loss={eval_loss} eval_acc={eval_acc}"
        )
        for batch in pbar:
            _, loss = model(batch["image_tensor"], batch["caption"],
                            quantize_coordinates(batch["target_coordinates"]), training=True)
            running_average = running_average * 0.99 + loss * 0.01
            iteration += 1
            if iteration % eval_interval == 0:
                eval_loss, eval_acc = eval(model, eval_dataset)
                if eval_acc > max_eval_acc:
                    best_model = model.state_dict()
            pbar.set_description(
                f"train_loss={running_average} eval_loss={eval_loss} eval_acc={eval_acc}"
            )
    return best_model


if __name__ == '__main__':
    model = RGListener().cuda()
    # model.load_state_dict(torch.load("model.pt"))
    dataset = ReferentialGamePyTorchDataset(
        directory="/data/hzhu2/referential-game/train",
        filename="generated_1M_vol",
        volume=list(range(3))
    )
    eval_dataset = ReferentialGamePyTorchDataset(
        directory="/data/hzhu2/referential-game/dev",
        filename="generated_10K_vol",
        volume=10)
    best_model = train(model, dataset, eval_dataset)
    torch.save(best_model, "model.pt")
