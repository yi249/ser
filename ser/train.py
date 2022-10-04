from torch import optim
import torch
import torch.nn.functional as F

from ser.model import Net

import visdom

vis = visdom.Visdom()

loss_plot = vis.line(X=torch.zeros([1]), Y=torch.zeros([1]))
accuracy_plot = vis.line(X=torch.zeros([1]), Y=torch.zeros([1]))

def train(run_path, params, train_dataloader, val_dataloader, device):
    # setup model
    model = Net().to(device)

    # setup params
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # train
    for epoch in range(params.epochs):
        _train_batch(model, train_dataloader, optimizer, epoch, device)
        _val_batch(model, val_dataloader, device, epoch)

    # save model and save model params
    torch.save(model, run_path / "model.pt")


def _train_batch(model, dataloader, optimizer, epoch, device):
    correct = 0
    for i, (images, labels) in enumerate(dataloader):
        images, labels = images.to(device), labels.to(device)
        model.train()
        optimizer.zero_grad()
        output = model(images)
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
        print(
            f"Train Epoch: {epoch} | Batch: {i}/{len(dataloader)} "
            f"| Loss: {loss.item():.4f}"
        )
        vis.line(X=torch.ones((1,1)).cpu()*i+(epoch*60),Y=torch.Tensor([loss]).unsqueeze(0).cpu(),win=loss_plot, update='append')
    accuracy = correct / len(dataloader.dataset)
    vis.line(X=torch.ones((1,1)).cpu()*epoch,Y=torch.Tensor([accuracy]).unsqueeze(0).cpu(),win=accuracy_plot, update='append', name="train")



@torch.no_grad()
def _val_batch(model, dataloader, device, epoch):
    val_loss = 0
    correct = 0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        model.eval()
        output = model(images)
        val_loss += F.nll_loss(output, labels, reduction="sum").item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(labels.view_as(pred)).sum().item()
    val_loss /= len(dataloader.dataset)
    accuracy = correct / len(dataloader.dataset)
    print(f"Val Epoch: {epoch} | Avg Loss: {val_loss:.4f} | Accuracy: {accuracy}")
    vis.line(X=torch.ones((1,1)).cpu()*epoch,Y=torch.Tensor([accuracy]).unsqueeze(0).cpu(),win=accuracy_plot, update='append', name="validation")
