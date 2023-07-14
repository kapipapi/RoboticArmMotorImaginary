import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from utils.dataset import EEGDataset


def train(args, model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def evaluate(model, device, criterion, test_loader):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    parser = argparse.ArgumentParser()

    # TODO: Change default
    parser.add_argument("--model", default="", help="Model to train")
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size")
    parser.add_argument("--epochs", default=100, help="Number of Epochs")
    parser.add_argument('--lr', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--seed', default=None, type=int, help='Seed for initializing training. ')
    parser.add_argument('--dry-run', default=False, help='Single model pass')
    parser.add_argument('--gamma', type=float, default=0.7, help='Scheduler gamma')
    parser.add_argument('--save_model', default=False, help='Save model')

    args = parser.parse_args()

    # TODO: Add model options
    if args.model == "":
        pass
    else:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # TODO: Change model
    model = ""
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    batch_size = args.batch_size
    epochs = args.epochs

    # TODO: Change dir
    dataset = EEGDataset("/root_dir")

    train_set, test_set = torch.utils.data.random_split(dataset, [0.7, 0.3])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

    try:
        for epoch in range(1, epochs + 1):
            train(args, model, device, train_loader, optimizer, criterion, epoch)
            evaluate(model, device, criterion, test_loader)
            scheduler.step()
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    if args.save_model:
        torch.save(model.state_dict(), f"{args.model}.pt")


if __name__ == '__main__':
    main()
