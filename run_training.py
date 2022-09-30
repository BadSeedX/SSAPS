from pathlib import Path

import numpy as np
from tqdm import tqdm
import datetime
import argparse

import torch
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


from dataset import MVTecAT, Repeat
from augmented_patches import AugPatch
from model import APSeg
from eval import eval_model

def run_training(data_type="screw",
                 model_dir="models",
                 epochs=300,
                 pretrained=True,
                 test_epochs=10,
                 freeze_model=20,
                 learninig_rate=0.03,
                 batch_size=64,
                 device = "cuda",
                 workers=8,
                 size = 384):
    torch.multiprocessing.freeze_support()

    weight_decay = 0.00003
    momentum = 0.9
    model_name = f"model-{data_type}" + '-{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now() )

    after_cutpaste_transform = transforms.Compose([])
    after_cutpaste_transform.transforms.append(transforms.ToTensor())
    after_cutpaste_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

    train_transform = transforms.Compose([])
    train_transform.transforms.append(transforms.Resize((size,size)))
    train_transform.transforms.append(AugPatch(transform = after_cutpaste_transform))

    train_data = MVTecAT("/home/BadSeedX/Datasets/MVTEC/MVTec-AD/", data_type, transform = train_transform, size=size)
    dataloader = DataLoader(Repeat(train_data, 3000), batch_size=batch_size, drop_last=True,
                                 shuffle=True, num_workers=workers, persistent_workers=True, pin_memory=True, prefetch_factor=5)

    writer = SummaryWriter(Path("logdirs") / model_name)

    model = APSeg(pretrained=pretrained)
    model.to(device)
    model.train()

    if freeze_model > 0 and pretrained:
        model.freeze_resnet()

    loss_fn = nn.BCELoss().to(device)

    optimizer = optim.SGD(model.parameters(), lr=learninig_rate, momentum=momentum,  weight_decay=weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, epochs)

    def get_data_inf():
        while True:
            for out in enumerate(dataloader):
                yield out
    dataloader_inf =  get_data_inf()

    train_loss = 0
    sota = 0
    for step in tqdm(range(epochs)):
        epoch = int(step / 1)
        if epoch == freeze_model:
            model.unfreeze()

        batch_idx, data = next(dataloader_inf)
        imgs, labels = data[0], data[1]

        torch.set_printoptions(threshold=np.sys.maxsize)

        xs = [x.to(device) for x in imgs]
        xl = [x.to(device) for x in labels]

        optimizer.zero_grad()
        xc = torch.cat(xs, axis=0)
        masks = torch.cat(xl, axis=0)

        maps, embed = model(xc)
        maps = torch.sigmoid(maps)
        loss = loss_fn(maps, masks)

        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if scheduler is not None:
            scheduler.step(epoch)
        
        writer.add_scalar('loss', loss.item(), step)

        if scheduler is not None:
            writer.add_scalar('lr', scheduler.get_last_lr()[0], step)

        if test_epochs > 0 and epoch % test_epochs == 0:
            model.eval()
            roc_auc= eval_model(model_name, data_type, device=device,
                                size=size,
                                model=model)
            model.train()
            writer.add_scalar('eval_auc', roc_auc, step)
            if roc_auc > sota:
                sota = roc_auc
                torch.save(model.state_dict(), model_dir / f"{model_name}.tch")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=300, type=int,
                        help='number of epochs to train the model')
    parser.add_argument('--model_dir', default="models",
                        help='output folder of the models')
    parser.add_argument('--no-pretrained', dest='pretrained', default=True, action='store_false',
                        help='use pretrained values to initalize ResNet18 , (default: True)')
    parser.add_argument('--test_epochs', default=3, type=int,
                        help='interval to calculate the auroc during trainig')
    parser.add_argument('--freeze_resnet', default=20, type=int,
                        help='number of epochs to freeze resnet')
    parser.add_argument('--lr', default=0.03, type=float,
                        help='learning rate')
    parser.add_argument('--optim', default="sgd",
                        help='optimizer')
    parser.add_argument('--batch_size', default=8, type=int,
                        help='batch size, real batch_size is  2x batch_size')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use cuda')
    parser.add_argument('--workers', default=1, type=int,
                        help="number of workers in data_loader")

    args = parser.parse_args()
    all_types = ['bottle', 'cable', 'carpet', 'leather', 'screw', 'grid', 'pill', 'capsule', 'transistor', 'metal_nut',
                 'hazelnut', 'zipper', 'toothbrush', 'tile', 'wood']
    
    device = "cuda" if args.cuda else "cpu"

    Path(args.model_dir).mkdir(exist_ok=True, parents=True)
    for data_type in all_types:
        run_training(data_type,
                     model_dir=Path(args.model_dir),
                     epochs=args.epochs,
                     pretrained=args.pretrained,
                     test_epochs=args.test_epochs,
                     freeze_model=args.freeze_resnet,
                     learninig_rate=args.lr,
                     batch_size=args.batch_size,
                     device=device,
                     workers=args.workers)
