from sklearn.metrics import roc_curve, auc
from torchvision import transforms
from torch.utils.data import DataLoader
import torch

from dataset import MVTecAT
from model import APSeg
import argparse
from pathlib import Path
from collections import defaultdict
from density import GaussianDensitySklearn, GaussianDensityTorch
import pandas as pd

test_data_eval = None
test_transform = None
cached_type = None

# get noemal patterns
def get_train_embeds(model, size, defect_type, transform, device):
    valid_data = MVTecAT("/home/BadSeedX/Datasets/MVTEC/MVTec-AD/", defect_type, size, transform=transform, mode="valid")
    dataloader_train = DataLoader(valid_data, batch_size=16, shuffle=False, num_workers=0)

    train_embed = []
    with torch.no_grad():
        for x in dataloader_train:
            score, embed = model(x.to(device))
            train_embed.append(embed.cpu())
    train_embed = torch.cat(train_embed)
    return train_embed

def eval_model(modelname, defect_type, device="cpu", size=384, model=None, density=GaussianDensityTorch()):
    # create test dataset
    global test_data_eval,test_transform, cached_type

    if test_data_eval is None or cached_type != defect_type:
        cached_type = defect_type
        test_transform = transforms.Compose([])
        test_transform.transforms.append(transforms.Resize((size,size)))
        test_transform.transforms.append(transforms.ToTensor())
        test_transform.transforms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                            std=[0.229, 0.224, 0.225]))
        test_data_eval = MVTecAT("/home/BadSeedX/Datasets/MVTEC/MVTec-AD/", defect_type, size, transform = test_transform, mode="test")

    dataloader_test = DataLoader(test_data_eval, batch_size=16, shuffle=False, num_workers=0)

    # create model
    if model is None:
        weights = torch.load(modelname, map_location='cpu')
        model = APSeg(pretrained=False, num_classes=2, device=device)
        model.load_state_dict(weights)
        model.to(device)
        model.eval()

    labels = []
    scores = []
    with torch.no_grad():
        for x, label in dataloader_test:
            maps, embed = model(x.to(device))

            labels.append(label.cpu())
            scores.append(embed.cpu())

    labels = torch.cat(labels)
    scores = torch.cat(scores)

    train_embed = get_train_embeds(model, size, defect_type, test_transform, device)
    scores = torch.nn.functional.normalize(scores, p=2, dim=1)
    train_embed = torch.nn.functional.normalize(train_embed, p=2, dim=1)

    # anomaly detection
    density.fit(train_embed)
    distances = density.predict(scores)

    fpr, tpr, _ = roc_curve(labels, distances)
    roc_auc = auc(fpr, tpr)

    return roc_auc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='eval models')

    parser.add_argument('--model_dir', default="models",
                        help=' directory containing models to evaluate')

    parser.add_argument('--cuda', default=True, type=bool,
                        help='use cuda for model predictions')

    parser.add_argument('--density', default="torch",
                        help='density implementation to use')


    args = parser.parse_args()

    all_types = ['grid', 'screw', 'pill', 'capsule', 'transistor', 'metal_nut', 'bottle', 'hazelnut', 'zipper',
                 'toothbrush', 'cable', 'carpet', 'leather', 'tile', 'wood']

    device = "cuda" if args.cuda else "cpu"

    density = GaussianDensityTorch

    model_names = [list(Path(args.model_dir).glob(f"model-{data_type}*"))[0] for data_type in all_types if
                   len(list(Path(args.model_dir).glob(f"model-{data_type}*"))) > 0]
    if len(model_names) < len(all_types):
        print("warning: not all types present in folder")

    obj = defaultdict(list)
    for model_name, data_type in zip(model_names, all_types):
        roc_auc = eval_model(model_name, data_type, save_plots=args.save_plots, device=device,
                             head_layer=args.head_layer, density=density())
        obj["defect_type"].append(data_type)
        obj["roc_auc"].append(roc_auc)

    # save pandas dataframe
    eval_dir = Path("eval") / args.model_dir
    eval_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(obj)
    df.to_csv(str(eval_dir) + "_perf.csv")
