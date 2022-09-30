import torch.nn as nn
import timm


class APSeg(nn.Module):
    def __init__(self, pretrained=True):
        super(APSeg, self).__init__()

        # code with timm
        self.fe = timm.create_model('convnext_base_384_in22ft1k', pretrained=pretrained, features_only=True)
        self.n_class = 2
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.deconv5 = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.classifier5 = nn.Conv2d(64, self.n_class, kernel_size=1)

        self.classifier = nn.Conv2d(64, self.n_class, kernel_size=1)

        self.norm = nn.LayerNorm(1024, eps=1e-6)
        self.deconv = nn.ConvTranspose2d(2, 2, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.linear = nn.Linear(1024, 512)


    def forward(self, x):

        embeds = self.fe(x)

        emb = self.norm(embeds[-1].mean([-2, -1]))
        # emb = self.linear(emb)

        # RDN module
        score = self.bn1(self.relu(self.deconv1(embeds[3])))
        score = score + embeds[2]
        score = self.bn2(self.relu(self.deconv2(score)))
        score = score + embeds[1]
        score = self.bn3(self.relu(self.deconv3(score)))
        score = score + embeds[0]
        score = self.bn4(self.relu(self.deconv4(score)))

        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier5(score)

        return score, emb
    
    def freeze_resnet(self):
        for param in self.fe.parameters():
            param.requires_grad = False


    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True
