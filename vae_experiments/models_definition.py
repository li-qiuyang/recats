import torch
from torch import nn
from torch.nn import functional as F

class Encoder(nn.Module):
    def __init__(self, args, input_features):
        # 编码器
        super(Encoder, self).__init__()
        self.input_features = input_features
        self.lstm = nn.LSTM(
            input_size=input_features,
            hidden_size=128,
            bidirectional=True,
            batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 64))
        self.fc1 = nn.Linear(self.input_features, 256)
        # self.b1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        # self.b2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        # self.b3 = nn.BatchNorm1d(64)
        self.mu_x = nn.Linear(64, 200)
        self.logvar_x = nn.Linear(64, 200)
        self.mu_w = nn.Linear(64, 150)
        self.logvar_w = nn.Linear(64, 150)
        self.qz = nn.Linear(64, args.K)
    def forward(self, X):
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        X = F.relu(X)
        X = self.fc3(X)
        X = F.relu(X)
        qz = F.softmax(self.qz(X), dim=-1)

        mu_x = self.mu_x(X)
        logvar_x = self.logvar_x(X)
        mu_w = self.mu_w(X)
        logvar_w = self.logvar_w(X)
        return qz, mu_x, logvar_x, mu_w, logvar_w
class Decoder(nn.Module):
    def __init__(self,latent_size,input_features ,device):
        # 解码器
        super(Decoder, self).__init__()
        self.device = device
        self.latent_size = latent_size
        self.input_features = input_features
        self.fc4 = nn.Linear(200, 128)
        # self.b4 = nn.BatchNorm1d(128)
        self.fc5 = nn.Linear(128, 256)
        # self.b5 = nn.BatchNorm1d(256)
        self.fc6 = nn.Linear(256, self.input_features)

    def forward(self, x):
        h = self.fc4(x)
        h = F.relu(h)
        h = self.fc5(h)
        h = F.relu(h)
        Y = self.fc6(h)
        return Y

class GMVAE(nn.Module):
    def __init__(self, args):
        super(GMVAE, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=500, num_heads=4, batch_first=True)

        self.device = args.device
        self.latent_size = args.gen_latent_size
        if args.dataset =="PSM":
            self.input_features = 25
        elif args.dataset =="SWaT":
            self.input_features = 51
        elif args.dataset =="weather":
            self.input_features = 8
        elif args.dataset =="SMAP":
            self.input_features = 25
        elif args.dataset =="MSL":
            self.input_features = 55
        elif args.dataset =="SMD":
            self.input_features = 38
        elif args.dataset =="GECCO":
            self.input_features = 9
        self.encoder = Encoder(args, self.input_features).to(args.device)
        self.decoder = Decoder(self.latent_size,self.input_features,device=args.device)
        self.K=args.K

        # 先验生成器
        self.h2 = nn.Linear(150, 500)
        self.mu_px = nn.ModuleList([nn.Linear(500, 200) for _ in range(self.K)])
        self.logvar_px = nn.ModuleList([nn.Linear(500, 200) for _ in range(self.K)])


    def priorGenerator(self, w_sample):
        batchSize = w_sample.size(0)
        h = torch.tanh(self.h2(w_sample))
        h = h.unsqueeze(1)  # shape [B, 1, 500]
        h, _ = self.attn(h, h, h)
        h = h.squeeze(1)
        # h, _ = self.attn(h, h, h)
        mu_px = torch.empty(batchSize, 200, self.K, device=self.device, requires_grad=False)
        logvar_px = torch.empty(batchSize, 200, self.K, device=self.device, requires_grad=False)
        for i in range(self.K):
            mu_px[:, :, i] = self.mu_px[i](h)
            logvar_px[:, :, i] = self.logvar_px[i](h)
        return mu_px, logvar_px


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def forward(self, X):
        qz, mu_x, logvar_x, mu_w, logvar_w = self.encoder(X)

        w_sample = self.reparameterize(mu_w, logvar_w)
        mu_px, logvar_px = self.priorGenerator(w_sample)
        x_sample = self.reparameterize(mu_x, logvar_x)

        recon_x = self.decoder(x_sample)
        return mu_x, logvar_x, mu_px, logvar_px, qz, recon_x, mu_w, logvar_w

