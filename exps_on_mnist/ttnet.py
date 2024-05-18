import torch
from torch import nn


class BilinearUnit(nn.Module):
    def __init__(self, in1_dim, in2_dim, out_dim):
        super().__init__()
        self.unit = torch.nn.Bilinear(in1_dim, in2_dim, out_dim, bias=False)

    def forward(self, x, y):
        return self.unit(x, y)


class TTBlock(nn.Module):
    def __init__(self, feat_dim, hid_dim, depth, num_classes):
        super().__init__()
        self.feat_dim = feat_dim
        self.depth = depth
        self.hid_dim = hid_dim
        self.units = nn.ModuleList(
            [BilinearUnit(1, feat_dim, hid_dim)] +
            [BilinearUnit(hid_dim, feat_dim, hid_dim) for i in range(depth - 2)] +
            [BilinearUnit(hid_dim, feat_dim, num_classes)]
        )

    def forward(self, x):
        # print(x.shape)
        # x = torch.unsqueeze(x, -1)
        # x = self.feature_map(x)
        # x = x[:, :, None]
        batch_size = x.shape[0]
        res = torch.ones(batch_size, 1)
        for i in range(self.depth):
            #print(x[:, i, :].shape)
            #print("Res", res.shape)
            # print(x[:, :, i].shape)
            res = self.units[i](res, x[:, :, i])
        logits = res
        return logits

    
    def predict_proba(self, x):
        logits = self.forward(x)
        return nn.functional.softmax(logits, dim=-1)


    def predict(self, x):
        probas = self.predict_proba(x)
        return torch.argmax(probas, dim=-1) 


class TTEqNetForImages(nn.Module):
    def __init__(self, feature_extractor, in_channels, feat_dim, hid_dim, depth, num_classes, batch_norm_use=None):
        super().__init__()
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        self.batch_norm_use = batch_norm_use
        
        self.feature_extractor = feature_extractor
        
        self.depth = depth
        # self.tt_block = TTBlock(feat_dim, hid_dim, depth, num_classes)
        
        # self.tt_block = nn.Sequential(
        #     nn.Flatten(start_dim=-2),
        #     nn.Linear(depth * feat_dim, num_classes)
        # )
        
        self.first_unit = nn.Bilinear(1, feat_dim, hid_dim, bias=False)
        self.middle_unit = nn.Bilinear(hid_dim, feat_dim, hid_dim, bias=False)
        self.last_unit = nn.Bilinear(hid_dim, feat_dim, num_classes, bias=False)
        self.bn = nn.BatchNorm1d(hid_dim)
        self.bns = nn.ModuleList([nn.BatchNorm1d(hid_dim) for _ in range(depth-2)])
        

    def forward(self, x, device=torch.device("cpu")):
        batch_size = x.shape[0]
        features = self.feature_extractor(x)
        assert self.depth == features.shape[-1]
        # print("features", features.shape)
        # features = features.view(features.shape[0], features.shape[1], self.depth)
        # features = features.transpose(-2, -1)
        # print("features", features.shape)
        # logits = self.first_unit(features)
        
        logits = torch.ones(batch_size, 1).to(device)
        mask = torch.eye(self.depth).unsqueeze(0).unsqueeze(0).to(device)
        # print((features).shape)
        # print(((features * mask[:, :, :, 0]).sum(-1) == features[:, :, 0]).all())
        logits = self.first_unit(logits, (features * mask[:, :, :, 0]).sum(-1))
        #print(logits[0])
        for i in range(self.depth-2):
            # print(((features * mask[:, :, :, i]).sum(-1) == features[:, :, i]).all())
            if self.batch_norm_use is None:
                logits = self.middle_unit(logits, (features * mask[:, :, :, i]).sum(-1))
            elif self.batch_norm_use == 'same':
                logits = self.bn(self.middle_unit(logits, (features * mask[:, :, :, i]).sum(-1)))
            else:
                logits = self.bns[i](self.middle_unit(logits, (features * mask[:, :, :, i]).sum(-1)))
            #print(logits[0])
        # print(((features * mask[:, :, :, -1]).sum(-1) == features[:, :, -1]).all())
        logits = self.last_unit(logits, (features * mask[:, :, :, -1]).sum(-1))
        # logits = self.tt_block(features)
        return logits

    
    def predict_proba(self, x, device=torch.device("cpu")):
        logits = self.forward(x, device=device)
        return nn.functional.softmax(logits, dim=-1)


    def predict(self, x, device=torch.device("cpu")):
        probas = self.predict_proba(x, device=device)
        return torch.argmax(probas, dim=-1) 


class TTNetForImages(nn.Module):
    def __init__(self, feature_extractor, in_channels, feat_dim, hid_dim, depth, num_classes):
        super().__init__()
        self.feat_dim = feat_dim
        self.hid_dim = hid_dim
        
        self.feature_extractor = feature_extractor
        
        self.depth = depth
        # self.tt_block = TTBlock(feat_dim, hid_dim, depth, num_classes)
        
        # self.tt_block = nn.Sequential(
        #     nn.Flatten(start_dim=-2),
        #     nn.Linear(depth * feat_dim, num_classes)
        # )
        
        self.first_unit = nn.Bilinear(1, feat_dim, hid_dim, bias=False)
        self.middle_units = nn.ModuleList([nn.Bilinear(hid_dim, feat_dim, hid_dim, bias=False) for _ in range(depth-2)])
        self.bns = nn.ModuleList([nn.BatchNorm1d(hid_dim) for _ in range(depth-2)])
        self.last_unit = nn.Bilinear(hid_dim, feat_dim, num_classes, bias=False)
        

    def forward(self, x, device=torch.device("cpu")):
        batch_size = x.shape[0]
        features = self.feature_extractor(x)
        assert self.depth == features.shape[-1]
        # print("features", features.shape)
        # features = features.view(features.shape[0], features.shape[1], self.depth)
        # features = features.transpose(-2, -1)
        # print("features", features.shape)
        # logits = self.first_unit(features)
        
        logits = torch.ones(batch_size, 1).to(device)
        mask = torch.eye(self.depth).unsqueeze(0).unsqueeze(0).to(device)
        # print((features).shape)
        # print(((features * mask[:, :, :, 0]).sum(-1) == features[:, :, 0]).all())
        logits = self.first_unit(logits, (features * mask[:, :, :, 0]).sum(-1))
        #print(logits[0])
        for i in range(self.depth-2):
            # print(((features * mask[:, :, :, i]).sum(-1) == features[:, :, i]).all())
            logits = self.bns[i](self.middle_units[i](logits, (features * mask[:, :, :, i]).sum(-1)))
            #print(logits[0])
        # print(((features * mask[:, :, :, -1]).sum(-1) == features[:, :, -1]).all())
        logits = self.last_unit(logits, (features * mask[:, :, :, -1]).sum(-1))
        # logits = self.tt_block(features)
        return logits

    
    def predict_proba(self, x, device=torch.device("cpu")):
        logits = self.forward(x, device=device)
        return nn.functional.softmax(logits, dim=-1)


    def predict(self, x, device=torch.device("cpu")):
        probas = self.predict_proba(x, device=device)
        return torch.argmax(probas, dim=-1) 