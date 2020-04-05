import torch
import torch.nn as nn

class FRNN(nn.Module):
    def __init__(self, num_steps, emb_dim):
        super().__init__()
        self.i2h = nn.Linear(emb_dim, emb_dim)
        self.h2h = nn.Linear(emb_dim, emb_dim)

        self.num_steps = num_steps
        self.relu = nn.functional.relu 
    
    def forward(self, txt):
        res = []
        for i in range(self.num_steps):
            i2h = self.i2h(txt[:,i]).unsqueeze(1)
            if i == 0:
                output = self.relu(i2h)
            else:
                h2h = self.h2h(res[i-1])
                output = self.relu(h2h)
            res.append(output)
        
        res = torch.cat(res, dim=1)
        res = torch.mean(res, dim=1)
        return res

class FGRU(nn.Module):
    def __init__(self, num_steps, emb_dim):
        super().__init__()
        self.i2h_up = nn.Linear(emb_dim, emb_dim)
        self.h2h_up = nn.Linear(emb_dim, emb_dim)
        self.i2h_reset = nn.Linear(emb_dim, emb_dim)
        self.h2h_reset = nn.Linear(emb_dim, emb_dim)
        self.i2h = nn.Linear(emb_dim, emb_dim)
        self.h2h = nn.Linear(emb_dim, emb_dim)
        
        self.num_steps = num_steps
    
    def forward(self, txt):
        res = []
        for i in range(self.num_steps):
            if i==0:
                output = torch.tanh(self.i2h(txt[:,i])).unsqueeze(1)
            else:
                update = torch.sigmoid(self.i2h_up(txt[:,i])+self.h2h_up(res[i-1]))
                reset = torch.sigmoid(self.i2h_reset(txt[:,i])+self.h2h_reset(res[i-1]))

                gated_hidden = reset*res[i-1]
                p1 = self.i2h(txt[:,i])
                p2 = self.h2h(gated_hidden)
                hidden_cand = torch.tanh(p1+p2)
                zh = update*hidden_cand
                zhm1 = ((update*-1)+1)*res[i-1]
                output = zh+zhm1
            res.append(output)
        
        res = torch.cat(res, dim=1)
        res = torch.mean(res, dim=1)
        return res

class charCNNRNN(nn.Module):
    def __init__(self, dataset, model_type):
        super().__init__()
        assert model_type in ['cvpr', 'icml'], 'model_type shoud be (cvpr|icml)'
        if model_type == 'cvpr':
            emb_dim = 256
            use_maxpool3 = True
            rnn = FRNN
            num_steps = 8
        else:
            emb_dim = 512
            use_maxpool3 = False
            rnn = FGRU
            num_steps = 18
        self.dataset = dataset
        self.model_type = model_type
        self.use_maxpool3 = use_maxpool3

        self.cnn = nn.Sequential(
            nn.Conv1d(70, 384, kernel_size=4),
            nn.Threshold(1e-6, 0),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(384, 512, kernel_size=4),
            nn.Threshold(1e-6, 0),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Conv1d(512, emb_dim, kernel_size=4),
            nn.Threshold(1e-6, 0)
        )
        if use_maxpool3:
            self.cnn.add_module('maxpool3', nn.MaxPool1d(kernel_size=3, stride=2))
        self.rnn = rnn(num_steps, emb_dim)
        self.emb_proj = nn.Linear(emb_dim, 512)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight,-0.08, 0.08)
                nn.init.uniform_(m.bias,-0.08, 0.08)
            elif isinstance(m, nn.Conv1d):
                nn.init.uniform_(m.weight,-0.08, 0.08)
                nn.init.uniform_(m.bias,-0.08, 0.08)

    def forward(self, txt):
        out = self.cnn(txt)
        out = out.permute(0,2,1)
        out = self.rnn(out)
        out = self.emb_proj(out)
        return out
