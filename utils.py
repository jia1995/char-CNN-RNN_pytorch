import torch

def prepare_text(string, max_str_len=201):
    '''
    Converts a text description from string format to one-hot tensor format.
    '''
    labels = str_to_labelvec(string, max_str_len)
    one_hot = labelvec_to_onehot(labels)
    return one_hot

def str_to_labelvec(string, max_str_len):
    string = string.lower()
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
    alpha_to_num = {k:v+1 for k,v in zip(alphabet, range(len(alphabet)))}
    labels = torch.zeros(max_str_len).long()
    max_i = min(max_str_len, len(string))
    for i in range(max_i):
        labels[i] = alpha_to_num.get(string[i], alpha_to_num[' '])

    return labels

def labelvec_to_onehot(labels):
    labels = torch.LongTensor(labels).unsqueeze(1)
    one_hot = torch.zeros(labels.size(0), 71).scatter_(1, labels, 1.)
    # ignore zeros in one-hot mask (position 0 = empty one-hot)
    one_hot = one_hot[:, 1:]
    one_hot = one_hot.permute(1,0)
    return one_hot

def onehot_to_labelvec(tensor):
    labels = torch.zeros(tensor.size(1), dtype=torch.long)
    val, idx = torch.nonzero(tensor).split(1, dim=1)
    labels[idx] = val+1
    return labels

def labelvec_to_str(labels):
    '''
    Converts a text description from one-hot tensor format to string format.
    '''
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{} "
    string = [alphabet[x-1] for x in labels if x > 0]
    string = ''.join(string)
    return string

def SJELoss(feat1, feat2):
    #print(feat1.size(), feat2.size())
    scores = torch.matmul(feat2, feat1.t())
    diagonal = scores.diag().view(scores.size(0), 1)
    diagonal = diagonal.expand_as(scores)
    cost = (1+scores-diagonal).clamp(min=0)
    cost[torch.eye(cost.size(0), dtype=torch.uint8)] = 0
    denom = cost.size(0)*cost.size(1)
    loss = cost.sum()/denom
    max_ids = torch.argmax(scores, dim=1)
    ground_truths = torch.LongTensor(range(scores.size(0))).to(feat1.device)
    num_correct = (max_ids == ground_truths).sum().float()
    accuracy = 100*num_correct/cost.size(0)
    return loss, accuracy
