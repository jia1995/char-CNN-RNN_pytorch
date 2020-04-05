import os
import argparse

from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

import torch
from dataset import MultimodalDataset
from torch.utils.data import DataLoader

from loss import sje_loss




def main(args):
    rng_init(args.seed)
    device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    dataset = MultimodalDataset(args.image_dir, args.captions_json, image_json=args.image_json)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=1, pin_memory=True)
    loader_len = len(loader)

    os.makedirs(os.path.join(args.checkpoint_dir, args.save_file), exist_ok=True)
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    model_name = '{}_{:.5f}_{}_{}_{}.pth'.format(args.save_file,
            args.learning_rate, args.symmetric, args.train_split, timestamp)
    ckpt_path = os.path.join(args.checkpoint_dir, args.save_file, model_name)
    writer = SummaryWriter(os.path.join(args.checkpoint_dir, args.save_file))

    net_txt = ccr.char_cnn_rnn(args.dataset, args.model_type).to(device)
    net_txt.apply(init_weights)

    optim_txt = torch.optim.RMSprop(net_txt.parameters(), lr=args.learning_rate)
    sched_txt = torch.optim.lr_scheduler.ExponentialLR(optim_txt,args.learning_rate_decay)

    acc1_smooth = acc2_smooth = 0

    for epoch in tqdm(range(args.epochs), position=1):
        for i, data in enumerate(tqdm(loader, position=0)):
            iter_num = (epoch * loader_len) + i + 1

            net_txt.train()
            img = data['img'].to(device)
            txt = data['txt'].to(device)
            feat_txt = net_txt(txt)
            feat_img = img

            loss1, acc1 = sje_loss(feat_txt, feat_img)
            loss2 = acc2 = 0
            if args.symmetric:
                loss2, acc2 = sje_loss(feat_img, feat_txt)
            loss = loss1 + loss2

            acc1_smooth = 0.99 * acc1_smooth + 0.01 * acc1
            acc2_smooth = 0.99 * acc2_smooth + 0.01 * acc2

            net_txt.zero_grad()
            loss.backward()
            optim_txt.step()

            writer.add_scalar('train/loss1', loss1.item(), iter_num)
            writer.add_scalar('train/loss2', loss2.item(), iter_num)
            writer.add_scalar('train/acc1', acc1, iter_num)
            writer.add_scalar('train/acc2', acc2, iter_num)
            writer.add_scalar('train/acc1_smooth', acc1_smooth, iter_num)
            writer.add_scalar('train/acc2_smooth', acc2_smooth, iter_num)
            writer.add_scalar('train/lr', sched_txt.get_lr()[0], iter_num)

            if (iter_num % args.print_every) == 0:
                run_info = (
                        'epoch: [{:3d}/{:3d}] | step: [{:4d}/{:4d}] | '
                        'loss: {:.4f} | loss1: {:.4f} | loss2: {:.4f} | '
                        'acc1: {:.2f} | acc2: {:.2f} | '
                        'acc1_smooth: {:.3f} | acc2_smooth: {:.3f} | '
                        'lr: {:.8f}'
                ).format(epoch+1, args.epochs, (i+1), loader_len,
                        loss, loss1, loss2,
                        acc1, acc2,
                        acc1_smooth, acc2_smooth,
                        sched_txt.get_lr()[0])
                tqdm.write(run_info)

        net_txt.eval()
        tqdm.write('Saving checkpoint to: {}'.format(ckpt_path))
        torch.save(net_txt.state_dict(), ckpt_path)

        sched_txt.step()

    writer.close()




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, required=True,
            choices=['cvpr', 'icml'],
            help='Model type')
    parser.add_argument('--image_dir', type=str, required=True,
            help='Image directory')
    parser.add_argument('--captions_json', type=str, required=True,
            help='Captions json file')
    parser.add_argument('--image_json', type=str, required=True,
            help='Image json file')
    parser.add_argument('--epochs', type=int,
            default=300,
            help='Number of epochs')
    parser.add_argument('--batch_size', type=int,
            default=40,
            help='Training batch size')
    parser.add_argument('--symmetric', type=bool,
            default=True,
            help='Whether or not to use symmetric form of SJE')
    parser.add_argument('--learning_rate', type=float,
            default=0.0004,
            help='Learning rate')
    parser.add_argument('--learning_rate_decay', type=float,
            default=0.98,
            help='Learning rate decay')

    parser.add_argument('--seed', type=int, required=True,
            help='RNG seed')
    parser.add_argument('--use_gpu', type=bool,
            default=True,
            help='Whether or not to use GPU')
    parser.add_argument('--print_every', type=int,
            default=100,
            help='How many steps/mini-batches between printing out the loss')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
            help='Output directory where model checkpoints get saved at')
    parser.add_argument('--save_file', type=str, required=True,
            help='Name to autosave model checkpoint to')

    args = parser.parse_args()
    main(args)
