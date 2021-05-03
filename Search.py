import os

import nni
from nni.algorithms.compression.pytorch.pruning import *
from transformer.data import *
import torch
from transformer.Models import Transformer
import argparse
from trainer import  *
config_list = [{
    'initial_sparsity': 0.0,
    'final_sparsity': 0.6,
    'start_epoch': 0,
    'end_epoch': 200,
    'frequency': 1,
    'op_types': ['default']
}]
device = torch.device('cuda')
trg_PAD_IDX = en_vocab.stoi['<pad>']
src_PAD_IDX = de_vocab.stoi['<pad>']
def get_model_and_checkpoint(checkpoint_path, n_gpu=1):
    if checkpoint_path:
        print('loading {}...'.format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_opt = checkpoint['settings']
        print("The model options :",model_opt)
        model = Transformer(
            model_opt.src_vocab_size,
            model_opt.trg_vocab_size,
            model_opt.src_pad_idx,
            model_opt.trg_pad_idx,
            trg_emb_prj_weight_sharing=model_opt.proj_share_weight,
            emb_src_trg_weight_sharing=model_opt.embs_share_weight,
            d_k=model_opt.d_k,
            d_v=model_opt.d_v,
            d_model=model_opt.d_model,
            d_word_vec=model_opt.d_word_vec,
            d_inner=model_opt.d_inner_hid,
            n_layers=model_opt.n_layers,
            n_head=model_opt.n_head,
            dropout=model_opt.dropout).to(device)
    model.load_state_dict(checkpoint['model'])
    print('[Info] Trained model state loaded.')
    return model,model_opt
def patch_src(src, pad_idx):
    src = src.transpose(0, 1)
    return src


def patch_trg(trg, pad_idx):
    trg = trg.transpose(0, 1)
    trg, gold = trg[:, :-1], trg[:, 1:].contiguous().view(-1)
    return trg, gold

def test(args, model, device, criterion, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            src_seq = patch_src(data,src_PAD_IDX).to(device)
            trg_seq, gold = map(lambda x: x.to(device), patch_trg(target, trg_PAD_IDX))
            output = model(src_seq,trg_seq)
            print("OUTPUTS:",output)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    acc = 100 * correct / len(test_loader.dataset)

    print('Test Loss: {}  Accuracy: {}%\n'.format(
        test_loss, acc))
    return acc

def parse_args():
    parser = argparse.ArgumentParser(description='AMC search script')
    parser.add_argument('--model_type', default='transformer', type=str, choices=['transformer', 'mobilenetv2', 'resnet18', 'resnet34', 'resnet50'],
        help='model to prune')
    parser.add_argument('--dataset', default='m30k_deen_shr.pkl', type=str, choices=['m30k_deen_shr.pkl', 'imagenet'], help='dataset to use (cifar/imagenet)')
    parser.add_argument('--batch_size', default=8, type=int, help='number of data batch size')
    parser.add_argument('--data_root', default='./data', type=str, help='dataset path')
    parser.add_argument('--flops_ratio', default=0.5, type=float, help='target flops ratio to preserve of the model')
    parser.add_argument('--lbound', default=0.2, type=float, help='minimum sparsity')
    parser.add_argument('--rbound', default=1., type=float, help='maximum sparsity')
    parser.add_argument('--ckpt_path', default=None, type=str, help='manual path of checkpoint')
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-embs_share_weight', action='store_true')
    parser.add_argument('-proj_share_weight', action='store_true')
    parser.add_argument('-scale_emb_or_prj', type=str, default='prj')
    parser.add_argument('--train_episode', default=800, type=int, help='number of training episode')
    parser.add_argument('--n_gpu', default=1, type=int, help='number of gpu to use')
    parser.add_argument('--n_worker', default=16, type=int, help='number of data loader worker')
    parser.add_argument('--suffix', default=None, type=str, help='suffix of auto-generated log directory')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    training_data,validation_data,test_data,trainLoader,valLoader,testLoader = DataInit()
    model,model_opt = get_model_and_checkpoint(checkpoint_path=args.ckpt_path, n_gpu=args.n_gpu)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)
    pruner = AGPPruner(model, config_list, optimizer, pruning_algorithm='level')
    model = pruner.compress()
    torch.save(model.state_dict(), os.path.join("output", 'model_speed_up.pth'))
    for epoch in range(1, args.epochs + 1):
        pruner.update_epoch(epoch)
        train(args, model, device, trainLoader, optimizer, 50)
        test(model, device, testLoader)
    print(model.state_dict())
    acc = train(model,valLoader,optimizer,criterion,1)
    print("ACCURACY: ",acc)
    nni.report_final_results(acc)
    pruner.export_model(model_path='output/Result.pth', mask_path='output/masks.pth')

#python3 Search.py --model_type transformer --ckpt model/model.chkpt