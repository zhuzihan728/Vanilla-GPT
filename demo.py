import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from utils.utils import set_seed, load_config, get_network_paras_amount
from utils.saver import Saver
from bunny_GPT2 import GPT2
import argparse

set_seed(42)


class SortDataset(Dataset):
    
    def __init__(self, n_samples=10000, n_seq=5, vocab_size=3):
        self.n_samples = n_samples
        self.n_seq = n_seq
        self.block_size = n_seq * 2 - 1
        self.vocab_size = vocab_size
        
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        seq = torch.randint(0, self.vocab_size, (self.n_seq,))
        sorted_seq = seq.sort()[0]
        x = torch.hstack((seq, sorted_seq))[:-1].clone()
        y = torch.hstack((seq, sorted_seq))[1:].clone()
        y[:self.n_seq-1] = -1
        return x, y

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="path to the config file")
    return parser.parse_args(args=args, namespace=namespace)
    
    
if __name__ == "__main__":
    cmd = parse_args()
    
    # load config
    args = load_config(cmd.config)
    
    train_dataset = SortDataset(n_samples=10000)
    test_dataset = SortDataset(n_samples=1000)
    
    vocab_size = train_dataset.vocab_size
    block_size = train_dataset.block_size
    embed_dim = args.embed_dim
    num_heads = args.num_heads
    num_blocks = args.num_blocks
    batch_size = args.batch_size
    n_epochs = args.n_epochs
    lr = args.lr
    save_dir = args.save_dir
    log_interval = args.log_interval
    val_interval_epoch = args.val_interval_epoch

    train_loader = DataLoader(train_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)
    val_interval = len(train_loader) * val_interval_epoch
    
    sort_model = GPT2(vocab_size=vocab_size, block_size=block_size, embed_dim=embed_dim, num_heads=num_heads, num_blocks=num_blocks)
    optimizer = optim.AdamW(sort_model.parameters(), lr=lr)

    # args = DotDict({'save_dir': save_dir})
    saver = Saver(args)
    for k, v in args.items():
        saver.log_info(f'> {k}: {v}')
    
    params_count = get_network_paras_amount({'model': sort_model})
    saver.log_info('--- model size ---')
    saver.log_info(params_count)
    saver.log_info('======= start training =======')
    
    sort_model.train()
    for epoch in range(n_epochs):
        for i, data in enumerate(train_loader):
            saver.global_step_increment()
            x, y = data

            optimizer.zero_grad()
            logits = sort_model(x)

            loss = sort_model.compute_loss(logits, y)

            loss.backward()
            optimizer.step()

            if saver.global_step % log_interval == 0:
                saver.log_info(
                    'epoch: {} | {:3d}/{:3d} |batch/s: {:.2f} | loss: {:.3f} | time: {} | step: {}'.format(
                        epoch,
                        epoch*len(train_loader)+i+1,
                        n_epochs*len(train_loader),
                        log_interval/saver.get_interval_time(),
                        loss.item(),
                        saver.get_total_time(),
                        saver.global_step
                    )
                )
            
                saver.log_value({
                    'train/loss': loss.item()
                })
                    
                
            # validation
            if saver.global_step % val_interval == 0 or (epoch == n_epochs-1 and i == len(train_loader)-1):
                # run testing set
                sort_model.eval()
                test_loss= 0
                test_acc = []
                with torch.no_grad():
                    for i_test, data in enumerate(test_loader):
                        x, y = data
                        logits = sort_model(x)
                        loss = sort_model.compute_loss(logits, y)
                        test_loss += loss.item()

                        gt = y[:, test_dataset.n_seq-1:].detach().numpy()
                        pred = torch.argmax(logits, dim=-1)[:, test_dataset.n_seq-1:].detach().numpy()
                        correct = (gt == pred).all(axis=-1)
                        test_acc.extend(correct)

                        if i_test == 0:
                            log_text = ''
                            for j in range(10):
                                input_seq = x[j, :test_dataset.n_seq]
                                pred = sort_model.generate(input_seq, test_dataset.n_seq)
                                log_text += f"input: {input_seq.detach().numpy()} | pred: {pred['input_ids'][0]}<br><br>"
                            saver.log_text(f'text result at epoch {epoch}', log_text)

                test_loss /= len(test_loader)
                test_acc = sum(test_acc) / len(test_acc)
                
                # log loss
                saver.log_info(
                    ' --- validation --- \nloss: {:.3f} |acc: {:.3f} '.format(
                        test_loss,
                        test_acc
                    )
                )

                saver.log_value({
                    'val/loss': test_loss,
                    'val/acc': test_acc
                })
                sort_model.train()

    saver.save_model(sort_model, optimizer, postfix=f'{saver.global_step}')
    