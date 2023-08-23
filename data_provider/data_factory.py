from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_M4, PSMSegLoader, \
    MSLSegLoader, SMAPSegLoader, SMDSegLoader, SWATSegLoader, UEAloader
from data_provider.data_loader_bytedance import Dataset_IAAS_minute
from data_provider.uea import collate_fn
from torch.utils.data import DataLoader
from functools import partial
# import numpy as np
import torch


data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'custom': Dataset_Custom,
    'm4': Dataset_M4,
    'PSM': PSMSegLoader,
    'MSL': MSLSegLoader,
    'SMAP': SMAPSegLoader,
    'SMD': SMDSegLoader,
    'SWAT': SWATSegLoader,
    'UEA': UEAloader,
    'IAAS': Dataset_IAAS_minute
}


def data_provider(args, flag):
    Data = data_dict[args.data]
    timeenc = 0 if args.embed != 'timeF' else 1

    if flag == 'test':
        shuffle_flag = False
        drop_last = False
        if args.task_name == 'anomaly_detection' or args.task_name == 'classification':
            batch_size = args.batch_size
        else:
            batch_size = 1  # bsz=1 for evaluation
        freq = args.freq
    elif flag == 'val':
        shuffle_flag = False
        drop_last = False
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid
        freq = args.freq

    if args.task_name == 'anomaly_detection':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            win_size=args.seq_len,
            flag=flag,
            train_len=args.train_len,
            val_len=args.val_len,
            test_len=args.test_len,
            pred_len=args.pred_len,
            label_len=args.label_len,
            seq_len=args.seq_len
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
    elif args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            root_path=args.root_path,
            flag=flag,
            train_len=args.train_len,
            val_len=args.val_len,
            test_len=args.test_len,
            pred_len=args.pred_len,
            label_len=args.label_len,
            seq_len=args.seq_len
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        if args.data == 'm4':
            drop_last = False
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            train_len=args.train_len,
            val_len=args.val_len,
            test_len=args.test_len,
            pred_len=args.pred_len,
            label_len=args.label_len,
            seq_len=args.seq_len,
            features=args.features,
            target=args.target,
            timeenc=timeenc,
            freq=freq,
            seasonal_patterns=args.seasonal_patterns
        )
        print(flag, len(data_set))
        if(args.ensemble_num <= 1):
            data_loader = DataLoader(
                data_set,
                batch_size=batch_size,
                shuffle=shuffle_flag,
                num_workers=args.num_workers,
                drop_last=drop_last)
        else:
            if(flag == "train"):
                data_loader = DataLoader(
                    data_set,
                    batch_size=batch_size,
                    shuffle=shuffle_flag,
                    num_workers=args.num_workers,
                    drop_last=drop_last,
                    collate_fn=partial(collate_training, batch_size=batch_size, ensemble_num=args.ensemble_num)
                    )
            else:
                data_loader = DataLoader(
                    data_set,
                    batch_size=batch_size,
                    shuffle=shuffle_flag,
                    num_workers=args.num_workers,
                    drop_last=drop_last,
                    collate_fn=partial(collate_test, ensemble_num=args.ensemble_num)
                    )
        return data_set, data_loader
    

# (x, y, x_mark, y_mark)
# ensemble_num x -> 1 x
#
def collate_training(batch, batch_size, ensemble_num):
    # get get x, y, x_mark, and y_mark
    x = [torch.tensor(i[0]) for i in batch] # (B,seq_len,C)
    y = [torch.tensor(i[1]) for i in batch] # (B,pred_len,C)
    x_mark = [torch.tensor(i[2]) for i in batch] # (B, seq,len_, d_emb)
    y_mark = [torch.tensor(i[3]) for i in batch] # (B, pred_len, d_emb)
    
    # separate data to ensembles
    batch_range = list(range(0, batch_size-ensemble_num))
    ensembles_x = [x[i : i + ensemble_num] for i in batch_range] # (B-ensemble_num, ensemble_num, seq_len, C)
    ensembles_y = [y[i:i + ensemble_num] for i in batch_range] # (B-ensemble_num, ensemble_num, pred_len, C)
    ensembles_x_mark = [x_mark[i:i + ensemble_num] for i in batch_range]
    ensembles_y_mark = [y_mark[i:i + ensemble_num] for i in batch_range]
    # Concatenate tensors to the ensembles
    try:
        ensembles_x = torch.stack([torch.cat(i, dim=1) for i in ensembles_x]) # (B-ensemble_num, seq_len, C*ensemble_num)
        ensembles_y = torch.stack([torch.cat(i, dim=1) for i in ensembles_y])
        ensembles_x_mark = torch.stack([torch.cat(i, dim=1) for i in ensembles_x_mark])
        ensembles_y_mark = torch.stack([torch.cat(i, dim=1) for i in ensembles_y_mark])
        # print(ensembles_x.shape)
    except:
        print(f'Collate error for train data.')

    return [ensembles_x, ensembles_y, ensembles_x_mark, ensembles_y_mark]

def collate_test(batch, ensemble_num):
    # get x, y, x_mark, and y_mark
    x = [torch.tensor(i[0]) for i in batch]
    y = torch.tensor([(i[1]) for i in batch])
    x_mark = [torch.tensor(i[2]) for i in batch]
    y_mark = [torch.tensor(i[3]) for i in batch]

    
    x_mimo = torch.stack([torch.cat([i]*ensemble_num, dim=1) for i in x])
    # y_mimo = torch.stack([torch.cat([i]*ensemble_num, dim=1) for i in y])
    # y_mimo = torch.stack(y)
    y_mimo = y
    # print(y_mimo.shape)
    x_mark_mimo = torch.stack([torch.cat([i]*ensemble_num, dim=1) for i in x_mark])
    y_mark_mimo = torch.stack([torch.cat([i]*ensemble_num, dim=1) for i in y_mark])
    # print("x: {}, y: {}, x_mark: {}, y_mark:{}".format(x_mimo.shape, y_mimo.shape, x_mark_mimo.shape, y_mark_mimo.shape))
    
    return [x_mimo, y_mimo, x_mark_mimo, y_mark_mimo]