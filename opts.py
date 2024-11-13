import sys
import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_path', default='/content/drive/My Drive/CMU MOSEI/final/result', type=str, help='Result directory path')
    parser.add_argument('--store_name', default='model', type=str, help='Name to store checkpoints')
    parser.add_argument('--dataset', default='MOSEI', type=str, help='Used dataset.')
    parser.add_argument('--n_classes', default=3, type=int, help='Number of classes')
    parser.add_argument('--csv_path1', default='/content/drive/MyDrive/CMU MOSEI/Audio/Raw 4/Raw4-2/train.csv', type=str, help='CSV file of all audio and related text')
    parser.add_argument('--csv_path2', default='/content/drive/MyDrive/CMU MOSEI/Audio/Raw 4/Raw4-2/valid.csv', type=str, help='CSV file of all audio and related text')
    parser.add_argument('--csv_path3', default='/content/drive/MyDrive/CMU MOSEI/Audio/Raw 4/Raw4-2/test.csv', type=str, help='CSV file of all audio and related text')
    parser.add_argument('--root_audio_folder1', default='/content/drive/MyDrive/CMU MOSEI/Audio/Raw 4/Raw4-2/train1', type=str, help='Folder of all audio files')
    parser.add_argument('--root_audio_folder2', default='/content/drive/MyDrive/CMU MOSEI/Audio/Raw 4/Raw4-2/valid1', type=str, help='Folder of all audio files')
    parser.add_argument('--root_audio_folder3', default='/content/drive/MyDrive/CMU MOSEI/Audio/Raw 4/Raw4-2/test1', type=str, help='Folder of all audio files')
    parser.add_argument('--bert_path1', default = '/content/drive/MyDrive/CMU MOSEI/Distil_bert/combined_data.npy', type = str, help = "bert output numpy file")
    parser.add_argument('--bert_path2', default = '/content/drive/MyDrive/CMU MOSEI/Distil_bert/valid42.npy', type = str, help = "bert output numpy file")
    parser.add_argument('--bert_path3', default = '/content/drive/MyDrive/CMU MOSEI/Distil_bert/combined_data2.npy', type = str, help = "bert output numpy file")

    parser.add_argument('--model', default='multimodal', type=str, help='')
    parser.add_argument('--num_heads', default=1, type=int, help='number of heads, in the paper 1 or 4')
    parser.add_argument('--device', default='cuda', type=str, help='Specify the device to run. Defaults to cuda, fallsback to cpu')


    parser.add_argument('--sample_duration', default=15, type=int, help='Temporal duration of inputs, ravdess = 18')

    parser.add_argument('--learning_rate', default=0.04, type=float, help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    parser.add_argument('--lr_steps', default=[40, 55, 65, 70, 200, 250], type=float, nargs="+", metavar='LRSteps', help='epochs to decay learning rate by 10')
    parser.add_argument('--dampening', default=0.9, type=float, help='dampening of SGD')
    parser.add_argument('--weight_decay', default=1e-3, type=float, help='Weight Decay')
    parser.add_argument('--lr_patience', default=10, type=int, help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.')
    parser.add_argument('--batch_size', default=20, type=int, help='Batch Size')
    parser.add_argument('--n_epochs', default=15, type=int, help='Number of total epochs to run')

    parser.add_argument('--begin_epoch', default=1, type=int, help='Training begins at this epoch. Previous trained model indicated by resume_path is loaded.')
    parser.add_argument('--resume_path', default='/content/drive/MyDrive/CMU MOSEI/final/result/MOSEI_multimodal_15_checkpoint0.pth', type=str, help='Save data (.pth) of previous training')
    parser.add_argument('--no_train', action='store_true', help='If true, training is not performed.,True,False')
    parser.set_defaults(no_train=False)
    parser.add_argument('--no_val', action='store_true', help='If true, validation is not performed.')
    parser.set_defaults(no_val=False)
    parser.add_argument('--test', action='store_true', help='If true, test is performed.')
    parser.set_defaults(test=True)
    parser.add_argument('--test_subset', default='test', type=str, help='Used subset in test (val | test)')

    parser.add_argument('--n_threads', default=16, type=int, help='Number of threads for multi-thread loading')

    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--fusion', default='ia', type=str)
    parser.add_argument('--mask', type=str, default='softhard')
    args = parser.parse_args(args=[] if sys.argv[1:] else [
    '--result_path', '/content/drive/My Drive/CMU MOSEI/final/result',
    '--store_name', 'model',
    '--dataset', 'MOSEI',
    '--n_classes', '3',
    '--csv_path', '/content/drive/MyDrive/CMU MOSEI/Audio/Raw 4/Raw4-2/train.csv',
    '--root_audio_folder', '/content/drive/MyDrive/CMU MOSEI/Audio/Raw 4/Raw4-2/train1',
    '--bert_path', '/content/drive/MyDrive/CMU MOSEI/Distil_bert/combined_data.npy',

    '--model', 'multimodal',
    '--num_heads', '1',
    '--device', 'cpu',
    '--sample_duration', '15',
    '--learning_rate', '0.04',
    '--momentum', '0.9',
    '--lr_steps', '230', '270', '295', '320', '350', '375',
    '--dampening', '0.9',
    '--weight_decay', '0.001',
    '--lr_patience', '10',
    '--batch_size', '10',
    '--n_epochs', '2050',
    '--begin_epoch', '1',
    '--resume_path', '',

    '--no_train',
    '--no_val',
    '--test',
    '--test_subset', 'test',
    '--n_threads', '16',
    '--manual_seed', '1',
    '--fusion', 'ia',
    '--mask', 'softhard'
])


    return args
