from Dataset import MOSEI
def get_training_set(opt, spatial_transform=None, audio_transform=None):
    assert opt.dataset in ['MOSEI'], print('Unsupported dataset: {}'.format(opt.dataset))

    if opt.dataset == 'MOSEI':
        training_data = MOSEI(opt.csv_path1, 'train', opt.root_audio_folder1,opt.bert_path1, data_type='audiotext')
    return training_data

def get_test_set(opt, spatial_transform=None, audio_transform=None):
    assert opt.dataset in ['MOSEI'], print('Unsupported dataset: {}'.format(opt.dataset))

    if opt.dataset == 'MOSEI':
        training_data = MOSEI(opt.csv_path3, 'test', opt.root_audio_folder3,opt.bert_path3, data_type='audiotext')
    return training_data

def get_validation_set(opt, spatial_transform=None, audio_transform=None):
    assert opt.dataset in ['MOSEI'], print('Unsupported dataset: {}'.format(opt.dataset))

    if opt.dataset == 'MOSEI':
        training_data = MOSEI(opt.csv_path2, 'valid', opt.root_audio_folder2,opt.bert_path2, data_type='audiotext')
    return training_data
