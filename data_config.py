import os


class DataConfig(dict):
    data_name = ""
    root_dir = ""
    label_transform = "norm"
    label_folder_name = 'label'
    img_folder_name = ['A']
    img_folder_names = ['A', 'B']
    n_class = 2

    def get_data_config(self, data_name):
        self.data_name = data_name
        if data_name == 'get_start':
            self.root_dir = 'samples'
        elif data_name == 'inria256':
            # put your data root here
            self.root_dir = r'G:\tmp_data\inria_cut256'
        elif data_name == 'LEVIR':
            # put your data root here
            self.root_dir = ''
        else:
            raise TypeError('%s has not defined' % data_name)
        return self


def get_pretrained_path(pretrained):
    out = None
    if pretrained is not None:
        if os.path.isfile(pretrained):
            out = pretrained
        elif pretrained == 'imagenet':
            out = pretrained
        elif pretrained == 'None' or pretrained == 'none':
            out = None
        else:
            raise NotImplementedError(pretrained)
    else:
        out = None
    return out
