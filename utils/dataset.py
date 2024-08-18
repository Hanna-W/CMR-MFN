from video_records import UESTC_MMEA_CL_VideoRecord
import torch.utils.data as data


class UESTC_MMEA_CL(data.Dataset):
    def __init__(self, list_file):
        self.list_file = list_file

        self._parse_list()

    def _parse_list(self):
        
        tmp = [x.strip().split(' ') for x in open(self.list_file)]
        tmp = [item for item in tmp if int(item[1]) >= 3]
        self.video_list = [UESTC_MMEA_CL_VideoRecord(item) for item in tmp]
        # print('video number:%d' % (len(self.video_list)))

    def __len__(self):
        return len(self.video_list)
