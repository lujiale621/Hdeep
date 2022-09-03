import numpy as np
from torch.utils.data import Dataset
import torch
from ludeep_datasets.datasetpre.dataprocess import getseqMatrix_Label, getMatrixLabel, getpssmMatrix, getdsspMatrix, \
    getmembraneMatrix

class LuDataset(Dataset):
    def __init__(self, train_neg_file=None, train_pos_file=None, train_file_name=None, window_size=51):
        super(LuDataset, self).__init__()
        glist = []
        multiaccessdata = []
        if train_file_name != None:
            # 获取位点短序列one-hot矩阵
            print("# 获取位点短序列one-hot矩阵")
            seqmatrix, label = getseqMatrix_Label(train_file_name=train_file_name, window_size=window_size)
            # 获取蛋白质pssm矩阵
            print("# 获取蛋白质pssm矩阵")
            pssms = getpssmMatrix(train_file_name)
            dssps = getdsspMatrix(train_file_name)
            print("# 获取蛋白图节点 蛋白cmap矩阵")
            # 获取蛋白图节点 蛋白cmap矩阵
            glist, emd = getmembraneMatrix(train_file_name)


        else:
            seqmatrixneg, labelneg = getseqMatrix_Label(train_file_name=train_neg_file, window_size=window_size)
            seqmatrixpos, labelpos = getseqMatrix_Label(train_file_name=train_pos_file, window_size=window_size)
            pssmsneg = getpssmMatrix(train_neg_file)
            pssmpos = getpssmMatrix(train_pos_file)
            dsspsneg = getdsspMatrix(train_neg_file)
            dsspspos = getdsspMatrix(train_pos_file)
            gneg, emdneg = getmembraneMatrix(train_neg_file)
            gpos, emdpos = getmembraneMatrix(train_pos_file)
            emd = np.append(emdneg, emdpos, axis=0)
            for i in gneg:
                glist.append(i)
            for i in gpos:
                glist.append(i)
            # seqmatrix.append(seqmatrixneg.append(seqmatrixpos))
            seqmatrix = np.append(seqmatrixneg, seqmatrixpos, axis=0)
            label = np.append(labelneg, labelpos, axis=0)
            pssms = np.append(pssmsneg, pssmpos, axis=0)
            dssps = np.append(dsspsneg, dsspspos, axis=0)
        for (seq, pssm, dssp) in zip(seqmatrix, pssms, dssps):
            concatdata = np.concatenate([seq, pssm, dssp], axis=1)
            multiaccessdata.append(concatdata)
        concatdata = torch.Tensor(np.array(multiaccessdata))
        # concatdata = torch.unsqueeze(concatdata, 1)
        self.emd = emd
        self.graph = glist
        # self.cmap= torch.Tensor(cmap)
        print("# 打包 seqmatrix pssms dssps concatdata")
        self.seqmatrix, self.label, self.pssms, self.dssps, self.concatdata = torch.Tensor(seqmatrix), label, torch.Tensor(pssms), torch.Tensor(dssps), concatdata

    def __getitem__(self, item):

        return self.seqmatrix[item], self.label[item], self.pssms[item], self.dssps[item], self.concatdata[item], \
               self.emd[item], self.graph[item]

    def __len__(self):
        return len(self.label)
