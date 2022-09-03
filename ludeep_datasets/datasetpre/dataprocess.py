import csv
import os
from  config import *
from ludeep_datasets.feature_computation.dssp_computation.compute import loaddsspfile
from ludeep_datasets.feature_computation.pssm_computation.compute import load_fasta_and_compute
from utils.emdata import dictdata
import torch
import numpy as np
import scipy.sparse as spp
import dgl
import zlib
from utils.emdata import Emdata, Embdict
dictdata = Emdata()
embdict = Embdict()
letterDict = {}
letterDict["A"] = 0
letterDict["C"] = 1
letterDict["D"] = 2
letterDict["E"] = 3
letterDict["F"] = 4
letterDict["G"] = 5
letterDict["H"] = 6
letterDict["I"] = 7
letterDict["K"] = 8
letterDict["L"] = 9
letterDict["M"] = 10
letterDict["N"] = 11
letterDict["P"] = 12
letterDict["Q"] = 13
letterDict["R"] = 14
letterDict["S"] = 15
letterDict["T"] = 16
letterDict["V"] = 17
letterDict["W"] = 18
letterDict["Y"] = 19
letterDict["*"] = 20
def decompre(data):
    st = zlib.decompress(bytes(data)).decode("utf-8")
    return st


def testsplit(st):
    lines = st.strip().split("\n")
    return lines

def openfile(url):
    with open(url, 'rb') as rf:
        data = rf.read()
        st = decompre(data)
        lines = testsplit(st)
    return lines

def getcmap(url, cmdata):
    lines = openfile(url)

    ret = np.zeros((len(lines), len(lines)))

    for x in range(len(lines)):
        # z = 0
        raw_row = lines[x].strip().split(' ')
        for y in range(len(lines)):
            if raw_row[y] == '1':
                ret[x][y] = 1
    return ret
def checkinfo(sseq, protein, position):
    # global mapdata
    Tag = True
    # 检查序列是否存在
    if sseq == 'none':
        Tag = False

    # 检查蛋白质cmap
    url = cpath + protein + '.cm'
    if os.path.exists(url) == False:
        Tag = False
    # 检查embed是否存在

    mappostion = dictdata.getTag(protein)
    if mappostion == -1:
        Tag = False
    if position > len(sseq):
        Tag = False
    return Tag

def listclass_to_one_hot(list, isnumpy=True):
    list_len = len(list)
    li = []
    for i in range(list_len):
        li.append([list[i]])
    one_hot_list = torch.LongTensor(li)
    # 标签独热编码的形式
    if isnumpy:
        one_hot_list = torch.zeros(list_len, 2).scatter_(1, one_hot_list, 1).numpy()
    else:
        one_hot_list = torch.zeros(list_len, 2).scatter_(1, one_hot_list, 1)
    return one_hot_list
def getdsspMatrix(train_file_name, window_size=51):
    prot = []  # list of protein name
    dssps = []
    half_len = (window_size - 1) / 2

    with open(train_file_name, 'r') as rf:
        reader = csv.reader(rf)
        header = next(reader)
        for row in reader:
            sseq = row[3]
            protein = row[1]
            position = int(row[2])
            if checkinfo(sseq, protein, position) == False:
                continue
            dssp = loaddsspfile(dssp_fn, protein + '_' + str(position), len(sseq))
            dssps.append((np.transpose(dssp)))
    return dssps

def getpssmMatrix(train_file_name, window_size=51, empty_aa="*"):
    prot = []  # list of protein name
    pssms = []
    half_len = (window_size - 1) / 2
    with open(train_file_name, 'r') as rf:
        reader = csv.reader(rf)
        header = next(reader)
        for row in reader:
            sseq = row[3]
            protein = row[1]
            position = int(row[2])
            if checkinfo(sseq, protein, position) == False:
                continue
            if position < (window_size - 1) / 2:
                start = 0
                l_padding = (window_size - 1) / 2 - position
            else:
                start = position - (window_size - 1) / 2
                l_padding = 0
            if position > len(sseq) - (window_size - 1) / 2:
                end = len(sseq)
                r_padding = (window_size - 1) / 2 - (len(sseq) - position)
            else:
                end = position + (window_size - 1) / 2
                r_padding = 0
            # if position > len(sseq):
            #     continue

            prot.append(protein)
            seq_fn = pssm_root_url + "/" + str(protein) + '.fasta'
            if not os.path.exists(seq_fn):
                fp = open(seq_fn, "w")
                fp.write('>' + str(protein) + '\n')
                fp.write(sseq)
                fp.close()
            out_base_fn = pssm_root_url
            raw_pssm_dir = pssm_fn
            pssm = load_fasta_and_compute(protein, position, raw_pssm_dir, start, end, l_padding, r_padding)
            pssms.append(np.transpose(pssm))

    return pssms
def getmembraneMatrix(train_file_name):
    # global mapdata
    cmdatalist = []
    cmaplist = []
    emblist = []
    Glist = []
    embed_data = []
    loaditer = 0
    saveiter = 0
    with open(train_file_name, 'r') as rf:
        reader = csv.reader(rf)
        header = next(reader)
        for row in reader:
            cmdata = []
            sseq = row[3]
            position = int(row[2])
            protein = row[1]
            url = cpath + protein + '.cm'

            if checkinfo(sseq, protein, position) == False:
                continue
            # print("protein:" + protein)
            g_embed = embdict.getTag(protein_name=protein)
            g_embed = torch.from_numpy(g_embed)
            # mappostion = dictdata.getTag(protein)
            # tag = os.path.exists('D:/emb/' + protein+".npz")
            # if tag == False:
            #     emb_url = emb_data + str(mappostion) + ".npz"
            #     embed_data = np.load(emb_url)
            #     g_embed = torch.tensor(embed_data[protein][:]).float()
            # else:
            #     # g_embed=np.loadtxt('D:/emb/' + protein + '.em',delimiter=',')
            #     # g_embed=torch.from_numpy(g_embed)
            #     g_embed=np.load('D:/emb/'+protein+".npz")
            #     g_embed = torch.from_numpy(g_embed['data'])
            #     loaditer=loaditer+1
            #     print("loadproteinemb:No:"+str(loaditer)+":"+protein)
            cmdata = getcmap(url, cmdata)

            # if tag == False:
            #     saveiter=saveiter+1
            #     save_txt = embed_data[protein][:]
            #     numpy.savez('D:/emb/'+protein, data=save_txt)
            #     # np.savetxt('D:/emb/'+protein+'.em',save_txt, fmt='%.2f', delimiter=',', encoding='utf-8')
            #     print("saveproteinemb:No:" + str(saveiter) + ":" + protein)
            if len(g_embed) != len(cmdata):
                cmdata = cmdata[:len(g_embed), :len(g_embed)]
            adj = spp.coo_matrix(cmdata)
            G = dgl.DGLGraph(adj)
            G.ndata['feat'] = g_embed.float()
            # (edgecuts, parts) = metis.part_graph(G, 3)
            Glist.append(G)
            nodenum = len(g_embed)
            if nodenum > 1000:
                textembed = g_embed[:1000]
            elif nodenum < 1000:
                textembed = np.concatenate((g_embed, np.zeros((1000 - nodenum, 1024))))
                textembed = torch.from_numpy(textembed)
            cmaplist.append(cmdata)
            emblist.append(textembed)
    return Glist, emblist
def getseqMatrix_Label(train_file_name, window_size=51, empty_aa='*'):
    prot = []  # list of protein name
    pos = []  # list of position with protein name
    rawseq = []
    all_label = []
    short_seqs = []
    temp_row = []
    half_len = (window_size - 1) / 2

    with open(train_file_name, 'r', encoding='utf-8') as rf:
        reader = csv.reader(rf)
        header = next(reader)
        for row in reader:
            temp_row = row
            position = int(row[2])
            sseq = row[3]
            protein = row[1]
            # if sseq == 'none':
            #     continue
            # if position>len(sseq):
            #     continue
            if checkinfo(sseq, protein, position) == False:
                continue
            rawseq.append(row[3])
            center = sseq[position - 1]
            all_label.append(int(row[0]))
            prot.append(row[1])
            pos.append(row[2])
            # short seq
            if position - half_len > 0:
                start = position - int(half_len)
                left_seq = sseq[start - 1:position - 1]
            else:
                left_seq = sseq[0:position - 1]

            end = len(sseq)
            if position + half_len < end:
                end = position + half_len
            right_seq = sseq[position:int(end)]

            if len(left_seq) < half_len:
                nb_lack = half_len - len(left_seq)
                left_seq = ''.join([empty_aa for count in range(int(nb_lack))]) + left_seq

            if len(right_seq) < half_len:
                nb_lack = half_len - len(right_seq)
                right_seq = right_seq + ''.join([empty_aa for count in range(int(nb_lack))])
            shortseq = left_seq + center + right_seq
            short_seqs.append(shortseq)
        targetY = listclass_to_one_hot(all_label)
        ONE_HOT_SIZE = 21
        # _aminos = 'ACDEFGHIKLMNPQRSTVWY*'
        Matr = np.zeros((len(short_seqs), window_size, ONE_HOT_SIZE))
        samplenumber = 0
        for seq in short_seqs:
            AANo = 0
            for AA in seq:
                index = letterDict[AA]
                Matr[samplenumber][AANo][index] = 1
                AANo = AANo + 1
            samplenumber = samplenumber + 1
    return Matr, targetY

def getMatrixLabel(train_file_name, window_size=51, empty_aa='*'):
    # input format   label, proteinName, postion,shortsequence,
    prot = []  # list of protein name
    pos = []  # list of position with protein name
    rawseq = []
    all_label = []
    short_seqs = []
    half_len = (window_size - 1) / 2
    with open(train_file_name, 'r') as rf:
        reader = csv.reader(rf)
        for row in reader:
            position = int(row[2])
            sseq = row[3]
            protein = row[1]
            if checkinfo(sseq, protein, position) == False:
                continue
            rawseq.append(row[3])
            center = sseq[position - 1]
            all_label.append(int(row[0]))
            prot.append(row[1])
            pos.append(row[2])

            # short seq
            if position - half_len > 0:
                start = position - int(half_len)
                left_seq = sseq[start - 1:position - 1]
            else:
                left_seq = sseq[0:position - 1]

            end = len(sseq)
            if position + half_len < end:
                end = position + half_len
            right_seq = sseq[position:int(end)]

            if len(left_seq) < half_len:
                nb_lack = half_len - len(left_seq)
                left_seq = ''.join([empty_aa for count in range(int(nb_lack))]) + left_seq

            if len(right_seq) < half_len:
                nb_lack = half_len - len(right_seq)
                right_seq = right_seq + ''.join([empty_aa for count in range(int(nb_lack))])
            shortseq = left_seq + center + right_seq
            short_seqs.append(shortseq)
        targetY = listclass_to_one_hot(all_label)
        ONE_HOT_SIZE = 21

        Matr = np.zeros((len(short_seqs), window_size, ONE_HOT_SIZE))
        samplenumber = 0
        for seq in short_seqs:
            AANo = 0
            for AA in seq:
                index = letterDict[AA]
                Matr[samplenumber][AANo][index] = 1
                AANo = AANo + 1
            samplenumber = samplenumber + 1

    return Matr, targetY