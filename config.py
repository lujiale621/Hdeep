lr=0.05
num_epochs=100
root='~/'
train_file = 'ludeep_datasets/datasetpre/PhosVarDeep_datasets/S,T/set1/S,T-train-set1-test.csv'
val_file = 'ludeep_datasets/datasetpre/PhosVarDeep_datasets/S,T/set1/S,T-val-set1-test.csv'
test_file = 'ludeep_datasets/datasetpre/PhosVarDeep_datasets/S,T/set1/S,T-test-set1-test.csv'
cpath = root+"data/cmap/cmap/"
embdict_path = root+"data/embdict.npz"
emb_path=root+'data/emb/'
emb_map = root+"data/uniprot_sprot.map"
pssm_root_url =root+ 'data/pssm/out'
pssm_fn = root+"data/pssm/matrix_25"
dssp_root_url = root+'data/dssp/dssp'
dssp_fn = root+"data/dssp/dssp_matrix_25"