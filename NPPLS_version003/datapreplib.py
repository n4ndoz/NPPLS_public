# DataPreparator class
# Author: Fernando Limoeiro
# Desc.:
# This class is responsible for all the input data (ProteiNet Records) prep
# for the ML/DL procedures. This class takes two files (train and val) and
# processes a (mini)Batch object. All the preparations included on the
# Jupyter Notebook (Input_preparation_Module) is contained here.
#
# This version (v003) even though clearner and more error proof,
# is not memory efficient. Since we use dictionaries for the entire
# process of data preping, the lookup, plus the many calls to dicts,
# will be memory consuming. I dont know why, but when preping the dist pipeline
# with or without padding, it consumes a helluvah lot of swap memory.
# Currently I suspect the the scipy solution is the bottleneck
# for the dict issue, refer to https://www.jessicayung.com/python-lists-vs-dictionaries-the-space-time-tradeoff/)


import numpy as np
from Utils import Utils as utils # no need to import like this
# Doing this to import the right tqdm interface
if utils.isnotebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm
from angle_utils_v3 import Dihedral_Calculator
# Will deprecate this record_parse method
#from angle_utils_v3 import record_parser as rparse
dhc = Dihedral_Calculator()
from tensorflow.keras.utils import to_categorical as tc
import re
import os
NUM_DIMENSIONS=3


class switch(object):
    """Switch statement for Python, based on recipe from Python Cookbook."""

    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration
    
    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5
            self.fall = True
            return True
        else:
            return False


class Data_Prep_Pipeline:
    # This class holds methods to parse the ProteiNet dataset
    # Usage is:
    # insert protein
    '''
    params dictionary format:
    
    '''
    def __init__(self, seq_cutoff=256, mask_them=False,no_mr=True,
                 output_dir='', load_all=False, input_file='',n_feat=46):
        self.seq_cutoff=seq_cutoff
        self.mask_them = mask_them
        self.no_mr = no_mr
        self.load_all = load_all
        self.input_file = input_file
        self.aa_dict = {'C':'0','F':'1','L':'2','W':'3','V':'4',
                       'I':'5','M':'6','Y':'7','A':'8','P':'9',
                       'H':'10','G':'11','N':'12','T':'13','S':'14',
                       'R':'15','Q':'16','D':'17','K':'18','E':'19'}
        # Changed aa_dict description dict as of 09/08/2020
        # to adequate some calculations done at some papers.
        # we are indexing "hydrophoobic" residues first and
        #self.aa_dict = {'A': '0', 'C': '1', 'D': '2', 'E': '3',
        #                'F': '4', 'G': '5', 'H': '6', 'I': '7',
        #                'K': '8', 'L': '9', 'M': '10', 'N': '11',
        #                'P': '12', 'Q': '13', 'R': '14', 'S': '15',
        #                'T': '16', 'V': '17', 'W': '18', 'Y': '19'}
        self.rev_aa_dict= { v:k for (k,v) in self.aa_dict.items() }
        self.batch_num = 0
        self.n_feat=n_feat

    def letter_to_num(self, string, dict_):
        """ Convert string of letters to list of ints """
        #dict_= {'A': '0', 'C': '1', 'D': '2', 'E': '3', 'F': '4', 'G': '5', 'H': '6', 'I': '7', 'K': '8', 'L': '9', 'M': '10', 'N': '11', 'P': '12', 'Q': '13', 'R': '14', 'S': '15', 'T': '16', 'V': '17', 'W': '18', 'Y': '19'}
        patt = re.compile('[' + ''.join(dict_.keys()) + ']')
        num_string = patt.sub(lambda m: dict_[m.group(0)] + ' ', string)
        #print(num_string)
        #print(type(num_string))
        num = [int(i) for i in num_string.split()]
        return num
    
    def open_dataset_file(self,pth):
        try:
            # this data_file will be used, no matter which dataset
            # we will be processing, no matter if it is training or validation
            # sets.
            self.input_data_file = open(pth)
            #return data_file
        except IOError:
            print('Could not open the file {}. Try again please.'.format(pth))
    def pairwise_distance(self, arr,measured_atom=1):
        # calculates the distance between every measured_atoms
        # args:
        # arr: array with shape (3L,3) The same as read by the record parser
        from scipy.spatial.distance import pdist, squareform
        arr = arr.reshape((-1,3,3))[:,measured_atom,:]
        return squareform(pdist(arr))

    def make_loss_mask(self, input_shape, max_len):
        '''
        this function makes a simple mask for calculating the loss over
        the protein. MR in the midle of the sequence are not considered.
        This needs some reimplementation.
        '''
        mask = np.zeros((max_len,max_len))
        mask[:input_shape[0],:input_shape[0]].fill(1)
        return mask
    
    def read_record(self, file_, num_evo_entries,get_data,mode='dist',return_seq=True):
        """ Read a Mathematica protein record from file and convert into dict. """
        # This method and the Switch Class were taken from the original ProteinNet repo.
        # arg is a open file, num_evo_entries is 20 by default
        #
        # Strip the dict and insert lists for each of the types of entries
        desc_dict = utils.load_obj('desc_dict.pkl')
        desc_dict_rev = {int(self.aa_dict[k]):v for (k,v) in desc_dict.items()}

        # this will be stripped soon.
        if get_data == 'all':
            get_data = ['id','primary','evolutionary','secondary','tertiary','mask']
        aa_dict = {'A': '0', 'C': '1', 'D': '2', 'E': '3', 'F': '4', 'G': '5', 'H': '6', 'I': '7', 'K': '8', 'L': '9', 'M': '10', 'N': '11', 'P': '12', 'Q': '13', 'R': '14', 'S': '15', 'T': '16', 'V': '17', 'W': '18', 'Y': '19'}
        _mask_dict = {'-': '0', '+': '1'}
        while True:
            next_line = file_.readline()
            for case in switch(next_line):
                if case('[ID]' + '\n'):
                    if 'id' in get_data:
                        id_ = file_.readline()[:-1]
                        
                elif case('[PRIMARY]' + '\n'):
                    if 'primary' in get_data:                        
                        prim = file_.readline()[:-1]
                        primary = self.letter_to_num(prim, self.aa_dict)
                        
                  
                elif case('[EVOLUTIONARY]' + '\n'):
                    if 'evo' in get_data:
                        evolutionary = []
                        for residue in range(num_evo_entries): evolutionary.append(np.asarray([float(step) for step in file_.readline().split()]))
                        evolutionary = np.array(evolutionary)
                        evolutionary = evolutionary.T # this will turn evo into an array of shape (-1, 20) Fuck yeah
                        
                elif case('[TERTIARY]' + '\n'):
                    if 'tert' in get_data:
                        tertiary = []
                        for axis in range(3): tertiary.append([float(coord) for coord in file_.readline().split()])
                        
                elif case('[MASK]' + '\n'):
                    if 'mask' in get_data:
                        mask = file_.readline()[:-1]
                        mask = self.letter_to_num(mask, _mask_dict)
                
                # ends reading a Single record
                elif case('\n'):
                    # perform preprocessing
                    if 0 in mask:
                        return -1
                    if len(primary) > self.seq_cutoff:
                        return -1
                    prop = utils.make_prop_array(primary,desc_dict_rev)
                    x = np.concatenate([tc(primary,num_classes=20),evolutionary,prop],axis=1)
                    tertiary = dhc.get_backbone_coords(np.array(tertiary))
                    if mode == 'dih':
                        y = dhc.fix_array(dhc.get_phi_psi(tertiary))
                        y = y.astype('float32')
                    elif mode =='dist':
                        y = self.pairwise_distance(tertiary)
                    elif mode == 'zmat':
                        tertiary = tertiary.reshape((1,-1,3,3))
                        tertiary = tertiary[:,:,1,:].reshape((1,-1,1,3)).astype('float32')/100
                        dist = dhc.calc_dist_vector(tertiary).numpy().reshape((1,-1,1))
                        ang = np.radians(dhc.calc_angle_vector(tertiary).numpy().reshape((1,-1,1)))
                        dih = dhc.calc_dihedral_vector(tertiary).numpy()
                        y = np.concatenate([dist,ang,dih],axis=-1)
                    elif mode == 'tert':
                        #tertiary = tertiary.reshape((1,-1,3,3))
                        tertiary = tertiary.reshape((-1,3,3))
                        #tertiary = tertiary[:,:,1,:].reshape((1,-1,1,3)).astype('float32')/100
                        #y = self.pairwise_distance(tertiary)
                        return [x.astype('float32',copy=False), tertiary.astype('float32',copy=False), np.asarray(id_), primary]
                    if return_seq:
                        return [x.astype('float32',copy=False), y.astype('float32',copy=False), np.asarray(id_), tertiary.astype('float32',copy=False), primary]
                    else:# if anything changes, i will be replacing this with a more pythonic way soon enough
                        # if I really need tertiary structure at anytime, i just code it her
                        return [x.astype('float32',copy=False), y.astype('float32',copy=False), np.asarray(id_)]
                    
                elif case(''):
                    return None
   
    def prep_data(self, mode='dih', pad_all=True,save_dir='dist_train_test'):
        self.open_dataset_file(self.input_file)
        data_file=self.input_data_file
        os.makedirs(save_dir,exist_ok=True)
        self.batch_num = 1
        import gc
        while True:
            
            num_evo_entries = 20
            read_ = self.read_record(data_file, num_evo_entries,['tert','primary','evo','mask', 'id'],mode)
            if type(read_) == list:
                if pad_all and mode == 'dih':
                    read_[0]= np.pad(read_[0],[(0,self.seq_cutoff-read_[0].shape[0]),(0,0)],mode='constant',constant_values=0.)
                    read_[1]= np.pad(read_[1],[(0,self.seq_cutoff-read_[1].shape[0]),(0,0)],mode='constant',constant_values=0.)
                elif pad_all and mode == 'dist':
                    read_[0], mask = utils.prep_feature_matrix(read_[0].reshape((1, *read_[0].shape)), self.seq_cutoff)
                    read_[1] = np.pad(read_[1].reshape((1,*read_[1].shape, 1)), [(0,0),(0,self.seq_cutoff-read_[1].shape[1]), (0, self.seq_cutoff-read_[1].shape[1]),(0,0)],
                                mode='constant', constant_values=0.)
                if pad_all and mode == 'zmat':
                    read_[0]= np.pad(read_[0],[(0,self.seq_cutoff-read_[0].shape[0]),(0,0)],mode='constant',constant_values=0.)
                    read_[1]= np.pad(read_[1][0],[(0,self.seq_cutoff-read_[1][0].shape[0]),(0,0)],mode='constant',constant_values=0.)
                elif mode=='tert':
                    # x.astype('float32',copy=False), tertiary.astype('float32',copy=False), np.asarray(id_), primary
                    np.save(os.path.join(save_dir,'mask_{:04d}.npy'.format(self.batch_num)),
                            self.make_loss_mask(read_[0].shape, self.seq_cutoff))
                    read_[0]= np.pad(read_[0],[(0,self.seq_cutoff-read_[0].shape[0]),(0,0)],mode='constant',constant_values=0.)
                    read_[1] = np.pad(read_[1],[(0,self.seq_cutoff-read_[1].shape[0]),(0,0),(0,0)],mode='constant',constant_values=0.)
                    np.save(os.path.join(save_dir,'x_{:04d}.npy'.format(self.batch_num)), read_[0])
                    np.save(os.path.join(save_dir,'tert_{:04d}.npy'.format(self.batch_num)), read_[1])
                    np.save(os.path.join(save_dir,'id_{:04d}.npy'.format(self.batch_num)), read_[2])
                    np.save(os.path.join(save_dir,'seq_{:04d}.npy'.format(self.batch_num)), read_[3])
                    
                    
                # Mudei aqui rapidinho, só pra testar um lance
                #np.save(os.path.join(save_dir,'x_{:04d}.npy'.format(self.batch_num)), read_[0])
                #np.save(os.path.join(save_dir,'y_{:04d}.npy'.format(self.batch_num)), read_[1])
                '''
                if len(read_) == 4:
                    # in case we want to save the tertiary
                    np.save(os.path.join(save_dir,'tert_{:04d}.npy'.format(self.batch_num)), read_[3])
                #return read_
                '''
                if self.batch_num % 2 == 0:
                    print("# Got already {} batches".format(self.batch_num), end='\r')
                self.batch_num+=1
                gc.collect()
                
            elif read_ == None:
                print('# Done Processing data.')
                break
            
                    

    
    
