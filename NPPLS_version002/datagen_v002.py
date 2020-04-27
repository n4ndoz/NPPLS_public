# DataPreparator class
# Author: Fernando Limoeiro
# (Working for TF2)(Well, trying to..)
# Desc.:
# This class is responsible for all the input data (ProteiNet Records) prep
# for the ML/DL procedures. This class takes two files (train and val) and
# processes a (mini)Batch object. All the preparations included on the
# Jupyter Notebook (Input_preparation_Module) is contained here.
#
# Refer to: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
# This will give some hints on future updates

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
    def __init__(self, seq_cutoff=500, mask_them=False,no_mr=True,
                 output_dir='', load_all=False, batch_size=10, input_file=''):
        self.seq_cutoff=seq_cutoff
        self.mask_them = mask_them
        self.no_mr = no_mr
        self.load_all = load_all
        self.input_file = input_file
        self.batch_size  = (None if load_all else batch_size)
        self.aa_dict = {'A': '0', 'C': '1', 'D': '2', 'E': '3',
                                  'F': '4', 'G': '5', 'H': '6', 'I': '7',
                                  'K': '8', 'L': '9', 'M': '10', 'N': '11',
                                  'P': '12', 'Q': '13', 'R': '14', 'S': '15',
                                  'T': '16', 'V': '17', 'W': '18', 'Y': '19'}
        self.rev_aa_dict= { v:k for (k,v) in self.aa_dict.items() }
        self.batch_num = 0

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
    def calc_dist_matrix(self, arr,measured_atom=1):
        # calculates the distance between every measured_atoms
        # args:
        # arr: array with shape (3L,3) The same as read by the record parser
        from scipy.spatial.distance import pdist, squareform
        arr = arr.reshape((-1,3,3))[:,measured_atom,:]
        return squareform(pdist(arr))
    
    def read_record(self, file_, num_evo_entries,get_data):
        """ Read a Mathematica protein record from file and convert into dict. """
        # arg is a open filed, num_evo_entries is 20 by default
        #
        # Strip the dict and insert lists for each of the types of entries
        prim_list = []
        tert_list = []
        mask_list = []
        evol_list = []
        dict_ = {}
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
                        #dict_.update({'id': id_})
                        
                elif case('[PRIMARY]' + '\n'):
                    if 'primary' in get_data:
                        
                        #aa_dict = {'A': '0', 'C': '1', 'D': '2', 'E': '3', 'F': '4', 'G': '5', 'H': '6', 'I': '7', 'K': '8', 'L': '9', 'M': '10', 'N': '11', 'P': '12', 'Q': '13', 'R': '14', 'S': '15', 'T': '16', 'V': '17', 'W': '18', 'Y': '19'}
                        #primary = file_.readline()[:-1]
                        primary = self.letter_to_num(file_.readline()[:-1], self.aa_dict)
                        #dict_.update({'primary': primary})
                        prim_list = primary
                elif case('[EVOLUTIONARY]' + '\n'):
                    if 'evo' in get_data:
                        evolutionary = []
                        for residue in range(num_evo_entries): evolutionary.append(np.asarray([float(step) for step in file_.readline().split()]))
                        evolutionary = np.array(evolutionary)
                        evolutionary = evolutionary.T # this will turn evo into an array of shape (-1, 20) Fuck yeah
                        #dict_.update({'evo': evolutionary})
                        evol_list = evolutionary
                elif case('[SECONDARY]' + '\n'):
                    if 'sec' in get_data:
                        secondary = letter_to_num(file_.readline()[:-1], _dssp_dict)
                        dict_.update({'secondary': secondary})
                elif case('[TERTIARY]' + '\n'):
                    if 'tert' in get_data:

                        tertiary = []
                        for axis in range(NUM_DIMENSIONS): tertiary.append([float(coord) for coord in file_.readline().split()])
                        #dict_.update({'tertiary': tertiary})
                        tert_list = tertiary 
                elif case('[MASK]' + '\n'):
                    if 'mask' in get_data:
                        mask = file_.readline()[:-1]
                        mask = self.letter_to_num(mask, _mask_dict)
                        #dict_.update({'mask': mask})
                        mask_list = mask
                # ends reading a Single record
                elif case('\n'):
                    #print('cu')
                    return [tertiary, primary, evolutionary, mask,id_]
                    
                elif case(''):
                    #print('xota')
                    return None
    # this is now a Keras generator for now
    # This is just the preparation
    
    def parse_data_block(self, data_limit=None,
                         close_file=False,num_evo_entries=20):
        # if load_all = True user must supply the total number of proteins on the
        # data set

        # This function takes care of reading all the data set and parsing
        # it to a series of lists.
        # The objective of this setup is to save the dataset to a numpy array
        # or NPZ file (with multiple arrays inside)
        data_file=self.input_data_file # forces the use of data_file only by this method
        # More easily to control now.
        #print(os.getcwd())
        desc_dict = utils.load_obj('desc_dict.pkl')
        
        # revert AA titles to numeric/integer this way we can properly work with the input arrays we get as input
        desc_dict_rev = {int(self.aa_dict[k]):v for (k,v) in desc_dict.items()}
        ids = []
        X=[]
        dihs=[]
        dists = []
        masks = []

        if data_file.closed:
            raise Exception('Data file already closed')
        
        #pbar = tqdm(total=data_limit)
        #for i in tqdm(range(data_limit),desc='Calculating Dihedral_Angles for {} proteins.'.format(p_number)):# tqdm(range(p_number),desc='Loading Protein Files'):
        #with tqdm(total = data_limit) as pbar:
        with tqdm(total = data_limit,desc='Processing batch {} with {} entries'.format(self.batch_num,data_limit),leave=False) as pbar:
            #print(len(X))
            while len(X) < data_limit and len(X) != None:
                read_ = self.read_record(data_file, num_evo_entries,['tert','primary','evo','mask', 'id'])
                if self.no_mr and read_ != None:
                    if 0 in read_[3]:
                        read_=-1
                
                
                if read_ is not None and read_ != -1:
                    # dont include huge sequences for the sake of RAM (for now)
                    if len(read_[1])<=self.seq_cutoff:
                        # now it stores the processed
                        # processes the Tert entry from ProteiNet and
                        # transforms to a np.array with shape recognizable
                        # by DHC
                        # output_shape: (L, 3, 3) | XYZ of each NCaC for each residue
                        read_[0] = dhc.get_backbone_coords(np.array(read_[0]))
                        dists.append(self.calc_dist_matrix(read_[0]))
                        
                        #np.save('obj/teste_read0',read_[0])
                        # Calculates DIH for ptotein and apppends to dihs list
                        # output shape: (L,2)
                        dihs.append(dhc.fix_array(dhc.get_phi_psi(read_[0])))

                        # calculates molecular properties given a pickled
                        # properties dictionary
                        # It's values are already MinMax scaled
                        # output shape: (L,prop_number)
                        #print(read_[1])
                        #print(desc_dict.keys())
                        #print(desc_dict.values())
                        prop = utils.make_prop_array(read_[1],desc_dict_rev)

                        # one-hot encodes each residue on a protein sequence
                        # output shape: (L, 20)
                        read_[1] = tc(read_[1],num_classes=20)

                        # Appends the already read mask to a different array
                        # output shape: (L,)
                        masks.append(read_[3])

                        # Append the properties vector to the read_ list
                        #read_.append(prop)

                        # Finally, makes the instance of X by concatenating
                        # 1-hot sequence, evolutionary info and properties
                        # on axis 1
                        # output shape: (L,20+20+N) | where N is the number of properties
                        temp_=np.concatenate([read_[1],read_[2],prop],axis=1)

                        # applies masking to the entire sequence
                        # In other words, it applies a Zero column to the columns where
                        # there are missing residues. This turns one single missing residue
                        # into an empty entry.
                        # I wont use it for now, since it dropped model performance overtime
                        # But it will be a good tool to look at in the future
                        if self.mask_them:
                            temp_ = utils.apply_mask_to_array(temp_,read_[3])
                        
                        #reads.append(read_)
                        # appends the unpadded entries
                        X.append(temp_)
                        ids.append(read_[-1])
                        # updates the progress bar from TQDM
                    pbar.update(1)
                        
                # if it reaches the EOF, returns none
                # It will obviously return a last tensor with length
                # smaller than the previous ones.
                # TODO: to aim this problem I will add random samples of the
                # other distributions.
                elif read_ == None:
                    # Do not close the file now
                    # The file handling is performed by another
                    # method
                    #data_file.close()
                    # this will break the while
                    pbar.close()
                    return X, dihs, dists, ids
        if close_file:
            data_file.close() # same here. gotta make another method
        # reads is a list containing tensors into which the
        pbar.close()
        return X, dihs, dists, ids # X, y 

    def make_batch_2(self, data_limit = None, pad_to_cutoff=True, at_once=False, chunk_size=100, to_file=True):
        ''' This method performs the parsing of an entire file or chunks of it.
            Args:
                data_limit: must be int or None. If int is given, the method will
                get data_limit entries from the dataset.

                pad_to_cutoff: bool: if True, apply padding to all entries (L, 46) on axis 1, adding max_length - L elements.

                at_once: bool: if True, performs the preprocessing all at once, saving to a file (See to_file) or returning
                the chunk.

                chunk_size: int: Refers to the size of each chunk to be returned. It is defaulted to 100, but one can change that.

            Outputs:

                X: array or list (if not padded) containing entries. If padded, X has dimensions (B,sequence_ length, aa_features)
                (sequence_length == max_length, if padded)where B is chunk_size, in case at_once is set to False, or len(X) otherwise.

                y: same type of X, but with shapes (L,2) or (B, L, 2). (Will change as soon as I implement the contact part)

                ids: list of strings, containing the PDBID of each entry. The PDBID follows the standard proposed by RCSB.
                
        '''
        # default lists
        #Xs = []
        #Ys = []
        #ids = []
        # primeiro processa X, y, ids e depois salva ou retorna!
        pass

    def make_batch(self,chunk_size=100, to_file=True, pad_to_cutoff=True,dims=[0,500,46],folder_name='latest_train',save_chunks=True):
        # implementar depois uma maneira de carregar tudo na memória e embaralhar pra salvar em chunks
        #eu podia implementar isso como um monte de métodos diferentes, mas vai poluir o código demais
        # Quer saber? Vou tirar essa palhaçada.
        # Das duas uma:
        # 1- Carrego tudo de uma vez, shuffle e depois salvar em baldes
        # 2- Carrego de chunck_size em chunk_size, aleatorizo chunk_size e depois implemento um
        # shuffle Hadoop-like (com vários arquivos).

        # Ta confuso. Vou modificar isso depois, pois por hora funciona
        # Essa variável save_chunks é lixo tbm
        
        # Posso mudar a lógica também pra: carrega de vários arquivos
        self.open_dataset_file(self.input_file)
        self.batch_num = 1

        X = []
        Y = []
        dists = []
        IDs = []
        
        
        while True:
            x, y, dists, ids= self.parse_data_block(chunk_size)
            if len(x) == 0:
                print("Finished loading everything")
                break
            if pad_to_cutoff:
                x = utils.pad_input_list(x,max_length=self.seq_cutoff)
                y = utils.pad_input_list(y,max_length=self.seq_cutoff)
            
            #print(x.shape)
            
            # ao invés de usar pad_to_cutoff eu posso simplesmente usar
            # o tipo, né?
            '''
            if pad_to_cutoff:
                if to_file and save_chunks:
                    utils.save_batch({'x':x,'y':y, 'dists': dists, 'ids': ids}, folder_name,self.batch_num)
                elif to_file and not save_chunks:
                    X = np.append(X,x,axis=0)
                    #print(X.shape)
                    Y = np.append(Y,y,axis=0)
            elif not pad_to_cutoff:
                if to_file and save_chunks:
                    utils.save_batch({'x':x,'y':y, 'dists': dists, 'ids': ids},folder_name,self.batch_num)
                elif to_file and not save_chunks:
                    X.append(x)
                    Y.append(y)
            IDs.extend(ids)
            '''
                     
            if save_chunks:
                utils.save_batch({'x':x,'y':y, 'dists': dists, 'ids': ids},folder_name,self.batch_num)
            else:
                X.append(x)
                Y.append(y)
            IDs.extend(ids)
            #self.batch_num+=1

            if len(x) < chunk_size:
                print("Finished loading everything")
                break
            
            self.batch_num+=1
            
        if not save_chunks:
            utils.save_batch({'x':X,'y':Y, 'dists': dists, 'ids': IDs},folder_name,self.batch_num)
            
            #print('to_file')
            # padronizar 'lateste pra %day%month%year[:2] URGENTE, fi
            '''
            utils.save_obj(X,'X_{}_{}'.format(folder_name),'latest')
            utils.save_obj(Y,'Y_{}_{}'.format(folder_name),'latest')
            utils.save_obj(ids,'IDs_{}_{}'.format(folder_name),'latest')
            '''
            
            return None
        else:
            return None
        # From now on, it will only save to files
        # next version I will remove to_file from default vars
        # 13/04/2020
        #else:
        #    print('a')
        #    return X, Y, IDs
            
            
            
        












    
