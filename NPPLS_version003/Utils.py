import pickle
import os
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import pdist, squareform

class Utils:
    
    # link to folder creation
    # Object saver
    # obj/ folder holds the binaries for the saved  stuff (dicts, arrays, etc etc)
    @staticmethod
    def save_obj(obj, name, folder='obj'):
        """
        # saves a given object. If ndarray, save as npy, else just pickle
        # Noteworthy is the fact that this function will try to create a directory
        named folder, if it is not present. In this way, maybe it is the case of
        reviewing this process to not create a folder. Or sometimes, root priviledge
        is demanded to execute it correctly.
        
        args:
            obj: is a savable object (array, list, tuple, integer, etc)
            name: is the file name (will be appended of a file extension throughout
            the process
            folder: the saving directory.
        """
        os.makedirs(os.path.dirname('./'+folder+'/'), exist_ok=True)
        # diminuir isso depois pois vai comer tempo com operação desnecessária
        
        #assert type(obj) in [list,np.ndarray], "Trying to save the unsavable. Type not available"
        if type(obj) == np.ndarray:
            name = folder+'/{}.{}'.format(name,'npy')
            np.save(name, obj)
        else:
            name = folder+'/{}.{}'.format(name,'pkl')       
            with open(name, 'wb') as f:
                pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    @staticmethod
    def save_batch(data, folder='obj',descriptor=''):
        """
        save_batch is a function for saving several objects at once
        args:
            data: a dictionary containing the names of files as keys
            and the savable objects as values.
            folder: the directory where to save all files
            descriptor: a simple piece of information to append to file title.
            currently we are using only integers, with standard printing
            trail of 4 0's
        """
        assert type(data)==dict, "Data provided must be a dict in the form {'x':x, 'y':y, 'desc':desc...}"
        for key in data.keys():
            Utils.save_obj(data[key],'{}_{:04d}'.format(key,descriptor),folder)
            
            

    # load a pickled object that is inside obj folder
    @staticmethod
    def load_obj(name, folder='obj'):
        """
        load_obj function:
        ##################
        args:
            name: name of the file you want to load (pkl or npy formats accepted)
            folder: location of the folder

        Returns:
            object: the opened object, depending on the format there will be different
            shapes, etc.
        """
        name = folder+'/'+name
        if 'npy' in name:
            return np.load(name)
        elif '.pkl' in name:
            with open(name, 'rb') as f:
                return pickle.load(f)

    # Scans the obj directory for pickled objects
    @staticmethod
    def scan_obj(cur,by_formats=['pkl','npy']):
        """
        Just a simple function to print all objects that belongs to the given format
        This function will die asap, since I don't need it anymore.
        """
        if os.path.isdir(cur):
            for f in os.listdir(cur):
                ext = f.partition('.')[2]
                print(ext)
                if ext in by_formats:
                    size = os.path.getsize(cur+f)/(1024*1024.0)
                    print('{} | format: {} | {:.3f} MB'.format(f.partition('.')[0],ext,size))
        else:
            ## use exception instead
            ## add exception for no obj folder
            ## add exception for empty folder as well
            print("## No obj/ folder in place")
            ## add option to create?

    # Pads the input data
    # receives a list object containing both protein representations and
    # dihedral angles.
    # Any 3D list can be used as input
    @staticmethod
    def pad_input_list(list_in,value=0.0,dtype=np.float32,max_length=None):
        """
        This function performs the padding of a given 2D array.
        As of now (12/05/2020) I'm performing the padding directly with
        np.pad and with no more fancy checks. Will update this function asap.

        args:
            list_in: a 2D array in the shape (timesteps, Features)
            value: the value with which to pad the input.
            dtype: corresponds to the dtype you use to perform calculations
            max_length: the maximum length to which pad the timesteps dimension

        Returns:
            padded: a padded np.ndarray with shape (1, max_length, Features)
        """
    # waits for a list of 2D unpaded nd.arrays (mx46) 
        if max_length == None:
            max_length = np.max([len(a) for a in list_in])
        return np.asarray([np.pad(seq,[(0,max_length-len(seq)),(0,0)],mode='constant',constant_values=0.) for seq in list_in])

    
    @staticmethod
    def apply_mask_to_array(array,mask):
        """
        # this functions applies a 0. mask on the data
        # this is not being used right now, since we are training with only complete
        # (no missing residues) proteins. But it will come in handy ASAP.
        # expects a array of shape (m,46) and a mask of shape (m,)
        # won't make assertions or exceptions, since it's not expected to someone to mess the code
        # so be careful when playing around, mate!
        it will basically get a binary mask and multiply a 2D array (timesteps, Features)
        by it. This is done to zero out the features at time steps where there are
        missing residues.
        I'm not dealing with Mr right now, so this wont be used anytime soon.
        """
        mask=np.asarray(mask)
        return array * mask.reshape((mask.shape[0],1))

    @staticmethod
    def make_prop_array(sequence,desc):
        """
        # function to create a prop array
        # All the properties are MinMax scaled to range [0,1]
        # These descriptors were taken from AAIndex dataBase and are
        # more thorougly detailed on the Readme (to Do)
        """
        return np.asarray([desc[aa] for aa in sequence])

    @staticmethod
    def get_str_sequence(sequence,aas):
        seq = []
        for res in sequence:
            seq.append(aas[np.argmax(res)])
        return seq

    @staticmethod
    def isnotebook():
        ''' This solution was drawn from Gustavo Bezerra's comment on
            stackoverflow question: https://stackoverflow.com/questions/15411967/how-can-i-check-if-code-is-executed-in-the-ipython-notebook
        '''
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False      # Probably standard Python interpreter

    @staticmethod
    def OuterConcatenate(x):
        """
        This operation performs pairwise feature building.
        Will detail more on this help, but for now refer to
        Raptor-X Contact paper for more information.
        """
        seqLen = x.shape[1]
        input2 = np.transpose(x,(1,0,2))
        #print(input2.shape)
        mesh = np.mgrid[0:seqLen, 0:seqLen]
        #print(mesh.shape)
        out=input2[mesh]
        #print(out.shape)
        output = np.concatenate((out[0],out[1]),axis=-1)
        #print(output.shape)
        return np.transpose(output,(2,0,1,3))
    
    @staticmethod
    def OuterConcatenate_tf(x):
        """
        # performs an operation similar to the outer product
        # but instead of multiplaying, it concatenates.
        # Input shape: (batch_size, sequence_length, features)
        # Output shape: (batch_size, sequence_length, sequence_length, 2*features)
        # This functions was first implemented on Jinbo's et al. paper:
        # DOI:
        """
        idx_except_last = tf.meshgrid(*[tf.range(s) for s in [x.shape[1],x.shape[1]]], indexing='ij')
        d2 = tf.transpose(x,(1,0,2))
        idx = tf.stack(idx_except_last)
        a = tf.gather(d2, idx)
        b = tf.concat([a[0],a[1]],axis=-1)
        b = tf.transpose(b,(2,0,1,3))
        return b
    '''
    @staticmethod
    def prep_feature_matrix(feat_matrix, max_len, padding_val=0.):
        # It will first outerconcat the the feat matrix
        # then pad
        assert feat_matrix.shape[1] <= max_len, "Cannot proceed with array of shape {} > {}".format(feat_matrix.shape[1], max_len)
        outcat = Utils.OuterConcatenate(feat_matrix) 
        padded = np.pad(outcat, [(0,0),(0,max_len-outcat.shape[1]), (0, max_len-outcat.shape[1]),(0,0)],
                        mode='constant', constant_values=padding_val) #
        
        mask = padded != padding_val
        print(mask.shape)
        mask = mask[...,0].astype('int')
        # In this way we add 0's to the diagonal of the masking matrix "padding the diagonals"
        print(mask.shape)
        print(mask[0].shape)
        print(mask[0])
        np.fill_diagonal(mask[0],0.)
        return padded, mask
    '''
    @staticmethod
    def prep_feature_matrix(feat_matrix, max_len, padding_val=0.):
        """
        Prepares the Feature matrix for Convolution in 2 dimensions.
        It will perform OuterConcatenation, pad and produce a binary mask.
        args:
            feat_matrix: a 3D array with shape (1, timesteps, Features).
            max_length: the maximum length to pad along axes [1,2] of the
            outerconcatenated feature_matrix
            padding_val: the value with which to pad the input
        Returns:
            padded: the padded, outerconcatenated feature matrix (shape = (1, max_length, max_length, 2*Features)
            mask: a binary mask, padded, with shape (1, max_length, max_length, 1)
        """            
        assert feat_matrix.shape[1] <= max_len, "Cannot proceed with array of shape {} > {}".format(feat_matrix.shape[1], max_len)
        outcat = Utils.OuterConcatenate(feat_matrix) 
        mask = np.ones((outcat.shape[1], outcat.shape[1]))
        np.fill_diagonal(mask, 0.)
        mask = np.expand_dims(mask, axis=[0,-1])
        padded = np.pad(outcat, [(0,0),(0,max_len-outcat.shape[1]), (0, max_len-outcat.shape[1]),(0,0)],
                        mode='constant', constant_values=padding_val) #
        
        mask = np.pad(mask, [(0,0),(0,max_len-mask.shape[1]), (0, max_len-mask.shape[1]),(0,0)],
                        mode='constant', constant_values=padding_val)
        # In this way we add 0's to the diagonal of the masking matrix "padding the diagonals"
    #     np.fill_diagonal(mask[0,],0.)
        return padded, mask

    @staticmethod
    def calc_self_dist_matrix(arr,measured_atom=1):
        '''
        Takes as input an array of Tert. Intended use is to select the
        measued_atom axis, detach from array and pass it through pdist.
        This procedure yields an (L*L) pairwise distance matrix in
        aproximatly 52.1 µs for our test systems, which is superb. For
        bow it is just calculating pdist and reformating in squareform.
        Squareform is important since preserving the sparsity and the symmetric
        information is also important when dealing with proteins.
        '''
        # calculates the self distance between every measured_atoms
        # args:
        # arr: array with shape (3L,3) The same as read by the record parser
        arr = arr.reshape((-1,3))#[:,measured_atom,:]
        return squareform(pdist(arr))
    
    @staticmethod
    def calc_contact_matrix(dist_mat,cutoffs=[0,1]):
        #results = np.zeros(dist_mat.shape)
        if len(cutoffs)==1 or type(cutoffs) == int:
            # contact map for a single cutoff
            # produces a single (L,L) contact array
            results = (dist_mat <= cutoffs[0]).astype('float32')
            
        else:
            # aqui tem um vetor0 com chape (L,L,1)
            # quando cutoff for == 0, eu tenho que fazer com que
            # 
            # else, produces a (L,L) array with len(cutoff) channels
            # or, of shape (L,L,len(cutoffs))
            results = np.zeros((len(cutoffs),*dist_mat.shape))
            for i, cutoff in enumerate(cutoffs):
                d_0 = cutoffs[0]-1 if i == 0 else cutoffs[i-1]
                results[i,] = ((dist_mat>d_0)&(dist_mat <= cutoff)).astype('float32')

                # nesse caso, transforma o primeiro canal (que eh zero)
                # uma matriz contato
                
        return results

    @staticmethod
    # Modificar RRC20 para tratar distâncias (????)
    def get_cebola(seq, contacts, normalize=False):
        dict_seq = {i:aa for i,aa in enumerate(seq)}
        # contacts tem que vir na forma (L,L,n_buckets)
        tab = np.zeros((contacts.shape[0],20,20))
        for i,d in enumerate(contacts):
            x = np.argwhere(d == 1)
            for e in x:
                tab[i,dict_seq[e[0]],dict_seq[e[1]]]+=1
        # transforms frequence into probability
        if normalize:
            #gambiarra
            # this is not meant to be done this way
            # but otherwise it would output a tensor full of nan's
            # and I really don't want a tensor full of nan's
            # if you are reading this, please find a way to avoid this, my friend
            t_sum = tab.sum(axis=0)
            t_sum[np.where(t_sum==0)]=0.01
            tab = tab/t_sum
            del t_sum
        return tab
 
