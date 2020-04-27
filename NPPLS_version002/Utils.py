import pickle
import os
import numpy as np

class Utils:
    
    # link to folder creation
    # Object saver
    # obj/ folder holds the binaries for the saved  stuff (dicts, arrays, etc etc)
    @staticmethod
    def save_obj(obj, name, folder='obj'):
        # saves a given object. If ndarray, save as npy, else just pickle
        # reformular essa parte dos paths, tá feio demaaaaais!
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
        # descriptor é uma informação qualquer que
        # se queira colocar no nome do arquivo.
        # esse assert para o programa se mandar data errado
        # tem que importar um dict cada chave sendo o nome do arquivo
        assert type(data)==dict, "Data provided must be a dict in the form {'x':x, 'y':y, 'desc':desc...}"
        for key in data.keys():
            Utils.save_obj(data[key],'{}_{:04d}'.format(key,descriptor),folder)
            
            

    # load a pickled object that is inside obj folder
    @staticmethod
    def load_obj(name, folder='obj'):
        name = folder+'/'+name
        if 'npy' in name:
            return np.load(name)
        elif '.pkl' in name:
            with open(name, 'rb') as f:
                return pickle.load(f)

    # Scans the obj directory for pickled objects
    @staticmethod
    def scan_obj(cur,by_formats=['pkl','npy']):
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
    # waits for a list of 2D unpaded nd.arrays (mx46) 
        if max_length == None:
            max_length = np.max([len(a) for a in list_in])
        return np.asarray([np.pad(seq,[(0,max_length-len(seq)),(0,0)],mode='constant',constant_values=0.) for seq in list_in])

    # this functions applies a 0. mask on the data
    # this is not being used right now, since we are training with only complete
    # (no missing residues) proteins. But it will come in handy ASAP.
    @staticmethod
    def apply_mask_to_array(array,mask):
        # expects a array of shape (m,46) and a mask of shape (m,)
        # won't make assertions or exceptions, since it's not expected to someone to mess the code
        # so be careful when playing around, mate!
        mask=np.asarray(mask)
        return array * mask.reshape((mask.shape[0],1))

    @staticmethod
    # function to create a prop array
    # All the properties are MinMax scaled to range [0,1]
    # These descriptors were taken from AAIndex dataBase and are
    # more thorougly detailed on the Readme (to Do)
    def make_prop_array(sequence,desc):
        
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
