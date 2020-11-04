# Utils for the Angle Calculators

"""
Text-based parser for ProteinNet Records.
"""

__author__ = "Mohammed AlQuraishi"
__copyright__ = "Copyright 2019, Harvard Medical School"
__license__ = "MIT"

#!/usr/bin/python

# imports
import sys
import re
import numpy as np
import tensorflow as tf
#for now tqdm is out
#import tqdm 
# Constants
NUM_DIMENSIONS = 3



# ptn is a np.array of shape (NUM_DIMENSIONS,SEQ_LEN*3)
    # Why?
    # In ProteinNet, residue coordinates (tertiary struc) are represented solely by the backbone
    # atoms (N, Ca, C). When parsing, via the Quraishi's supplied scripts, we should group the tertiary
    # matrix so that each index corresponds to a tuple (N[x,y,z],Ca[x,y,z],C[x,y,z]) so we could apply
    # the praxeolitic formula to calculate Dihedral Angles.

    # A reminder:
    # This ain't no "TF record parser" utility. All utils here are solely for
    # Alphabet calculation.
    # Since we hope to examine carefully the statistics, we need a separate tool.
    # In the near future, this one script will be responsible for:
    # 1- Parsing PN record tertiary and primary structure;
    # 2- Calculating statistics (correlation between residue triplet and angle, clustering (KMEANS) among others)
    # 3- Outputing a detailed visualization of the statistics and ploting the ramachandram space
    

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

class record_parser:
    
    # those are static methods, since it wont be linked with objects or instances od the class
    # I need them to be standalone and organized on those classes
    
    def letter_to_num(self, string, dict_):
        """ Convert string of letters to list of ints """
        patt = re.compile('[' + ''.join(dict_.keys()) + ']')
        num_string = patt.sub(lambda m: dict_[m.group(0)] + ' ', string)
        num = [int(i) for i in num_string.split()]
        return num
    
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
                        primary = self.letter_to_num(file_.readline()[:-1], aa_dict)
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
                        
                        mask = self.letter_to_num(file_.readline()[:-1], _mask_dict)
                        #dict_.update({'mask': mask})
                        mask_list = mask
                elif case('\n'):
                    #return dict
                    if 'primary' in get_data:
                        #return dict_
                        #print(primary)
                        return [tertiary, primary, evolutionary, mask,id_]
                    else:
                        return tertiary
                elif case(''):
                    return None

# main. accepts two command-line arguments: input file and the number of entries in evo profiles, and outputs dicts to stdout

#import tensorflow as tf


#input_path      = sys.argv[1] 
#num_evo_entries = int(sys.argv[2]) if len(sys.argv) == 3 else 20 # default number of evo entries
class Dihedral_Calculator:
    
    def get_backbone_coords(self,ptn):
        # ptn is a np.array of shape (NUM_DIMENSIONS,SEQ_LEN*3)
        # Why?
        # In ProteinNet, residue coordinates (tertiary struc) are represented solely by the backbone
        # atoms (N, Ca, C). When parsing, via the Quraishi's supplied scripts, we should group the tertiary
        # matrix so that each index corresponds to a tuple (N[x,y,z],Ca[x,y,z],C[x,y,z]) so we could apply
        # the praxeolitic formula to calculate Dihedral Angles.

        # A reminder:
        # This ain't no "TF record parser" utility. All utils here are solely for
        # Alphabet calculation.
        # Since we hope to examine carefully the statistics, we need a separate tool.
        # In the near future, this one script will be responsible for:
        # 1- Parsing PN record tertiary and primary structure;
        # 2- Calculating statistics (correlation between residue triplet and angle, clustering (KMEANS) among others)
        # 3- Outputing a detailed visualization of the statistics and ploting the ramachandram space
        # 4- Store the given alphabets
        coords = [] # list to store the coordinates of each atom
        for i in range(ptn.shape[1]):
            
            coords.append(ptn[...,i])

        return (np.array(coords))

    
    
    def extract_dihedrals(self,p):
        # vou fazer à partir do código do Meow lá
        # melhor em TF, pq extrai e calcula mais rápido
        calculator = DihedralCalculator()
        #prot = p[1,:,:]
        prot_len = p.shape[1]/3
        dihedral_calculator = DihedralCalculator()
        true_dihedrals = dihedral_calculator.dihedral_pipeline(p, protein_length = int(p.shape[0]/3))

        with tf.Session() as sess:
            sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))    
            true_dihedrals_ = sess.run([true_dihedrals])
    def tf_rad2deg(self, rad):
        pi_on_180 = 0.017453292519943295
        return rad / pi_on_180

    # Author: Michal (refer to him and his implementation later)
    
    def dihedral_tf3(self,p,to_deg=False):
        p0 = tf.gather(p, 0, axis=2)
        p1 = tf.gather(p, 1, axis=2)
        p2 = tf.gather(p, 2, axis=2)
        p3 = tf.gather(p, 3, axis=2)
        
        b0 = -1.0 * (tf.subtract(p1, p0))
        b1 = tf.subtract(p2, p1)
        b2 = tf.subtract(p3, p2)
        
        b1 = tf.divide(b1, tf.norm(b1, axis=2, keepdims=True))
        #b1 = tf.where(tf.is_nan(b1), tf.ones_like(b1), b1)
        b1 = tf.where(tf.math.is_nan(b1), tf.ones_like(b1), b1)
        
        v = tf.subtract(b0, tf.einsum('bi,bij->bij', tf.einsum('bij,bij->bi', b0, b1), b1))
        w = tf.subtract(b2, tf.einsum('bi,bij->bij', tf.einsum('bij,bij->bi', b2, b1), b1))
        
        x = tf.reduce_sum( tf.multiply( v, w ), 2, keepdims=True )
        
        #y = tf.reduce_sum( tf.multiply( tf.cross(b1, v), w ), 2, keepdims=True )

        y = tf.reduce_sum( tf.multiply( tf.linalg.cross(b1, v), w ), 2, keepdims=True )
        if to_deg:
            return self.tf_rad2deg(tf.atan2(y,x))
        else:
            return tf.atan2(y,x)

    def get_phi_psi(self,ptn):
        #tf.reset_default_graph()
        ptn = np.expand_dims(ptn, axis=0)

        #p2= tf.convert_to_tensor(np.concatenate([ptn,ptn]))
        p2 = tf.convert_to_tensor(ptn)
        #print(p2.get_shape())

        # p2 = np.expand_dims(p2, axis=0)
        p2 = p2[:,:,:,None]
        #print(p2.get_shape())

        #
        '''
        p2 = tf.extract_image_patches(p2,
          ksizes=[1, 4, 3, 1],
          strides=[1, 1, 1, 1],
          rates=[1, 1, 1, 1],
          padding='VALID')

        '''
        p2 = tf.image.extract_patches(p2,
          sizes=[1, 4, 3, 1],
          strides=[1, 1, 1, 1],
          rates=[1, 1, 1, 1],
          padding='VALID')
        #print(p2.shape)
        p2 = tf.reshape(tf.squeeze(p2), [1, -1, 4, 3])

        angles3 = self.dihedral_tf3(p2)

        #with tf.Session() as sess:
        #    angles3_,p_shape,p2_shape = sess.run([angles3, tf.shape(p2), tf.shape(p2)])
        
        angles3_ = np.insert(angles3, 0, None, axis=1)
        #print(angles3_.shape)
        psi_and_phi_i = np.array(np.sort(list(range(0,angles3_.shape[1],3)) + list(range(1,angles3_.shape[1],3))))
        #print(phi_and_psi_i.shape)
        phi_and_psi = angles3_[0][psi_and_phi_i][:,0]
        #print(phi_and_psi.shape)
        phi_and_psi = np.append(phi_and_psi, None)
        #print(phi_and_psi.shape)
        angles = phi_and_psi.reshape(-1,2)
        #print(angles.shape)
        return angles

    def save_dihedrals_to_csv(self,angles,out_file):
    # a simple file writer 
        with open(out_file,'a') as out:
            #print(angles.shape[0])
            for i in range(angles.shape[0]):
                out.write('{},{}\n'.format(angles[i][0],angles[i][1]))
                        
    def fix_array(self,b):
        
        b = np.where(np.equal(b, None), np.nan, b)
        col_mean = np.nanmean(b,axis=0)
        col_mean
        b = b.astype('float64') 
        b[np.isnan(b)]=0.0
        
        return b

############# ZMat region ######################
    @staticmethod
    def calc_dist_vector(ptn):
        # Working!! 
        #Calculates inter Ca sequential distance
        ptn = tf.image.extract_patches(ptn,
          sizes=[1, 2, 1, 1],
          strides=[1, 1, 1, 1],
          rates=[1, 1, 1, 1], padding='VALID')
        # reshaping to (1, L-1, 2, 3 ) two consecutive atoms
        ptn = tf.reshape(ptn, (1,-1,2,3))
        # compute L2 norm distance between consecutive Cas
        ptn = tf.linalg.norm(ptn[:,:,1,:] - ptn[:,:,0,:], axis=-1)
        # Insert a 0 vector at the begining of ptn, to 
        # assert the length
        #ptn = tf.concat([tf.zeros((ptn.shape[0],1,2,3)),ptn],axis=1)
        ptn = tf.concat([tf.zeros((ptn.shape[0],1)),ptn],axis=1)
        # ptn will be a tensor where each i will be a 2,3 matrix containing Ca i-1, Ca i 
        return ptn
    @staticmethod
    def calc_angle_vector(prot):
        # Expects an array of shape (batch, L,1,3)
        ishape = prot.shape
        norm = tf.linalg.norm
        pi = np.pi
        prot = tf.image.extract_patches(prot,sizes=[1, 3, 1, 1],strides=[1, 1, 1, 1],rates=[1, 1, 1, 1], padding='VALID')
        prot = tf.reshape(prot,(prot.shape[0],-1,3,3))
        v1 = prot[:,:,1,:] - prot[:,:,0,:]
        u1 = v1 / tf.reshape(norm(v1,axis=-1),(v1.shape[0],v1.shape[1],1))
        v2 = prot[:,:,1,:] - prot[:,:,2,:]
        u2 = v2 / tf.reshape(norm(v2,axis=-1),(v1.shape[0],v1.shape[1],1))
        theta = tf.math.acos(tf.reduce_sum(tf.multiply(u1,u2),axis=2))#,0)#axes=[[0,2],[0,2]])
        theta = tf.concat((tf.zeros((ishape[0],2)),theta),axis=-1)*(180/np.pi)
        return theta

    @staticmethod
    def calc_dihedral_vector(p,to_deg=False):
        # Expects a array of shape (BatchSize,L,3,1)
        
        ishape = p.shape
        p = tf.image.extract_patches(p,sizes=[1, 4, 1, 1],strides=[1, 1, 1, 1],rates=[1, 1, 1, 1], padding='VALID')
        p= tf.reshape(p,(1,-1,4,3))
        p0 = tf.gather(p, 0, axis=2)
        p1 = tf.gather(p, 1, axis=2)
        p2 = tf.gather(p, 2, axis=2)
        p3 = tf.gather(p, 3, axis=2)

        b0 = -1.0 * (tf.subtract(p1, p0))
        b1 = tf.subtract(p2, p1)
        b2 = tf.subtract(p3, p2)

        b1 = tf.divide(b1, tf.norm(b1, axis=2, keepdims=True))
        #b1 = tf.where(tf.is_nan(b1), tf.ones_like(b1), b1)
        b1 = tf.where(tf.math.is_nan(b1), tf.ones_like(b1), b1)

        v = tf.subtract(b0, tf.einsum('bi,bij->bij', tf.einsum('bij,bij->bi', b0, b1), b1))
        w = tf.subtract(b2, tf.einsum('bi,bij->bij', tf.einsum('bij,bij->bi', b2, b1), b1))

        x = tf.reduce_sum( tf.multiply( v, w ), 2, keepdims=True )

        #y = tf.reduce_sum( tf.multiply( tf.cross(b1, v), w ), 2, keepdims=True )

        y = tf.reduce_sum( tf.multiply( tf.linalg.cross(b1, v), w ), 2, keepdims=True )
        if to_deg:
            return tf.concat((tf.zeros((ishape[0],ishape[1],1)),tf.atan2(y,x)),axis=1)/(np.pi/180)
        else:
            return tf.concat((tf.zeros((ishape[0],3,1)),tf.atan2(y,x)),axis=1)
