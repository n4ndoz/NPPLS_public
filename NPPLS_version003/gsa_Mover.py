# Implementação dos termos do GSA
import numpy as np
from math import gamma
import pyrosetta
from pyrosetta import Pose
class GSAMover(pyrosetta.rosetta.protocols.moves.Mover):
    """

    GSA Mover:
    This mover implements the Generalized Simulated Annealing, proposed by
    Tsalis et al. (1996). This implementation uses the generalizations of
    Pascutti et al. 2010, reducing the perturbation function to a sampling
    of a distribution given by a random number (or array) r_i and the temperature
    of the iteration. We implemented, at first, three types of minimization strategies,
    listed as follows:

    GSA types:
    1: x(t+1) = x + g(r), being x a L,2 dimensional vector
    of the phi and psi dihedrals. In this scenario when
    self.modifyPose() is called, an array dih with all psi/psi
    is created and the perturbation self.g() is applied to both
    angles of all residue, by a single value from the
    distribution g(r). r_i is int or shape (L,1)

    2: x(t+1) = x_i + g(r_i)*direction_mask, a modification of the second method
    but with addition of a random direction mask, that sets the direction of the angle
    perturbation. r_i shape is (L,1), (L,2)

    3: x(t+1) = x + g(r_i)*direction_mask, same as method 3, but the direction mask is retained
    in case the change in energy is positive. r_i shape is (L,1), (L,2)

    Modifier Types:
    1: g(r) is not modified by a direction mask.
    2: modifies g(r) by a random direction mask
    3: same as 2, but the mask is kept if the minimization was efficient
    (E(t+1) < E(t)).
    4: perturbates only a single residue, chosen at random.
    
    """
    def __init__(self, **kwargs):
        pyrosetta.rosetta.protocols.moves.Mover.__init__(self)
        self.max_iter = kwargs.pop('max_iter', 1000)
        self.gsa_type = kwargs.pop('gsa_type',1) # 3 types of GSA
        self.qv = kwargs.pop('qv', 1.1)
        self.qt = kwargs.pop('qt', 1.1)
        self.qa = kwargs.pop('qa', 1.1)
        self.tqt1 = kwargs.pop('tqt1', 1.1)
        self.epoch = 0
        modify_methods = {1: self.modify_type1, 2: self.modify_type2,
                          3: self.modify_type3, 4: self.modify_type4}

        assert type(self.r_i) == float and self.gsa_type not in [2,3],
        "r_i must be np.ndarray or Nonefor GSA types 2,3"

        self.modifyPose = modify_methods[self.gsa_type]
        
        # get the score, if no score given, raise exception (or let PyRosetta handle)
        
        
    def __term1(self):
        t0 = (self.qv-1)/np.pi
        d0 = (1-(1/2*(self.qv-1)))/(self.qv-1)
        n0 = (1/(self.qv-1))-0.5
        return t0*(gamma(d0)/2)/gamma(n0)    

    def __term2(self, r_i):
        self.tqt = self.temperature(self.epoch+1)
        d0 = self.tqt**(-1*(1/(3-self.qv)))
        n0 = ((r_i)**2)/(self.tqt**(2/(3-self.qv)))
        n0 = 1 + (self.qv-1)* (n0**((1/(self.qv-1))-0.5))
        return d0/n0

    def temperature(self, epoch):
        d0 = (2**(self.qt-1))-1
        n0 = ((1+epoch)**(self.qt-1))-1
        return self.tqt1*(d0/n0)

    def g(self, r_i):
        return self.__term2(r_i)*self.__term1()

    def Pqa(self, p1, p2):
        delta_E = self.score(p2) - self.score(p1)
        return [1 + (self.qa-1)*(delta_E)/self.temperature()] ** (1/(self.qa-1))
    
    def decisionCriteria(self,p1, p2):
        '''
        Decision criteria as applied by Pascutti et al, 2010.
        '''
        if self.score(p2) < self.score(p1):
            return p2
        else:
            if np.random.uniform(0,1) > self.Pqa():
                return p1
            else:
                return p2
            
    def __str__(self):
        return '''* GSA TERMS *\n q_v: {}\n q_t: {}\n q_a: {}\n max_iter: {}\n GSA type: {}
                 '''.format(self.qv, self.qt, self.qa, self.max_iter, self.gsa_type)

    def fetch_dihedrals(self, pose):
        dih = np.empty((len(pose.sequence()),2))
        for i in range(len(seq)):
            dih[i,] = np.asarray([pose.phi(i+1),pose.psi(i+1)])
        return dih
    
    def modify_type1(self, pose):
        dihs = self.fetch_dihedrals(pose)
        # x + g(r)
        dihs = np.add(dihs, np.degrees(self.g(r_i)))
        for i,dih in enumerate(dihs[1:]):
            pose.set_phi(i+2,dih[0])
            pose.set_psi(i+2,dih[1])
        return pose


    def modify_type2(self, pose):
        dihs = self.fetch_dihedrals(pose)
        # x + g(r)        
        self.direction_mask = np.random.randint(0,1,dihs.shape))
        self.direction_mask[np.where(self.direction_mask == 0)] = -1
        dihs = np.add(dihs, np.degrees(self.g(r_i))*self.direction_mask)
        for i,dih in enumerate(dihs[1:]):
            pose.set_phi(i+2,dih[0])
            pose.set_psi(i+2,dih[1])
        
        return pose

    def modify_type3(self, pose):
        dihs = self.fetch_dihedrals(pose)
        # x + g(r)
        if self.epoch == 0:
            self.direction_mask = np.random.randint(0,1,dihs.shape))
            self.direction_mask[np.where(self.direction_mask == 0)] = -1
        dihs = np.add(dihs, np.degrees(self.g(r_i))*self.direction_mask)
        for i,dih in enumerate(dihs[1:]):
            pose.set_phi(i+2,dih[0])
            pose.set_psi(i+2,dih[1])
            
        if self.score(pose) > self.score(self.previous_pose):
            self.direction_mask = np.random.randint(0,1,dihs.shape))
            self.direction_mask[np.where(self.direction_mask == 0)] = -1
        return pose
    
    def modify_type4(self, pose):
        index = random.randint(2, your_pose.total_residue())
        if type(self.r_i) == np.ndarray:
            gri = np.degrees(self.g(self.r_i[index]))
        else: gri = np.degrees(self.g(self.r_i))
        newPhi = pose.phi(index) + gri
        newPsi = pose.psi(index) + gri
        pose.set_phi(index,newPhi) 
        pose.set_psi(index,newPsi)
        return pose

    
    def get_name(self):
        """Return name of class."""
        return self.__class__.__name__

    def apply(self, pose):
        """Applies move to pose."""
        self.best_pose = Pose()
        self.previous_pose = Pose()
        pose = pose.get()
        for self.epoch in range(self.max_iter):
            if self.epoch == 0:
                self.best_pose.assign(pose)
                self.previous_pose.assign(pose)
            self.pose_zero = Pose()
            self.pose_zero.assign(pose)

            self.pose_modified = Pose()
            self.pose_modified.assign(self.modifyPose())
            
            pose.assign(self.decisionCriteria(self.pose_modified,
                                              self.pose_zero))

            if self.score(pose) < self.score(self.best_pose):
                self.best_pose.assign(pose)
                
            self.previous_pose.assign(pose)


