import kdl_parser_py.urdf as urdf
import PyKDL as kdl
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Wrench
import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D

# makes a point into a list [x,y,z]
def PointToList(pt):
    return [pt.x,pt.y,pt.z]

# get angle between two vectors
def absKDLVec(v):
    return (v[0]**2 + v[1]**2 + v[2]**2)**0.5

def unitKDLVec(v):
    return v/absKDLVec(v)


def angleKDLVec(v1,v2):
    axis = unitKDLVec(v1*v2)
    angle = np.arcsin(absKDLVec(v1*v2)/(absKDLVec(v1)*absKDLVec(v2)))
    return axis, angle

class GetTongsTransform:
    def __init__(self):
	dir = os.path.dirname(__file__)
        urdfFile = os.path.join(dir, "./withRootTongsAssembly.URDF")
        (ok, tree) = urdf.treeFromFile(urdfFile)
        self.tree = tree
        self.chain1 =  tree.getChain('base_link','upperForceSensor')
        self.fksolver1 = kdl.ChainFkSolverPos_recursive(self.chain1)
        self.chain2 = tree.getChain('base_link', 'lowerForceSensor')
        self.fksolver2 = kdl.ChainFkSolverPos_recursive(self.chain2)
        self.getTransforms(0)

    def getTransforms(self,angle):
        self.upperForceSensorTransform = kdl.Frame()
        jarr = kdl.JntArray(self.chain1.getNrOfJoints())
        self.fksolver1.JntToCart(jarr, self.upperForceSensorTransform, 2)
        jarr = kdl.JntArray(self.chain2.getNrOfJoints())
        jarr[0] = angle
        self.lowerForceSensorTransform = kdl.Frame()
        self.fksolver2.JntToCart(jarr, self.lowerForceSensorTransform, 3)
        # after some thought maybe we can use vive transform as the transform at the center
        self.pivotTransform = kdl.Frame()
        self.fksolver2.JntToCart(jarr, self.pivotTransform, 2)
        upperArmVec =  self.upperForceSensorTransform.p - self.pivotTransform.p
        lowerArmVec = self.lowerForceSensorTransform.p - self.pivotTransform.p

        axis, centerRotAngle = angleKDLVec(lowerArmVec,upperArmVec)

        self.centerTransform = kdl.Frame()
        self.centerTransform.p = (self.upperForceSensorTransform.p + self.lowerForceSensorTransform.p) / 2
        self.centerTransform.M =  self.upperForceSensorTransform.M

        self.centerTransform.M = self.centerTransform.M * kdl.Rotation.RotZ(-np.pi / 2)
        self.centerTransform.M = self.centerTransform.M*kdl.Rotation.RotY(-np.pi)
        self.centerTransform.M = self.centerTransform.M*kdl.Rotation.RotX(-centerRotAngle/2)

    def getTransformsVive(self,angle,viveFullPose):
        vivePos = kdl.Vector(viveFullPose.position.x,\
                             viveFullPose.position.y,\
                             viveFullPose.position.z)
        viveRot = kdl.Rotation.Quaternion(viveFullPose.orientation.x, \
                                 viveFullPose.orientation.y, \
                                 viveFullPose.orientation.z, \
                                 viveFullPose.orientation.w)
        viveFrame = kdl.Frame(viveRot, vivePos)
        self.getTransforms(angle)
        self.upperForceSensorTransform = viveFrame*self.upperForceSensorTransform
        self.lowerForceSensorTransform = viveFrame*self.lowerForceSensorTransform
        self.centerTransform = viveFrame*self.centerTransform
    '''
    def computeForceVectors(self,upperForceBar,lowerForceBar):
        # upperForceBar and lowerForceBar are the local force vectors wrt the force sensors
        if(type(upperForceBar)!=type(Point())):
            print "Force1 should be of Point type"
            return
        if (type(lowerForceBar) != type(Point())):
            print "Force2 should be of Point type"
            return
        uFB = kdl.Vector(upperForceBar.x,upperForceBar.y,upperForceBar.z)
        lFB = kdl.Vector(lowerForceBar.x, lowerForceBar.y, lowerForceBar.z)
        self.upperForce = self.upperForceSensorTransform.M*uFB
        self.lowerForce = self.lowerForceSensorTransform.M*lFB
        self.centerForce = self.upperForce + self.lowerForce
    '''

    def computeForceVectors(self,upperForceBar,lowerForceBar):
        # upperForceBar and lowerForceBar are the local force vectors wrt the force sensors
        '''
        if(type(upperForceBar)!=type(Point())):
            print "Force1 should be of Point type"
            return
        if (type(lowerForceBar) != type(Point())):
            print "Force2 should be of Point type"
            return
        '''
        uFB = kdl.Vector(upperForceBar[0],upperForceBar[1],upperForceBar[2])
        lFB = kdl.Vector(lowerForceBar[0], lowerForceBar[1], lowerForceBar[2])
        self.upperForce = self.upperForceSensorTransform.M*uFB
        self.lowerForce = self.lowerForceSensorTransform.M*lFB
        self.centerForce = self.upperForce + self.lowerForce

    def computeWrenchAboutCenter(self,upperForceBar,lowerForceBar):
        self.computeForceVectors(self, upperForceBar, lowerForceBar)
        Forces = self.centerForce
        #find moment arms:
        upperMomentArm = self.upperForceSensorTransform.p - self.centerTransform.p
        lowerMomentArm = self.lowerForceSensorTransform.p - self.centerTransform.p

        Moments = np.cross(PointToList(upperMomentArm),PointToList(self.upperForce))
        Moments = Moments + np.cross(PointToList(lowerMomentArm),PointToList(self.lowerForce))
        self.computedWrench = Wrench()
        self.computedWrench.force = Forces
        self.computedWrench.torque = Moments
        return self.computedWrench

    def getCenterPosition(self, pos, quat, angle):
        '''
        get the center position between the pads of the tongs
        :param pos: position as vector [x y z]
        :param quat: orientation as quaternion [w x y z]
        :param angle: Full angle of the tongs
        :return: center position [x y z]
        '''

        viveFullPose = Pose()
        viveFullPose.position.x = pos[0]
        viveFullPose.position.y = pos[1]
        viveFullPose.position.z = pos[2]
        viveFullPose.orientation.x = quat[1]
        viveFullPose.orientation.y = quat[2]
        viveFullPose.orientation.z = quat[3]
        viveFullPose.orientation.w = quat[0]

        self.getTransformsVive(angle, viveFullPose)
        pos = self.centerTransform.p
        return [pos[0],pos[1],pos[2]]

    def getUpperAndLowerPosition(self, pos, quat, angle):
        '''
        get the center position between the pads of the tongs
        :param pos: position as vector [x y z]
        :param quat: orientation as quaternion [w x y z]
        :param angle: Full angle of the tongs
        :return: center position [x y z]
        '''

        viveFullPose = Pose()
        viveFullPose.position.x = pos[0]
        viveFullPose.position.y = pos[1]
        viveFullPose.position.z = pos[2]
        viveFullPose.orientation.x = quat[1]
        viveFullPose.orientation.y = quat[2]
        viveFullPose.orientation.z = quat[3]
        viveFullPose.orientation.w = quat[0]

        self.getTransformsVive(angle, viveFullPose)
        lower = self.lowerForceSensorTransform.p
        upper = self.upperForceSensorTransform.p
        return [upper[0],upper[1],upper[2]], [lower[0],lower[1],lower[2]]

    def getTongsDistance(self, pos, quat, angle):
        '''
        returns the distance between the
        :param pos:
        :param quat:
        :param angle:
        :return:
        '''
        upper, lower = self.getUpperAndLowerPosition(pos, quat, angle)
        return np.linalg.norm(np.array(upper) - np.array(lower))


if __name__ == "__main__":
    #### Please use reference: http://www.orocos.org/kdl/usermanual/geometric-primitives
    tongstfs = GetTongsTransform()
    viveFullPose = Pose()
    viveFullPose.position.x = 0
    viveFullPose.position.y = 0
    viveFullPose.position.z = 0
    viveFullPose.orientation.x = 0
    viveFullPose.orientation.y = 0
    viveFullPose.orientation.z = 0
    viveFullPose.orientation.w = 1

    p = Point()
    p.x = 0
    p.y = 0
    p.z = 0

    tongstfs.getTransformsVive(6.5,viveFullPose)


    print tongstfs.upperForceSensorTransform.p
    print tongstfs.lowerForceSensorTransform.p
    print tongstfs.centerTransform.p
    print tongstfs.upperForceSensorTransform.M

    orientCenterStraight = tongstfs.centerTransform.p + tongstfs.centerTransform.M * kdl.Vector(0, 0.025, 0)
    orientCenterUp = tongstfs.centerTransform.p + tongstfs.centerTransform.M * kdl.Vector(0, 0, 0.025)

    fig = plt.figure()
    plt.plot(viveFullPose.position.y,viveFullPose.position.z)

    plt.plot([tongstfs.pivotTransform.p[1],tongstfs.upperForceSensorTransform.p[1]],\
             [tongstfs.pivotTransform.p[2],tongstfs.upperForceSensorTransform.p[2]])

    plt.plot([tongstfs.pivotTransform.p[1], tongstfs.lowerForceSensorTransform.p[1]], \
             [tongstfs.pivotTransform.p[2], tongstfs.lowerForceSensorTransform.p[2]])

    plt.plot([tongstfs.pivotTransform.p[1], tongstfs.centerTransform.p[1]], \
             [tongstfs.pivotTransform.p[2], tongstfs.centerTransform.p[2]])

    plt.plot([viveFullPose.position.y, tongstfs.pivotTransform.p[1]], \
             [viveFullPose.position.z, tongstfs.pivotTransform.p[2]])

    plt.plot([orientCenterStraight[1], tongstfs.centerTransform.p[1]], \
             [orientCenterStraight[2], tongstfs.centerTransform.p[2]])

    plt.plot([orientCenterUp[1], tongstfs.centerTransform.p[1]], \
             [orientCenterUp[2], tongstfs.centerTransform.p[2]])

    plt.axis('equal')
    plt.show()

