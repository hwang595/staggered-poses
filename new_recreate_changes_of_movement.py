from numpy import NaN, Inf, arange, isscalar, asarray, array
from usingRosBag_linear_version3 import RosBagParser
from scipy import interpolate
from scipy import signal
from tongsCenter import GetTongsTransform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.signal import argrelextrema
from scipy.signal import butter, lfilter, freqz, group_delay
from numpy import linalg as LA
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import Wrench
import kdl_parser_py.urdf as urdf
from dipy.tracking import metrics as tm
import sklearn.preprocessing as skp
import argparse
import numpy as np
import os
import math
import random

#parameter of the lowpass filter
ORDER = 2
FS = 1000       # sample rate, Hz
# desired cutoff frequency of the filter, Hz
LOWCUT = 6.666666667
HIGHCUT = 400.0

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cut_off_freq, fs, order=5):
    b, a = butter_lowpass(cut_off_freq, fs, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def peakdet(v, delta, x = None):
    maxtab = []
    mintab = []
    if x is None:
        x = arange(len(v))
    v = asarray(v)
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')
    if not isscalar(delta):
        sys.exit('Input argument delta must be a scalar')
    if delta <= 0:
        sys.exit('Input argument delta must be positive')
    mn, mx = Inf, -Inf
    mnpos, mxpos = NaN, NaN
    lookformax = True    
    for i in arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]
        if lookformax:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True
    return array(maxtab), array(mintab)

def dot(vector_1, vector_2):
	ret_val = 0
	assert len(vector_1) == len(vector_2)
	for i in range(len(vector_1)):
		ret_val += (vector_1[i] * vector_2[i])
	return ret_val

def preprocess_curve_signal(T, N, signal):
	ret_signal = np.array([])
	for i in range(len(signal) - 1):
		norm_vec_n = skp.normalize(N[i].reshape(1, -1))
		motion_trial = signal[i+1] - signal[i]
		n_direction = np.dot(motion_trial, norm_vec_n[0]) / LA.norm(motion_trial)
		ret_signal = np.append(ret_signal, abs(n_direction))
	return ret_signal

def signal_scale(signal): #scale the signal from its original range to 0-1
	mu = np.mean(signal)
	scale_val = []
	for i in range(len(signal)):
		signal[i] = signal[i] - mu
		scale_val.append(abs(signal[i]))
	to_scale = max(scale_val)
	for i in range(len(signal)):
		signal[i] /= to_scale
		signal[i] += 1
	for i in range(len(signal)):
		signal[i] /= 2
	return signal

def split_traj_for_plot(max_loc, traj_x, traj_y, traj_z):
	sub_traj_x = []
	sub_traj_y = []
	sub_traj_z = []
	sub_color = []
	former_item = 0
	for items in max_loc:
		sub_traj_x.append(traj_x[former_item:int(items[0])])
		sub_traj_y.append(traj_y[former_item:int(items[0])])
		sub_traj_z.append(traj_z[former_item:int(items[0])])
		former_item = int(items[0])

		rand_R_val = random.randint(0, 100) / float(100)
		rand_G_val = random.randint(0, 100) / float(100)
		rand_B_val = random.randint(0, 100) / float(100)
		sub_color.append((rand_R_val, rand_G_val, rand_B_val))
	sub_traj_x.append(traj_x[former_item:])
	sub_traj_y.append(traj_y[former_item:])
	sub_traj_z.append(traj_z[former_item:])
	rand_R_val = random.randint(0, 100) / float(100)
	rand_G_val = random.randint(0, 100) / float(100)
	rand_B_val = random.randint(0, 100) / float(100)
	sub_color.append((rand_R_val, rand_G_val, rand_B_val))
	return sub_traj_x, sub_traj_y, sub_traj_z, sub_color

def interpolate_from_key_frame(max_loc, traj, obj_time_line):
	#create a time list for key_frame
	ori_time_list = []
	key_frame = []
	key_frame_time = []
	ori_time_list.append(obj_time_line[0])
	key_frame.append(traj[0])
	key_frame_time.append(obj_time_line[0])
	for item in max_loc:
		ori_time_list.append(obj_time_line[int(item[0])])
		key_frame.append(traj[int(item[0])])
		key_frame_time.append(obj_time_line[int(item[0])])

	ori_time_list.append(obj_time_line[-1])
	key_frame_time.append(obj_time_line[-1])
	key_frame.append(traj[-1])
#	f = interpolate.interp1d(ori_time_list, key_frame, kind='cubic')
#	recreated_signal = f(obj_time_line)
	f = interpolate.PchipInterpolator(key_frame_time, key_frame)
	recreated_signal = f(obj_time_line)
	return recreated_signal

def unit_vector(data, axis=None, out=None):
    """Return ndarray normalized by length, i.e. Euclidean norm, along axis.

    >>> v0 = numpy.random.random(3)
    >>> v1 = unit_vector(v0)
    >>> numpy.allclose(v1, v0 / numpy.linalg.norm(v0))
    True
    >>> v0 = numpy.random.rand(5, 4, 3)
    >>> v1 = unit_vector(v0, axis=-1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=2)), 2)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = unit_vector(v0, axis=1)
    >>> v2 = v0 / numpy.expand_dims(numpy.sqrt(numpy.sum(v0*v0, axis=1)), 1)
    >>> numpy.allclose(v1, v2)
    True
    >>> v1 = numpy.empty((5, 4, 3))
    >>> unit_vector(v0, axis=1, out=v1)
    >>> numpy.allclose(v1, v2)
    True
    >>> list(unit_vector([]))
    []
    >>> list(unit_vector([1]))
    [1.0]

    """
    if out is None:
        data = np.array(data, dtype=np.float64, copy=True)
        if data.ndim == 1:
            data /= math.sqrt(np.dot(data, data))
            return data
    else:
        if out is not data:
            out[:] = np.array(data, copy=False)
        data = out
    length = np.atleast_1d(np.sum(data*data, axis))
    np.sqrt(length, length)
    if axis is not None:
        length = np.expand_dims(length, axis)
    data /= length
    if out is None:
        return data

def quaternion_slerp(quat0, quat1, fraction, spin=0, shortestpath=True):
    """Return spherical linear interpolation between two quaternions.

    >>> q0 = random_quaternion()
    >>> q1 = random_quaternion()
    >>> q = quaternion_slerp(q0, q1, 0)
    >>> numpy.allclose(q, q0)
    True
    >>> q = quaternion_slerp(q0, q1, 1, 1)
    >>> numpy.allclose(q, q1)
    True
    >>> q = quaternion_slerp(q0, q1, 0.5)
    >>> angle = math.acos(numpy.dot(q0, q))
    >>> numpy.allclose(2, math.acos(numpy.dot(q0, q1)) / angle) or \
        numpy.allclose(2, math.acos(-numpy.dot(q0, q1)) / angle)
    True

    """
    _EPS = np.finfo(float).eps * 4.0
    q0 = unit_vector(quat0[:4])
    q1 = unit_vector(quat1[:4])
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if abs(abs(d) - 1.0) < _EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        np.negative(q1, q1)
    angle = math.acos(d) + spin * math.pi
    if abs(angle) < _EPS:
        return q0
    isin = 1.0 / math.sin(angle)
    q0 *= math.sin((1.0 - fraction) * angle) * isin
    q1 *= math.sin(fraction * angle) * isin
    q0 += q1
    return q0

def quart_slerp_interpolate(quat_w, quat_x, quat_y, quat_z, key_time_stamp, global_time_stamp):
    ret_list = []
    temp_quart = []
    time_interval_obj = global_time_stamp
    iter_op_0 = zip(quat_w, quat_x, quat_y, quat_z)
    for it_0 in iter_op_0:
        temp_quart.append(it_0)
    range_counter = 0
    i_counter = 0
    for t_idx in range(len(time_interval_obj)):
        time_comp = round(key_time_stamp[range_counter], 3)
        last_time_comp = round(key_time_stamp[range_counter-1], 3)
        if time_interval_obj[t_idx] < time_comp:
            fraction = (float(time_interval_obj[t_idx]) - last_time_comp) / (time_comp - last_time_comp)
            ret_list.append(
            	quaternion_slerp(temp_quart[range_counter-1], temp_quart[range_counter], fraction))
        elif (time_interval_obj[t_idx] - time_comp) < 1e-8:   
            range_counter += 1
            if range_counter == len(temp_quart):
            	ret_list.append(temp_quart[-1])
            else:
                fraction = 0
                ret_list.append(
            	    quaternion_slerp(temp_quart[range_counter-1], temp_quart[range_counter], fraction))
        else:
            while time_interval_obj[t_idx] > time_comp:
                range_counter += 1
                time_comp = time_stamp[range_counter]
            ret_list.append(
            	quaternion_slerp(temp_quart[range_counter-1], temp_quart[range_counter], fraction))
    return ret_list

#define the staggered poses class for implementing the staggered poses method
class staggered_poses:
	def __init__(self, cut_off_freq=1.75):
		self.cut_off_freq = cut_off_freq
		self.resampleRate = 1000
		self.ros_bag_data = []
		self.time_stamp = []
		self.recreated_pos_x = []
		self.recreated_pos_y = []
		self.recreated_pos_z = []
		self.recreated_quat_w = []
		self.recreated_quat_x = []
		self.recreated_quat_y = []
		self.recreated_quat_z = []
		self.recreated_encoder = []
		self.max_loc = []
		self.center_traj_x = []
		self.center_traj_y = []
		self.center_traj_z = []
		self.staggered_poses_pos_x = []
		self.staggered_poses_pos_y = []
		self.staggered_poses_pos_z = []
		self.staggered_poses_quat_w = []
		self.staggered_poses_quat_x = []
		self.staggered_poses_quat_y = []
		self.staggered_poses_quat_z = []
		self.staggered_poses_time_stamp = []

	def load_data(self, ros_bag_file, resampleRate=1000):
		bag_data = RosBagParser(resampleRate)
		empty_flag_list = bag_data.parseTongsBag(ros_bag_file)
		self.ros_bag_data = bag_data
		self.time_stamp = self.ros_bag_data.resample_time_stamp
		self.resampleRate = resampleRate

	def search_staggered_poses_and_recreate(self):
		#extract position signals
		pos_x = np.array([a[0] for a in self.ros_bag_data.vivePos_interpolated])
		pos_y = np.array([a[1] for a in self.ros_bag_data.vivePos_interpolated])
		pos_z = np.array([a[2] for a in self.ros_bag_data.vivePos_interpolated])
		#extract quaternion signals
		quat_w = np.array([b[0] for b in self.ros_bag_data.viveQuat_interpolated])
		quat_x = np.array([b[1] for b in self.ros_bag_data.viveQuat_interpolated])
		quat_y = np.array([b[2] for b in self.ros_bag_data.viveQuat_interpolated])
		quat_z = np.array([b[3] for b in self.ros_bag_data.viveQuat_interpolated])
		#extract force sensor value 1:
		force1_x = [c[0] for c in self.ros_bag_data.forceSensor1array_interpolated]
		force1_y = [c[1] for c in self.ros_bag_data.forceSensor1array_interpolated]
		force1_z = [c[2] for c in self.ros_bag_data.forceSensor1array_interpolated]
		#extract force sensor value 2:
		force2_x = [d[0] for d in self.ros_bag_data.forceSensor2array_interpolated]
		force2_y = [d[1] for d in self.ros_bag_data.forceSensor2array_interpolated]
		force2_z = [d[2] for d in self.ros_bag_data.forceSensor2array_interpolated]

		force1_x_filtered = butter_lowpass_filter(
			force1_x, self.cut_off_freq, self.resampleRate, 6)
		force1_y_filtered = butter_lowpass_filter(
			force1_y, self.cut_off_freq, self.resampleRate, 6)
		force1_z_filtered = butter_lowpass_filter(
			force1_z, self.cut_off_freq, self.resampleRate, 6)
		force2_x_filtered = butter_lowpass_filter(
			force2_x, self.cut_off_freq, self.resampleRate, 6)
		force2_y_filtered = butter_lowpass_filter(
			force2_y, self.cut_off_freq, self.resampleRate, 6)
		force2_z_filtered = butter_lowpass_filter(
			force2_z, self.cut_off_freq, self.resampleRate, 6)
		#preprocess the quaternion signal
		encoder_signal = self.ros_bag_data.encoderarray_interpolated

		tongstfs = GetTongsTransform()
		viveFullPose = Pose()
		p = Point()
		p.x = 0
		p.y = 0
		p.z = 0
		center_traj_x = np.array([])
		center_traj_y = np.array([])
		center_traj_z = np.array([])
		for i in range(len(self.time_stamp)):
			cur_encoder_val = encoder_signal[i]
			viveFullPose.position.x = pos_x[i]
			viveFullPose.position.y = pos_y[i]
			viveFullPose.position.z = pos_z[i]
			viveFullPose.orientation.x = quat_x[i]
			viveFullPose.orientation.y = quat_y[i]
			viveFullPose.orientation.z = quat_z[i]
			viveFullPose.orientation.w = quat_w[i]
			tongstfs.getTransformsVive(cur_encoder_val, viveFullPose)
			center_traj_x = np.append(
				center_traj_x, tongstfs.centerTransform.p[0])
			center_traj_y = np.append(
				center_traj_y, tongstfs.centerTransform.p[1])
			center_traj_z = np.append(
				center_traj_z, tongstfs.centerTransform.p[2])

		self.center_traj_x = center_traj_x
		self.center_traj_y = center_traj_y
		self.center_traj_z = center_traj_z
		center_traj_xyz = np.vstack((center_traj_x, center_traj_y, center_traj_z)).T
		T, N, B, curvature, torsion = tm.frenet_serret(center_traj_xyz)

		kk = preprocess_curve_signal(T, N, center_traj_xyz)
		kk_filtered = butter_lowpass_filter(kk, self.cut_off_freq, self.resampleRate, 6)

		pos_x_std = signal_scale(pos_x)
		pos_y_std = signal_scale(pos_y)
		pos_z_std = signal_scale(pos_z)
		encoder_signal_std = signal_scale(encoder_signal)
		quat_signal_std = signal_scale(kk_filtered)
		force1_x_std = signal_scale(force1_x_filtered)
		force1_y_std = signal_scale(force1_y_filtered) 
		force1_z_std = signal_scale(force1_z_filtered)
		force2_x_std = signal_scale(force2_x_filtered)
		force2_y_std = signal_scale(force2_y_filtered)
		force2_z_std = signal_scale(force2_z_filtered)
		aggregated_pos = []
		quat_signal_std = np.append(quat_signal_std, 0)

		encoder_peak, _ = peakdet(encoder_signal_std, 0.005)
		force1_x_peak, _ = peakdet(force1_x_std, 0.005)
		force1_y_peak, _ = peakdet(force1_y_std, 0.005)
		force1_z_peak, _ = peakdet(force1_z_std, 0.005)
		force2_x_peak, _ = peakdet(force2_x_std, 0.005)
		force2_y_peak, _ = peakdet(force2_y_std, 0.005)
		force2_z_peak, _ = peakdet(force2_z_std, 0.005)
		quat_peak, _ = peakdet(quat_signal_std, 0.005)
		all_frame = [int(a[0]) for a in encoder_peak]
		all_frame.extend([int(a[0]) for a in force1_x_peak])
		all_frame.extend([int(a[0]) for a in force1_y_peak])
		all_frame.extend([int(a[0]) for a in force1_z_peak])
		all_frame.extend([int(a[0]) for a in force2_x_peak])
		all_frame.extend([int(a[0]) for a in force2_y_peak])
		all_frame.extend([int(a[0]) for a in force2_z_peak])
		all_frame.extend([int(a[0]) for a in quat_peak])
		all_frame = sorted(list(set(all_frame)))
		all_frame_for_calc = []
		for frame in all_frame:
			all_frame_for_calc.append([frame, 0])
		all_frame_for_calc = all_frame_for_calc[1:]

		ops = zip(quat_signal_std, encoder_signal_std,
				force1_x_std, force1_y_std, force1_z_std, 
				force2_x_std, force2_y_std, force2_z_std)
		for o_p in ops:
			aggregated_pos.append(sum(o_p))
		aggregated_pos = np.array(aggregated_pos)
		#lowpass filtering the aggregated signal
		filtered_signal = butter_lowpass_filter(aggregated_pos, LOWCUT, self.resampleRate, ORDER)
		#extract peak value of aggregated signal
		max_loc, min_loc = peakdet(filtered_signal, 0.05)
		self.max_loc = max_loc
		#======================try some interesting things====================================
		staggered_poses_pos_x = []
		staggered_poses_pos_y = []
		staggered_poses_pos_z = []
		staggered_poses_quat_w = []
		staggered_poses_quat_x = []
		staggered_poses_quat_y = []
		staggered_poses_quat_z = []
		key_time_stamp = []
		staggered_poses_pos_x.append(center_traj_x[0])
		staggered_poses_pos_y.append(center_traj_y[0])
		staggered_poses_pos_z.append(center_traj_z[0])
		staggered_poses_quat_w.append(quat_w[0])
		staggered_poses_quat_x.append(quat_x[0])
		staggered_poses_quat_y.append(quat_y[0])
		staggered_poses_quat_z.append(quat_z[0])
		key_time_stamp.append(self.time_stamp[0])		
		min_loc_list = []
		for loc in quat_peak:
			min_loc_list.append(int(loc[0]))
		for loc in max_loc:
			key_time_stamp.append(self.time_stamp[int(loc[0])])
			staggered_poses_pos_x.append(center_traj_x[int(loc[0])])
			staggered_poses_pos_y.append(center_traj_y[int(loc[0])])
			staggered_poses_pos_z.append(center_traj_z[int(loc[0])])
			staggered_poses_quat_w.append(quat_w[int(loc[0])])
			staggered_poses_quat_x.append(quat_x[int(loc[0])])
			staggered_poses_quat_y.append(quat_y[int(loc[0])])
			staggered_poses_quat_z.append(quat_z[int(loc[0])])
		key_time_stamp.append(self.time_stamp[-1])
		staggered_poses_pos_x.append(center_traj_x[-1])
		staggered_poses_pos_y.append(center_traj_y[-1])
		staggered_poses_pos_z.append(center_traj_z[-1])
		staggered_poses_quat_w.append(quat_w[-1])
		staggered_poses_quat_x.append(quat_x[-1])
		staggered_poses_quat_y.append(quat_y[-1])
		staggered_poses_quat_z.append(quat_z[-1])
		self.staggered_poses_pos_x = staggered_poses_pos_x
		self.staggered_poses_pos_y = staggered_poses_pos_y
		self.staggered_poses_pos_z = staggered_poses_pos_z
		self.staggered_poses_quat_w = staggered_poses_quat_w
		self.staggered_poses_quat_x = staggered_poses_quat_x
		self.staggered_poses_quat_y = staggered_poses_quat_y
		self.staggered_poses_quat_z = staggered_poses_quat_z
		self.staggered_poses_time_stamp = key_time_stamp
		#=====================================================================================
		#segmenting the whole trajectory using only key_frame value
		sub_traj_x, sub_traj_y, sub_traj_z, sub_color = split_traj_for_plot(
											max_loc, center_traj_x, center_traj_y, center_traj_z)
		#segmenting the whole trajectory using all key_frame value
		sub_all_x, sub_all_y, sub_all_z, sub_all_color = split_traj_for_plot(
								all_frame_for_calc, center_traj_x, center_traj_y, center_traj_z)

		obj_time_line = self.time_stamp
		recreated_pos_x = interpolate_from_key_frame(
			self.max_loc, self.center_traj_x, obj_time_line)
		recreated_pos_y = interpolate_from_key_frame(
			self.max_loc, self.center_traj_y, obj_time_line)
		recreated_pos_z = interpolate_from_key_frame(
			self.max_loc, self.center_traj_z, obj_time_line)
		recreated_encoder = interpolate_from_key_frame(
			self.max_loc, self.ros_bag_data.encoderarray_interpolated, obj_time_line)
		self.recreated_encoder = recreated_encoder
		#=======================================================================================
		sub_recreate_x, sub_recreate_y, sub_recreate_z, sub_recreate_color = split_traj_for_plot(
										self.max_loc, recreated_pos_x, recreated_pos_y, recreated_pos_z)
		self.recreated_pos_x = recreated_pos_x
		self.recreated_pos_y = recreated_pos_y
		self.recreated_pos_z = recreated_pos_z

		#recreate the 3-d trajectory through simple interpolation using all frame
		recreated_all_x = interpolate_from_key_frame(
			all_frame_for_calc, center_traj_x, obj_time_line)
		recreated_all_y = interpolate_from_key_frame(
			all_frame_for_calc, center_traj_y, obj_time_line)
		recreated_all_z = interpolate_from_key_frame(
			all_frame_for_calc, center_traj_z, obj_time_line)
		sub_recreate_all_x, sub_recreate_all_y, sub_recreate_all_z, sub_recreate_all_color = split_traj_for_plot(
										all_frame_for_calc, recreated_all_x, recreated_all_y, recreated_all_z)

		#recreate the rotations using only staggered poses
		key_frame_quat_w = []
		key_frame_quat_x = []
		key_frame_quat_y = []
		key_frame_quat_z = []
		key_frame_quat_w.append(quat_w[0])
		key_frame_quat_x.append(quat_x[0])
		key_frame_quat_y.append(quat_y[0])
		key_frame_quat_z.append(quat_z[0])
		for item in self.max_loc:
			key_frame_quat_w.append(quat_w[int(item[0])])
			key_frame_quat_x.append(quat_x[int(item[0])])
			key_frame_quat_y.append(quat_y[int(item[0])])
			key_frame_quat_z.append(quat_z[int(item[0])])
		key_frame_quat_w.append(quat_w[-1])
		key_frame_quat_x.append(quat_x[-1])
		key_frame_quat_y.append(quat_y[-1])
		key_frame_quat_z.append(quat_z[-1])

		recreated_quat = quart_slerp_interpolate(
			key_frame_quat_w, key_frame_quat_x, key_frame_quat_y, 
			key_frame_quat_z, key_time_stamp, obj_time_line)

		self.recreated_quat_w = [m[0] for m in recreated_quat]
		self.recreated_quat_x = [m[1] for m in recreated_quat]
		self.recreated_quat_y = [m[2] for m in recreated_quat]
		self.recreated_quat_z = [m[3] for m in recreated_quat] 

		key_frame_time_max = []
		key_frame_time_quat = []
		for item in max_loc:
			key_frame_time_max.append(self.time_stamp[int(item[0])])
		for item in quat_peak:
			key_frame_time_quat.append(self.time_stamp[int(item[0])])

		'''
		plt.figure(1)
	#	plt.subplot(211)
		plt.grid(True)
		plt.plot(self.time_stamp, filtered_signal, 'b--', linewidth=2)
		for item in key_frame_time_max:
			plt.axvline(item, 0, 1, linewidth=1, color='r')

		plt.figure(6)
		plt.grid(True)
	#	plt.plot(self.time_stamp[0:-1], kk, 'b--', linewidth=2)
		plt.plot(self.time_stamp[0:-1], kk, 'k--', linewidth=1)
		plt.plot(self.time_stamp[0:-1], kk_filtered, 'r-', linewidth=1)
		for item in key_frame_time_quat:
			plt.axvline(item, 0, 1, linewidth=2, color='g')
		plt.show()
		'''
		'''
		plt.subplot(212)
		plt.grid(True)
		plt.plot(self.time_stamp, encoder_signal_std, linewidth=2)
		plt.xlabel("aggregated signal")

		plt.figure(6)
		plt.grid(True)
		plt.plot(self.time_stamp, quat_signal_std,'r-', linewidth=2)
		'''

		fig = plt.figure(3)
		ax = fig.add_subplot(111, projection='3d')
		for i in range(len(sub_traj_x)):
			ax.plot(sub_traj_x[i], sub_traj_y[i], sub_traj_z[i], c=sub_color[i], linewidth=2)

		fig_2 = plt.figure(2)
		bx = fig_2.add_subplot(111, projection='3d')
		for j in range(len(sub_recreate_x)):
			bx.plot(sub_recreate_x[j], sub_recreate_y[j], sub_recreate_z[j], c=sub_color[j], linewidth=2)

		fig4 = plt.figure(4)
		cx = fig4.add_subplot(111, projection='3d')
		for i in range(len(sub_all_x)):
			cx.plot(sub_all_x[i], sub_all_y[i], sub_all_z[i], c=sub_all_color[i], linewidth=2)

		fig5 = plt.figure(5)
		dx = fig5.add_subplot(111, projection='3d')
		for i in range(len(sub_recreate_all_x)):
			dx.plot(sub_recreate_all_x[i], 
				sub_recreate_all_y[i], sub_recreate_all_z[i], c=sub_all_color[i], linewidth=2)
		plt.show()
		'''
		plt.figure(8)
		plt.subplot(411)
		plt.plot(self.time_stamp, quat_w, 'k-', linewidth=2)
		plt.plot(self.time_stamp, self.recreated_quat_w, 'k--', linewidth=1)
		plt.subplot(412)
		plt.plot(self.time_stamp, quat_x, 'r-', linewidth=2)
		plt.plot(self.time_stamp, self.recreated_quat_x, 'r--', linewidth=1)
		plt.subplot(413)
		plt.plot(self.time_stamp, quat_y, 'g-', linewidth=2)
		plt.plot(self.time_stamp, self.recreated_quat_y, 'g--', linewidth=1)	
		plt.subplot(414)
		plt.plot(self.time_stamp, quat_z, 'b-', linewidth=2)
		plt.plot(self.time_stamp, self.recreated_quat_z, 'b--', linewidth=1)
		'''
		'''
		plt.figure(7)
		plt.title("force sensor 1")
		plt.grid(True)
		plt.subplot(311)
		plt.plot(self.time_stamp, force1_x_std, 'r-', linewidth=2)
		plt.plot(self.time_stamp, force1_x, 'y--', linewidth=2)
		plt.subplot(312)
		plt.plot(self.time_stamp, force1_y_std, 'g-', linewidth=2)
		plt.plot(self.time_stamp, force1_y, 'y--', linewidth=2)
		plt.subplot(313)
		plt.plot(self.time_stamp, force1_z_std, 'b-', linewidth=2)
		plt.plot(self.time_stamp, force1_z, 'y--', linewidth=2)

		plt.figure(8)
		plt.title("force sensor 2")
		plt.grid(True)
		plt.subplot(311)
		plt.plot(self.time_stamp, force2_x_std, 'r-', linewidth=2)
		plt.plot(self.time_stamp, force2_x, 'y--', linewidth=2)
		plt.subplot(312)
		plt.plot(self.time_stamp, force2_y_std, 'g-', linewidth=2)
		plt.plot(self.time_stamp, force2_y, 'y--', linewidth=2)
		plt.subplot(313)
		plt.plot(self.time_stamp, force2_z_std, 'b-', linewidth=2)
		plt.plot(self.time_stamp, force2_z, 'y--', linewidth=2)

		plt.show()
		'''
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--filename', help='The file name/path of the bag.', required=True)
	args = vars(parser.parse_args())
	bagname = args["filename"]
	#define a staggered poses class and initialize it using ros bag data
	sp = staggered_poses(cut_off_freq=1.5)
	sp.load_data(ros_bag_file=bagname, resampleRate=1000)
	sp.search_staggered_poses_and_recreate()
