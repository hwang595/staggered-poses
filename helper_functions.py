from numpy import NaN, Inf, arange, isscalar, asarray, array
from usingRosBag import RosBagParser
from scipy import interpolate
from scipy import signal
from tongsCenter import GetTongsTransform
import matplotlib.pyplot as plt
import matplotlib as mpl
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
import csv
import os
import math
import random
import transformations
import pickle

def parse_dataTable(data_table):
    '''
    parse data matrix into 3D or 4D elements
    if data matrix is 4D based it will return elements like: vec_w, vec_x, vec_y, vec_z
    if data matrix is 3D based it will return elements like: vec_x, vec_y, vec_z
    parameter:
    data_table: data matrix that contain signals

    return:
    3D elements:
    vec_w, vec_x, vec_y, vec_z
    or
    4D elements:
    vec_x, vec_y, vec_z
    '''
    if len(data_table[0]) == 3:
        vec_x = [data_row[0] for data_row in data_table]
        vec_y = [data_row[1] for data_row in data_table]
        vec_z = [data_row[2] for data_row in data_table]
        return np.array(vec_x), np.array(vec_y), np.array(vec_z)
    elif len(data_table[0]) == 4:
        vec_w = [data_row[0] for data_row in data_table]
        vec_x = [data_row[1] for data_row in data_table]
        vec_y = [data_row[2] for data_row in data_table]
        vec_z = [data_row[3] for data_row in data_table]
        return np.array(vec_w), np.array(vec_x), np.array(vec_y), np.array(vec_z)

def force_filtering(forceTable, cut_off_freq, resampleRate, order):
    force_x, force_y, force_z = parse_dataTable(forceTable)
    force_x_filtered = butter_lowpass_filter(force_x, cut_off_freq, resampleRate, order)
    force_y_filtered = butter_lowpass_filter(force_y, cut_off_freq, resampleRate, order)
    force_z_filtered = butter_lowpass_filter(force_z, cut_off_freq, resampleRate, order)
    return force_x_filtered, force_y_filtered, force_z_filtered

def threeDArray_filtering(element_x, element_y, element_z, cut_off_freq, resampleRate, order):
    filetered_elem_x = butter_lowpass_filter(element_x, cut_off_freq, resampleRate, order)
    filetered_elem_y = butter_lowpass_filter(element_y, cut_off_freq, resampleRate, order)
    filetered_elem_z = butter_lowpass_filter(element_z, cut_off_freq, resampleRate, order)
    return filetered_elem_x, filetered_elem_y, filetered_elem_z

def take_force_deriv(force_x, force_y, force_z, cut_off_freq, resampleRate):
    deriv_force_x = calc_derivative(cut_off_freq, resampleRate, force_x)
    deriv_force_y = calc_derivative(cut_off_freq, resampleRate, force_y)
    deriv_force_z = calc_derivative(cut_off_freq, resampleRate, force_z)
    return deriv_force_x, deriv_force_y, deriv_force_z

def generate_quat_keyFrame(quat_table, max_loc):
    key_frame_quat_w = []
    key_frame_quat_x = []
    key_frame_quat_y = []
    key_frame_quat_z = []
    quat_w, quat_x, quat_y, quat_z = parse_dataTable(quat_table)
    key_frame_quat_w.append(quat_w[0])
    key_frame_quat_x.append(quat_x[0])
    key_frame_quat_y.append(quat_y[0])
    key_frame_quat_z.append(quat_z[0])
    for item in max_loc:
        key_frame_quat_w.append(quat_w[int(item[0])])
        key_frame_quat_x.append(quat_x[int(item[0])])
        key_frame_quat_y.append(quat_y[int(item[0])])
        key_frame_quat_z.append(quat_z[int(item[0])])
    key_frame_quat_w.append(quat_w[-1])
    key_frame_quat_x.append(quat_x[-1])
    key_frame_quat_y.append(quat_y[-1])
    key_frame_quat_z.append(quat_z[-1])
    return key_frame_quat_w, key_frame_quat_x, key_frame_quat_y, key_frame_quat_z

def write_to_csvFile(fileName, position, quaternion, sp, dataType):
    pos_x, pos_y, pos_z = parse_dataTable(position)
    quat_w, quat_x, quat_y, quat_z = parse_dataTable(quaternion)
    if dataType == "stag_pos":
        encoder_val_array =  sp._recreated_encoder
        time_stamp_array = sp._time_stamp
    elif dataType == "raw":
        encoder_val_array = sp._ros_bag_data.encoderarray_interpolated
        time_stamp_array = sp._ros_bag_data.resample_time_stamp
    with open(fileName, 'w') as out_file:
        csv_writer = csv.writer(out_file, dialect='excel')
        for i in range(len(sp._recreated_pos_x)):
            csv_writer.writerow([time_stamp_array[i], [pos_x[i], pos_y[i], pos_z[i]], 
                [quat_w[i], quat_x[i], quat_y[i], quat_z[i]], 
                encoder_val_array[i]])

def find_axis_angle(quaternion):
    '''
    find the axis angles: rx, ry. rz based on quaternion
    parameter:
    quaternion: in qw, qx, qy, qz format

    return:
    axis angle found, in rx, ry, rz format
    '''
    axis_angle = np.zeros(3)
    q_w = quaternion[0]
    q_x = quaternion[1]
    q_y = quaternion[2]
    q_z = quaternion[3]

    angle = 2 * math.acos(q_w)
    angle_x = float(q_x) / math.sqrt(1 - (q_w * q_w))
    angle_y = float(q_y) / math.sqrt(1 - (q_w * q_w))
    angle_z = float(q_z) / math.sqrt(1 - (q_w * q_w))
    axis_angle[0] = angle_x
    axis_angle[1] = angle_y
    axis_angle[2] = angle_z
    return axis_angle

def process_matrix_for_lib(matrix):
    ret_matrix = np.zeros((4, 4))
    for row_idx in range(len(matrix)):
        ret_matrix[row_idx] = np.append(matrix[row_idx], 0)
    ret_matrix[3] = np.array([0, 0, 0, 1])
    return ret_matrix 

def transform_to_robot_coor(pos, quaternion):
    pkl_file = open('M_vm_rev.pkl', 'rb')
    pkl_file2 = open('M_mr_mod.pkl', 'rb')
    ret_pos = np.zeros((len(pos), 3))
    ret_axis_angle = np.zeros((len(pos), 3))
    M_vm = np.zeros((3, 3))
    M_mr = np.zeros((3, 3))
    p_vm = np.array([])
    p_mr = np.array([])
    #split the rotation matrix to M and p
    T_vm = pickle.load(pkl_file)
    for r_idx in range(len(T_vm)-1):
        row = T_vm[r_idx]
        M_vm[r_idx] = row[0:3]
        p_vm = np.append(p_vm, row[3])

    T_mr = pickle.load(pkl_file2)
    for r_idx in range(len(T_mr)-1):
        row = T_mr[r_idx]
        M_mr[r_idx] = row[0:3]
        p_mr = np.append(p_mr, row[3])

    M_vm_nor = M_vm
    M_mr_nor = M_mr
    #do transformation for positions
    for p_idx in range(len(pos)):
        points = pos[p_idx]
        tmp_1 = np.dot(M_vm_nor, points)
        tmp_2 = np.dot(M_mr_nor, tmp_1)
        tmp_3 = tmp_2 + p_vm
        tmp_4 = tmp_3 + p_mr
        ret_pos[p_idx] = tmp_4
    
    for q_idx in range(len(quaternion)):
        q_for_tran = quaternion[q_idx]
        t_for_tran = transformations.quaternion_matrix_rev(q_for_tran)
        tmp_M_1 = np.dot(M_vm_nor, t_for_tran)
        tmp_M_2 = np.dot(M_mr_nor, tmp_M_1)
        tmp_M_3 = process_matrix_for_lib(tmp_M_2)
        transformed_q =  transformations.quaternion_from_matrix(tmp_M_2)
        axis_angle = fine_axis_angle(transformed_q)
        ret_axis_angle[q_idx] = axis_angle
    return ret_pos, ret_axis_angle

def transform_to_robot_direct(pos, quaternion):
    pkl_file = open('pkl_files/moCap_robot_calib.pkl', 'rb')
    ret_pos = np.zeros((len(pos), 3))
    ret_quat = np.zeros((len(quaternion), 4))
#    ret_axis_angle = np.zeros((len(pos), 3))
    MVR = np.zeros((3, 3))
    PMR = np.array([])
    MVR_226 = pickle.load(pkl_file)
    pos = np.transpose(pos)
    CC = np.dot(MVR_226, pos)
    CC = np.transpose(CC)
    CC_x = [a[0] for a in CC]
    CC_y = [a[1] for a in CC]
    CC_z = [a[2] for a in CC]
    ret_pos = np.vstack((CC_x, CC_y, CC_z)).T

    for r_idx in range(len(MVR_226)-1):
        row = MVR_226[r_idx]
        MVR[r_idx] = row[0:3]
        PMR = np.append(PMR, row[3])

    for q_idx in range(len(quaternion)):
        q_for_tran = quaternion[q_idx]
        t_for_tran = transformations.quaternion_matrix_rev(q_for_tran)
        tmp_M_1 = np.dot(MVR, t_for_tran)
        transformed_q =  transformations.quaternion_from_matrix(tmp_M_1)
        ret_quat[q_idx] = transformed_q
    return ret_pos, ret_quat

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
    counter = 0
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

    if len(max_loc)!=0:
        if max_loc[0][0] != 0:
            ori_time_list.append(obj_time_line[0])
            key_frame.append(traj[0])
            key_frame_time.append(obj_time_line[0])
    else:
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
                time_comp = global_time_stamp[range_counter]
            ret_list.append(
            	quaternion_slerp(temp_quart[range_counter-1], temp_quart[range_counter], fraction))
    return ret_list

def calc_derivative(cut_off_freq, resampleRate, signal):
    """
    calculate the derivate of a certain channel of signal
    after computing the derivative, low-pass filter the signal
    parameters:
    cut_off_freq: cut-off frequency of the low pass filter
    resampleRate: resample rate of low-pass filter
    signal: the signal that is going to be take derivative
    return:
    derivative value of signal (which have same length of the original signal)
    """
    signal_deriv = np.gradient(signal)
    signal_deriv = np.absolute(signal_deriv)
    signal_deriv_filtered = butter_lowpass_filter(
    	signal_deriv, cut_off_freq, resampleRate, 6)
    signal_deriv_filtered = signal_scale(signal_deriv_filtered)
    return signal_deriv_filtered

def switch_function(signal_table, flag_list):
    '''
    The signal table is like this:
    quat_signal_std, encoder_deriv_filtered,
    force1_x_deriv, force1_y_deriv, force1_z_deriv, 
    force2_x_deriv, force2_y_deriv, force2_z_deriv, 
    mag_force1_std, mag_force2_std,
    tran_force1_z_std, tran_force2_z_std
    ==============================================
    The flag list is like this:
    QUAT,
    ENCODER,
    FORCE1_GLOBAL, 
    FORCE2_GLOBAL,
    FORCE1_MAGNITUDE,
    FORCE2_MAGNITUDE,
    TRANSFORMED_FORCE_Z
    '''
    ret_list = []
    if flag_list[0]:
        ret_list.append(signal_table[0])
    if flag_list[1]:
        ret_list.append(signal_table[1])
    if flag_list[2]:
        for i in range(2, 5):
            ret_list.append(signal_table[i])
    if flag_list[3]:
        for i in range(5, 8):
            ret_list.append(signal_table[i])
    if flag_list[4]:
        ret_list.append(signal_table[8])
    if flag_list[5]:
        ret_list.append(signal_table[9])
    if flag_list[6]:
        for i in range(10, 12):
            ret_list.append(signal_table[i])
    if flag_list[7]:
        for i in range(12, 14):
            ret_list.append(signal_table[i])
    return ret_list