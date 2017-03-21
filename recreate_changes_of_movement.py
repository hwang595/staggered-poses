from helper_functions import *
import writeScript
import csv

#parameter of the lowpass filter
ORDER = 2
FS = 1000       # sample rate, Hz
# desired cutoff frequency of the filter, Hz
LOWCUT = 6.666666667
HIGHCUT = 400.0

QUAT = True
ENCODER = True
FORCE1_GLOBAL = True 
FORCE2_GLOBAL = True
FORCE1_MAGNITUDE = True
FORCE2_MAGNITUDE = True
TRANSFORMED_FORCE_Z = False
MERGED_FORCE = True
SCRIPT_NAME = "demo_to_show2"
mpl.rc('font', family='serif', serif='Times New Roman')

#define the staggered poses class for implementing the staggered poses method
class staggered_poses:
	def __init__(self, cut_off_freq=1.75):
		self._cut_off_freq = cut_off_freq
		self._resampleRate = 1000
		self._time_stamp = []
		self._recreated_pos_x = []
		self._recreated_pos_y = []
		self._recreated_pos_z = []
		self._recreated_quat_w = []
		self._recreated_quat_x = []
		self._recreated_quat_y = []
		self._recreated_quat_z = []
		self._recreated_encoder = []
		self._max_loc = []
		self._center_traj_x = []
		self._center_traj_y = []
		self._center_traj_z = []
		self._staggered_poses_pos_x = []
		self._staggered_poses_pos_y = []
		self._staggered_poses_pos_z = []
		self._staggered_poses_quat_w = []
		self._staggered_poses_quat_x = []
		self._staggered_poses_quat_y = []
		self._staggered_poses_quat_z = []
		self._staggered_poses_time_stamp = []

	def load_data(self, ros_bag_file, resampleRate=1000):
		bag_data = RosBagParser(resampleRate)
		empty_flag_list, encoderarray = bag_data.parseTongsBag(ros_bag_file)
		self._ros_bag_data = bag_data
		self._time_stamp = self._ros_bag_data.resample_time_stamp
		self._resampleRate = resampleRate

	def search_staggered_poses_and_recreate(self):
		merged_force1 = []
		merged_force2 = []
		aggregated_pos = []
		transformed_force1_x = []
		transformed_force1_y = []
		transformed_force1_z = []
		transformed_force2_x = []
		transformed_force2_y = []
		transformed_force2_z = []
		#extract position signals
		pos_x, pos_y, pos_z = parse_dataTable(self._ros_bag_data.vivePos_interpolated)
		#extract quaternion signals
		quat_w, quat_x, quat_y, quat_z = parse_dataTable(self._ros_bag_data.viveQuat_interpolated)
		#extract force sensor value 1:
		force1_x, force1_y, force1_z = parse_dataTable(self._ros_bag_data.forceSensor1array_interpolated)
		#extract force sensor value 2:
		force2_x, force2_y, force2_z = parse_dataTable(self._ros_bag_data.forceSensor2array_interpolated)

		force1_x_filtered = butter_lowpass_filter(
			force1_x, 5, self._resampleRate, 6)
		force1_y_filtered = butter_lowpass_filter(
			force1_y, 5, self._resampleRate, 6)
		force1_z_filtered = butter_lowpass_filter(
			force1_z, 5, self._resampleRate, 6)
		force2_x_filtered = butter_lowpass_filter(
			force2_x, 5, self._resampleRate, 6)
		force2_y_filtered = butter_lowpass_filter(
			force2_y, 5, self._resampleRate, 6)
		force2_z_filtered = butter_lowpass_filter(
			force2_z, 5, self._resampleRate, 6)
		encoder_signal = self._ros_bag_data.encoderarray_interpolated
		#preprocess the quaternion signal
		encoder_deriv_filtered = calc_derivative(self._cut_off_freq, self._resampleRate, encoder_signal)
		#preprocess the force signal
		force1_x_deriv = calc_derivative(self._cut_off_freq, self._resampleRate, force1_x_filtered)
		force1_y_deriv = calc_derivative(self._cut_off_freq, self._resampleRate, force1_y_filtered)
		force1_z_deriv = calc_derivative(self._cut_off_freq, self._resampleRate, force1_z_filtered)
		force2_x_deriv = calc_derivative(self._cut_off_freq, self._resampleRate, force2_x_filtered)
		force2_y_deriv = calc_derivative(self._cut_off_freq, self._resampleRate, force2_y_filtered)
		force2_z_deriv = calc_derivative(self._cut_off_freq, self._resampleRate, force2_z_filtered)

		tongstfs = GetTongsTransform()
		viveFullPose = Pose()
		p = Point()
		p.x = 0
		p.y = 0
		p.z = 0
		center_traj_x = np.array([])
		center_traj_y = np.array([])
		center_traj_z = np.array([])
		for i in range(len(self._time_stamp)):
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

		center_traj_x_filtered = butter_lowpass_filter(
					center_traj_x, 3.33, self._resampleRate, 6)
		center_traj_y_filtered = butter_lowpass_filter(
					center_traj_y, 3.33, self._resampleRate, 6)
		center_traj_z_filtered = butter_lowpass_filter(
					center_traj_z, 3.33, self._resampleRate, 6)

		fil_freq0 = 0.7
		fil_freq1 = 2.5
		center_traj_x_filtered0 = butter_lowpass_filter(
					center_traj_x, fil_freq0, self._resampleRate, 6)
		center_traj_y_filtered0 = butter_lowpass_filter(
					center_traj_y, fil_freq0, self._resampleRate, 6)
		center_traj_z_filtered0 = butter_lowpass_filter(
					center_traj_z, fil_freq0, self._resampleRate, 6)

		center_traj_x_filtered1 = butter_lowpass_filter(
					center_traj_x, fil_freq1, self._resampleRate, 6)
		center_traj_y_filtered1 = butter_lowpass_filter(
					center_traj_y, fil_freq1, self._resampleRate, 6)
		center_traj_z_filtered1 = butter_lowpass_filter(
					center_traj_z, fil_freq1, self._resampleRate, 6)

		self._center_traj_x = center_traj_x
		self._center_traj_y = center_traj_y
		self._center_traj_z = center_traj_z
		#Transforming the coordination of the force vector according to the Tongs
		for i in range(len(force1_x_filtered)):
			temp_upper_force = [force1_x_filtered[i], force1_y_filtered[i], force1_z_filtered[i]]
			temp_lower_force = [force2_x_filtered[i], force2_y_filtered[i], force2_z_filtered[i]]
			tongstfs.computeForceVectors(temp_upper_force, temp_lower_force)
			transformed_force1_x.append(tongstfs.upperForce[0])
			transformed_force1_y.append(tongstfs.upperForce[1])
			transformed_force1_z.append(tongstfs.upperForce[2])
			transformed_force2_x.append(tongstfs.lowerForce[0])
			transformed_force2_y.append(tongstfs.lowerForce[1])
			transformed_force2_z.append(tongstfs.lowerForce[2])

		#standardize transformed for force sensor1 value:
		tran_force1_x_std  = signal_scale(transformed_force1_x)
		tran_force1_y_std  = signal_scale(transformed_force1_y)
		tran_force1_z_std  = signal_scale(transformed_force1_z)
		tran_force2_x_std  = signal_scale(transformed_force2_x)
		tran_force2_y_std  = signal_scale(transformed_force2_y)
		tran_force2_z_std  = signal_scale(transformed_force2_z)
		#use the magnitude of the force
		mag_force1 = []
		mag_force2 = []
		ops_force1 = zip(force1_x_deriv, force1_y_deriv, force1_z_deriv)
		ops_force2 = zip(force2_x_deriv, force2_y_deriv, force2_z_deriv)
		for op_f1 in ops_force1:
			mag_force1.append(np.linalg.norm(op_f1))
		for op_f2 in ops_force2:
			mag_force2.append(np.linalg.norm(op_f2))
		mag_force1_std = signal_scale(mag_force1)
		mag_force2_std = signal_scale(mag_force2)

		#merge force sensor value together in x and y directions
		merge_force1_ops = zip(force1_x_filtered, force1_y_filtered)
		merge_force2_ops = zip(force2_x_filtered, force2_y_filtered)
		for m_f1_op in merge_force1_ops:
			merged_force1.append(sum(m_f1_op))
		for m_f2_op in merge_force2_ops:
			merged_force2.append(sum(m_f2_op))

		merged_force1_std = signal_scale(merged_force1)
		merged_force2_std = signal_scale(merged_force2)

		center_traj_xyz = np.vstack((center_traj_x, center_traj_y, center_traj_z)).T
		
		center_traj_xyz_fil = np.vstack(
			(center_traj_x_filtered, center_traj_y_filtered, center_traj_z_filtered)).T

		T, N, B, curvature, torsion = tm.frenet_serret(center_traj_xyz)

		T_f, N_f, B_f, _, _ = tm.frenet_serret(center_traj_xyz_fil)
		T_x, T_y, T_z = parse_dataTable(T)

		kk_f = preprocess_curve_signal(T_f, N_f, center_traj_xyz_fil)
		kk_ff = butter_lowpass_filter(kk_f, self._cut_off_freq, self._resampleRate, 6)

		kk = preprocess_curve_signal(T, N, center_traj_xyz)
		kk_filtered = butter_lowpass_filter(kk, self._cut_off_freq, self._resampleRate, 6)

		encoder_signal_std = signal_scale(encoder_signal)
		quat_signal_std = signal_scale(kk_filtered)
		force1_x_std = signal_scale(force1_x_filtered)
		force1_y_std = signal_scale(force1_y_filtered) 
		force1_z_std = signal_scale(force1_z_filtered)
		force2_x_std = signal_scale(force2_x_filtered)
		force2_y_std = signal_scale(force2_y_filtered)
		force2_z_std = signal_scale(force2_z_filtered)		
		quat_signal_std = np.append(quat_signal_std, 0)

		encoder_peak, aa = peakdet(encoder_deriv_filtered, 0.1)
		#calculate the transformed force value peak
		merged_force1_peak, _ = peakdet(merged_force1, 0.05)
		merged_force2_peak, _ = peakdet(merged_force2, 0.05)
		force1_z_peak, _ = peakdet(tran_force1_z_std, 0.05)
		force2_z_peak, _ = peakdet(tran_force2_z_std, 0.05)

		quat_peak, _ = peakdet(quat_signal_std, 0.1)
		all_frame = [int(a[0]) for a in encoder_peak]
		all_frame.extend([int(a[0]) for a in quat_peak])
		all_frame = sorted(list(set(all_frame)))
		all_frame_for_calc = []
		for frame in all_frame:
			all_frame_for_calc.append([frame, 0])
		all_frame_for_calc = all_frame_for_calc[1:]
		'''
		ops = zip(quat_signal_std, encoder_deriv_filtered,
				merged_force1, tran_force1_z_std, 
				merged_force2, tran_force2_z_std)
		'''
		signal_table = [
				quat_signal_std, encoder_deriv_filtered,
				force1_x_deriv, force1_y_deriv, force1_z_deriv, 
				force2_x_deriv, force2_y_deriv, force2_z_deriv, 
				mag_force1_std, mag_force2_std,
				tran_force1_z_std, tran_force2_z_std,
				merged_force1_std, merged_force2_std 
				]
		flag_list = [
						QUAT,
						ENCODER,
						FORCE1_GLOBAL, 
						FORCE2_GLOBAL,
						FORCE1_MAGNITUDE,
						FORCE2_MAGNITUDE,
						TRANSFORMED_FORCE_Z,
						MERGED_FORCE
					]

		#ops = switch_function(signal_table, flag_list)
		
		ops = zip(
				quat_signal_std, encoder_deriv_filtered,
				force1_x_deriv, force1_y_deriv, force1_z_deriv, 
				force2_x_deriv, force2_y_deriv, force2_z_deriv, 
				mag_force1_std, mag_force2_std,
				merged_force1_std, merged_force2_std 
				) #tran_force1_z_std, tran_force2_z_std
		#quat_signal_std, encoder_deriv_filtered,encoder_deriv_filtered, 
		'''
		force1_x_deriv, force1_y_deriv, force1_z_deriv, 
		force2_x_deriv, force2_y_deriv, force2_z_deriv, 
		mag_force1_std, mag_force2_std,
		merged_force1_std, merged_force2_std
		'''
		for o_p in ops:
			aggregated_pos.append(sum(o_p))
		'''
		for idx in range(len(ops[0])):
			tmp = []
			for element in ops:
				tmp.append(element[idx])
			aggregated_pos.append(sum(tmp))
		'''
		aggregated_pos = np.array(aggregated_pos)
		#lowpass filtering the aggregated signal
		filtered_signal = butter_lowpass_filter(aggregated_pos, LOWCUT, self._resampleRate, ORDER)
		#extract peak value of aggregated signal
		max_loc, min_loc = peakdet(filtered_signal, 0.1)
		self._max_loc = max_loc
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
		key_time_stamp.append(self._time_stamp[0])		
		min_loc_list = []
		for loc in quat_peak:
			min_loc_list.append(int(loc[0]))
		for loc in max_loc:
			key_time_stamp.append(self._time_stamp[int(loc[0])])
			staggered_poses_pos_x.append(center_traj_x[int(loc[0])])
			staggered_poses_pos_y.append(center_traj_y[int(loc[0])])
			staggered_poses_pos_z.append(center_traj_z[int(loc[0])])
			staggered_poses_quat_w.append(quat_w[int(loc[0])])
			staggered_poses_quat_x.append(quat_x[int(loc[0])])
			staggered_poses_quat_y.append(quat_y[int(loc[0])])
			staggered_poses_quat_z.append(quat_z[int(loc[0])])
		key_time_stamp.append(self._time_stamp[-1])
		staggered_poses_pos_x.append(center_traj_x[-1])
		staggered_poses_pos_y.append(center_traj_y[-1])
		staggered_poses_pos_z.append(center_traj_z[-1])
		staggered_poses_quat_w.append(quat_w[-1])
		staggered_poses_quat_x.append(quat_x[-1])
		staggered_poses_quat_y.append(quat_y[-1])
		staggered_poses_quat_z.append(quat_z[-1])
		self._staggered_poses_pos_x = staggered_poses_pos_x
		self._staggered_poses_pos_y = staggered_poses_pos_y
		self._staggered_poses_pos_z = staggered_poses_pos_z
		self._staggered_poses_quat_w = staggered_poses_quat_w
		self._staggered_poses_quat_x = staggered_poses_quat_x
		self._staggered_poses_quat_y = staggered_poses_quat_y
		self._staggered_poses_quat_z = staggered_poses_quat_z
		self._staggered_poses_time_stamp = key_time_stamp
		#=====================================================================================
		threeD_arrow = []
		for item in max_loc:
			threeD_arrow.append(T[int(item[0])])
		#segmenting the whole trajectory using only key_frame value
		sub_traj_x, sub_traj_y, sub_traj_z, sub_color = split_traj_for_plot(
											max_loc, center_traj_x, center_traj_y, center_traj_z)
		
		obj_time_line = self._time_stamp
		recreated_pos_x = interpolate_from_key_frame(
			self._max_loc, self._center_traj_x, obj_time_line)
		recreated_pos_y = interpolate_from_key_frame(
			self._max_loc, self._center_traj_y, obj_time_line)
		recreated_pos_z = interpolate_from_key_frame(
			self._max_loc, self._center_traj_z, obj_time_line)
		recreated_encoder = interpolate_from_key_frame(
			self._max_loc, self._ros_bag_data.encoderarray_interpolated, obj_time_line)
		#=======================================================================================
		sub_recreate_x, sub_recreate_y, sub_recreate_z, sub_recreate_color = split_traj_for_plot(
										self._max_loc, recreated_pos_x, recreated_pos_y, recreated_pos_z)
		self._recreated_pos_x = recreated_pos_x
		self._recreated_pos_y = recreated_pos_y
		self._recreated_pos_z = recreated_pos_z
		self._recreated_encoder = recreated_encoder

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
		for item in self._max_loc:
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

		self._recreated_quat_w, self._recreated_quat_x, self._recreated_quat_y, self._recreated_quat_z = parse_dataTable(recreated_quat)

		key_frame_time_max = []
		key_frame_time_quat = []
		for item in max_loc:
			key_frame_time_max.append(self._time_stamp[int(item[0])])
		for item in quat_peak:
			key_frame_time_quat.append(self._time_stamp[int(item[0])])

		#======================doing some test here===============================
		merged_force1_peak_mark = []
		merged_force2_peak_mark = []
		force1_z_peak_mark = []
		force2_z_peak_mark = []
		encoder_peak_mark = []
		quat_peak_mark = []

		for item in merged_force1_peak:
			merged_force1_peak_mark.append(
				[center_traj_x[int(item[0])], center_traj_y[int(item[0])], center_traj_z[int(item[0])]])
		for item in merged_force2_peak:
			merged_force2_peak_mark.append(
				[center_traj_x[int(item[0])], center_traj_y[int(item[0])], center_traj_z[int(item[0])]])

		for item in force1_z_peak:
			force1_z_peak_mark.append(
				[center_traj_x[int(item[0])], center_traj_y[int(item[0])], center_traj_z[int(item[0])]])
		for item in force2_z_peak:
			force2_z_peak_mark.append(
				[center_traj_x[int(item[0])], center_traj_y[int(item[0])], center_traj_z[int(item[0])]])

		for item in encoder_peak:
			encoder_peak_mark.append(
				[center_traj_x[int(item[0])], center_traj_y[int(item[0])], center_traj_z[int(item[0])]])

		for item in quat_peak:
			quat_peak_mark.append(
					[center_traj_x[int(item[0])], center_traj_y[int(item[0])], center_traj_z[int(item[0])]])					
		#=========================================================================	

		plt.figure(2)
		plt.subplot(511)
		plt.grid(True)
		plt.plot(self._time_stamp, encoder_deriv_filtered, 'b-', linewidth=2)
		plt.subplot(512)
		plt.grid(True)
	#	plt.plot(self._time_stamp[0:-1], kk, 'b--', linewidth=2)
		plt.plot(self._time_stamp[0:-1], kk_filtered, 'b-', linewidth=2)

		plt.subplot(513)
		plt.grid(True)
		plt.plot(self._time_stamp, force1_x_deriv, 'r-', linewidth=2, label='x')
		plt.plot(self._time_stamp, force1_y_deriv, 'g-', linewidth=2, label='y')
		plt.plot(self._time_stamp, tran_force1_z_std, 'b-', linewidth=2, label='z')
		plt.legend()	
		
		plt.subplot(514)
		plt.grid(True)
		plt.plot(self._time_stamp, force2_x_deriv, 'r-', linewidth=2, label='x')
		plt.plot(self._time_stamp, force2_y_deriv, 'g-', linewidth=2, label='y')
		plt.plot(self._time_stamp, tran_force1_z_std, 'b-', linewidth=2, label='z')
		plt.legend()

		plt.subplot(515)
		plt.grid(True)
		plt.plot(self._time_stamp, filtered_signal, 'b-', linewidth=2)
		for item in key_frame_time_max:
			plt.axvline(item, 0, 1, linewidth=1, color='r', linestyle='--')

		'''
		plt.figure(6)
		plt.grid(True)
	#	plt.plot(self._time_stamp[0:-1], kk, 'b--', linewidth=2)
		plt.plot(self._time_stamp[0:-1], kk, 'k--', linewidth=1)
		plt.plot(self._time_stamp[0:-1], kk_filtered, 'r-', linewidth=1)
		for item in key_frame_time_quat:
			plt.axvline(item, 0, 1, linewidth=2, color='g')

		plt.figure(7)
		plt.title("force sensor 1")
		plt.grid(True)
		plt.subplot(311)
		plt.plot(self._time_stamp, force1_x_deriv, 'r-', linewidth=2)
		plt.subplot(312)
		plt.plot(self._time_stamp, force1_y_deriv, 'g-', linewidth=2)
		plt.subplot(313)
		plt.plot(self._time_stamp, tran_force1_z_std, 'b-', linewidth=2)		
		
		plt.figure(8)
		plt.title("force sensor 2")
		plt.grid(True)
		plt.subplot(311)
		plt.plot(self._time_stamp, force2_x_deriv, 'r-', linewidth=2)
		plt.subplot(312)
		plt.plot(self._time_stamp, force2_y_deriv, 'g-', linewidth=2)
		plt.subplot(313)
		plt.plot(self._time_stamp, tran_force1_z_std, 'b-', linewidth=2)
		'''


		'''
		ax.scatter(
			sub_traj_x[0], center_traj_y[0], center_traj_y[0], c='r', s=70, marker='^')
		ax.scatter(
			center_traj_x[-1], center_traj_y[-1], center_traj_y[-1], c='k', s=70, marker='^')
		'''
		'''
		#encoder:			
		for i in range(len(encoder_peak)):
			ax.scatter(
				encoder_peak_mark[i][0], encoder_peak_mark[i][1], encoder_peak_mark[i][2], c='k', s=70, marker='^')
		for i in range(len(merged_force1_peak)):
			ax.scatter(
				merged_force1_peak_mark[i][0], merged_force1_peak_mark[i][1], merged_force1_peak_mark[i][2], c='r', s=70, marker='s')
		for i in range(len(merged_force2_peak)):
			ax.scatter(
				merged_force2_peak_mark[i][0], merged_force2_peak_mark[i][1], merged_force2_peak_mark[i][2], c='r', s=70)
		for i in range(len(force1_z_peak)):
			ax.scatter(
				force1_z_peak_mark[i][0], force1_z_peak_mark[i][1], force1_z_peak_mark[i][2], c='g', s=70, marker='s')
		for i in range(len(force2_z_peak)):
			ax.scatter(
				force2_z_peak_mark[i][0], force2_z_peak_mark[i][1], force2_z_peak_mark[i][2], c='g', s=70)
		'''		
		'''
		for i in range(len(quat_peak)):
			ax.scatter(
				quat_peak_mark[i][0], quat_peak_mark[i][1], quat_peak_mark[i][2], c='y', s=70, marker='h')
		'''
		'''
		fig = plt.figure(14)
		ax = fig.add_subplot(111, projection='3d')
		ax.plot(center_traj_x, center_traj_y, center_traj_z, c=[0.172, 0.498, 0.721], linewidth=4)

		ax.set_title("Raw Trajectory", fontname="Times New Roman")
		ax.set_xlabel('x_axis', fontname="Times New Roman")
		ax.set_ylabel('y_axis', fontname="Times New Roman")
		ax.set_zlabel('z_axis', fontname="Times New Roman")
		ax.grid(b=False)

		figg = plt.figure(15)
		bx = figg.add_subplot(111, projection='3d')
		for j in range(len(sub_recreate_x)):
			bx.plot(sub_recreate_x[j], sub_recreate_y[j], sub_recreate_z[j], c=sub_color[j], linewidth=4)
		bx.set_title("Recreated Trajectory", fontname="Times New Roman")
		bx.set_xlabel('x_axis', fontname="Times New Roman")
		bx.set_ylabel('y_axis', fontname="Times New Roman")
		bx.set_zlabel('z_axis', fontname="Times New Roman")
		bx.grid(b=False)
		'''


		fig = plt.figure(13)
		ax = fig.add_subplot(221, projection='3d')

		for i in range(len(sub_traj_x)):
			ax.plot(sub_traj_x[i], sub_traj_y[i], sub_traj_z[i], c=sub_color[i], linewidth=2)

		for item in max_loc:
			idx = int(item[0])
			ax.scatter(center_traj_x[idx], center_traj_y[idx], center_traj_z[idx], c='r', marker='^', s=40)
		ax.set_title("Raw Trajectory", fontname="Times New Roman")
		ax.set_xlabel('x_axis', fontname="Times New Roman")
		ax.set_ylabel('y_axis', fontname="Times New Roman")
		ax.set_zlabel('z_axis', fontname="Times New Roman")
		ax.grid(b=False)
		bx = fig.add_subplot(222, projection='3d')
		for j in range(len(sub_recreate_x)):
			bx.plot(sub_recreate_x[j], sub_recreate_y[j], sub_recreate_z[j], c=sub_color[j], linewidth=2)
		bx.set_title("Recreated Trajectory", fontname="Times New Roman")
		bx.set_xlabel('x_axis', fontname="Times New Roman")
		bx.set_ylabel('y_axis', fontname="Times New Roman")
		bx.set_zlabel('z_axis', fontname="Times New Roman")
		bx.grid(b=False)

		cx = fig.add_subplot(223, projection='3d')
		cx.plot(center_traj_x_filtered0, center_traj_y_filtered0, center_traj_z_filtered0, c=[0.172, 0.498, 0.721], linewidth=2)
		cx.set_title("Filtered with 0.75Hz Low Pass Filter", fontname="Times New Roman")
		cx.set_xlabel('x_axis', fontname="Times New Roman")
		cx.set_ylabel('y_axis', fontname="Times New Roman")
		cx.set_zlabel('z_axis', fontname="Times New Roman")
		cx.grid(b=False)

		dx = fig.add_subplot(224, projection='3d')
		dx.plot(center_traj_x_filtered1, center_traj_y_filtered1, center_traj_z_filtered1, c=[0.172, 0.498, 0.721], linewidth=2)
		dx.set_title("Filtered with 2.5Hz Low Pass Filter", fontname="Times New Roman")
		dx.set_xlabel('x_axis', fontname="Times New Roman")
		dx.set_ylabel('y_axis', fontname="Times New Roman")
		dx.set_zlabel('z_axis', fontname="Times New Roman")
		dx.grid(b=False)

		'''
		fig_2 = plt.figure(2)
		bx = fig_2.add_subplot(111, projection='3d')
		for j in range(len(sub_recreate_x)):
			bx.plot(sub_recreate_x[j], sub_recreate_y[j], sub_recreate_z[j], c=sub_color[j], linewidth=2)

		fig_12 = plt.figure(12)
		zx = fig_12.add_subplot(111, projection='3d')
		zx.plot(center_traj_x_filtered0, center_traj_y_filtered0, center_traj_z_filtered0, c='k', linewidth=2)

		fig_11 = plt.figure(11)
		zx = fig_12.add_subplot(111, projection='3d')
		zx.plot(center_traj_x_filtered1, center_traj_y_filtered1, center_traj_z_filtered1, c='k', linewidth=2)
		'''
	
		
		'''
		fig4 = plt.figure(4)
		cx = fig4.add_subplot(111, projection='3d')
		for i in range(len(sub_all_x)):
			cx.plot(sub_all_x[i], sub_all_y[i], sub_all_z[i], c=sub_all_color[i], linewidth=2)

		fig5 = plt.figure(5)
		dx = fig5.add_subplot(111, projection='3d')
		for i in range(len(sub_recreate_all_x)):
			dx.plot(sub_recreate_all_x[i], 
				sub_recreate_all_y[i], sub_recreate_all_z[i], c=sub_all_color[i], linewidth=2)
		'''
		'''
		plt.figure(8)
		plt.subplot(411)
		plt.plot(self._time_stamp, quat_w, 'k-', linewidth=2)
		plt.plot(self._time_stamp, self._recreated_quat_w, 'k--', linewidth=1)
		plt.subplot(412)
		plt.plot(self._time_stamp, quat_x, 'r-', linewidth=2)
		plt.plot(self._time_stamp, self._recreated_quat_x, 'r--', linewidth=1)
		plt.subplot(413)
		plt.plot(self._time_stamp, quat_y, 'g-', linewidth=2)
		plt.plot(self._time_stamp, self._recreated_quat_y, 'g--', linewidth=1)	
		plt.subplot(414)
		plt.plot(self._time_stamp, quat_z, 'b-', linewidth=2)
		plt.plot(self._time_stamp, self._recreated_quat_z, 'b--', linewidth=1)
		'''
		'''
		plt.figure(7)
		plt.title("force sensor 1")
		plt.grid(True)
		plt.subplot(311)
		plt.plot(self._time_stamp, force1_x_deriv, 'r-', linewidth=2)
		plt.plot(self._time_stamp, force1_x_filtered, 'y--', linewidth=2)
		plt.subplot(312)
		plt.plot(self._time_stamp, force1_y_deriv, 'g-', linewidth=2)
		plt.plot(self._time_stamp, force1_y_filtered, 'y--', linewidth=2)
		plt.subplot(313)
		plt.plot(self._time_stamp, tran_force1_z_std, 'b-', linewidth=2)
		plt.plot(self._time_stamp, force1_z_filtered, 'y--', linewidth=2)
		
		
		plt.figure(8)
		plt.title("force sensor 2")
		plt.grid(True)
		plt.subplot(311)
		plt.plot(self._time_stamp, force2_x_deriv, 'r-', linewidth=2)
		plt.plot(self._time_stamp, force2_x_filtered, 'y--', linewidth=2)
		plt.subplot(312)
		plt.plot(self._time_stamp, force2_y_deriv, 'g-', linewidth=2)
		plt.plot(self._time_stamp, force2_y_filtered, 'y--', linewidth=2)
		plt.subplot(313)
		plt.plot(self._time_stamp, tran_force1_z_std, 'b-', linewidth=2)
		plt.plot(self._time_stamp, force2_z_filtered, 'y--', linewidth=2)
		'''
		'''
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
	unit = np.ones(len(sp.recreated_pos_x))
	pos_viv_coor = np.vstack((sp.recreated_pos_x, sp.recreated_pos_y, sp.recreated_pos_z, unit)).T

	quat_viv_coor = np.vstack((sp.recreated_quat_w, sp.recreated_quat_x, sp.recreated_quat_y, sp.recreated_quat_z, unit)).T
	unit2 = np.ones(len(sp.staggered_poses_pos_x))
	spPos_vive = np.vstack((sp.staggered_poses_pos_x, sp.staggered_poses_pos_y, sp.staggered_poses_pos_z, unit2)).T
	spQuat_vive = np.vstack((
		sp.staggered_poses_quat_w, sp.staggered_poses_quat_x, sp.staggered_poses_quat_y, sp.staggered_poses_quat_z)).T
#	transformed_pos, transformed_axis_angle = transform_to_robot_coor(pos_viv_coor, quat_viv_coor)
	transformed_pos, transformed_quat = transform_to_robot_direct(pos_viv_coor, quat_viv_coor)

	raw_pos = sp.ros_bag_data.vivePos_interpolated
	raw_quat = sp.ros_bag_data.viveQuat_interpolated

	raw_pos_x = sp.center_traj_x
	raw_pos_y = sp.center_traj_y
	raw_pos_z = sp.center_traj_z
	unit_raw =  np.ones(len(raw_pos_x))

	raw_quat_w, raw_quat_x, raw_quat_y, raw_quat_z = parse_dataTable(raw_quat)
	pos_raw = np.vstack((raw_pos_x, raw_pos_y, raw_pos_z, unit_raw)).T
	quat_raw = np.vstack((raw_quat_w, raw_quat_x, raw_quat_y, raw_quat_z, unit_raw)).T
	tran_raw_pos, tran_raw_quat = transform_to_robot_direct(pos_raw, quat_raw)

	tran_raw_x, tran_raw_y, tran_raw_z = parse_dataTable(tran_raw_pos)
	tran_raw_q_w, tran_raw_q_x, tran_raw_q_y, tran_raw_q_z = parse_dataTable(tran_raw_quat)
	tran_x, tran_y, tran_z = parse_dataTable(transformed_pos)
	tran_q_w, tran_q_x, tran_q_y, tran_q_z = parse_dataTable(transformed_quat)
	'''
	with open("set_down_plate1.csv", 'w') as out_file:
		csv_writer = csv.writer(out_file, dialect='excel')
		for i in range(len(sp.recreated_pos_x)):
			#print([sp.time_stamp[i], pos_viv_coor, quat_viv_coor])
			csv_writer.writerow([sp.time_stamp[i], [tran_x[i], tran_y[i], tran_z[i]], 
				[tran_q_w[i], tran_q_x[i], tran_q_y[i], tran_q_z[i]], 
				sp.recreated_encoder[i]])

	with open("set_down_plate1_raw.csv", 'w') as out_file:
		csv_writer = csv.writer(out_file, dialect='excel')
		for i in range(len(sp.recreated_pos_x)):
			#print([sp.time_stamp[i], pos_viv_coor, quat_viv_coor])
			csv_writer.writerow([sp.ros_bag_data.resample_time_stamp[i], [tran_raw_x[i], tran_raw_y[i], tran_raw_z[i]], 
				[tran_raw_q_w[i], tran_raw_q_x[i], tran_raw_q_y[i], tran_raw_q_z[i]], 
				sp.ros_bag_data.encoderarray_interpolated[i]])
	'''
	'''
	fig = plt.figure(1)
	bx = fig.add_subplot(111, projection='3d')
	bx.plot(sp.recreated_pos_x, sp.recreated_pos_y, sp.recreated_pos_z, c='r', linewidth=2)
	bx.plot(tran_x, tran_y, tran_z, c='b', linewidth=2)
	bx.set_xlabel('X')
	bx.set_ylabel('Y')
	bx.set_zlabel('Z')
	'''

	fig = plt.figure(1)
	bx = fig.add_subplot(111, projection='3d')
	bx.plot(raw_pos_x, raw_pos_y, raw_pos_z, c='r', linewidth=2)
	bx.plot(tran_raw_x, tran_raw_y, tran_raw_z, c='b', linewidth=2)
	bx.set_xlabel('X')
	bx.set_ylabel('Y')
	bx.set_zlabel('Z')

	fig_21 = plt.figure(21)
	plt.subplot(311)
	plt.plot(sp.time_stamp, sp.ros_bag_data.encoderarray_interpolated, c='r')
	plt.subplot(312)
	plt.plot(sp.time_stamp, sp.recreated_encoder, c='g')

	'''
	plt.figure(8)
	plt.subplot(311)
	plt.plot(interval, tran_x, c='r')
	plt.subplot(312)
	plt.plot(interval, tran_y, c='g')
	plt.subplot(313)
	plt.plot(interval, tran_z, c='b')
	'''
	
	plt.show()


