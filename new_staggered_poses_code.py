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
			force1_x, 5, self.resampleRate, 6)
		force1_y_filtered = butter_lowpass_filter(
			force1_y, 5, self.resampleRate, 6)
		force1_z_filtered = butter_lowpass_filter(
			force1_z, 5, self.resampleRate, 6)
		force2_x_filtered = butter_lowpass_filter(
			force2_x, 5, self.resampleRate, 6)
		force2_y_filtered = butter_lowpass_filter(
			force2_y, 5, self.resampleRate, 6)
		force2_z_filtered = butter_lowpass_filter(
			force2_z, 5, self.resampleRate, 6)
		encoder_signal = self.ros_bag_data.encoderarray_interpolated
		#preprocess the quaternion signal
		encoder_deriv_filtered = calc_derivative(self.cut_off_freq, self.resampleRate, encoder_signal)
		#preprocess the force signal
		force1_x_deriv = calc_derivative(self.cut_off_freq, self.resampleRate, force1_x_filtered)
		force1_y_deriv = calc_derivative(self.cut_off_freq, self.resampleRate, force1_y_filtered)
		force1_z_deriv = calc_derivative(self.cut_off_freq, self.resampleRate, force1_z_filtered)
		force2_x_deriv = calc_derivative(self.cut_off_freq, self.resampleRate, force2_x_filtered)
		force2_y_deriv = calc_derivative(self.cut_off_freq, self.resampleRate, force2_y_filtered)
		force2_z_deriv = calc_derivative(self.cut_off_freq, self.resampleRate, force2_z_filtered)

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

		center_traj_x_filtered = butter_lowpass_filter(
					center_traj_x, 3.33, self.resampleRate, 6)
		center_traj_y_filtered = butter_lowpass_filter(
					center_traj_y, 3.33, self.resampleRate, 6)
		center_traj_z_filtered = butter_lowpass_filter(
					center_traj_z, 3.33, self.resampleRate, 6)

		fil_freq0 = 0.7
		fil_freq1 = 2.5
		center_traj_x_filtered0 = butter_lowpass_filter(
					center_traj_x, fil_freq0, self.resampleRate, 6)
		center_traj_y_filtered0 = butter_lowpass_filter(
					center_traj_y, fil_freq0, self.resampleRate, 6)
		center_traj_z_filtered0 = butter_lowpass_filter(
					center_traj_z, fil_freq0, self.resampleRate, 6)

		center_traj_x_filtered1 = butter_lowpass_filter(
					center_traj_x, fil_freq1, self.resampleRate, 6)
		center_traj_y_filtered1 = butter_lowpass_filter(
					center_traj_y, fil_freq1, self.resampleRate, 6)
		center_traj_z_filtered1 = butter_lowpass_filter(
					center_traj_z, fil_freq1, self.resampleRate, 6)

		self.center_traj_x = center_traj_x
		self.center_traj_y = center_traj_y
		self.center_traj_z = center_traj_z
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
		T_x = [a[0] for a in T]
		T_y = [a[1] for a in T]
		T_z = [a[1] for a in T]

		kk_f = preprocess_curve_signal(T_f, N_f, center_traj_xyz_fil)
		kk_ff = butter_lowpass_filter(kk_f, self.cut_off_freq, self.resampleRate, 6)

		kk = preprocess_curve_signal(T, N, center_traj_xyz)
		kk_filtered = butter_lowpass_filter(kk, self.cut_off_freq, self.resampleRate, 6)

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
		#		tran_force1_z_std, tran_force2_z_std,
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
		threeD_arrow = []
		for item in max_loc:
			threeD_arrow.append(T[int(item[0])])
		#segmenting the whole trajectory using only key_frame value
		sub_traj_x, sub_traj_y, sub_traj_z, sub_color = split_traj_for_plot(
											max_loc, center_traj_x, center_traj_y, center_traj_z)
		#segmenting the whole trajectory using all key_frame value
		'''
		sub_all_x, sub_all_y, sub_all_z, sub_all_color = split_traj_for_plot(
								all_frame_for_calc, center_traj_x, center_traj_y, center_traj_z)
		'''
		obj_time_line = self.time_stamp
		recreated_pos_x = interpolate_from_key_frame(
			self.max_loc, self.center_traj_x, obj_time_line)
		recreated_pos_y = interpolate_from_key_frame(
			self.max_loc, self.center_traj_y, obj_time_line)
		recreated_pos_z = interpolate_from_key_frame(
			self.max_loc, self.center_traj_z, obj_time_line)
		recreated_encoder = interpolate_from_key_frame(
			self.max_loc, self.ros_bag_data.encoderarray_interpolated, obj_time_line)
		#=======================================================================================
		sub_recreate_x, sub_recreate_y, sub_recreate_z, sub_recreate_color = split_traj_for_plot(
										self.max_loc, recreated_pos_x, recreated_pos_y, recreated_pos_z)
		self.recreated_pos_x = recreated_pos_x
		self.recreated_pos_y = recreated_pos_y
		self.recreated_pos_z = recreated_pos_z
		self.recreated_encoder = recreated_encoder

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

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--filename', help='The file name/path of the bag.', required=True)
	args = vars(parser.parse_args())
	bagname = args["filename"]
	#define a staggered poses class and initialize it using ros bag data
	sp = staggered_poses(cut_off_freq=1.5)
	sp.load_data(ros_bag_file=bagname, resampleRate=1000)
	sp.search_staggered_poses_and_recreate()

