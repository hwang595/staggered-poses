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
		'''
		load data from rosBag files
		data is saved in class members
		parameters:
		rog_bag_file: name of rosBag files, sparsed from command line
		resampleRate: the signal synchronized frequency

		return:
		None
		'''
		bag_data = RosBagParser(resampleRate)
		empty_flag_list, encoderarray = bag_data.parseTongsBag(ros_bag_file)
		self._ros_bag_data = bag_data
		self._time_stamp = self._ros_bag_data.resample_time_stamp
		self._resampleRate = resampleRate

	def search_staggered_poses_and_recreate(self, do_plot=False):
		'''
		search the key-frames using staggered poses algorithm
		parameter:
		do_plot: True or False visulize the performance of staggered pose if True
		         Do nothing is false except assign staggered pose data

		return:
		None
		'''
		staggered_poses_pos_x = []
		staggered_poses_pos_y = []
		staggered_poses_pos_z = []
		staggered_poses_quat_w = []
		staggered_poses_quat_x = []
		staggered_poses_quat_y = []
		staggered_poses_quat_z = []
		key_time_stamp = []
		merged_force1 = []
		merged_force2 = []
		aggregated_pos = []
		#extract position signals
		pos_x, pos_y, pos_z = parse_dataTable(self._ros_bag_data.rigidBodyPos_interpolated)
		#extract quaternion signals
		quat_w, quat_x, quat_y, quat_z = parse_dataTable(self._ros_bag_data.rigidBodyQuat_interpolated)
		#extract force sensor value 1:
		force1_x, force1_y, force1_z = parse_dataTable(self._ros_bag_data.forceSensor1array_interpolated)
		#extract force sensor value 2:
		force2_x, force2_y, force2_z = parse_dataTable(self._ros_bag_data.forceSensor2array_interpolated)

		force1_x_filtered, force1_y_filtered, force1_z_filtered = force_filtering(forceTable=self._ros_bag_data.forceSensor1array_interpolated,
																				  cut_off_freq=5, 
																				  resampleRate=self._resampleRate, 
																				  order=6)
		force2_x_filtered, force2_y_filtered, force2_z_filtered = force_filtering(forceTable=self._ros_bag_data.forceSensor2array_interpolated,
																				  cut_off_freq=5, 
																				  resampleRate=self._resampleRate, 
																				  order=6)
		encoder_signal = self._ros_bag_data.encoderarray_interpolated
		#preprocess the quaternion signal
		encoder_deriv_filtered = calc_derivative(self._cut_off_freq, self._resampleRate, encoder_signal)
		#preprocess the force signal
		force1_x_deriv, force1_y_deriv, force1_z_deriv = take_force_deriv(
								force1_x_filtered, force1_y_filtered, force1_z_filtered, 
								cut_off_freq=self._cut_off_freq, 
								resampleRate=self._resampleRate)
		force2_x_deriv, force2_y_deriv, force2_z_deriv = take_force_deriv(
								force2_x_filtered, force2_y_filtered, force2_z_filtered, 
								cut_off_freq=self._cut_off_freq, 
								resampleRate=self._resampleRate)

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

		center_traj_x_filtered, center_traj_y_filtered, center_traj_z_filtered = threeDArray_filtering(
							  element_x=center_traj_x, element_y=center_traj_y, element_z=center_traj_z, 
							  cut_off_freq=3.33, 
							  resampleRate=self._resampleRate, 
							  order=6)

		#to make comparison between staggered poses regenerated signal and naive low-pass filter
		#and even make comparison with even stronger filter
		center_traj_x_filtered0, center_traj_y_filtered0, center_traj_z_filtered0 = threeDArray_filtering(
					  element_x=center_traj_x, 
					  element_y=center_traj_y, 
					  element_z=center_traj_z, 
					  cut_off_freq=0.7, 
					  resampleRate=self._resampleRate, 
					  order=6)
		center_traj_x_filtered1, center_traj_y_filtered1, center_traj_z_filtered1 = threeDArray_filtering(
					  element_x=center_traj_x, 
					  element_y=center_traj_y, 
					  element_z=center_traj_z, 
					  cut_off_freq=2.5, 
					  resampleRate=self._resampleRate, 
					  order=6)
		self._center_traj_x = center_traj_x
		self._center_traj_y = center_traj_y
		self._center_traj_z = center_traj_z

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
		quat_signal_std = np.append(quat_signal_std, 0)
		encoder_peak, aa = peakdet(encoder_deriv_filtered, 0.1)
		#calculate the transformed force value peak
		merged_force1_peak, _ = peakdet(merged_force1, 0.05)
		merged_force2_peak, _ = peakdet(merged_force2, 0.05)

		quat_peak, _ = peakdet(quat_signal_std, 0.1)
		all_frame = [int(a[0]) for a in encoder_peak]
		all_frame.extend([int(a[0]) for a in quat_peak])
		all_frame = sorted(list(set(all_frame)))
		all_frame_for_calc = []
		for frame in all_frame:
			all_frame_for_calc.append([frame, 0])
		all_frame_for_calc = all_frame_for_calc[1:]
		
		ops = zip(
					quat_signal_std, encoder_deriv_filtered,
					force1_x_deriv, force1_y_deriv, force1_z_deriv, 
					force2_x_deriv, force2_y_deriv, force2_z_deriv, 
					mag_force1_std, mag_force2_std,
					merged_force1_std, merged_force2_std 
				) #tran_force1_z_std, tran_force2_z_std

		for o_p in ops:
			aggregated_pos.append(sum(o_p))

		aggregated_pos = np.array(aggregated_pos)
		#lowpass filtering the aggregated signal
		filtered_signal = butter_lowpass_filter(aggregated_pos, LOWCUT, self._resampleRate, ORDER)
		#extract peak value of aggregated signal
		max_loc, min_loc = peakdet(filtered_signal, 0.1)
		self._max_loc = max_loc
		#======================try some interesting things====================================
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
		key_frame_quat_w, key_frame_quat_x, key_frame_quat_y, key_frame_quat_z = generate_quat_keyFrame(
							   quat_table=self._ros_bag_data.rigidBodyQuat_interpolated, 
							   max_loc=self._max_loc)

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

		#start visulization things for performance of staggered poses
		if do_plot:
			plt.figure(2)
			plt.subplot(511)
			plt.grid(True)
			plt.plot(self._time_stamp, encoder_deriv_filtered, 'b-', linewidth=2)
			plt.subplot(512)
			plt.grid(True)
			plt.plot(self._time_stamp[0:-1], kk_filtered, 'b-', linewidth=2)

			plt.subplot(513)
			plt.grid(True)
			plt.plot(self._time_stamp, force1_x_deriv, 'r-', linewidth=2, label='x')
			plt.plot(self._time_stamp, force1_y_deriv, 'g-', linewidth=2, label='y')
			plt.plot(self._time_stamp, force1_z_deriv, 'b-', linewidth=2, label='z')
			plt.legend()	
			
			plt.subplot(514)
			plt.grid(True)
			plt.plot(self._time_stamp, force2_x_deriv, 'r-', linewidth=2, label='x')
			plt.plot(self._time_stamp, force2_y_deriv, 'g-', linewidth=2, label='y')
			plt.plot(self._time_stamp, force2_z_deriv, 'b-', linewidth=2, label='z')
			plt.legend()

			plt.subplot(515)
			plt.grid(True)
			plt.plot(self._time_stamp, filtered_signal, 'b-', linewidth=2)
			for item in key_frame_time_max:
				plt.axvline(item, 0, 1, linewidth=1, color='r', linestyle='--')

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

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('-f', '--filename', help='The file name/path of the bag.', required=True)
	args = vars(parser.parse_args())
	bagname = args["filename"]
	#define a staggered poses class and initialize it using ros bag data
	sp = staggered_poses(cut_off_freq=1.5)
	sp.load_data(ros_bag_file=bagname, resampleRate=1000)
	sp.search_staggered_poses_and_recreate(do_plot=True)
	unit = np.ones(len(sp._recreated_pos_x))
	pos_viv_coor = np.vstack((sp._recreated_pos_x, sp._recreated_pos_y, sp._recreated_pos_z, unit)).T

	quat_viv_coor = np.vstack((sp._recreated_quat_w, sp._recreated_quat_x, sp._recreated_quat_y, sp._recreated_quat_z, unit)).T
	unit2 = np.ones(len(sp._staggered_poses_pos_x))
	spPos_vive = np.vstack((sp._staggered_poses_pos_x, sp._staggered_poses_pos_y, sp._staggered_poses_pos_z, unit2)).T
	spQuat_vive = np.vstack((
		sp._staggered_poses_quat_w, sp._staggered_poses_quat_x, sp._staggered_poses_quat_y, sp._staggered_poses_quat_z)).T

	#do transformation for calibration
	transformed_pos, transformed_quat = transform_to_robot_direct(pos_viv_coor, quat_viv_coor)

	raw_pos = sp._ros_bag_data.rigidBodyPos_interpolated
	raw_quat = sp._ros_bag_data.rigidBodyQuat_interpolated

	raw_pos_x = sp._center_traj_x
	raw_pos_y = sp._center_traj_y
	raw_pos_z = sp._center_traj_z
	unit_raw =  np.ones(len(raw_pos_x))

	raw_quat_w, raw_quat_x, raw_quat_y, raw_quat_z = parse_dataTable(raw_quat)
	pos_raw = np.vstack((raw_pos_x, raw_pos_y, raw_pos_z, unit_raw)).T
	quat_raw = np.vstack((raw_quat_w, raw_quat_x, raw_quat_y, raw_quat_z, unit_raw)).T
	tran_raw_pos, tran_raw_quat = transform_to_robot_direct(pos_raw, quat_raw)

	raw_csv_fileName = "out_csv_file/raw.csv"
	stagPos_csv_fileName = "out_csv_file/stagPos.csv"
	write_to_csvFile(raw_csv_fileName, tran_raw_pos, tran_raw_quat, sp, 
					dataType="raw")
	write_to_csvFile(stagPos_csv_fileName, transformed_pos, transformed_quat, sp, 
					dataType="stag_pos")

	fig = plt.figure(33)
	bx = fig.add_subplot(111, projection='3d')
	bx.plot([x[0] for x in pos_raw], 
			[x[1] for x in pos_raw], 
			[x[2] for x in pos_raw], 
			c='r', 
			linewidth=2)
	bx.plot([x[0] for x in tran_raw_pos], 
			[x[1] for x in tran_raw_pos], 
			[x[2] for x in tran_raw_pos], 
			c='b', 
			linewidth=2)
	bx.set_xlabel('X')
	bx.set_ylabel('Y')
	bx.set_zlabel('Z')
	plt.show()