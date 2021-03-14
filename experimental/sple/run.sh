# preprocessing, 
# python processing.py #number_of_data_parts #number_of_angles_per_part #binning #file_name_without_h5
python processing.py 1 1200 1 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Myers/Sple1_Phase_1201prj_interlaced_1s_010
# reconstruction with alignment
# python admm.py #number_of_data_parts #number_of_angles_per_part #file_name_without_h5
python admm.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Myers/Sple1_Phase_1201prj_interlaced_1s_010
# reconstruction without alignment
# python cg.py #number_of_data_parts #number_of_angles_per_part #file_name_without_h5
python cg.py 1 1200 /data/staff/tomograms/vviknik/tomoalign_vincent_data/2020-07/Myers/Sple1_Phase_1201prj_interlaced_1s_010
