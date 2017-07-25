import cPickle as pickle

merge_dc = False
if merge_dc:
    vid_content = pickle.load(open('predict_dc.p', 'rb'))
    input_path = 'predict_d.txt'
    fout = open('predict_dc.txt', 'w')
    new_col_title = 'dc_wp'

merge_du = False
if merge_du:
    vid_content = pickle.load(open('predict_du.p', 'rb'))
    input_path = 'predict_dc.txt'
    # fout = open('predict_du.txt', 'w')
    fout = open('predict_du.txt', 'w')
    new_col_title = 'du_wp'

merge_duc = False
if merge_duc:
    vid_content = pickle.load(open('predict_duc.p', 'rb'))
    input_path = 'predict_du.txt'
    fout = open('predict_duc.txt', 'w')
    new_col_title = 'duc_wp'

merge_dut = False
if merge_dut:
    vid_content = pickle.load(open('predict_dut.p', 'rb'))
    input_path = 'predict_duc.txt'
    fout = open('predict_dut.txt', 'w')
    new_col_title = 'dut_wp'

merge_duct = True
if merge_duct:
    vid_content = pickle.load(open('predict_duct.p', 'rb'))
    input_path = 'predict_dut.txt'
    fout = open('predict_duct.txt', 'w')
    new_col_title = 'duct_wp'

# vid_channel_content = pickle.load(open('vid_channel_content_rbf.p', 'rb'))
# vid_channel_topic = pickle.load(open('vid_channel_topic.p', 'rb'))
# vid_channel_all = pickle.load(open('vid_channel_all.p', 'rb'))

# tmp_dict = {}
# with open('vid_topic.txt', 'r') as fin:
#    for line in fin:
#        vid, topic_wp = line.rstrip().split()
#	tmp_dict[vid] = topic_wp

to_write_header = True
with open(input_path, 'r') as fin:
    header = fin.readline().rstrip()
    if to_write_header:
        fout.write('{0}\t{1}\n'.format(header, new_col_title))
        to_write_header = False
    for line in fin:
        vid, dump = line.rstrip().split('\t', 1)
        _, d_wp, dc_wp = line.rstrip().rsplit('\t', 2)
        if vid_content[vid] == 'NA':
            du_wp = float(d_wp)
            # du_wp = 'NA'
        else:
            du_wp = vid_content[vid]
        fout.write('{0}\t{1}\n'.format(line.rstrip(), du_wp))
fout.close()
