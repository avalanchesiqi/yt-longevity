#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Extract freebase topic type from latest data dump.

Usage: python extract_freebase_type.py freebase_dump_path output_path
Example: python extract_freebase_type.py [change path if necessary]
Time: ~4H
"""

import gzip, re, time, datetime

if __name__ == '__main__':
    # setting parameters
    print('>>> Start to extract freebase mid and corresponding type...')
    start_time = time.time()

    # load dataset
    with gzip.open('../freebase-rdf-latest.gz', 'r') as fin, open('freebase_mid_type.txt', 'w') as fout:
        for line in fin:
            subject, predicate, object, _ = line.split('\t')
            if predicate == '<http://rdf.freebase.com/ns/type.type.instance>':
                try:
                    mid = '/m/' + re.split('/|>', object)[4][2:]

                    ftype = re.split('/|>', subject)[4]
                    ftype_list = ftype.split('.')
                    if ftype_list[0] == 'freebase' or ftype_list[0] == 'aareas':
                        continue
                    elif ftype_list[0] == 'user' and ftype_list[2] == 'default_domain':
                        ftype_output = ftype_list[-1]
                    else:
                        ftype_output = ftype_list[-2]

                    fout.write('{0}\t{1}\n'.format(mid, ftype_output))
                except:
                    print('Exception line', line)
                    continue

    # get running time
    print('\n>>> Total running time: {0}'.format(str(datetime.timedelta(seconds=time.time() - start_time)))[:-3])
