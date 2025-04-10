import os
import sys
from os.path import join

ROOT = 'data/scenefun3d/benchmark_file_lists'
SPLIT = sys.argv[1] # train or val

with open(join(ROOT,'train_val_set.csv')) as f:
    all_visits = f.readlines()[1:]

with open(join(ROOT,SPLIT + '_scenes.txt')) as f:
    visit_ids = set(visit.strip('\n') for visit in f.readlines())

with open(join(ROOT,SPLIT + '_set.csv'),'w') as f:

    f.write('visit_id,video_id\n')

    for line in all_visits:
        visit_id, video_id = line.split(',')
        if visit_id in visit_ids:
            f.write(f'{visit_id},{video_id}')


