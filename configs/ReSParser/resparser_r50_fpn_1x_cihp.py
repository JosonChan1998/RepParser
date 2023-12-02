_base_ = [
    '../_base_/datasets/cihp.py',
    '../_base_/models/resparser_r50_fpn.py',
    '../_base_/schedules/schedule_cihp_1x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    test_cfg=dict(
       save_root='work_dirs/resparser_r50_fpn_1x_cihp'))

optimizer = dict(lr=0.005)
evaluation=dict(interval=100, metric='bbox')
