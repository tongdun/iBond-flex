from functools import partial



from caffeine.feature.feature_engineering.selection.guest import FeatureSelectionGuest
from caffeine.feature.feature_engineering.selection.host import FeatureSelectionHost
from caffeine.feature.feature_engineering.selection.coordinator import FeatureSelectionCoord
from caffeine_tests.feature.feature_engineering.selection.hetero_mainflow import (hetero_case_template, 
        hetero_run_processes, hetero_case_template_multihost,
        multihost_hetero_run_processes)

##############################################################################
case_prefix = 'selection_iv_'

selector = [
    FeatureSelectionGuest,
    FeatureSelectionHost,
    FeatureSelectionCoord
]

# hetero_sec_param = {
#     OTP_PN_FL: [["onetime_pad", {"key_length": 512}],],
#     KS_FFS: [["paillier", {"key_length": 1024}],],
# }

configs = [
    {'engine': 'light', 'warehouse': '/mnt/nfs/tmp/caffeine/models/feature_eng/model', 
            'bulletin':{'bond_dag_task_uuid': 'zyj_test_fea_eng', "report_url":"http://10.58.14.31:8089","bond_monitor_id":"14786",
                        'name': 'guest', 'model_id': 'guest_feature_process'}},
    {'engine': 'light', 'warehouse': '/mnt/nfs/tmp/caffeine/models/feature_eng/model', 
            'bulletin':{'bond_dag_task_uuid': 'zyj_test_fea_eng', "report_url":"http://10.58.14.31:8089","bond_monitor_id":"14786",
                        'name': 'host1', 'model_id': 'host1_feature_process'}},
    {'engine': 'light', 'warehouse': '/mnt/nfs/tmp/caffeine/models/feature_eng/model', 
            'bulletin':{'bond_dag_task_uuid': 'zyj_test_fea_eng', "report_url":"http://10.58.14.31:8089","bond_monitor_id":"14786",
                        'name': 'host2', 'model_id': 'host2_feature_process'}},
    {'engine': 'light', 'warehouse': '/mnt/nfs/tmp/caffeine/models/feature_eng/model', 
            'bulletin':{'bond_dag_task_uuid': 'zyj_test_fea_eng', "report_url":"http://10.58.14.31:8089","bond_monitor_id":"14786",
                        'name': 'arbiter', 'model_id': 'arbiter_feature_process'}},
]

meta_params = {
    "train_param":{
        'process_method': 'hetero',
        'pipeline': ['Iv'],
        'configs': {
            'Ks': {
                'iv_thres': 0.2,
                'top_k': 10,
            }
        },
        'common_params': {
            'down_feature_num': 2,
            'max_num_col': 200,
            'max_feature_num_col': 500,
            'use_multiprocess': False,
        }
    },
    # "security_param": hetero_sec_param,
    "predict_param": None
}

test_cases = {
    'hetero_new_finance': hetero_case_template(
        train_files = [
            '/mnt/nfs/datasets/shap_finance/multi_host/train_guest.csv',
            '/mnt/nfs/datasets/shap_finance/multi_host/train_host1.csv',
            '/mnt/nfs/datasets/shap_finance/multi_host/train_host2.csv',
        ],
        test_files = [
            '/mnt/nfs/datasets/shap_finance/multi_host/test_guest.csv',
            '/mnt/nfs/datasets/shap_finance/multi_host/test_host1.csv',
            '/mnt/nfs/datasets/shap_finance/multi_host/test_host2.csv',
        ],
        feat_infos = [
            '/mnt/nfs/yijing.zhou/data/shap_finance/train_guest_attrib_iv.json',
            '/mnt/nfs/yijing.zhou/data/shap_finance/train_host1_attrib_iv.json',
            dict()
        ],
        descs = [
            {'id_desc': ['ID'], 'y_desc': ['Label']},
            {'id_desc': ['ID'], 'y_desc': []},
            {'id_desc': ['ID'], 'y_desc': []},
        ],
        meta_params = meta_params,
        selector = selector,
        config = configs,
        timeout = 1200
    )
}

# test_cases_multihost = {
#     'hetero_new_finance_multi': hetero_case_template_multihost(
#         train_files = [
#             '/mnt/nfs/datasets/shap_finance/multi_host/train_guest.csv',
#             '/mnt/nfs/datasets/shap_finance/multi_host/train_host1.csv',
#             '/mnt/nfs/datasets/shap_finance/multi_host/train_host2.csv',
#         ],
#         test_files = [
#             '/mnt/nfs/datasets/shap_finance/multi_host/test_guest.csv',
#             '/mnt/nfs/datasets/shap_finance/multi_host/test_host1.csv',
#             '/mnt/nfs/datasets/shap_finance/multi_host/test_host2.csv',
#         ],
#         feat_infos = [
#             '/mnt/nfs/yijing.zhou/data/shap_finance/train_guest_attrib_iv.json',
#             '/mnt/nfs/yijing.zhou/data/shap_finance/train_host1_attrib_iv.json',
#             '/mnt/nfs/yijing.zhou/data/shap_finance/train_host2_attrib_iv.json',
#             None
#         ],
#         descs = [
#             {'id_desc': ['ID'], 'y_desc': ['Label']},
#             {'id_desc': ['ID'], 'y_desc': []},
#             {'id_desc': ['ID'], 'y_desc': []},
#         ],
#         meta_params = meta_params,
#         selector = selector,
#         config = configs,
#         timeout = 1200
#     )
# }
##############################################################################

for case_name, case in test_cases.items():
    locals()[f'test_{case_prefix}{case_name}'] = partial(multihost_hetero_run_processes, {case_name: case})

# for case_name, case in test_cases_multihost.items():
#     locals()[f'test_{case_prefix}{case_name}'] = partial(multihost_hetero_run_processes, {case_name: case})
#


if __name__ == '__main__':
    for case_name, case in test_cases.items():
        eval(f'test_{case_prefix}{case_name}()')
    # for case_name, case in test_cases_multihost.items():
    #     eval(f'test_{case_prefix}{case_name}()')
