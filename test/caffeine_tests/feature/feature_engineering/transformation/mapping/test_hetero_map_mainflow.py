from functools import partial

from flex.constants import (OTP_PN_FL, HE_DT_FB)

from caffeine.feature.feature_engineering.transformation.guest import FeatureTransformGuest
from caffeine.feature.feature_engineering.transformation.host import FeatureTransformHost
from caffeine.feature.feature_engineering.transformation.coordinator import FeatureTransformCoord
from caffeine_tests.feature.feature_engineering.transformation.hetero_mainflow import (hetero_case_template, 
        hetero_run_processes, hetero_case_template_multihost,
        multihost_hetero_run_processes)

##############################################################################
case_prefix = 'transform_woe_map_'

selector = [
    FeatureTransformGuest,
    FeatureTransformHost,
    FeatureTransformCoord
]

hetero_sec_param = {
    OTP_PN_FL: [["onetime_pad", {"key_length": 512}]],
    HE_DT_FB: [['paillier', {"key_length": 1024}]]
}

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
    'train_param': {
        'pipeline': ['dt_bin', 'woe_map'],
        'configs': {
            'dt_bin': {
                'equal_num_bin': 50,
                'min_bin_num': 4,
                'max_bin_num': 6,
            },
            'woe_map': {
                'check_monotone': True,
                'map_to_woe': True
            }
        },
        'process_method': 'hetero',
        'common_params':{
            'use_multiprocess': False,
        },
    },
    'security_param': hetero_sec_param,
    'predict_param': None
}


test_cases = {
    # 'hetero_dtype': hetero_case_template(
    #     train_files = [
    #         '/mnt/nfs/datasets/shap_finance/multi_host/train_guest_dtype.csv',
    #         '/mnt/nfs/datasets/shap_finance/multi_host/train_host1.csv',
    #         '/mnt/nfs/datasets/shap_finance/multi_host/train_host2.csv',
    #     ],
    #     test_files = [
    #         '/mnt/nfs/datasets/shap_finance/multi_host/train_guest_dtype.csv',
    #         '/mnt/nfs/datasets/shap_finance/multi_host/test_host1.csv',
    #         '/mnt/nfs/datasets/shap_finance/multi_host/test_host2.csv',
    #     ],
    #     feat_infos = [
    #         '/mnt/nfs/yijing.zhou/data/shap_finance/train_guest_dtype_attrib.json',
    #         '/mnt/nfs/yijing.zhou/data/shap_finance/train_host1_attrib.json',
    #         None
    #     ],
    #     descs = [
    #         {'id_desc': ['ID'], 'y_desc': ['Label']},
    #         {'id_desc': ['ID'], 'y_desc': []},
    #         {'id_desc': ['ID'], 'y_desc': []},
    #     ],
    #     meta_params = meta_params,
    #     selector = selector,
    #     config = configs,
    #     timeout = 1200
    # ),
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
            '/mnt/nfs/yijing.zhou/data/shap_finance/train_guest_attrib.json',
            '/mnt/nfs/yijing.zhou/data/shap_finance/train_host1_attrib.json',
            None
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
    ),
    # 'hetero_new_finance_corr': hetero_case_template(
    #     train_files = [
    #         '/mnt/nfs/datasets/shap_finance/shap_corr/train_guest_corr.csv',
    #         '/mnt/nfs/datasets/shap_finance/shap_corr/train_host_corr.csv',
    #         '/mnt/nfs/datasets/shap_finance/shap_corr/train_host.csv',
    #     ],
    #     test_files = [
    #         '/mnt/nfs/datasets/shap_finance/shap_corr/test_guest_corr.csv',
    #         '/mnt/nfs/datasets/shap_finance/shap_corr/test_host_corr.csv',
    #         '/mnt/nfs/datasets/shap_finance/shap_corr/test_host.csv',
    #     ],
    #     feat_infos = [
    #         '/mnt/nfs/yijing.zhou/data/shap_finance/train_guest_corr_attrib.json',
    #         '/mnt/nfs/yijing.zhou/data/shap_finance/train_host_corr_attrib.json',
    #         None
    #     ],
    #     descs = [
    #         {'id_desc': ['ID'], 'y_desc': ['Label']},
    #         {'id_desc': ['ID'], 'y_desc': []},
    #         {'id_desc': ['ID'], 'y_desc': []},
    #     ],
    #     meta_params = meta_params,
    #     selector = selector,
    #     config = configs,
    #     timeout = 1200
    # ),
    # 'hetero_new_finance_nan': hetero_case_template(
    #     train_files = [
    #         '/mnt/nfs/datasets/shap_finance/shap_null/shap_guest_null.csv',
    #         '/mnt/nfs/datasets/shap_finance/shap_null/shap_host_null.csv',
    #     ],
    #     test_files = [
    #         '/mnt/nfs/datasets/shap_finance/shap_null/test_guest_null.csv',
    #         '/mnt/nfs/datasets/shap_finance/shap_null/test_host_null.csv',
    #     ],
    #     feat_infos = [
    #         '/mnt/nfs/yijing.zhou/data/shap_finance/train_guest_nan.json',
    #         '/mnt/nfs/yijing.zhou/data/shap_finance/train_host_nan.json',
    #         None
    #     ],
    #     descs = [
    #         {'id_desc': ['ID'], 'y_desc': ['Label']},
    #         {'id_desc': ['ID'], 'y_desc': []},
    #         {'id_desc': ['ID'], 'y_desc': []},
    #     ],
    #     meta_params = meta_params,
    #     selector = selector,
    #     config = configs,
    #     timeout = 1200
    # ),
    # 'hetero_data_united_40w': hetero_case_template(
    #     train_files = [
    #         '/mnt/nfs/datasets/data_united/train_hetero_guest.csv',
    #         # '/mnt/nfs/datasets/data_united/train_hetero_guest.csv',
    #         '/mnt/nfs/datasets/data_united/train_hetero_host.csv',
    #         '/mnt/nfs/datasets/data_united/train_hetero_host.csv',
    #     ],
    #     test_files = [
    #         '/mnt/nfs/datasets/data_united/test_5t_guest.csv',
    #         # '/mnt/nfs/datasets/data_united/test_5t_guest.csv',
    #         '/mnt/nfs/datasets/data_united/test_5t_host.csv',
    #         '/mnt/nfs/datasets/data_united/test_5t_host.csv',
    #     ],
    #    feat_infos = [
    #        '/mnt/nfs/yijing.zhou/data/data_united/guest_data_attrib.json',
    #        '/mnt/nfs/yijing.zhou/data/data_united/host_data_attrib.json',
    #        '/mnt/nfs/yijing.zhou/data/data_united/host_data_attrib.json',
    #    ],
    #     descs = [
    #         {'id_desc': ['ID'], 'y_desc': ['label']},
    #         {'id_desc': ['ID'], 'y_desc': []},
    #         {'id_desc': ['ID'], 'y_desc': []},
    #     ],
    #     meta_params = meta_params,
    #     selector = selector,
    #     config = configs,
    #     timeout = 12000
    # ),
    # 'hetero_large_data_1w_dim': hetero_case_template(
    #     train_files = [
    #         '/mnt/nfs/yijing.zhou/data/large_data/lending_club/test_aa_1w_2000.csv',
    #         '/mnt/nfs/yijing.zhou/data/large_data/lending_club/test_bb_1w_8000.csv',
    #         '/mnt/nfs/yijing.zhou/data/large_data/lending_club/test_bb_1w_8000.csv',
    #     ],
    #     test_files = [
    #         '/mnt/nfs/yijing.zhou/data/large_data/lending_club/test_aa_1w_2000.csv',
    #         '/mnt/nfs/yijing.zhou/data/large_data/lending_club/test_bb_1w_8000.csv',
    #         '/mnt/nfs/yijing.zhou/data/large_data/lending_club/test_bb_1w_8000.csv',
    #     ],
    #    feat_infos = [
    #        '/mnt/nfs/yijing.zhou/data/large_data/test_aa_data_attrib.json',
    #        '/mnt/nfs/yijing.zhou/data/large_data/test_bb_data_attrib.json',
    #        None
    #    ],
    #     descs = [
    #         {'id_desc': ['ID'], 'y_desc': ['Status']},
    #         {'id_desc': ['ID'], 'y_desc': []},
    #         {'id_desc': ['ID'], 'y_desc': []},
    #     ],
    #     meta_params = meta_params,
    #     selector = selector,
    #     config = configs,
    #     timeout = 12000
    # ),
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
#             '/mnt/nfs/yijing.zhou/data/shap_finance/train_guest_attrib.json',
#             '/mnt/nfs/yijing.zhou/data/shap_finance/train_host1_attrib.json',
#             '/mnt/nfs/yijing.zhou/data/shap_finance/train_host2_attrib.json',
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
#     ),
#     'hetero_data_united_40w_multi': hetero_case_template_multihost(
#         train_files = [
#             '/mnt/nfs/datasets/data_united/train_hetero_guest.csv',
#             '/mnt/nfs/datasets/data_united/train_hetero_host.csv',
#             '/mnt/nfs/datasets/data_united/train_hetero_host.csv',
#         ],
#         test_files = [
#             '/mnt/nfs/datasets/data_united/test_5t_guest.csv',
#             '/mnt/nfs/datasets/data_united/test_5t_host.csv',
#             '/mnt/nfs/datasets/data_united/test_5t_host.csv',
#         ],
#         feat_infos = [
#             '/mnt/nfs/yijing.zhou/data/data_united/guest_data_attrib.json',
#             '/mnt/nfs/yijing.zhou/data/data_united/host_data_attrib.json',
#             '/mnt/nfs/yijing.zhou/data/data_united/host_data_attrib.json',
#             None
#         ],
#         descs = [
#             {'id_desc': ['ID'], 'y_desc': ['label']},
#             {'id_desc': ['ID'], 'y_desc': []},
#             {'id_desc': ['ID'], 'y_desc': []},
#         ],
#         meta_params = meta_params,
#         selector = selector,
#         config = configs,
#         timeout = 2400
#     ),
# }
##############################################################################

from functools import partial
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
