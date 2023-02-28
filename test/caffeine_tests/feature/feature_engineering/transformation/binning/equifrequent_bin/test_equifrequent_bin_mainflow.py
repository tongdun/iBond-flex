from functools import partial

from flex.constants import (OTP_PN_FL, SS_STATS, COUNTER_MERGE, HE_CHI_IV_FBI,
            CORR_FFS, KS_FFS, SCC_FFS, FMC, OTP_CS_BSR, HE_CHI_FB, HE_DT_FB)

from caffeine.feature.feature_engineering.transformation.guest import FeatureTransformGuest
from caffeine.feature.feature_engineering.transformation.host import FeatureTransformHost
from caffeine.feature.feature_engineering.transformation.coordinator import FeatureTransformCoord
from caffeine_tests.feature.feature_engineering.transformation.hetero_mainflow import (hetero_case_template, 
        hetero_run_processes, hetero_case_template_multihost,
        multihost_hetero_run_processes)

##############################################################################
case_prefix = 'transform_equifrequent_bin_'

selector = [
    FeatureTransformGuest,
    FeatureTransformHost,
    FeatureTransformCoord
]


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
        'pipeline': ['equifrequent_bin'],
        'configs': {
            'equifrequent_bin': {
                'equal_num_bin': 30, 
            },
        },
        'process_method': 'local',
        'common_params':{
            'use_multiprocess': False,
        }
    },
    'security_param': None,
    'predict_param': None
}

test_cases = {
    'hetero_dtype': hetero_case_template(
        train_files = [
            '/mnt/nfs/datasets/shap_finance/multi_host/train_guest_dtype.csv',
            '/mnt/nfs/datasets/shap_finance/multi_host/train_host1.csv',
            '/mnt/nfs/datasets/shap_finance/multi_host/train_host2.csv',
        ],
        test_files = [
            '/mnt/nfs/datasets/shap_finance/multi_host/train_guest_dtype.csv',
            '/mnt/nfs/datasets/shap_finance/multi_host/test_host1.csv',
            '/mnt/nfs/datasets/shap_finance/multi_host/test_host2.csv',
        ],
        feat_infos = [
            '/mnt/nfs/yijing.zhou/data/shap_finance/train_guest_dtype_attrib.json',
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
    ),
    'hetero_large_data_1w_dim': hetero_case_template(
        train_files = [
            '/mnt/nfs/yijing.zhou/data/large_data/lending_club/test_aa_1w_2000.csv',
            '/mnt/nfs/yijing.zhou/data/large_data/lending_club/test_bb_1w_8000.csv',
            '/mnt/nfs/yijing.zhou/data/large_data/lending_club/test_bb_1w_8000.csv',
        ],
        test_files = [
            '/mnt/nfs/yijing.zhou/data/large_data/lending_club/test_aa_1w_2000.csv',
            '/mnt/nfs/yijing.zhou/data/large_data/lending_club/test_bb_1w_8000.csv',
            '/mnt/nfs/yijing.zhou/data/large_data/lending_club/test_bb_1w_8000.csv',
        ],
        feat_infos = [
           '/mnt/nfs/yijing.zhou/data/large_data/test_aa_data_attrib.json',
           '/mnt/nfs/yijing.zhou/data/large_data/test_bb_data_attrib.json',
           dict()
        ],
        descs = [
            {'id_desc': ['ID'], 'y_desc': ['Status']},
            {'id_desc': ['ID'], 'y_desc': []},
            {'id_desc': ['ID'], 'y_desc': []},
        ],
        meta_params = meta_params,
        selector = selector,
        config = configs,
        timeout = 1200
    ),
}

test_cases_multihost = {
    'hetero_new_finance_multi': hetero_case_template_multihost(
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
            '/mnt/nfs/yijing.zhou/data/shap_finance/train_host2_attrib.json',
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
    'hetero_data_united_40w_multi': hetero_case_template_multihost(
        train_files = [
            '/mnt/nfs/datasets/data_united/train_hetero_guest.csv',
            '/mnt/nfs/datasets/data_united/train_hetero_host.csv',
            '/mnt/nfs/datasets/data_united/train_hetero_host.csv',
        ],
        test_files = [
            '/mnt/nfs/datasets/data_united/test_5t_guest.csv',
            '/mnt/nfs/datasets/data_united/test_5t_host.csv',
            '/mnt/nfs/datasets/data_united/test_5t_host.csv',
        ],
        feat_infos = [
            '/mnt/nfs/yijing.zhou/data/data_united/guest_data_attrib.json',
            '/mnt/nfs/yijing.zhou/data/data_united/host_data_attrib.json',
            '/mnt/nfs/yijing.zhou/data/data_united/host_data_attrib.json',
        ],
        descs = [
            {'id_desc': ['ID'], 'y_desc': ['label']},
            {'id_desc': ['ID'], 'y_desc': []},
            {'id_desc': ['ID'], 'y_desc': []},
        ],
        meta_params = meta_params,
        selector = selector,
        config = configs,
        timeout = 2400
    ),
}
##############################################################################

from functools import partial
for case_name, case in test_cases.items():
    locals()[f'test_{case_prefix}{case_name}'] = partial(multihost_hetero_run_processes, {case_name: case})

for case_name, case in test_cases_multihost.items():
    locals()[f'test_{case_prefix}{case_name}'] = partial(multihost_hetero_run_processes, {case_name: case})



if __name__ == '__main__':
    for case_name, case in test_cases.items():
        eval(f'test_{case_prefix}{case_name}()')
    for case_name, case in test_cases_multihost.items():
        eval(f'test_{case_prefix}{case_name}()')
