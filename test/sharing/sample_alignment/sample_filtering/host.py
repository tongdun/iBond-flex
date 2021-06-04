import numpy as np

from flex.api import make_protocol
from flex.crypto.id_filter.id_filter import IDFilter
from flex.constants import BF_SF

from test.fed_config_example import fed_conf_host


def test():
    federal_info = fed_conf_host

    sec_param = []

    algo_param = {
        "log2_len": 23
    }

    # inputdata
    input_data = list(range(6100))

    # step1 bloomfilter map of inputdata
    fltr = IDFilter(algo_param['log2_len'])
    fltr.update(input_data)
    Map = {}
    for i in input_data:
        t = i % (1 << algo_param['log2_len'])
        Map.setdefault(t, []).append(i)

    # compute intersect filter
    interset_fltr = make_protocol(BF_SF, federal_info, sec_param)
    sf_res = interset_fltr.intersect(fltr)
    inter_fltr = np.unpackbits(sf_res.filter)

    # step9 return interset result
    interset = set()
    inter_lis2 = np.where(inter_fltr == 1)[0].tolist()
    for item in inter_lis2:
        if item in Map.keys():
            interset.update(Map[item])
    print(interset)


if __name__ == '__main__':
    test()
