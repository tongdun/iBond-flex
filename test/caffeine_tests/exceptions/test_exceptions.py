import json
import os
from multiprocessing import Process

from caffeine.utils.exceptions import *


def run():
    try:
        raise EmptyDataError
    except EmptyDataError as error:
        error.report()


if __name__ == '__main__':
    json_path = '../../caffeine/utils/exceptions'
    all_filenames = os.listdir(json_path)
    filenames = [i for i in all_filenames if os.path.splitext(i)[1] == '.json']

    exception_config = {}
    for filename in filenames:
        with open(os.path.join(json_path, filename), 'r') as f:
            exception_config.update(json.load(f))
    run()
