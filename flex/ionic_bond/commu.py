"""main commu wrapper api
"""
#
#  Copyright 2020 The FLEX Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import sys
import threading

UnitTest = False
LocalTest = False

this = sys.modules[__name__]


def init(*args, **kwargs):
    """
    Init
    """
    print("=" * 100)
    print(f"Initializing federation with config={args},{kwargs}")
    print(f"UnitTest={UnitTest}, LocalTest={LocalTest}")
    print("=" * 100)
    if LocalTest:
        from .ion_local import Ion
        if not hasattr(this, 'runtime_instance'):
            this.runtime_instance = None
        if this.runtime_instance is None:
            this.runtime_instance = Ion(*args, **kwargs)
    elif UnitTest:
        if not hasattr(this, 'runtime_instance'):
            this.runtime_instance = {}
        from .dummpy_ion import Ion
        if threading.get_ident not in this.runtime_instance:
            this.runtime_instance[threading.get_ident] = Ion(*args, **kwargs)
    else:
        from .ion import Ion
        if not hasattr(this, 'runtime_instance'):
            this.runtime_instance = None
        if this.runtime_instance is None:
            this.runtime_instance = Ion(*args, **kwargs)


def check():
    """
    make sure inited
    """
    if UnitTest:
        if threading.get_ident not in this.runtime_instance:
            raise RuntimeError("Init Commu before using it.")
    else:
        if this.runtime_instance is None:
            raise RuntimeError("Init Commu before using it.")

def get_instance():
    if UnitTest:
        return this.runtime_instance[threading.get_ident]
    else:
        return this.runtime_instance

def get_local_id():
    """
    get local id
    """
    check()
    return get_instance().local_id

def get_local_role():
    """
    get local role
    """
    check()
    return get_instance().role

def get_job_id():
    """
    get job id
    """
    check()
    return get_instance().job_id

def get_federation():
    """
    get federation
    """
    check()
    return get_instance().federation

def get_federation_members():
    """
    get federation_members
    """
    check()
    return get_instance().federation_members

def send(*args, **kwargs):
    """
    warpped send
    """
    check()
    return get_instance().send(*args, **kwargs)

def recv(*args, **kwargs):
    """
    wrapped recv
    """
    check()
    return get_instance().recv(*args, **kwargs)
