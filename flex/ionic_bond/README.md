# FLEX Bond

This is a toy implementation of framework for message delivery.

## How to install

```python
pip install .
```

## Start server

```python
python message_server.py
```

## How to config federation

see flex_bond/configuration.py

## Demo codes

### Init your communicate configuration.

```python
import json
from flex_bond import commu

with open('config.json') as conf_file:
    config = json.load(conf_file)

commu.init(config) # Only once for every process.

#Some useful api.
commu.get_job_id()
commu.get_local_id()
commu.get_fedration_members()
```

### Use VariableChannel

```python
from flex_bond.channel import VariableChannel

remote_id = "zhibang-d-014011"
var_chan = VariableChannel(
    name="Test_var",
    remote_id=remote_id)

# How to send a var
var = 123
var_chan.send(var)

# How to receive a var
remote_var = var_chan.recv()

# How to swap a var (send my var and get remote var at the same time.)
remote_var = var_chan.swap(var)

# This is a context mannager style usage,
with VariableChannel(name="Test_var",
                     remote_id=remote_id) as var_chan:
    var_chan.send(var)
    remote_var = var_chan.recv()
```

### Use BroadcastChannel

```python
from flex_bond.channel import make_broadcast_channel

var_chan = make_broadcast_channel(
    name="Test_var",
    root="zhibang-d-014012",
    remote_group=["zhibang-d-014010",
                  "zhibang-d-014011"])

# On root process
var = 123
var_chan.broadcast(var)
remote_var = var_chan.gather() # remote_var is a list of len(remote_group)
var_chan.scatter([a, b])

# On remote_group process
var = 123
root_var = var_chan.broadcast()
var_chan.gather(var)
root_var = var_chan.scatter() # if local_id is "zhibang-d-014010" root_var==a
```

### Use your own tag instead of auto_offset

```python
from flex_bond.channel import VariableChannel

remote_id = "zhibang-d-014011"
var_chan = VariableChannel(
    name="Test_var",
    remote_id=remote_id,
    auto_offset=False)
my_tag = 'aaa'
# How to send a var
var = 123
var_chan.send(var, tag=my_tag)

# How to receive a var
remote_var = var_chan.recv(tag=my_tag)

# How to swap a var (send my var and get remote var at the same time.)
remote_var = var_chan.swap(var, tag=my_tag)
```
