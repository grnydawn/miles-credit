# How to add a new postblock

In this example we will be going over adding a new postblock, `Foo` to CREDIT.


## Create code for new postblock

One can add a new class to `credit/postblock.py` or define a new module and import it into `credit/postblock.py`. See `credit/skebs.py` for an example of the latter.

The parser will add the `data` and `model` fields from the main config to `post_conf`. Inside the class `Foo` you will be able to access these.



```python
from torch import nn

class Foo(nn.Module):
    def __init__(self, post_conf):
        super().__init__()
        self.bar = post_conf["foo"]["bar"]
        
        # accessing data or model conf
        lead_time_periods = post_conf["data"]["lead_time_periods"] 

    def forward(self, x):
        # x will be a dictionary of the previous state x and y_pred
        # of the model up to this point
        # both tensors will be in the transformed space

        y_pred = x["y_pred"]
        x_prev = x["x"]

        # do stuff ...

        x["y_pred"] = y_pred   # pack back into the dictionary
        return x

```

## Define config fields

Inside of your config you will need to add a new field for your postblock. 

```yaml
model:
    ...
    post_conf:
        ...
        foo:
            activate: True
            bar: 1.0
            ...
```

## Add to postblock module

Inside `credit/postblock.py`, append your postblock to the list of postblock operations `self.operations`, the order that you want it.

```python
from credit.skebs import SKEBS

class PostBlock(nn.Module):
    def __init__(self, post_conf):
        ...
        # SKEBS
        if post_conf["skebs"]["activate"]:
            logger.info("SKEBS registered")
            opt = SKEBS(post_conf)
            self.operations.append(opt)
        ...
        if post_conf["foo"]["activate"]:
            logger.info("foo registered")
            opt = Foo(post_conf)
            self.operations.append(opt)
        ...
```


