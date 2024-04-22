import yaml
import torch
from torchvision.transforms import *

#import sys
#sys.path.append("../credit")
from credit.data import CONUS404Dataset
#from data import CONUS404Dataset
from credit.transforms import ToTensor

config = "../config/conus404.yml"

with open(config) as cf:
    conf = yaml.load(cf, Loader=yaml.FullLoader)

print(conf)

#varlist = ["U850", "V850", "Q850", "U250", "U500", "V250", "V500"]
varlist = ["U850", "V850", "U500"]

transform = Compose([
    #NormalizeState(conf["data"]["mean_path"],conf["data"]["std_path"]),
    ToTensor(history_len=conf['data']['history_len'], forecast_len=conf['data']['forecast_len'], variables=[], surface_variables=varlist)
])

#todo: transform = Compose([ToTensor(conf)])

print(transform)


dataset = CONUS404Dataset(
    zarrpath="/glade/campaign/ral/risc/DATA/conus404/zarr",
    varnames = varlist,
    history_len=conf['data']['history_len'],
    forecast_len=conf['data']['forecast_len'],
    transform = transform
)

print(dataset)
#print(dataset.__getitem__(0))



# Dataloader

train_batch_size = 2

train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=False,
        #sampler=train_sampler,
        pin_memory=True,
        #persistent_workers=True if thread_workers > 0 else False,
        num_workers=8,
        drop_last=True
    )

print(train_loader)

#for batch in train_loader:
#    print(batch)
#    raise
