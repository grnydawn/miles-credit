import yaml
import torch
from torchvision.transforms import *

#import sys
#sys.path.append("../credit")
from credit.data404 import CONUS404Dataset
#from data import CONUS404Dataset
from credit.transforms404 import ToTensor

config = "/glade/work/mcginnis/ML/GWC/miles-credit/config/conus404.yml"

with open(config) as cf:
    conf = yaml.load(cf, Loader=yaml.FullLoader)

print("\nconf:\n", conf)

transform = Compose([
    #NormalizeState(conf["data"]["mean_path"],conf["data"]["std_path"]),
    ToTensor(conf)
])


print("\ntransform:\n", transform)


dataset = CONUS404Dataset(
    zarrpath=conf['data']['zarrpath'], #"/glade/campaign/ral/risc/DATA/conus404/zarr",
    varnames = conf['data']['variables'],
    history_len=conf['data']['history_len'],
    forecast_len=conf['data']['forecast_len'],
    start = "2000-01-01",
    finish = "2001-01-01",
    transform = transform
)

print("\ndataset\n", dataset)
print("\nlen(dataset)\n", len(dataset))
#print("\ndataset[0]\n", dataset.__getitem__(0))



# Dataloader

train_batch_size = 2

train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=train_batch_size,
        shuffle=False,
        #sampler=train_sampler,
        pin_memory=True,
        #persistent_workers=True if thread_workers > 0 else False,
        num_workers=1,
        drop_last=True
    )

print("\ntrain_loader:\n", train_loader)

#for batch in train_loader:
#    print("\nbatch:\n", batch)
#    raise
