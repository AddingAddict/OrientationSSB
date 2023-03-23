## Network model for development of orientation selective Receptive Fields

TBC

### How to run the simulation

The network is implementated in tensorflow and numpy. The numpy implementation can be found in the files with a _np suffix.

The two layer network can be started on the cluster using 

> python start_twolayer_jobs.py

Giving a list of parameters would start a list of jobs on the cluster.

For testing purposes the two layer network, as well as the individual layers can be run locally via

(running Layer 2/3)

> python run_layer23.py [args]

(running Layer 4)

> python run_onelayer.py [args]

(Running both Layers)

> python run_twolayer.py [args]


The default network configuration can be found in data/params_default.yaml. The network configuration following Antolik et al 2013 can be found in data/params_antolik_etal.yaml.

In tools/parse_args.py the list of arguments to give to the network simulation can be found.


