# Image creation

1. boot AWS instance
2. run aws_deploy.sh
3. shut down instance
4. create image

Please be considerate of others' bandwidth and create an image rather than running aws_deploy.sh each time you boot an instance.

# Instance selection

qFit works well on c5 instances.
A c5.12xlarge is appropriately-sized for moderately-sized proteins.

# AWS ParallelCluster

qFit has been tested with AWS ParallelCluster using the Slurm scheduler and on-demand instances.
Note that the AWSBatch scheduler is not compatible with custom images.
