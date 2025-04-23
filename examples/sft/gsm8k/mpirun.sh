# hosts.txt contains the list of hosts, 1 per line
# Runpod does some special ssh forwarding so you need to get the IP addresses from /etc/hosts
mpirun --npernode 4 --hostfile hosts.txt bash -c 'echo "Hello, world from $OMPI_COMM_WORLD_NODE_RANK"' -V