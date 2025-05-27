# WAFL-MLP-Asynchronous
-------
Summary
-------
This is an asynchronous implementation of the WAFL-MLP project.
(https://github.com/jo2lxq/wafl/tree/main/WAFL-MLP)
All the devices are assigned identifiers from 0 onwards 
(Node0.py is 'device' 0, Node1.py is 'device' 1, and so on).

1) Number of Devices (can be expanded): 4
2) Topology used (can be modified): Line Topology
3) Communication Protocol: TCP/IP
4) Scope of IP Addresses (can be expanded): Currently just LOCALHOST
5) Communication Scheme: 
    Each python script spawns a separate server thread for sharing model parameters
    in serialized form whenever another peer device requests data from it. Since
    only a single IP address is being used at present, all the nodes are dynamically
    assigned port numbers starting with 65430 (for node 0, it is 65430 + i for Node i).

-------
Communication and Logging-related codes:
-------
1) MDLREQ: Message for requesting model parameters.
2) REQUESTING, RECV, and RETRY: Possible states of the client requesting parameters.
3) BOUND, SENT and WAITING: Possible states of the server threads.


Other parameters such as the number of training epochs for both phases
can be modified in the configuration section of the individual NodeX.py files.
The process for the generation and the splitting of the overall MNIST dataset into separate datasets, one for each Node, for simplicity, has been kept the same as in the original MLP project (along with the non-independent-and-identically-distributed-data filter configuration and usage section).
However, separation has been introduced by segmenting the datasets and having the nodes load their assigned datasets independently.

-------
Process of Execution:
-------
1) Run the requirements.txt file and ensure compatibility.
2) Run the Initializer.py file (after loading the data folder with the sub-directories
    from the WAFL-MLP standard project).
    Initializer.py prepares, segments and stores the datasets for the nodes in serialized form.
3) Run the N Node python scripts in separate terminal windows.
4) Run the accuracy trend and confusion matrix generation files for evaluation.

-------
Current Priorities:
-------
1) Simplified modification of project parameters.
2) Automated node python script generation and deployment.
3) Standardization of the communication scheme and expansion to
    enable transfer of other important information between the nodes.


