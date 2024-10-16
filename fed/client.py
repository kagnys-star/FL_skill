from spirl.components.params import get_args
import flwr as fl
from cl_model import *

if __name__ == '__main__':
    client = FS_Clients(args=get_args()).to_client()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)
