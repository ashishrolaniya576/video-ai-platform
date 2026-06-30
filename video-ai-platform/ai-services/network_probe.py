import socket
import os
import sys

def check_tcp(host, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(2.0)
        s.connect((host, port))
        s.close()
        return True
    except Exception as e:
        return str(e)

if __name__ == "__main__":
    targets = [
        ("localhost", 8554),
        ("127.0.0.1", 8554),
        ("host.docker.internal", 8554),
        ("172.17.0.1", 8554)
    ]
    
    print("=== NETWORK REACHABILITY DIAGNOSTICS ===")
    print(f"Hostname: {socket.gethostname()}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Python: {sys.executable}")
    
    for host, port in targets:
        res = check_tcp(host, port)
        status = "Connected" if res is True else f"Connection refused ({res})"
        print(f"tcp://{host}:{port} -> {status}")
