import subprocess

def main():
    print(subprocess.check_output(["lscpu"]).decode('utf-8'))  


if __name__ == "__main__":
    main()