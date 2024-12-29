# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "requests",
#     "python-dotenv",
#     "rich",
#     "pyyaml",
# ]
# ///

import json
import os
import sys
import subprocess
import requests
import yaml
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from rich.console import Console
from rich import print as rprint

# Constants
API_URL = "https://api.runpod.io/graphql"
DEFAULT_CONFIG_PATH = "config.yaml"

# Initialize Rich console
console = Console()

# Exception Classes
class ConfigurationError(Exception):
    pass

class DeploymentError(Exception):
    pass

# Helper Functions
def run_graphql_query(query: str, api_key: str) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    url = f"{API_URL}?api_key={api_key}"
    data = json.dumps({"query": query})

    response = requests.post(url, headers=headers, data=data, timeout=30)

    if response.status_code == 401:
        raise DeploymentError("Unauthorized request, please check your API key.")
    if "errors" in response.json():
        raise DeploymentError(response.json()["errors"][0]["message"])
    return response.json()

def get_gpus(api_key: str) -> list:
    query = """
    query GpuTypes {
        gpuTypes {
            id
            displayName
            memoryInGb
            securePrice
            communityPrice
            secureSpotPrice
            communitySpotPrice
        }
    }
    """
    response = run_graphql_query(query, api_key)
    return response["data"]["gpuTypes"]

def create_pod(api_key: str, name: str, image_name: str, gpu_type_id: str) -> dict:
    query = f"""
    mutation {{
        podFindAndDeployOnDemand(
            input: {{
                name: "{name}",
                imageName: "{image_name}",
                gpuTypeId: "{gpu_type_id}",
                cloudType: ALL,
                startSsh: true,
                supportPublicIp: true,
                gpuCount: 1,
                containerDiskInGb: 10,
                volumeInGb: 0,
                minVcpuCount: 1,
                minMemoryInGb: 1,
                ports: "22/tcp,8888/http"
            }}
        ) {{
            id
            name
            imageName
            desiredStatus
        }}
    }}
    """

    response = requests.post(
        f"{API_URL}?api_key={api_key}",
        json={"query": query},
        headers={"Content-Type": "application/json"},
        timeout=30
    )

    if response.status_code != 200 or "errors" in response.json():
        raise DeploymentError(f"Failed to create pod: {response.text}")

    return response.json()["data"]["podFindAndDeployOnDemand"]

def get_pod_status(api_key: str) -> list:
    query = """
    query {
        myself {
            pods {
                id
                name
                machineId
                desiredStatus
                imageName
                costPerHr
                gpuCount
                machine {
                    gpuDisplayName
                    podHostId
                }
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                    }
                }
            }
        }
    }
    """
    response = run_graphql_query(query, api_key)
    return response["data"]["myself"]["pods"]

def read_config(config_path: str) -> Dict[str, Any]:
    if not os.path.isfile(config_path):
        raise ConfigurationError(f"Configuration file not found at {config_path}")

    with open(config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Error parsing YAML file: {str(e)}")

def validate_config(config: Dict[str, Any]):
    required_fields = ['provider', 'gpu', 'image', 'script', 'ssh', 'upload']
    for field in required_fields:
        if field not in config:
            raise ConfigurationError(f"Missing required field in config: {field}")

    # Validate nested fields
    if 'path' not in config['script']:
        raise ConfigurationError("Missing 'path' under 'script' in config.")
    if 'local_dir' not in config['upload'] or 'remote_dir' not in config['upload']:
        raise ConfigurationError("Missing 'local_dir' or 'remote_dir' under 'upload' in config.")
    if 'key_path' not in config['ssh']:
        raise ConfigurationError("Missing 'key_path' under 'ssh' in config.")

def set_executable(file_path: str):
    """
    Ensures that the specified file has execute permissions.
    """
    if not os.path.isfile(file_path):
        raise DeploymentError(f"File not found: {file_path}")

    # Check if the file is already executable
    if os.access(file_path, os.X_OK):
        rprint(f"[green]File '{file_path}' is already executable.[/green]")
    else:
        # Set execute permissions
        try:
            st = os.stat(file_path)
            os.chmod(file_path, st.st_mode | 0o111)
            rprint(f"[green]Set execute permissions for '{file_path}'.[/green]")
        except Exception as e:
            raise DeploymentError(f"Failed to set execute permissions for '{file_path}': {str(e)}")

def upload_files_to_pod(api_key: str, pod: Dict[str, Any], local_dir: str, remote_dir: str, ssh_key_path: str, ssh_ip: str, ssh_port: int):
    # Ensure remote directory exists
    ssh_command = [
        "ssh",
        "-i", ssh_key_path,
        "-p", str(ssh_port),
        f"root@{ssh_ip}",
        f"mkdir -p {remote_dir}"
    ]
    try:
        subprocess.run(ssh_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        rprint(f"[green]Ensured remote directory exists: {remote_dir}[/green]")
    except subprocess.CalledProcessError as e:
        raise DeploymentError(f"Failed to create remote directory: {e.stderr}")

    # Use SCP to upload the contents of the local directory
    scp_command = [
        "scp",
        "-i", ssh_key_path,
        "-P", str(ssh_port),
        "-r", os.path.join(local_dir, '.'),  # Upload contents, not the directory itself
        f"root@{ssh_ip}:{remote_dir}"
    ]
    rprint(f"[green]Uploading contents of '{local_dir}' to '{remote_dir}' on the pod...[/green]")
    try:
        subprocess.run(scp_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        rprint("[green]Files uploaded successfully![/green]")
    except subprocess.CalledProcessError as e:
        raise DeploymentError(f"SCP upload failed: {e.stderr}")

def execute_script_on_pod(ssh_key_path: str, ssh_ip: str, ssh_port: int, script_path: str, env_vars: Dict[str, str]):
    # Construct environment variables export command
    env_exports = ' && '.join([f'export {key}="{value}"' for key, value in env_vars.items()])

    # Construct the SSH command to execute the script
    ssh_command = [
        "ssh",
        "-i", ssh_key_path,
        "-p", str(ssh_port),
        f"root@{ssh_ip}",
        f"{env_exports} && bash {script_path}"
    ]

    rprint(f"[green]Executing script: {script_path} on the pod...[/green]")
    try:
        process = subprocess.Popen(ssh_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Stream stdout
        for line in iter(process.stdout.readline, ''):
            rprint(f"[green]{line.rstrip()}[/green]")

        # Stream stderr
        for line in iter(process.stderr.readline, ''):
            rprint(f"[yellow]{line.rstrip()}[/yellow]")

        process.stdout.close()
        process.stderr.close()
        return_code = process.wait()
        if return_code != 0:
            raise DeploymentError(f"Script execution failed with exit code {return_code}")
        rprint("[green]Script executed successfully![/green]")
    except subprocess.CalledProcessError as e:
        raise DeploymentError(f"Script execution failed: {e.stderr}")

def terminate_pod(api_key: str, pod_id: str):
    query = f"""
    mutation {{
        podTerminate(input: {{ podId: "{pod_id}" }})
    }}
    """
    response = run_graphql_query(query, api_key)
    return response

def deploy_pod_from_config(api_key: str, config: Dict[str, Any]) -> Dict[str, Any]:
    gpu_id = config.get('gpu_id')
    gpu_display_name = config.get('gpu')
    image_name = config['image']
    script_path = config['script']['path']
    env_vars = config['script'].get('env', {})
    local_dir = config['upload']['local_dir']
    remote_dir = config['upload']['remote_dir']
    ssh_key_path = os.path.expanduser(config['ssh']['key_path'])

    # Validate SSH key path
    if not os.path.isfile(ssh_key_path):
        raise DeploymentError(f"SSH key not found at {ssh_key_path}")

    # Set execute permissions for the script
    full_script_path = os.path.join(local_dir, os.path.basename(script_path))
    set_executable(full_script_path)

    if gpu_id:
        selected_gpu_id = gpu_id
        rprint(f"[blue]Using GPU ID from config: {selected_gpu_id}[/blue]")
    else:
        # Get available GPUs
        gpus = get_gpus(api_key)

        selected_gpu_id = None
        gpu_display_name_lower = gpu_display_name.lower()
        for gpu in gpus:
            if gpu_display_name_lower in gpu['displayName'].lower() or gpu_display_name_lower in gpu['id'].lower():
                selected_gpu_id = gpu['id']
                break
        if not selected_gpu_id:
            raise DeploymentError(f"GPU type '{gpu_display_name}' not found.")

    # Create a unique pod name
    import time
    pod_name = f"auto-deploy-{int(time.time())}"

    # Initialize pod_id for tracking
    pod_id = None

    try:
        # Create pod
        pod = create_pod(api_key, pod_name, image_name, selected_gpu_id)
        pod_id = pod['id']
        rprint(f"[green]Pod '{pod_name}' created successfully with ID: {pod_id}[/green]")

        # Wait for pod to be ready with runtime information
        rprint("[yellow]Waiting for pod to be in RUNNING state with runtime information...[/yellow]")
        pod = wait_for_pod_running(api_key, pod_id)

        # Extract SSH details
        runtime = pod.get('runtime', {})
        ports = runtime.get('ports', [])
        ssh_port_info = next((port for port in ports if port['privatePort'] == 22 and port['isIpPublic']), None)
        if not ssh_port_info:
            raise DeploymentError("No public SSH port found for the pod.")

        ssh_ip = ssh_port_info['ip']
        ssh_port = ssh_port_info['publicPort']

        # Upload files
        upload_files_to_pod(api_key, pod, local_dir, remote_dir, ssh_key_path, ssh_ip, ssh_port)

        # Execute the bash script
        execute_script_on_pod(ssh_key_path, ssh_ip, ssh_port, script_path, env_vars)

        rprint("[green]Deployment completed successfully![/green]")
        return pod

    except Exception as e:
        rprint(f"[red]Error during deployment: {str(e)}[/red]")
        if pod_id:
            rprint("[yellow]Attempting to terminate the pod due to deployment error...[/yellow]")
            try:
                terminate_pod(api_key, pod_id)
                rprint("[green]Pod terminated successfully.[/green]")
            except Exception as termination_error:
                rprint(f"[red]Failed to terminate pod: {str(termination_error)}[/red]")
        sys.exit(1)

def wait_for_pod_running(api_key: str, pod_id: str, timeout: int = 120, interval: int = 15) -> Dict[str, Any]:
    """
    Waits until the pod reaches the RUNNING state and runtime information is available.

    Args:
        api_key (str): RunPod API key.
        pod_id (str): ID of the pod to monitor.
        timeout (int): Maximum time to wait in seconds.
        interval (int): Interval between status checks in seconds.

    Returns:
        dict: Updated pod information.

    Raises:
        DeploymentError: If the pod does not reach RUNNING state or runtime information within the timeout.
    """
    import time

    elapsed = 0
    while elapsed < timeout:
        pods = get_pod_status(api_key)
        pod = next((p for p in pods if p['id'] == pod_id), None)
        if not pod:
            raise DeploymentError(f"Pod with ID {pod_id} not found.")

        status = pod.get('desiredStatus', 'UNKNOWN')
        rprint(f"Current pod status: {status}")

        if status.lower() == 'running':
            runtime = pod.get('runtime')
            if runtime:
                return pod
            else:
                rprint("[yellow]Pod is running but runtime information is not yet available. Waiting...[/yellow]")
        elif status.lower() in ['failed', 'terminated']:
            raise DeploymentError(f"Pod has entered an invalid state: {status}")

        time.sleep(interval)
        elapsed += interval

    raise DeploymentError("Timed out waiting for pod to reach RUNNING state with runtime information.")


def automate_workflow(config_path: str):
    try:
        config = read_config(config_path)
        validate_config(config)
    except ConfigurationError as e:
        rprint(f"[red]Configuration Error: {str(e)}[/red]")
        sys.exit(1)

    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        rprint("[red]Error: RUNPOD_API_KEY environment variable not found[/red]")
        sys.exit(1)

    try:
        pod = deploy_pod_from_config(api_key, config)
    except DeploymentError as e:
        rprint(f"[red]Deployment Error: {str(e)}[/red]")
        sys.exit(1)

def main():
    load_dotenv()
    config_path = DEFAULT_CONFIG_PATH

    # Check if config.yaml exists
    if not os.path.isfile(config_path):
        rprint(f"[red]Error: '{config_path}' not found in the current directory.[/red]")
        rprint("[yellow]Please create a 'config.yaml' file based on the provided template.[/yellow]")
        sys.exit(1)

    # Start the automated workflow
    automate_workflow(config_path)

if __name__ == "__main__":
    main()
