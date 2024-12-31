# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "requests",
#     "python-dotenv",
#     "rich",
#     "pyyaml",
#     "cryptography",
#     "logging",
# ]
# ///

import logging
import json
import os
import sys
import subprocess
import requests
import yaml
import argparse
import signal
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from pathlib import Path

# Import logging setup
from logging_setup import setup_logging, get_logger

# Constants
API_URL = "https://api.runpod.io/graphql"
DEFAULT_CONFIG_PATH = "config.yaml"
DEFAULT_PROJECTS_DIR = "projects"

# Exception Classes
class ConfigurationError(Exception):
    pass

class DeploymentError(Exception):
    pass

def ensure_log_directory(log_file_path):
    log_dir = Path(log_file_path).parent
    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

def encrypt_env(config: dict) -> tuple[bytes, bytes]:
    needed_vars = {var[2:-1]: os.getenv(var[2:-1])
                  for var in config['script']['env'].values()
                  if isinstance(var, str) and var.startswith('${') and var.endswith('}')}
    key = Fernet.generate_key()
    return Fernet(key).encrypt(json.dumps(needed_vars).encode()), key

# Helper Functions
def run_graphql_query(query: str, api_key: str, logger: logging.Logger) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    url = f"{API_URL}?api_key={api_key}"
    data = json.dumps({"query": query})

    response = requests.post(url, headers=headers, data=data, timeout=30)

    if response.status_code == 401:
        logger.error("Unauthorized request, please check your API key.")
        raise DeploymentError("Unauthorized request, please check your API key.")
    if "errors" in response.json():
        error_message = response.json()["errors"][0]["message"]
        logger.error(f"GraphQL Error: {error_message}")
        raise DeploymentError(error_message)
    return response.json()

def get_gpus(api_key: str, logger: logging.Logger) -> list:
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
    response = run_graphql_query(query, api_key, logger)
    logger.debug("Fetched GPU types successfully.")
    return response["data"]["gpuTypes"]

def create_pod(api_key: str, name: str, image_name: str, gpu_type_id: str, logger: logging.Logger) -> dict:
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
                containerDiskInGb: 50,
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
        error_text = response.text
        logger.error(f"Failed to create pod: {error_text}")
        raise DeploymentError(f"Failed to create pod: {error_text}")

    pod = response.json()["data"]["podFindAndDeployOnDemand"]
    logger.info(f"Pod '{name}' created successfully with ID: {pod['id']}")
    return pod

def get_pod_status(api_key: str, logger: logging.Logger) -> list:
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
    response = run_graphql_query(query, api_key, logger)
    logger.debug("Fetched pod statuses successfully.")
    return response["data"]["myself"]["pods"]

def read_config(config_path: str, logger: logging.Logger) -> Dict[str, Any]:
    if not os.path.isfile(config_path):
        logger.error(f"Configuration file not found at {config_path}")
        raise ConfigurationError(f"Configuration file not found at {config_path}")

    with open(config_path, "r") as file:
        try:
            config = yaml.safe_load(file)

            # Ensure script paths are relative to the project directory
            project_dir = os.path.dirname(config_path)
            config["script"]["path"] = os.path.join(project_dir, config["script"]["path"])
            config["upload"]["local_dir"] = project_dir
            logger.debug("Configuration file read and paths adjusted successfully.")
            return config
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file: {str(e)}")
            raise ConfigurationError(f"Error parsing YAML file: {str(e)}")

def validate_config(config: Dict[str, Any], logger: logging.Logger):
    required_fields = ['provider', 'gpu', 'image', 'script', 'ssh', 'upload', 'budget']
    for field in required_fields:
        if field not in config:
            logger.error(f"Missing required field in config: {field}")
            raise ConfigurationError(f"Missing required field in config: {field}")

    # Validate nested fields
    if 'path' not in config['script']:
        logger.error("Missing 'path' under 'script' in config.")
        raise ConfigurationError("Missing 'path' under 'script' in config.")
    if 'local_dir' not in config['upload'] or 'remote_dir' not in config['upload']:
        logger.error("Missing 'local_dir' or 'remote_dir' under 'upload' in config.")
        raise ConfigurationError("Missing 'local_dir' or 'remote_dir' under 'upload' in config.")
    if 'key_path' not in config['ssh']:
        logger.error("Missing 'key_path' under 'ssh' in config.")
        raise ConfigurationError("Missing 'key_path' under 'ssh' in config.")
    if 'budget' in config:
        if 'max_dollars' not in config['budget'] or 'max_hours' not in config['budget']:
            logger.error("Missing 'max_dollars' or 'max_hours' under 'budget' in config.")
            raise ConfigurationError("Missing 'max_dollars' or 'max_hours' under 'budget' in config.")

    logger.debug("Configuration validated successfully.")

def set_executable(file_path: str, logger: logging.Logger):
    """
    Ensures that the specified file has execute permissions.
    """
    if not os.path.isfile(file_path):
        logger.error(f"File not found: {file_path}")
        raise DeploymentError(f"File not found: {file_path}")

    # Check if the file is already executable
    if os.access(file_path, os.X_OK):
        logger.info(f"File '{file_path}' is already executable.")
    else:
        # Set execute permissions
        try:
            st = os.stat(file_path)
            os.chmod(file_path, st.st_mode | 0o111)
            logger.info(f"Set execute permissions for '{file_path}'.")
        except Exception as e:
            logger.error(f"Failed to set execute permissions for '{file_path}': {str(e)}")
            raise DeploymentError(f"Failed to set execute permissions for '{file_path}': {str(e)}")

def upload_files_to_pod(api_key: str, pod: Dict[str, Any], local_dir: str, remote_dir: str, ssh_key_path: str, ssh_ip: str, ssh_port: int, logger: logging.Logger):
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
        logger.info(f"Ensured remote directory exists: {remote_dir}")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create remote directory: {e.stderr}")
        raise DeploymentError(f"Failed to create remote directory: {e.stderr}")

    # Use SCP to upload the contents of the local directory
    scp_command = [
        "scp",
        "-i", ssh_key_path,
        "-P", str(ssh_port),
        "-r", os.path.join(local_dir, '.'),  # Upload contents, including run_script.sh and script.py
        f"root@{ssh_ip}:{remote_dir}"
    ]
    logger.info(f"Uploading contents of '{local_dir}' to '{remote_dir}' on the pod...")
    try:
        subprocess.run(scp_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        logger.info("Files uploaded successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"SCP upload failed: {e.stderr}")
        raise DeploymentError(f"SCP upload failed: {e.stderr}")

def execute_script_on_pod(ssh_key_path: str, ssh_ip: str, ssh_port: int, script_path: str, env_vars: Dict[str, str], remote_dir: str, logger: logging.Logger):
    # Construct environment variables export command
    env_exports = ' && '.join([f'export {key}="{value}"' for key, value in env_vars.items()])

    # Change directory to the remote workspace and execute the script
    ssh_command = [
        "ssh",
        "-i", ssh_key_path,
        "-p", str(ssh_port),
        f"root@{ssh_ip}",
        f"cd {remote_dir} && {env_exports} && bash {script_path}"
    ]

    logger.info(f"Executing script: {script_path} in remote directory: {remote_dir}")
    try:
        process = subprocess.Popen(ssh_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Stream stdout
        for line in iter(process.stdout.readline, ''):
            logger.info(line.rstrip())

        # Stream stderr
        for line in iter(process.stderr.readline, ''):
            logger.warning(line.rstrip())

        process.stdout.close()
        process.stderr.close()
        return_code = process.wait()
        if return_code != 0:
            logger.error(f"Script execution failed with exit code {return_code}")
            raise DeploymentError(f"Script execution failed with exit code {return_code}")
        logger.info("Script executed successfully!")
    except subprocess.CalledProcessError as e:
        logger.error(f"Script execution failed: {e.stderr}")
        raise DeploymentError(f"Script execution failed: {e.stderr}")

def terminate_pod(api_key: str, pod_id: str, logger: logging.Logger):
    query = f"""
    mutation {{
        podTerminate(input: {{ podId: "{pod_id}" }})
    }}
    """
    response = run_graphql_query(query, api_key, logger)
    logger.info(f"Pod '{pod_id}' terminated successfully.")
    return response

def monitor_pod(api_key: str, pod_id: str, config: Dict[str, Any], state: Dict[str, Any], logger: logging.Logger):
    import time
    start_time = time.time()

    while True:
        try:
            pods = get_pod_status(api_key, logger)
            pod = next((p for p in pods if p['id'] == pod_id), None)

            if not pod:
                logger.error(f"Pod {pod_id} not found during monitoring")
                break

            hours_elapsed = (time.time() - start_time) / 3600
            cost = hours_elapsed * float(pod['costPerHr'])

            # Check budgets
            if hours_elapsed > config['budget']['max_hours']:
                logger.warning(f"Time budget exceeded: {hours_elapsed:.1f} hours")
                if not state['keep_alive']:
                    terminate_pod(api_key, pod_id, logger)
                break

            if cost > config['budget']['max_dollars']:
                logger.warning(f"Cost budget exceeded: ${cost:.2f}")
                if not state['keep_alive']:
                    terminate_pod(api_key, pod_id, logger)
                break

            time.sleep(60)  # Check every minute

        except Exception as e:
            logger.error(f"Error in pod monitoring: {str(e)}")
            break

def deploy_pod_from_config(api_key: str, config: Dict[str, Any], logger: logging.Logger, keep_alive: bool, state: Dict[str, Any]) -> Dict[str, Any]:
    project_name = config.get("project_name")
    if not project_name:
        logger.error("Missing 'project_name' in config.")
        raise ConfigurationError("Missing 'project_name' in config.")

    # Infer remote workspace dynamically
    remote_dir = config["upload"]["remote_dir"]

    local_dir = config["upload"]["local_dir"]
    script_path = config["script"]["path"]
    env_vars = config["script"].get("env", {})
    ssh_key_path = os.path.expanduser(config["ssh"]["key_path"])

    # Validate SSH key path
    if not os.path.isfile(ssh_key_path):
        logger.error(f"SSH key not found at {ssh_key_path}")
        raise DeploymentError(f"SSH key not found at {ssh_key_path}")

    # Set execute permissions for the script
    full_script_path = os.path.join(local_dir, os.path.basename(script_path))
    set_executable(full_script_path, logger)

    if "gpu_id" in config:
        selected_gpu_id = config["gpu_id"]
        logger.info(f"Using GPU ID from config: {selected_gpu_id}")
    else:
        # Get available GPUs
        gpus = get_gpus(api_key, logger)

        gpu_display_name_lower = config["gpu"].lower()
        selected_gpu_id = None
        for gpu in gpus:
            if gpu_display_name_lower in gpu["displayName"].lower() or gpu_display_name_lower in gpu["id"].lower():
                selected_gpu_id = gpu["id"]
                break
        if not selected_gpu_id:
            logger.error(f"GPU type '{config['gpu']}' not found.")
            raise DeploymentError(f"GPU type '{config['gpu']}' not found.")

    # Create a unique pod name
    import time
    pod_name = f"auto-deploy-{int(time.time())}"

    # Initialize pod_id for tracking
    pod_id = None
    try:
        # Create pod
        pod = create_pod(api_key, pod_name, config["image"], selected_gpu_id, logger)
        pod_id = pod["id"]
        state['pod_id'] = pod_id

        # Start monitor thread
        import threading
        monitor_thread = threading.Thread(
            target=monitor_pod,
            args=(api_key, pod_id, config, state, logger),
            daemon=True
        )
        monitor_thread.start()

        # Wait for pod to be ready with runtime information
        logger.info("Waiting for pod to be in RUNNING state with runtime information...")
        pod = wait_for_pod_running(api_key, pod_id, logger)

        # Extract SSH details
        runtime = pod.get("runtime", {})
        ports = runtime.get("ports", [])
        ssh_port_info = next((port for port in ports if port["privatePort"] == 22 and port["isIpPublic"]), None)
        if not ssh_port_info:
            logger.error("No public SSH port found for the pod.")
            raise DeploymentError("No public SSH port found for the pod.")

        ssh_ip = ssh_port_info["ip"]
        ssh_port = ssh_port_info["publicPort"]

        # SSH command
        ssh_command_str = f"ssh root@{ssh_ip} -p {ssh_port} -i {ssh_key_path}"
        logger.info("Pod ready for SSH access:")
        logger.warning(f"{ssh_command_str}")

        # Save SSH info to file
        ssh_info = {
            "command": ssh_command_str,
            "pod_id": pod_id,
            "ip": ssh_ip,
            "port": ssh_port
        }

        project_dir = config["upload"]["local_dir"]
        with open(os.path.join(project_dir, ".pod_ssh"), "w") as f:
            json.dump(ssh_info, f, indent=2)
        logger.debug("SSH information saved to .pod_ssh file.")

        # Upload files
        upload_files_to_pod(api_key, pod, local_dir, remote_dir, ssh_key_path, ssh_ip, ssh_port, logger)

        # Execute the bash script
        execute_script_on_pod(ssh_key_path, ssh_ip, ssh_port, os.path.basename(script_path), env_vars, remote_dir, logger)

        # After successful script execution
        if keep_alive:
            logger.info("Deployment completed successfully! Dropping into SSH session as '--keep-alive' is set.")
            logger.warning(f"\nTo connect to your pod:\n{ssh_command_str}")

            # Connect to SSH
            logger.warning("Connecting to pod via SSH...")
            try:
                subprocess.run([
                    "ssh",
                    f"root@{ssh_ip}",
                    "-p", str(ssh_port),
                    "-i", ssh_key_path
                ])
            except Exception as e:
                logger.error(f"Failed to establish SSH connection: {str(e)}")
                logger.warning("You can manually connect using the command above.")
        else:
            logger.info("Deployment completed successfully! Terminating the pod as '--keep-alive' is not set.")
            terminate_pod(api_key, pod_id, logger)
            logger.info("Pod terminated as per '--keep-alive' flag.")

        return pod

    except Exception as e:
        logger.error(f"Error during deployment: {str(e)}", exc_info=True)
        if pod_id:
            logger.warning("Attempting to terminate the pod due to deployment error...")
            try:
                terminate_pod(api_key, pod_id, logger)
                logger.info("Pod terminated successfully.")
            except Exception as termination_error:
                logger.error(f"Failed to terminate pod: {str(termination_error)}", exc_info=True)
                logger.error(f"Terminate your pod manually at https://www.runpod.io/console/pods")
        sys.exit(1)

def wait_for_pod_running(api_key: str, pod_id: str, logger: logging.Logger, timeout: int = 120, interval: int = 15) -> Dict[str, Any]:
    """
    Waits until the pod reaches the RUNNING state and runtime information is available.

    Args:
        api_key (str): RunPod API key.
        pod_id (str): ID of the pod to monitor.
        logger (logging.Logger): Logger instance.
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
        pods = get_pod_status(api_key, logger)
        pod = next((p for p in pods if p['id'] == pod_id), None)
        if not pod:
            logger.error(f"Pod with ID {pod_id} not found.")
            raise DeploymentError(f"Pod with ID {pod_id} not found.")

        status = pod.get('desiredStatus', 'UNKNOWN')
        logger.info(f"Current pod status: {status}")

        if status.lower() == 'running':
            runtime = pod.get('runtime')
            if runtime:
                logger.debug("Pod is running and runtime information is available.")
                return pod
            else:
                logger.warning("Pod is running but runtime information is not yet available. Waiting...")
        elif status.lower() in ['failed', 'terminated']:
            logger.error(f"Pod has entered an invalid state: {status}")
            raise DeploymentError(f"Pod has entered an invalid state: {status}")

        time.sleep(interval)
        elapsed += interval

    logger.error("Timed out waiting for pod to reach RUNNING state with runtime information.")
    raise DeploymentError("Timed out waiting for pod to reach RUNNING state with runtime information.")

def automate_workflow(config_path: str, logger: logging.Logger, keep_alive: bool, state: Dict[str, Any]):
    try:
        config = read_config(config_path, logger)
        validate_config(config, logger)
        logger.info("Configuration validated successfully.")
    except ConfigurationError as e:
        logger.error(f"Configuration Error: {str(e)}", exc_info=True)
        sys.exit(1)

    project_dir = Path(config['upload']['local_dir'])
    log_file_path = project_dir / "deployment.jsonl"
    ensure_log_directory(log_file_path)
    setup_logging(str(log_file_path))  # Initialize logging
    logger = get_logger('my_logger')  # Retrieve the logger

    # Encrypt environment variables and save keys
    encrypted_env, key = encrypt_env(config)
    (project_dir / '.env.encrypted').write_bytes(encrypted_env)
    (project_dir / '.env.key').write_bytes(key)
    logger.debug("Environment variables encrypted and saved.")

    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        logger.error("Error: RUNPOD_API_KEY environment variable not found.")
        sys.exit(1)

    try:
        pod = deploy_pod_from_config(api_key, config, logger, keep_alive, state)
    except DeploymentError as e:
        logger.error(f"Deployment Error: {str(e)}", exc_info=True)
        sys.exit(1)

def handle_signal(signum, frame, api_key: str, logger: logging.Logger, state: Dict[str, Any]):
    """
    Signal handler to gracefully terminate the pod upon receiving interrupt signals.

    Args:
        signum (int): Signal number.
        frame: Current stack frame.
        api_key (str): RunPod API key.
        logger (logging.Logger): Logger instance.
        state (dict): Mutable dictionary containing deployment state.
    """
    logger.warning(f"Received signal {signum}. Initiating graceful shutdown...")
    pod_id = state.get('pod_id')
    keep_alive = state.get('keep_alive', False)
    if pod_id:
        if not keep_alive:
            logger.info("Terminating pod as '--keep-alive' is not set.")
            try:
                terminate_pod(api_key, pod_id, logger)
                logger.info("Pod terminated successfully.")
            except DeploymentError as e:
                logger.error(f"Failed to terminate pod during shutdown: {str(e)}")
                logger.error(f"Terminate your pod manually at https://www.runpod.io/console/pods")
        else:
            logger.info("Pod will remain running as '--keep-alive' is set.")
    else:
        logger.info("No pod to terminate.")
    logger.info("Exiting the deployment script.")
    sys.exit(0)

def main():
    # Load environment variables from .env file
    load_dotenv()

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Deploy a RunPod project.")
    parser.add_argument(
        "--project",
        required=True,
        help="The project name. The script will look for the configuration in 'projects/{project_name}/config.yaml'.",
    )
    parser.add_argument(
        "--keep-alive",
        action='store_true',
        default=False,
        help="If set, keeps the pod running and drops into an SSH session after script execution. Otherwise, terminates the pod after the script runs."
    )
    args = parser.parse_args()
    project_name = args.project
    keep_alive = args.keep_alive

    # Construct the config path
    config_path = os.path.join(DEFAULT_PROJECTS_DIR, project_name, "config.yaml")

    # Check if config.yaml exists
    if not os.path.isfile(config_path):
        print(f"Error: Config file not found at '{config_path}'.")  # Use print since logging isn't set up yet
        sys.exit(1)

    # Initialize a state dictionary to track pod_id and keep_alive
    state = {
        'pod_id': None,
        'keep_alive': keep_alive
    }

    # Temporarily set up basic logging to capture signals before full logging is initialized
    basic_logger = logging.getLogger('basic_logger')
    basic_logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    basic_logger.addHandler(handler)

    # Load the configuration to get the API key for signal handling
    try:
        config = read_config(config_path, basic_logger)
        validate_config(config, basic_logger)
    except ConfigurationError as e:
        basic_logger.error(f"Configuration Error: {str(e)}")
        sys.exit(1)

    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        basic_logger.error("Error: RUNPOD_API_KEY environment variable not found.")
        sys.exit(1)

    # Register signal handlers
    signal.signal(signal.SIGINT, lambda s, f: handle_signal(s, f, api_key, basic_logger, state))
    signal.signal(signal.SIGTERM, lambda s, f: handle_signal(s, f, api_key, basic_logger, state))

    # Set up logging
    project_dir = os.path.dirname(config_path)
    log_file_path = os.path.join(project_dir, "deployment.jsonl")
    ensure_log_directory(log_file_path)
    setup_logging(log_file_path)  # Initialize logging with the log file path
    logger = get_logger('my_logger')  # Retrieve the logger

    # Update the state with keep_alive
    state['keep_alive'] = keep_alive

    logger.info(f"Starting deployment for project '{project_name}' with config at '{config_path}'.")
    logger.info(f"'--keep-alive' is set to {'True' if keep_alive else 'False'}.")

    # Start the automated workflow
    automate_workflow(config_path, logger, keep_alive, state)

if __name__ == "__main__":
    main()
