# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "requests",
#     "python-dotenv",
#     "rich",
# ]
# ///

import json
import os
import sys
import subprocess
import requests
from dotenv import load_dotenv
from typing import Optional, Dict, Any
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from rich.prompt import Prompt, IntPrompt

QUICK_GPU_OPTIONS = {
    "1": "NVIDIA A40",
    "2": "NVIDIA RTX 4090",
    "3": "NVIDIA A100 SXM",
    "4": "NVIDIA H100 SXM",
    "5": "Enter custom GPU ID"
}

DOCKER_IMAGES = {
    "1": "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04",
    "2": "runpod/pytorch:1.13.0-py3.10-cuda11.7.1-devel-ubuntu22.04",
    "3": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
    "4": "Enter custom image"
}

def run_graphql_query(query: str, api_key: str) -> Dict[str, Any]:
    api_url = "https://api.runpod.io/graphql"
    headers = {"Content-Type": "application/json"}
    url = f"{api_url}?api_key={api_key}"
    data = json.dumps({"query": query})

    response = requests.post(url, headers=headers, data=data, timeout=30)

    if response.status_code == 401:
        raise Exception("Unauthorized request, please check your API key.")
    if "errors" in response.json():
        raise Exception(response.json()["errors"][0]["message"])
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
        f"https://api.runpod.io/graphql?api_key={api_key}",
        json={"query": query},
        headers={"Content-Type": "application/json"},
        timeout=30
    )

    if response.status_code != 200 or "errors" in response.json():
        raise Exception(f"Failed to create pod: {response.text}")

    return response.json()["data"]["podFindAndDeployOnDemand"]

def display_gpu_table(gpus: list):
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")

    table.add_column("GPU Name")
    table.add_column("Memory")
    table.add_column("Secure")
    table.add_column("Community")
    table.add_column("Spot (Secure)")
    table.add_column("Spot (Community)")

    for gpu in gpus:
        table.add_row(
            gpu['displayName'],
            f"{gpu['memoryInGb']}GB",
            f"${gpu.get('securePrice', 'N/A')}/hr",
            f"${gpu.get('communityPrice', 'N/A')}/hr",
            f"${gpu.get('secureSpotPrice', 'N/A')}/hr",
            f"${gpu.get('communitySpotPrice', 'N/A')}/hr"
        )

    console.print(table)

def select_docker_image() -> str:
    rprint("\n[bold]Select Docker Image:[/bold]")
    for key, image in DOCKER_IMAGES.items():
        rprint(f"{key}. {image}")

    choice = Prompt.ask("Enter your choice", choices=list(DOCKER_IMAGES.keys()))
    if choice == "4":
        return Prompt.ask("Enter custom Docker image")
    return DOCKER_IMAGES[choice]

def select_gpu_type(gpus: list) -> str:
    rprint("\n[bold]Select GPU Type:[/bold]")
    for key, gpu in QUICK_GPU_OPTIONS.items():
        rprint(f"{key}. {gpu}")

    choice = Prompt.ask("Enter your choice", choices=list(QUICK_GPU_OPTIONS.keys()))
    if choice == "5":
        rprint("\nAvailable GPU IDs:")
        for gpu in gpus:
            rprint(f"- {gpu['id']}: {gpu['displayName']}")
        return Prompt.ask("Enter GPU ID")

    # Find the GPU ID that matches the display name
    selected_name = QUICK_GPU_OPTIONS[choice]
    for gpu in gpus:
        if gpu['displayName'] == selected_name:
            return gpu['id']
    raise Exception(f"GPU {selected_name} not found in available options")

def get_pod_creation_inputs(gpus: list) -> tuple:
    console = Console()
    # display_gpu_table(gpus)

    name = Prompt.ask("\nEnter pod name")
    image_name = select_docker_image()

    rprint("\nAvailable GPU IDs:")
    for gpu in gpus:
        rprint(f"- {gpu['id']}: {gpu['displayName']}")
    gpu_type_id = Prompt.ask("Enter GPU ID")

    return name, image_name, gpu_type_id

def stop_pod(api_key: str, pod_id: str):
    query = f"""
    mutation {{
        podStop(input: {{ podId: "{pod_id}" }}) {{
            id
            desiredStatus
        }}
    }}
    """
    return run_graphql_query(query, api_key)

def terminate_pod(api_key: str, pod_id: str):
    query = f"""
    mutation {{
        podTerminate(input: {{ podId: "{pod_id}" }})
    }}
    """
    return run_graphql_query(query, api_key)

def resume_pod(api_key: str, pod_id: str, gpu_count: int):
    query = f"""
    mutation {{
        podResume(input: {{ podId: "{pod_id}", gpuCount: {gpu_count} }}) {{
            id
            desiredStatus
        }}
    }}
    """
    return run_graphql_query(query, api_key)

def display_pod_info(pod_info: Dict[str, Any]):
    """
    Displays the Pod information and top-level user data in a structured table.

    Args:
        pod_info (dict): The dictionary containing user data and Pod data.
    """
    console = Console()
    table = Table(show_header=True, header_style="bold green")

    # Top-level User Data
    table.add_section()
    table.add_row("[bold underline]User Information[/bold underline]", "")
    table.add_row("User ID", pod_info['myself'].get('id', 'N/A'))
    table.add_row("Client Balance", f"${pod_info['myself'].get('clientBalance', 'N/A')}")

    # Savings Plans (if any)
    savings_plans = pod_info['myself'].get('savingsPlans', [])
    if savings_plans:
        table.add_row("Savings Plans", "")
        for plan in savings_plans:
            table.add_row(
                f"- Start: {plan.get('startTime', 'N/A')}",
                f"End: {plan.get('endTime', 'N/A')}, GPU Type ID: {plan.get('gpuTypeId', 'N/A')}, Cost/hr: ${plan.get('costPerHr', 'N/A')}"
            )
    else:
        table.add_row("Savings Plans", "None")

    # Specific Pod Data
    pod = pod_info['pod']
    table.add_section()
    table.add_row("[bold underline]Pod Information[/bold underline]", "")
    table.add_row("Pod ID", pod.get('id', 'N/A'))
    table.add_row("Name", pod.get('name', 'N/A'))
    table.add_row("Status", pod.get('desiredStatus', 'N/A'))
    table.add_row("Image Name", pod.get('imageName', 'N/A'))
    table.add_row("Cost per Hour", f"${pod.get('costPerHr', 'N/A')}/hr")
    table.add_row("GPU Count", str(pod.get('gpuCount', 'N/A')))
    table.add_row("GPU Type", pod.get('machine', {}).get('gpuDisplayName', 'N/A'))
    table.add_row("Uptime (seconds)", str(pod.get('runtime', {}).get('uptimeInSeconds', 'N/A')))
    table.add_row("Created At", pod.get('createdAt', 'N/A'))
    table.add_row("Updated At", pod.get('updatedAt', 'N/A'))

    # Ports Information
    ports = pod.get('runtime', {}).get('ports', [])
    if ports:
        ports_info = "\n".join([
            f"IP: {port['ip']}, Public: {port['isIpPublic']}, Private Port: {port['privatePort']}, Public Port: {port['publicPort']}"
            for port in ports
        ])
        table.add_row("Ports", ports_info)
    else:
        table.add_row("Ports", "No ports available.")

    console.print(table)


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

def display_pods_table(pods: list):
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("#")
    table.add_column("ID")
    table.add_column("Name")
    table.add_column("Status")
    table.add_column("SSH Command")

    for i, pod in enumerate(pods):
        pod_id = pod['machine']['podHostId']
        ssh_command = f"ssh {pod_id}@ssh.runpod.io -i ~/.ssh/id_ed25519"

        table.add_row(
            str(i),
            pod_id,
            pod['name'],
            pod.get('desiredStatus', 'UNKNOWN'),
            ssh_command
        )
    Console().print(table)

def deploy_files(pod: dict):
    runtime = pod.get('runtime')
    if not runtime:
        rprint("[red]Pod does not have runtime information.[/red]")
        rprint("[yellow]Here is the full Pod information for debugging purposes:[/yellow]")
        rprint(json.dumps(pod, indent=4))
        return

    ports = runtime.get('ports')
    if not ports:
        rprint("[red]Pod does not have exposed ports.[/red]")
        rprint("[yellow]Here is the runtime information for debugging purposes:[/yellow]")
        rprint(json.dumps(runtime, indent=4))
        return

    public_ports = [port for port in runtime['ports'] if port['isIpPublic']]
    if not public_ports:
        rprint("[red]No public ports available for this Pod.[/red]")
        rprint(json.dumps(runtime, indent=4))
        return

    # Assuming SSH is exposed via one of the ports
    ssh_port_info = None
    for port in public_ports:
        if port['privatePort'] == 22:
            ssh_port_info = port
            break

    if not ssh_port_info:
        # If SSH port is not directly found, take the first public port
        ssh_port_info = public_ports[0]

    ip = ssh_port_info['ip']
    port = ssh_port_info['publicPort']

    # Prompt user for SSH key path
    default_key_path = os.path.expanduser("~/.ssh/id_ed25519")
    key_path = default_key_path
    # key_path = Prompt.ask("Enter path to your SSH private key", default=default_key_path)

    # Prompt for local file path
    local_path = Prompt.ask("Enter the local file path to upload")

    if not os.path.isfile(local_path):
        rprint(f"[red]Local file not found at {local_path}[/red]")
        return

    # Prompt for remote destination path
    default_remote_path = f"/root/{os.path.basename(local_path)}"
    remote_path = Prompt.ask("Enter the remote destination path", default=default_remote_path)

    # Construct scp command
    scp_command = [
        "scp",
        "-P", str(port),
        "-i", key_path,
        local_path,
        f"root@{ip}:{remote_path}"
    ]

    rprint(f"\n[green]Executing SCP command:[/green] {' '.join(scp_command)}")

    try:
        result = subprocess.run(scp_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        rprint("[green]File uploaded successfully![/green]")
    except subprocess.CalledProcessError as e:
        rprint(f"[red]SCP command failed: {e.stderr}[/red]")

def interactive_ssh(pod: dict):
    """
    Launches an interactive SSH session to the selected pod.

    Args:
        pod (dict): The pod information dictionary.
    """
    pod_host_id = pod['machine']['podHostId']
    ssh_command = [
        "ssh",
        f"{pod_host_id}@ssh.runpod.io",
        "-i",
        os.path.expanduser("~/.ssh/id_ed25519")
    ]

    rprint(f"\n[green]Launching SSH session to {pod_host_id}@ssh.runpod.io[/green]")
    rprint("[yellow]Press Ctrl+D or type 'exit' to end the SSH session.[/yellow]")

    try:
        subprocess.run(ssh_command)
    except Exception as e:
        rprint(f"[red]Error launching SSH session: {str(e)}[/red]")

def manage_pod(api_key: str):
    pods = get_pod_status(api_key)
    if not pods:
        rprint("[yellow]No pods found[/yellow]")
        return

    display_pods_table(pods)
    pod_index = IntPrompt.ask("\nEnter pod number", default=0)

    if pod_index >= len(pods) or pod_index < 0:
        rprint("[red]Invalid pod number[/red]")
        return

    pod = pods[pod_index]
    pod_id = pod['id']
    pod_host_id = pod['machine']['podHostId']

    rprint("\n[bold]Pod Actions:[/bold]")
    rprint("1. SSH into pod")
    rprint("2. Stop pod")
    rprint("3. Terminate pod")
    rprint("4. Deploy files")
    rprint("5. Resume pod")
    rprint("6. Interactive SSH")
    rprint("0. Back to main menu")

    action = Prompt.ask("Select action", choices=["0", "1", "2", "3", "4", "5", "6"])

    if action == "0":
        return

    try:
        if action == "1":
            ssh_command = f"ssh {pod_host_id}@ssh.runpod.io -i ~/.ssh/id_ed25519"
            rprint(f"\n[green]SSH Command:[/green] {ssh_command}")
            rprint("[yellow]Note: Make sure you have added your SSH public key to RunPod[/yellow]")
        elif action == "2":
            stop_pod(api_key, pod_id)
            rprint("[green]Pod stopped successfully[/green]")
        elif action == "3":
            terminate_pod(api_key, pod_id)
            rprint("[green]Pod terminated successfully[/green]")
        elif action == "4":
            deploy_files(pod)
        elif action == "5":
            # Retrieve current GPU count
            current_gpu_count = pod.get('gpuCount', 1)
            rprint(f"\nCurrent GPU Count: {current_gpu_count}")
            # Prompt user for new GPU count with default as current_gpu_count
            gpu_count_str = Prompt.ask(
                "Enter GPU count to resume",
                default=str(current_gpu_count)
            )
            # Validate the input
            if gpu_count_str.isdigit() and int(gpu_count_str) > 0:
                gpu_count = int(gpu_count_str)
                resume_pod(api_key, pod_id, gpu_count)
                rprint("[green]Pod resumed successfully[/green]")
            else:
                rprint("[red]Invalid GPU count. Please enter a positive integer.[/red]")
        elif action == "6":
            interactive_ssh(pod)
        else:
            rprint("[red]Invalid action selected[/red]")
    except Exception as e:
        rprint(f"[red]Error: {str(e)}[/red]")

def display_menu():
    print("\nRunPod CLI Menu")
    print("1. List available GPUs")
    print("2. Create new pod")
    print("3. Manage pods")
    print("0. Exit")
    return input("Select an option: ")

def main():
    load_dotenv()
    api_key = os.getenv("RUNPOD_API_KEY")
    if not api_key:
        rprint("[red]Error: RUNPOD_API_KEY environment variable not found[/red]")
        sys.exit(1)

    while True:
        choice = display_menu()

        if choice == "1":
            try:
                gpus = get_gpus(api_key)
                display_gpu_table(gpus)
            except Exception as e:
                rprint(f"[red]Error: {str(e)}[/red]")

        elif choice == "2":
            try:
                gpus = get_gpus(api_key)
                name, image_name, gpu_type_id = get_pod_creation_inputs(gpus)
                pod = create_pod(api_key, name, image_name, gpu_type_id)
                rprint("\n[green]Pod created successfully![/green]")
                rprint(f"Pod ID: {pod['id']}")
                rprint(f"Status: {pod.get('desiredStatus', 'PENDING')}")
            except Exception as e:
                rprint(f"[red]Error: {str(e)}[/red]")

        elif choice == "3":
            try:
                manage_pod(api_key)
            except Exception as e:
                rprint(f"[red]Error: {str(e)}[/red]")

        elif choice == "0":
            print("Goodbye!")
            break

        else:
            print("Invalid option, please try again")

if __name__ == "__main__":
    main()
