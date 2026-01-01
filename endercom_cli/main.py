from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.panel import Panel

# --- Custom YAML Handling for CloudFormation ---
class CfnTag:
    def __init__(self, tag, value):
        self.tag = tag
        self.value = value

def cfn_constructor(loader, node):
    return CfnTag(node.tag, loader.construct_scalar(node))

def cfn_representer(dumper, data):
    return dumper.represent_scalar(data.tag, data.value)

# Register common CloudFormation short tags
for tag in ["!Sub", "!Ref", "!GetAtt", "!ImportValue", "!Join", "!Select", "!Split"]:
    yaml.SafeLoader.add_constructor(tag, cfn_constructor)
    yaml.SafeDumper.add_representer(CfnTag, cfn_representer)
# -----------------------------------------------

app = typer.Typer(no_args_is_help=True)
from rich.theme import Theme

# Retro Theme: Monochromatic
theme = Theme({
    "info": "bold white",
    "warning": "white dim",
    "error": "bold white on black",
    "success": "bold white",
    "command": "white dim",
})
console = Console(theme=theme)

LOGO = r"""
 ███████╗███╗   ██╗██████╗ ███████╗██████╗  ██████╗ ██████╗ ███╗   ███╗
 ██╔════╝████╗  ██║██╔══██╗██╔════╝██╔══██╗██╔════╝██╔═══██╗████╗ ████║
 █████╗  ██╔██╗ ██║██║  ██║█████╗  ██████╔╝██║     ██║   ██║██╔████╔██║
 ██╔══╝  ██║╚██╗██║██║  ██║██╔══╝  ██╔══██╗██║     ██║   ██║██║╚██╔╝██║
 ███████╗██║ ╚████║██████╔╝███████╗██║  ██║╚██████╗╚██████╔╝██║ ╚═╝ ██║
 ╚══════╝╚═╝  ╚═══╝╚═════╝ ╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚═════╝ ╚═╝     ╚═╝
"""

from rich.text import Text

def print_banner():
    # Gradient effect: white to dark gray
    lines = LOGO.strip('\n').split('\n')
    colors = ["#ffffff", "#dddddd", "#bbbbbb", "#999999", "#777777", "#555555"]
    
    text = Text(justify="center")
    for i, line in enumerate(lines):
        color = colors[i % len(colors)]
        text.append(line + "\n", style=f"bold {color}")
        
    text.append("\nAI AGENT DEPLOYMENT SYSTEM // V1.0", style="dim white")
    
    console.print(Panel(text, border_style="white", padding=(1, 2)))

TEMPLATE_DIR = Path(__file__).parent / "templates"

def run(cmd: list[str], cwd: Optional[Path] = None) -> None:
    console.print(f"[command]$ {' '.join(cmd)}[/command]")
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_cmd_exists(name: str) -> None:
    if shutil.which(name) is None:
        raise typer.BadParameter(
            f"Missing dependency: `{name}` not found on PATH. Install it and try again."
        )


def load_agent_cfg(project_dir: Path) -> dict:
    cfg_path = project_dir / ".endercom" / "agent.yaml"
    if not cfg_path.exists():
        raise typer.BadParameter(f"Missing {cfg_path}. Run `endercom init` first.")
    return yaml.safe_load(cfg_path.read_text())


@app.command()
def init(
    name: Optional[str] = typer.Argument(None, help="Agent name / stack name slug"),
    directory: Path = typer.Option(Path("."), "--dir", help="Target directory"),
    runtime: str = typer.Option("python3.12", help="Lambda runtime"),
    region: str = typer.Option("us-east-1", help="AWS region"),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", help="Run interactive wizard"),
) -> None:
    """
    Scaffold an Endercom Lambda agent project.
    """
    if interactive:
        print_banner()
        console.print("[info]CREATING NEW AGENT...[/info]")
        if not name:
            name = typer.prompt("Agent ID")
        
        # If user didn't provide overrides via flags, prompt them
        # Note: typer fills defaults if not provided, so we check if they are default values?
        # A bit tricky since we can't distinguish "user passed default" vs "default".
        # But we can just prompt with current value as default.
        
        region = typer.prompt("AWS Region", default=region)
        runtime = typer.prompt("Lambda Runtime", default=runtime)
        memory = typer.prompt("Memory (MB)", default=1024, type=int)
        timeout = typer.prompt("Timeout (seconds)", default=30, type=int)
        
        freq_id = typer.prompt("Frequency ID (optional - can be configured later)", default="", show_default=False)
        freq_key = typer.prompt("Frequency API Key (optional - can be configured later)", default="", show_default=False, hide_input=True)
    else:
        if not name:
            console.print("[red]Missing argument 'NAME'.[/red]")
            raise typer.Exit(code=1)
        memory = 1024
        timeout = 30
        freq_id = ""
        freq_key = ""

    if not name: # Should happen only if interactive prompt was empty which typer prevents for required input
         raise typer.Exit(code=1)

    project_dir = (directory / name).resolve()
    if project_dir.exists() and any(project_dir.iterdir()):
        if not typer.confirm(f"Directory {project_dir} is not empty. Continue?"):
             raise typer.Abort()
             
    project_dir.mkdir(parents=True, exist_ok=True)

    # Create .endercom/agent.yaml
    ender_dir = project_dir / ".endercom"
    ender_dir.mkdir(exist_ok=True)
    
    agent_cfg = {
        "name": name,
        "runtime": runtime,
        "region": region,
        "memory": memory,
        "timeout": timeout,
        "expose": {"type": "http", "auth": "none"},
        "env": {"FREQUENCY_ID": freq_id or "freq_...", "AGENT_ID": name},
        "secrets": ["FREQUENCY_API_KEY"],
    }
    (ender_dir / "agent.yaml").write_text(yaml.safe_dump(agent_cfg, sort_keys=False))

    # Copy templates
    for fname in ["app.py", "requirements.txt", "template.yaml", ".env.example"]:
        src = TEMPLATE_DIR / fname
        dst = project_dir / fname
        
        # We need to replace placeholders
        content = src.read_text()
        content = content.replace("{{AGENT_NAME}}", name)
        content = content.replace("{{RUNTIME}}", runtime)
        # Handle frequency ID placeholder (defaults to "freq_..." if empty for later config)
        f_id = freq_id if freq_id else "freq_..."
        content = content.replace("{{FREQUENCY_ID}}", f_id)
        
        dst.write_text(content)

    console.print(Panel.fit(f"PROJECT INITIALIZED AT: {project_dir}\n\nNEXT STEPS:\n  cd {project_dir}\n  endercom deploy", title="[bold white]STATUS[/bold white]", border_style="white"))
    
    if freq_key:
        console.print(f"\n[info]NOTE:[/info] Frequency API Key provided.")
        console.print(f"Run [bold]endercom configure --frequency-key <KEY>[/bold] inside to save to Secrets Manager.")
    else:
        console.print("[warning]IMPORTANT:[/warning] Put secrets in SSM/Secrets Manager. Never hardcode keys.")


@app.command()
def configure(
    directory: Path = typer.Option(Path("."), "--dir", help="Project directory"),
    profile: Optional[str] = typer.Option(None, "--profile", help="AWS profile name"),
    frequency_id: Optional[str] = typer.Option(None, "--frequency-id", help="Set FREQUENCY_ID in agent.yaml"),
    frequency_key: Optional[str] = typer.Option(None, "--frequency-key", help="Set FREQUENCY_API_KEY in Secrets Manager"),
) -> None:
    """
    Update configuration (agent.yaml or Secrets Manager).
    """
    project_dir = directory.resolve()
    
    # Update agent.yaml
    if frequency_id:
        cfg_path = project_dir / ".endercom" / "agent.yaml"
        if not cfg_path.exists():
            console.print(f"[red]Missing {cfg_path}. Are you in the right directory?[/red]")
            raise typer.Exit(1)
            
        cfg = yaml.safe_load(cfg_path.read_text())
        cfg.setdefault("env", {})["FREQUENCY_ID"] = frequency_id
        cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        console.print(f"[green]Updated FREQUENCY_ID in {cfg_path}[/green]")
        
        # Also update template.yaml if it exists
        tpl_path = project_dir / "template.yaml"
        if tpl_path.exists():
            content = tpl_path.read_text()
            # Replace FREQUENCY_ID: "..." with new value
            # This regex looks for FREQUENCY_ID: followed by optional whitespace and a quoted string
            # We replace it with the new value
            if "FREQUENCY_ID:" in content:
                # Replace the line with the new value
                # We use a regex to be safe about whitespace, but replace the whole line to ensure valid YAML
                # Pattern: start of line (or after newline) + whitespace + FREQUENCY_ID: + whitespace + "old" or old
                content = re.sub(r'(\s*FREQUENCY_ID:\s*).*', f'\\1"{frequency_id}"', content)
                tpl_path.write_text(content)
                console.print(f"[green]Updated FREQUENCY_ID in {tpl_path}[/green]")

    # Update Secrets Manager
    if frequency_key:
        ensure_cmd_exists("aws")
        cfg = load_agent_cfg(project_dir)
        stack = cfg["name"]
        region = cfg.get("region", "us-east-1")
        
        env = os.environ.copy()
        if profile:
            env["AWS_PROFILE"] = profile
            
        secret_name = f"endercom/{stack}/FREQUENCY_API_KEY"
        
        create_cmd = [
            "aws", "secretsmanager", "create-secret",
            "--name", secret_name,
            "--secret-string", frequency_key,
            "--region", region,
        ]
        put_cmd = [
            "aws", "secretsmanager", "put-secret-value",
            "--secret-id", secret_name,
            "--secret-string", frequency_key,
            "--region", region,
        ]

        console.print("[bold]Storing FREQUENCY_API_KEY in Secrets Manager[/bold]")
        try:
            subprocess.run(create_cmd, cwd=str(project_dir), env=env, check=True, capture_output=True)
            console.print(f"[green]Created secret {secret_name}[/green]")
        except subprocess.CalledProcessError as e:
            # If create fails, check if it exists or is scheduled for deletion
            err_msg = e.stderr.decode() if e.stderr else str(e)
            
            if "ResourceExistsException" in err_msg or "InvalidRequestException" in err_msg:
                # Check if we should restore (if the user wants to update it, we should ensure it is active)
                restore_cmd = [
                    "aws", "secretsmanager", "restore-secret",
                    "--secret-id", secret_name,
                    "--region", region
                ]
                
                # Try restore blindly
                try:
                   subprocess.run(restore_cmd, cwd=str(project_dir), env=env, check=True, capture_output=True)
                   console.print(f"[cyan]Restored secret {secret_name} from deletion.[/cyan]")
                except subprocess.CalledProcessError:
                    pass # It was probably not pending deletion
                
                console.print(f"[yellow]Secret exists. Updating {secret_name}...[/yellow]")
                try:
                    subprocess.run(put_cmd, cwd=str(project_dir), env=env, check=True, capture_output=True)
                    console.print(f"[green]Updated secret {secret_name}[/green]")
                except subprocess.CalledProcessError as e_put:
                    err_msg_put = e_put.stderr.decode() if e_put.stderr else str(e_put)
                    console.print(f"[red]Failed to update secret:[/red] {err_msg_put}")
            else:
                console.print(f"[red]Failed to create secret:[/red] {err_msg}")


import ast
import sys

def scan_imports(app_path: Path) -> set[str]:
    """Scan app.py for imports using AST, excluding stdlib."""
    if not app_path.exists():
        return set()
        
    try:
        tree = ast.parse(app_path.read_text())
    except SyntaxError:
        return set()

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])
                
    # Filter stdlib (approximate list for py3.12 + common built-ins)
    # sys.stdlib_module_names is available in 3.10+
    stdlib = sys.stdlib_module_names if hasattr(sys, "stdlib_module_names") else set(sys.builtin_module_names)
    
    return {pkg for pkg in imports if pkg not in stdlib and pkg not in sys.builtin_module_names}

# Knowledge base of libraries and their implicit env vars
LIBRARY_ENV_VARS = {
    "openai": ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "pinecone": ["PINECONE_API_KEY"],
    "google": ["GOOGLE_API_KEY"],
    "llama_api_client": ["LLAMA_API_KEY"]
}

def scan_env_vars(app_path: Path) -> set[str]:
    """Scan app.py for os.environ calls using AST and check for implicit library usage."""
    if not app_path.exists():
        return set()
    
    try:
        tree = ast.parse(app_path.read_text())
    except SyntaxError:
        return set()
        
    env_vars = set()
    imports = set()

    for node in ast.walk(tree):
        # Collect imports to check against knowledge base
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split('.')[0])

        # Check for os.environ['VAR'] or os.environ.get('VAR') or os.getenv('VAR')
        
        # 1. Subscript: os.environ['VAR']
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Attribute):
                # We need to check if attribute is os.environ
                if isinstance(node.value.value, ast.Name) and node.value.value.id == 'os' and node.value.attr == 'environ':
                     # Get the key
                     if isinstance(node.slice, ast.Constant): # python 3.9+
                         env_vars.add(node.slice.value)
                     elif isinstance(node.slice, ast.Index) and isinstance(node.slice.value, ast.Str): # python < 3.9
                         env_vars.add(node.slice.value.s)

        # 2. Call: os.environ.get('VAR') or os.getenv('VAR')
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                # os.environ.get
                if isinstance(node.func.value, ast.Attribute) and \
                   isinstance(node.func.value.value, ast.Name) and \
                   node.func.value.value.id == 'os' and \
                   node.func.value.attr == 'environ' and \
                   node.func.attr == 'get':
                        if node.args and isinstance(node.args[0], ast.Constant):
                            env_vars.add(node.args[0].value)
                # os.getenv
                elif isinstance(node.func.value, ast.Name) and \
                     node.func.value.id == 'os' and \
                     node.func.attr == 'getenv':
                        if node.args and isinstance(node.args[0], ast.Constant):
                             env_vars.add(node.args[0].value)

    # Add implicit env vars from libraries
    for lib, vars_list in LIBRARY_ENV_VARS.items():
        if lib in imports:
            # We add them to suggested env vars
            # Note: libraries might not NEED all of them (e.g. optional ones).
            # But usually the API Key is mandatory.
            # We can be aggressive and suggest them.
            for v in vars_list:
                env_vars.add(v)

    return env_vars

def sync_template_env(project_dir: Path, cfg: dict) -> None:
    """
    Syncs environment variables and secrets from agent.yaml to template.yaml.
    """
    tpl_path = project_dir / "template.yaml"
    if not tpl_path.exists():
        console.print(f"[warning]Template {tpl_path} not found. Skipping env sync.[/warning]")
        return

    try:
        template = yaml.safe_load(tpl_path.read_text())
    except Exception as e:
        console.print(f"[warning]Failed to parse template.yaml: {e}. Skipping env sync.[/warning]")
        return

    # Navigate to Resources -> AgentFunction -> Properties -> Environment -> Variables
    try:
        # If any key is missing, we create it or fail gracefully if structure is totally different
        env_vars = template.setdefault("Resources", {}).setdefault("AgentFunction", {}).setdefault("Properties", {}).setdefault("Environment", {}).setdefault("Variables", {})
    except AttributeError:
        console.print("[warning]Template structure mismatch (Resources/AgentFunction/Properties/Environment/Variables). Skipping env sync.[/warning]")
        return
    
    # Update from 'env' section
    for key, value in cfg.get("env", {}).items():
        env_vars[key] = value
        
    # Update from 'secrets' section
    # Convention: secretsmanager:endercom/${AWS::StackName}/<KEY>
    for secret_key in cfg.get("secrets", []):
        # We construct the dynamic reference string
        # !Sub "{{resolve:secretsmanager:endercom/${AWS::StackName}/KEY:SecretString}}"
        # We must use CfnTag to represent !Sub
        ref_str = f"{{{{resolve:secretsmanager:endercom/${{AWS::StackName}}/{secret_key}:SecretString}}}}"
        env_vars[secret_key] = CfnTag("!Sub", ref_str)
        
    # Write back
    try:
        with tpl_path.open("w") as f:
            yaml.safe_dump(template, f, sort_keys=False)
        console.print(f"[info]Synced environment variables to {tpl_path}[/info]")
    except Exception as e:
        console.print(f"[error]Failed to write template.yaml: {e}[/error]")

@app.command()
def deploy(
    directory: Path = typer.Option(Path("."), "--dir", help="Project directory"),
    profile: Optional[str] = typer.Option(None, "--profile", help="AWS profile name"),
    guided: bool = typer.Option(False, "--guided", help="Run `sam deploy --guided`"),
    set_secret: Optional[str] = typer.Option(None, "--set-frequency-key", help="Set FREQUENCY_API_KEY into SSM SecureString (value)"),
) -> None:
    """
    Build + deploy the agent with SAM.
    """
    ensure_cmd_exists("sam")
    ensure_cmd_exists("aws")

    project_dir = directory.resolve()
    cfg = load_agent_cfg(project_dir)
    stack = cfg["name"]
    region = cfg.get("region", "us-east-1")

    # Auto-detect missing env vars
    app_path = project_dir / "app.py"
    if app_path.exists():
        detected_env = scan_env_vars(app_path)
        
        # Determine what's already defined
        defined_env = set(cfg.get("env", {}).keys())
        defined_secrets = set(cfg.get("secrets", []))
        all_defined = defined_env.union(defined_secrets)
        
        # Filter typical false positives or system vars if needed
        # For now, trust the AST scan
        missing = detected_env - all_defined
        
        # Always exclude standard AWS Lambda ones if detected (though user code rarely reads them via os.environ explicit dict access usually)
        # AWS_REGION, AWS_LAMBDA_FUNCTION_NAME etc.
        # But if user accesses them, they might expect them to be set or just reading them.
        # If they are reading them, they are available by default in Lambda.
        # We should probably filter out AWS_* variables.
        missing = {v for v in missing if not v.startswith("AWS_") and not v.startswith("_")}

        if missing:
            console.print(f"[warning]DETECTED UNDECLARED ENVIRONMENT VARIABLES: {', '.join(missing)}[/warning]")
            
            cfg_path = project_dir / ".endercom" / "agent.yaml"
            updated_cfg = False
            
            for var in missing:
                if typer.confirm(f"Add '{var}' to configuration?"):
                    is_secret = typer.confirm(f"Is '{var}' a secret?")
                    if is_secret:
                        # Prompt for value securely
                        val = typer.prompt(f"Enter value for secret {var}", hide_input=True)
                        
                        # Store in Secrets Manager immediately
                        # We reuse the logic from secrets command or just call AWS CLI
                        sec_name = f"endercom/{stack}/{var}"
                        console.print(f"[info]Saving {var} to Secrets Manager ({sec_name})...[/info]")
                        
                        # Just run AWS CLI directly here for simplicity
                        # Try create first
                        try:
                             subprocess.run(
                                 ["aws", "secretsmanager", "create-secret", "--name", sec_name, "--secret-string", val, "--region", region],
                                 check=True, capture_output=True
                             )
                        except subprocess.CalledProcessError:
                             # Try update
                             subprocess.run(
                                 ["aws", "secretsmanager", "put-secret-value", "--secret-id", sec_name, "--secret-string", val, "--region", region],
                                 check=True, capture_output=True
                             )
                        
                        # Update cfg
                        if "secrets" not in cfg:
                            cfg["secrets"] = []
                        if var not in cfg["secrets"]:
                            cfg["secrets"].append(var)
                            updated_cfg = True
                            
                    else:
                        # Regular env var
                        val = typer.prompt(f"Enter value for {var}")
                        if "env" not in cfg:
                            cfg["env"] = {}
                        cfg["env"][var] = val
                        updated_cfg = True
            
            if updated_cfg:
                cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                console.print(f"[success]Updated {cfg_path}[/success]")
                # Reload cfg to be fresh for sync_template_env
                cfg = load_agent_cfg(project_dir)

    # Sync environment variables before build
    console.print("[info]Syncing environment variables...[/info]")
    sync_template_env(project_dir, cfg)

    env = os.environ.copy()
    if profile:
        env["AWS_PROFILE"] = profile

    # Optionally store secret in SSM
    if set_secret:
        secret_name = f"endercom/{stack}/FREQUENCY_API_KEY"

        # Try create; if exists, put new version
        create_cmd = [
            "aws", "secretsmanager", "create-secret",
            "--name", secret_name,
            "--secret-string", set_secret,
            "--region", region,
        ]
        put_cmd = [
            "aws", "secretsmanager", "put-secret-value",
            "--secret-id", secret_name,
            "--secret-string", set_secret,
            "--region", region,
        ]

        console.print("[info]Storing FREQUENCY_API_KEY in Secrets Manager...[/info]")
        try:
            subprocess.run(create_cmd, cwd=str(project_dir), env=env, check=True, capture_output=True)
            console.print(f"[success]SECURE STORAGE: Created secret {secret_name}[/success]")
        except subprocess.CalledProcessError as e:
            # If create fails, check if it exists or is scheduled for deletion
            err_msg = e.stderr.decode() if e.stderr else str(e)
            
            if "ResourceExistsException" in err_msg or "InvalidRequestException" in err_msg:
                 # Check if we should restore (if the user wants to update it, we should ensure it is active)
                restore_cmd = [
                    "aws", "secretsmanager", "restore-secret",
                    "--secret-id", secret_name,
                    "--region", region
                ]
                
                # Try restore blindly; if it's not pending deletion, this might fail or succeed harmlessly?
                # Actually if it's not pending deletion, restore-secret throws InvalidRequestException.
                # So we wrap restore in try/except.
                try:
                   subprocess.run(restore_cmd, cwd=str(project_dir), env=env, check=True, capture_output=True)
                   console.print(f"[info]RESTORED secret {secret_name} from deletion.[/info]")
                except subprocess.CalledProcessError:
                    pass # It was probably not pending deletion
                
                console.print(f"[warning]Secret exists. Updating {secret_name}...[/warning]")
                try:
                    subprocess.run(put_cmd, cwd=str(project_dir), env=env, check=True, capture_output=True)
                    console.print(f"[success]UPDATED secret {secret_name}[/success]")
                except subprocess.CalledProcessError as e_put:
                     console.print(f"[error]Failed to update secret:[/error] {e_put}")
            else:
                 console.print(f"[error]Failed to create secret:[/error] {err_msg}")

        console.print(f"[success]SAVED[/success] {secret_name}")
    else:
        # Check if secret exists and is valid before deploying, otherwise CloudFormation might fail if it tries to resolve it
        # Actually CloudFormation fails if the secret is deleted but still referenced.
        # We should check if the secret is in 'deleted' state and warn the user or restore it if possible?
        # But we don't know the value to restore it to if the user didn't provide --set-frequency-key.
        # So we should just check if it is deleted.
        secret_name = f"endercom/{stack}/FREQUENCY_API_KEY"
        
        # Check secret status
        check_cmd = [
            "aws", "secretsmanager", "describe-secret",
            "--secret-id", secret_name,
            "--region", region
        ]
        
        try:
            # We use capture_output to inspect the JSON response if needed, but for now just check return code/stderr
            proc = subprocess.run(check_cmd, cwd=str(project_dir), env=env, capture_output=True, text=True)
            
            if proc.returncode == 0:
                # Secret exists. Check if it is marked for deletion?
                # The output JSON contains "DeletedDate" if it is marked for deletion.
                if '"DeletedDate":' in proc.stdout:
                    console.print(f"[error]CRITICAL: Secret {secret_name} is marked for deletion![/error]")
                    console.print("CloudFormation will fail to deploy. You must restore it or provide a new key.")
                    console.print(f"Run [bold]endercom deploy --set-frequency-key <KEY>[/bold] to restore and update it.")
                    raise typer.Exit(code=1)
            else:
                # Secret does not exist or other error
                # If it doesn't exist, CloudFormation will fail if the template references it.
                # The template references: {{resolve:secretsmanager:...}}
                # So we must ensure it exists.
                if "ResourceNotFoundException" in proc.stderr:
                     console.print(f"[error]CRITICAL: Secret {secret_name} does not exist![/error]")
                     console.print("The template requires this secret.")
                     console.print(f"Run [bold]endercom deploy --set-frequency-key <KEY>[/bold] to create it.")
                     raise typer.Exit(code=1)

        except FileNotFoundError:
             # aws cli not installed? ensure_cmd_exists handles this earlier but safe to ignore here
             pass


    # Auto-scan requirements
    app_path = project_dir / "app.py"
    req_path = project_dir / "requirements.txt"
    if app_path.exists() and req_path.exists():
        detected = scan_imports(app_path)
        existing = set()
        if req_path.read_text().strip():
            # Basic parsing of requirements.txt (ignores versions for now)
            # e.g. "openai==1.0" -> "openai"
            for line in req_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith('#'):
                    # Handle "openai==..." or "openai>=..."
                    pkg = re.split(r'[=<>~!]', line)[0].strip()
                    existing.add(pkg)
                    # Also handle case insensitivity? PyPI is usually lowercase normalized but imports might match.
                    # We'll stick to exact match for simplicity.

        missing = detected - existing
        # Filter out common false positives if necessary, e.g. "setuptools", "pkg_resources"
        # And mapping imports to packages (e.g. "yaml" -> "PyYAML", "PIL" -> "Pillow") is hard without external DB.
        # We will assume import name == package name for now, which covers 90% cases (openai, requests, numpy).
        
        if missing:
            console.print(f"[warning]DETECTED MISSING REQUIREMENTS: {', '.join(missing)}[/warning]")
            if typer.confirm("Add them to requirements.txt?"):
                with req_path.open("a") as f:
                    f.write("\n" + "\n".join(missing) + "\n")
                console.print(f"[success]UPDATED requirements.txt[/success]")

    # Build
    try:
        subprocess.run(["sam", "build"], cwd=str(project_dir), env=env, check=True)
    except subprocess.CalledProcessError:
        console.print("[error]BUILD FAILED.[/error]")
        raise typer.Exit(code=1)

    # Deploy
    deploy_cmd = ["sam", "deploy"]
    if guided:
        deploy_cmd.append("--guided")
    else:
        # Non-interactive defaults (you can make these smarter)
        deploy_cmd += [
            "--stack-name", stack,
            "--region", region,
            "--capabilities", "CAPABILITY_IAM",
            "--resolve-s3",
            "--no-confirm-changeset",
        ]
    
    try:
        subprocess.run(deploy_cmd, cwd=str(project_dir), env=env, check=True)
    except subprocess.CalledProcessError as e:
        # Check if the error is due to ROLLBACK_COMPLETE state
        # SAM deploy output goes to stdout/stderr. If check=True, we catch CalledProcessError.
        # But subprocess.run streams output to console, so we can't easily check output here unless we captured it.
        # However, we can catch the error and suggest a fix.
        console.print("[error]DEPLOYMENT FAILED.[/error]")
        console.print("If the stack is in [bold]ROLLBACK_COMPLETE[/bold] state (because initial creation failed), you must delete it first.")
        console.print(f"Run: [bold]endercom destroy --dir {directory}[/bold]")
        raise typer.Exit(code=1)

    console.print(Panel.fit("DEPLOYMENT SUCCESSFUL.\n\nNext: find the Invoke URL in API Gateway console OR add Outputs to template.yaml.", title="[bold white]COMPLETE[/bold white]", border_style="white"))


@app.command()
def secrets(
    action: str = typer.Argument(..., help="Action: set"),
    key: str = typer.Argument(..., help="Secret Key Name"),
    value: str = typer.Argument(..., help="Secret Value"),
    directory: Path = typer.Option(Path("."), "--dir", help="Project directory"),
    profile: Optional[str] = typer.Option(None, "--profile", help="AWS profile name"),
) -> None:
    """
    Manage secrets in AWS Secrets Manager.
    Usage: endercom secrets set MY_KEY "my-value"
    """
    if action != "set":
        console.print(f"[red]Unknown action: {action}. Only 'set' is supported currently.[/red]")
        raise typer.Exit(1)
        
    ensure_cmd_exists("aws")
    project_dir = directory.resolve()
    cfg = load_agent_cfg(project_dir)
    stack = cfg["name"]
    region = cfg.get("region", "us-east-1")
    
    env = os.environ.copy()
    if profile:
        env["AWS_PROFILE"] = profile
        
    secret_name = f"endercom/{stack}/{key}"
    
    console.print(f"[bold]Setting secret {secret_name}...[/bold]")
    
    # Check if secret exists first to decide create vs put-value
    # Or just try create, catch exists, then put-value (like in deploy/configure)
    
    create_cmd = [
        "aws", "secretsmanager", "create-secret",
        "--name", secret_name,
        "--secret-string", value,
        "--region", region,
    ]
    
    put_cmd = [
        "aws", "secretsmanager", "put-secret-value",
        "--secret-id", secret_name,
        "--secret-string", value,
        "--region", region,
    ]
    
    try:
        subprocess.run(create_cmd, cwd=str(project_dir), env=env, check=True, capture_output=True)
        console.print(f"[success]Created secret {secret_name}[/success]")
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.decode() if e.stderr else str(e)
        if "ResourceExistsException" in err_msg or "InvalidRequestException" in err_msg:
             # Try restore if needed
            restore_cmd = ["aws", "secretsmanager", "restore-secret", "--secret-id", secret_name, "--region", region]
            try:
                subprocess.run(restore_cmd, cwd=str(project_dir), env=env, check=True, capture_output=True)
                console.print(f"[cyan]Restored {secret_name} from deletion.[/cyan]")
            except subprocess.CalledProcessError:
                pass

            # Update value
            try:
                subprocess.run(put_cmd, cwd=str(project_dir), env=env, check=True, capture_output=True)
                console.print(f"[success]Updated secret {secret_name}[/success]")
            except subprocess.CalledProcessError as e_put:
                console.print(f"[error]Failed to update secret:[/error] {e_put.stderr.decode()}")
        else:
            console.print(f"[error]Failed to create secret:[/error] {err_msg}")
            
    # Remind user to add to agent.yaml
    if key not in cfg.get("secrets", []):
         console.print(f"\n[warning]REMINDER:[/warning] Add '{key}' to the 'secrets' list in {project_dir}/.endercom/agent.yaml so it gets injected into your Lambda.")


@app.command()
def logs(
    directory: Path = typer.Option(Path("."), "--dir", help="Project directory"),
    profile: Optional[str] = typer.Option(None, "--profile", help="AWS profile name"),
    tail: bool = typer.Option(True, "--tail/--no-tail", help="Tail logs"),
) -> None:
    """
    Tail Lambda logs via SAM.
    """
    ensure_cmd_exists("sam")
    project_dir = directory.resolve()
    cfg = load_agent_cfg(project_dir)
    stack = cfg["name"]
    region = cfg.get("region", "us-east-1")

    env = os.environ.copy()
    if profile:
        env["AWS_PROFILE"] = profile

    cmd = ["sam", "logs", "--stack-name", stack, "--region", region, "-n", "AgentFunction"]
    if tail:
        cmd.append("--tail")
    subprocess.run(cmd, cwd=str(project_dir), env=env, check=True)


@app.command()
def destroy(
    directory: Path = typer.Option(Path("."), "--dir", help="Project directory"),
    profile: Optional[str] = typer.Option(None, "--profile", help="AWS profile name"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation"),
) -> None:
    """
    Delete the deployed stack.
    """
    ensure_cmd_exists("sam")
    project_dir = directory.resolve()
    cfg = load_agent_cfg(project_dir)
    stack = cfg["name"]
    region = cfg.get("region", "us-east-1")

    env = os.environ.copy()
    if profile:
        env["AWS_PROFILE"] = profile

    cmd = ["sam", "delete", "--stack-name", stack, "--region", region]
    if yes:
        cmd += ["--no-prompts"]
    subprocess.run(cmd, cwd=str(project_dir), env=env, check=True)
    console.print("[success]STACK DESTROYED.[/success]")

    if not yes:
        # Get all secrets from agent.yaml
        secrets_list = cfg.get("secrets", [])
        
        # Always include FREQUENCY_API_KEY if not present, just in case
        if "FREQUENCY_API_KEY" not in secrets_list:
            secrets_list.append("FREQUENCY_API_KEY")
            
        if secrets_list:
            console.print(f"\n[info]Associated Secrets: {', '.join(secrets_list)}[/info]")
            if typer.confirm("Do you want to delete these secrets from Secrets Manager?"):
                for secret_key in secrets_list:
                    secret_name = f"endercom/{stack}/{secret_key}"
                    console.print(f"[white dim]Deleting secret {secret_name}...[/white dim]")
                    del_secret_cmd = [
                        "aws", "secretsmanager", "delete-secret",
                        "--secret-id", secret_name,
                        "--region", region,
                    ]
                    try:
                        subprocess.run(
                            del_secret_cmd, cwd=str(project_dir), env=env, check=True, capture_output=True
                        )
                        console.print(f"[bold green]DELETED secret {secret_name}.[/bold green]")
                    except subprocess.CalledProcessError as e:
                        # If secret doesn't exist, AWS CLI returns non-zero.
                        # We can check stderr to be nicer.
                        err_msg = e.stderr.decode().strip() if e.stderr else str(e)
                        if "ResourceNotFoundException" in err_msg:
                            console.print(f"[warning]Secret {secret_name} not found (already deleted?).[/warning]")
                        else:
                            console.print(f"[error]Failed to delete secret {secret_name}: {err_msg}[/error]")


