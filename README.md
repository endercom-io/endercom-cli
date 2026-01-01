# Endercom CLI

A command-line interface for deploying and managing AI agents on AWS Lambda using the Endercom framework.

## Installation

```bash
pip install endercom-cli
```

## Prerequisites

Before you begin, ensure you have:

1.  **An AWS Account**: You'll need credentials to deploy resources.
2.  **AWS CLI Configured**: Run `aws login` to set your credentials and default region.
    ```bash
    brew install awscli
    aws login
    ```
3.  **AWS SAM CLI**: Required for building and deploying the serverless stack.
    ```bash
    brew install aws-sam-cli
    ```

## Quick Start

### 1. Initialize an Agent Project

Create a new agent project structure.

```bash
endercom init
```

This scaffolds a directory with:

- `app.py`: Your agent logic
- `template.yaml`: AWS SAM infrastructure definition
- `requirements.txt`: Python dependencies
- `.endercom/agent.yaml`: CLI configuration

### 2. Configure Environment (Optional)

If your agent needs API keys (e.g., `OPENAI_API_KEY`), add them to the secrets manager.

```bash
cd my-agent
endercom secrets set OPENAI_API_KEY "sk-..."
```

### 3. Deploy

Build and deploy your agent to AWS.

```bash
endercom deploy
```

The CLI will:

- Check your code for undeclared environment variables.
- Sync your configuration to the SAM template.
- Build the agent using `sam build`.
- Deploy using `sam deploy`.

## Commands

| Command                   | Description                                            |
| ------------------------- | ------------------------------------------------------ |
| `init <name>`             | Create a new agent project.                            |
| `deploy`                  | Build and deploy the agent stack to AWS.               |
| `configure`               | Update project configuration.                          |
| `secrets set <key> <val>` | securely store secrets in AWS Secrets Manager.         |
| `logs`                    | Tail real-time logs from your Lambda function.         |
| `destroy`                 | Tear down the AWS stack and remove associated secrets. |

## Configuration

The source of truth for your agent's configuration is `.endercom/agent.yaml`.

```yaml
name: my-agent
runtime: python3.12
region: us-east-1
memory: 1024
timeout: 30
env:
  MY_PUBLIC_VAR: "production"
secrets:
  - OPENAI_API_KEY
```

- **env**: Standard environment variables injected at deployment.
- **secrets**: List of keys to resolve securely from AWS Secrets Manager.
