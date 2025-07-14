# Meta-Prompting Framework

A comprehensive framework for meta-prompt optimization and evaluation using Streamlit. This project provides tools for optimizing and evaluating prompts for code optimization tasks, with both local and project-based evaluation capabilities.

## Features

- **Meta-Prompt Optimization**: Generate and optimize prompts for code optimization tasks
- **Multiple Evaluation Methods**: 
  - Local benchmark-based evaluation
  - Project-based evaluation using Artemis Falcon API
- **Interactive UI**: Streamlit-based interface for:
  - Prompt generation and optimization
  - Code optimization evaluation
  - Performance scoring and visualization
- **Support for Multiple LLMs**: Compatible with various language models including:
  - GPT-4
  - Claude
  - Gemini
  - LLaMA

## Project Structure

```
.
├── Meta-Prompt Core
│   ├── meta_prompt_optimization.py           # Core optimization logic
│   ├── meta_prompt_optimization_by_project.py    # Project-based optimization
│   └── meta_prompt_optimization_enhanced_by_project.py  # Enhanced project optimization
│
├── Streamlit Applications
│   ├── streamlit_app_optimization.py         # Basic optimization UI
│   ├── streamlit_app_optimization_by_project.py    # Project-based UI
│   └── streamlit_app_optimization_enhanced_by_project.py  # Enhanced project UI
│
├── Scoring and Evaluation
│   ├── meta_prompt_scoring.py                # Scoring logic
│   ├── meta_prompt_scoring_by_project.py     # Project-based scoring
│   ├── streamlit_app_scoring.py              # Scoring UI
│   └── streamlit_app_scoring_by_project.py   # Project-based scoring UI
│
└── Utilities
    └── falcon_client.py                      # Artemis Falcon API client
```

## Prerequisites

- Python 3.8 or higher
- Streamlit
- Access to Artemis Falcon API (for project-based features)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JingTurintech/Meta_prompt_demo.git
   cd Meta_prompt_demo
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root with:

   FALCON_API_KEY=your_falcon_api_key
   VISION_API_KEY=your_vision_api_key
   THANOS_API_KEY=your_thanos_api_key

## Environment Setup: .env File

To run this project, you need to create a `.env` file in the project root directory with your API credentials and service endpoints. This file is used to configure Artemis and related services.

**Example .env file:**
```
THANOS_HOST=artemis.turintech.ai
THANOS_HTTPS=true
THANOS_PORT=443
THANOS_POSTFIX=/turintech-thanos/api
THANOS_CLIENT_ID=your_client_id
THANOS_CLIENT_SECRET=your_client_secret
THANOS_GRANT_TYPE=password
THANOS_USERNAME=your_username
THANOS_PASSWORD=your_password

FALCON_HOST=artemis.turintech.ai
FALCON_HTTPS=true
FALCON_PORT=443
FALCON_POSTFIX=/api

VISION_HOST=artemis.turintech.ai
VISION_HTTPS=true
VISION_PORT=443
VISION_POSTFIX=/api
```

- Replace the values with your actual credentials.
- **Do not commit your `.env` file to version control.**
- The application will automatically load these variables using `python-dotenv`.

## Usage

```bash
conda activate meta_prompt_env
streamlit run xxx.py
```

The framework provides two main workflows for evaluating and optimizing meta-prompts:

### 1. Code Optimization Workflow

This workflow focuses on optimizing code using meta-prompts and evaluating the optimization results.

#### Local Optimization
```bash
streamlit run streamlit_app_optimization.py
```
- Uses local benchmarks for optimization
- Suitable for initial prompt development and testing

#### Project-Based Optimization
```bash
streamlit run streamlit_app_optimization_by_project.py
```
- Integrates with Artemis Falcon API
- Uses real project context for optimization

#### Enhanced Project Optimization
```bash
streamlit run streamlit_app_optimization_enhanced_by_project.py
```
- Advanced features including:
  - Project information display
  - Enhanced visualization
  - Detailed performance metrics
  - Real-time optimization tracking

### 2. Scoring and Evaluation Workflow

This workflow focuses on evaluating and comparing the effectiveness of different prompts.

#### Local Scoring
```bash
streamlit run streamlit_app_scoring.py
```
- Evaluates prompts using local benchmarks
- Provides basic scoring metrics
- Suitable for quick prompt comparisons

#### Project-Based Scoring
```bash
streamlit run streamlit_app_scoring_by_project.py
```
- Evaluates prompts in the context of real projects
- Features:
  - Comparative analysis of different prompts
  - Performance visualization
  - Project-specific metrics
  - Integration with Artemis Falcon API

## Configuration

- Adjust LLM settings in the respective Python files
- Modify optimization parameters through the Streamlit interface
- Configure project-specific settings in the `.env` file

## Remote Artemis Runner Server Management

This project includes a remote Ubuntu server (35.189.66.83) running an Artemis custom runner in a Docker container.

### Server Connection
Connect to the remote server using the SSH key:
```bash
ssh -i my_rsa_key_with_email ubuntu@35.189.66.83
```

### Check Artemis Runner Status
Check if the artemis runner container is running:
```bash
# Check all running containers
ssh -i my_rsa_key_with_email ubuntu@35.189.66.83 "docker ps"


### View Artemis Runner Logs
Check recent logs to verify the runner is functioning:
```bash
# View last 20 log entries
docker logs --tail 20 artemis-stable-runner

# Follow logs in real-time
docker logs --tail 100 -f artemis-stable-runner
```

### Container Management
Start/stop/restart the artemis runner if needed:
```bash
# Stop the container
ssh -i my_rsa_key_with_email ubuntu@35.189.66.83 "docker stop artemis-stable-runner"

# Start the container
ssh -i my_rsa_key_with_email ubuntu@35.189.66.83 "docker start artemis-stable-runner"

# Restart the container
ssh -i my_rsa_key_with_email ubuntu@35.189.66.83 "docker restart artemis-stable-runner"
```
