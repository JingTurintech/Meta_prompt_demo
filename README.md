# Meta-Prompting Framework

A comprehensive framework for meta-prompt optimization and evaluation using Streamlit. This project provides tools for optimizing and evaluating prompts for code optimization tasks, with both local and project-based evaluation capabilities.

## Prerequisites

- Python 3.11 or higher
- Streamlit
- Access to Artemis Falcon API (for project-based features)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/JingTurintech/Meta_prompt_demo.git
   cd Meta_prompt_demo
   ```

2. Create and activate a conda virtual environment (recommended):
   ```bash
   conda create -n meta_prompt_env python=3.11
   conda activate meta_prompt_env
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
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

First, activate your conda environment:
```bash
conda activate meta_prompt_env
```

Then run the desired application:
```bash
streamlit run [app_name].py
```

The project provides four main workflows for different use cases:

## 1. 🚀 Large-scale Performance Evaluation Workflow

```bash
streamlit run artemis_performance_evaluation_app.py
```

**Key Features:**
- **Batch Recommendation Creation**: Generate recommendations for multiple constructs using meta-prompting templates
- **Batch Solution Creation**: Create solutions from recommendations or generate new ones at scale
- **Batch Solution Evaluation**: Execute and evaluate multiple solutions with performance metrics
- **Runtime Impact Analysis**: Analyze runtime performance impact of code recommendations vs original code
- **Multi-Project Support**: Process multiple projects simultaneously with intelligent resource management

**Use Cases:**
- Conduct large-scale performance evaluation experiments to evaluate the meta-prompting code optimization technique.

## 2. 📊 LLM Scoring Evaluation Workflow

```bash
streamlit run llm_scoring_evaluation_app.py
```

**Key Features:**
- Conduct LLM scoring experiments to evaluate the meta-prompting code optimization technique.
- Load existing LLM scoring experiment results (ELO ratings)


## 3. 🔍 Analysis & Utility Tools

### Collect scores from a project
```bash
python llm_score_collector.py
```

### Analyze correlation between runtime and LLM scores (unfinished)
```bash
streamlit run llm_runtime_correlation_analysis.py
```




## Falcon Client Functions Reference

This section provides a comprehensive overview of all Falcon client functions used throughout the application, their locations, and purposes.

| **Function** | **Location** | **Purpose** | **Usage Context** |
|-------------|-------------|-------------|-------------------|
| **`authenticate()`** | `meta_artemis_modules/evaluator.py:126` | Authenticate with Artemis platform | Initial setup when creating evaluator instance |
| **`get_project(project_id)`** | `meta_artemis_modules/evaluator.py:147` | Get project information and metadata | Fetching project details, name, description, language |
| **`get_constructs_info(project_id)`** | `meta_artemis_modules/evaluator.py:151`<br>`meta_artemis_modules/recommendations.py:689` | Get all constructs (code segments) in a project | Loading project structure, finding optimizable code segments |
| **`get_spec(spec_id, ...)`** | `meta_artemis_modules/evaluator.py:305, 429, 582`<br>`meta_artemis_modules/evaluator.py:1488, 1495` | Get detailed specification of a construct | Retrieving original and optimized code versions |
| **`get_optimisation(optimization_id)`** | `meta_artemis_modules/project_manager.py:51, 227`<br>`meta_artemis_modules/evaluator.py:711` | Get optimization configuration details | Loading optimization settings and parameters |
| **`get_solutions(optimization_id, ...)`** | `meta_artemis_modules/project_manager.py:56, 243` | Get all solutions for an optimization | Loading existing solutions for analysis and evaluation |
| **`get_solution(solution_id)`** | `artemis_performance_evaluation_app.py:2065`<br>`meta_artemis_modules/evaluator.py:732, 753, 782, 883, 904, 1006, 1027` | Get detailed solution information | Checking solution status, retrieving results and metrics |
| **`add_prompt(prompt_request, project_id)`** | `meta_artemis_modules/evaluator.py:361, 514, 1070` | Add meta-prompt to project | Creating optimization prompts for LLM-based optimization |
| **`execute_recommendation_task(request, ...)`** | `meta_artemis_modules/evaluator.py:377, 530` | Execute optimization recommendation | Running LLM-based code optimization tasks |
| **`get_process(process_id)`** | `meta_artemis_modules/evaluator.py:399, 552, 688, 809` | Get process status and results | Monitoring long-running optimization processes |
| **`add_solution(solution_request)`** | `meta_artemis_modules/solutions.py:209, 357, 422`<br>`meta_artemis_modules/evaluator.py:843, 964` | Create new solution in optimization | Adding optimized code solutions to projects |
| **`evaluate_solution(solution_id, ...)`** | `artemis_performance_evaluation_app.py:2053`<br>`meta_artemis_modules/evaluator.py:740, 870, 993` | Start solution evaluation/benchmarking | Running performance evaluations on solutions |
| **`get_ai_application(ai_run_id)`** | `meta_artemis_modules/project_manager.py:191`<br>`meta_artemis_modules/evaluator.py:1405` | Get AI application run details | Retrieving LLM execution information and metadata |
| **`get_prompt(prompt_id)`** | `meta_artemis_modules/evaluator.py:1413` | Get prompt details | Retrieving prompt text and configuration |

### Key Workflows

**Project Discovery Flow:**
1. `get_project()` → Get basic project information
2. `get_constructs_info()` → Load all code constructs
3. `get_spec()` → Get detailed code for specific constructs

**Solution Management Flow:**
1. `get_optimisation()` → Load optimization configuration
2. `get_solutions()` → Get all solutions in optimization
3. `get_solution()` → Get detailed solution information

**Optimization Pipeline:**
1. `add_prompt()` → Create meta-prompt
2. `execute_recommendation_task()` → Run LLM optimization
3. `get_process()` → Monitor optimization progress
4. `add_solution()` → Save optimized code

**Evaluation Workflow:**
1. `add_solution()` → Create solution (if needed)
2. `evaluate_solution()` → Start performance evaluation
3. `get_solution()` → Check evaluation results

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