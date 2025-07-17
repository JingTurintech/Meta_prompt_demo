# Meta-Prompting Framework

A comprehensive framework for meta-prompt optimization and evaluation using Streamlit. This project provides tools for optimizing and evaluating prompts for code optimization tasks, with both local and project-based evaluation capabilities.

## Project Structure

```
Meta_prompt_demo/
├── Core Applications/
│   ├── batch_meta_artemis_app.py              # Batch processing for large-scale operations
│   ├── benchmark_streamlit_app_meta_artemis.py # Comprehensive benchmarking platform
│   ├── benchmark_streamlit_app.py             # Basic benchmarking interface
│   └── benchmark_streamlit_app_Artemis.py     # Artemis-integrated benchmarking
├── Optimization Workflows/
│   ├── streamlit_app_optimization.py          # Basic optimization UI
│   ├── streamlit_app_optimization_by_project.py # Project-based optimization
│   └── streamlit_app_optimization_enhanced_by_project.py # Enhanced project optimization
├── Scoring & Evaluation/
│   ├── streamlit_app_scoring.py               # Basic scoring interface
│   └── streamlit_app_scoring_by_project.py    # Project-based scoring
├── Analysis & Utilities/
│   ├── llm_runtime_correlation_analysis_app.py # Runtime correlation analysis
│   ├── outlier_solution_analyzer.py           # Outlier solution analysis
│   ├── benchmark_collector.py                 # Benchmark data collection
│   └── llm_score_collector.py                 # LLM score collection
└── Core Modules/
    ├── meta_artemis_modules/                   # Modular components
    ├── benchmark_evaluator_meta_artemis.py    # Core evaluation engine
    ├── meta_prompt_optimization.py            # Optimization logic
    └── meta_prompt_scoring.py                 # Scoring algorithms
```

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

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with the appropriate application
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the application logs in the Streamlit interface
2. Verify your `.env` configuration
3. Ensure the Artemis runner server is operational
4. Create an issue in the GitHub repository

---

**🎯 Quick Start Guide:**
- **New users**: Start with `streamlit run streamlit_app_optimization.py`
- **Large-scale evaluation**: Use `streamlit run batch_meta_artemis_app.py`
- **Comprehensive benchmarking**: Try `streamlit run benchmark_streamlit_app_meta_artemis.py`
- **Performance analysis**: Run `streamlit run llm_runtime_correlation_analysis_app.py`
