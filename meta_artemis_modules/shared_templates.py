# Define available LLMs
from vision_models.service.llm import LLMType


AVAILABLE_LLMS = [
    LLMType("gpt-4-o"),
    LLMType("llama-3-1-405b"),
    LLMType("qwen-2-5-coder-32b"),
    LLMType("mistral-large-2"),
    LLMType("command-r-plus"),
    LLMType("llama-3-1-70b"),
    LLMType("gpt-4-o-mini"),
    LLMType("gemini-v15-flash"),
    LLMType("llama-3-1-8b"),
    LLMType("claude-v35-sonnet"),
    LLMType("claude-v37-sonnet"),
    # LLMType("claude-v4-sonnet"),
]

# Default LLM types
DEFAULT_META_PROMPT_LLM = "claude-v37-sonnet"
DEFAULT_CODE_OPTIMIZATION_LLM = "claude-v37-sonnet"
DEFAULT_SCORING_LLM = "claude-v37-sonnet"

# Define optimization tasks
OPTIMIZATION_TASKS = {
    "runtime_performance": {
        "description": "Optimize code for better runtime performance",
        "objective": "improving runtime performance",
        "default_prompt": "Improve the performance of the provided code. Try to find ways to reduce runtime, while keeping the main functionality of the code unchanged.",
        "instruction": "Generate an optimized version of the code that improves runtime performance while maintaining the same functionality.",
        "data_format": "\n\nOriginal Code:\n{}\n",
        "considerations": """1. Algorithmic complexity (Big O notation)
2. Data structure efficiency and access patterns
3. Loop optimizations and unnecessary iterations
4. Memory access patterns and caching
5. I/O operations and system calls
6. Parallel processing opportunities
7. Redundant computations"""
    },
    "memory_usage": {
        "description": "Optimize code for reduced memory consumption",
        "objective": "reducing memory usage",
        "default_prompt": "Improve the performance of the provided code. Try to find ways to reduce memory usage, while keeping the main functionality of the code unchanged.",
        "instruction": "Generate an optimized version of the code that reduces memory consumption while maintaining the same functionality.",
        "data_format": "\n\nOriginal Code:\n{}\n",
        "considerations": """1. Memory allocation and deallocation patterns
2. Memory leaks and resource cleanup
3. Data structure memory footprint
4. Buffer sizes and memory pools
5. Memory fragmentation
6. Garbage collection impact
7. Shared memory usage"""
    }
}

# Define meta-prompt templates - MPCO and ablation study variants for RQ3
META_PROMPT_TEMPLATES = {
    "mpco": {
        "name": "MPCO (Full Context)",
        "description": "Meta-Prompt for Code Optimization with full contextual information - the complete MPCO approach",
        "template": """You are an expert in code optimization. Please generate a prompt that will instruct the target LLM {target_llm} to optimize code for {objective}. Consider the project context, task context, and adapt the prompt complexity and style based on the target LLM's capabilities.

## Project Context
Project Name: {project_name}
Project Description: {project_description}
Primary Languages: {project_languages}

## Task Context
- Description: {task_description}
- Considerations: {task_considerations}

## Target LLM Context
- Target Model: {target_llm}
- For cost-efficient LLMs (e.g., gpt-4-o-mini, gemini-v15-flash, llama-3-1-8b): these models have limited internal chain-of-thought, so the generated prompt should give short, clear and succinct instructions, without internal reasoning.
- For larger LLMs (e.g., gpt-4-o, claude-v35-sonnet, claude-v37-sonnet): The generated prompt should allow for more complex and extensive internal reasoning, and encourage internal verification of any assumptions related to metrics based on the task description. 

NOTE: Your response should contain only the prompt, without any placeholders for the code, formatting instructions, or additional text. The generated prompt should not contain any additional text like placeholders for the code or formatting instructions.
"""
    },
    
    # === RQ3 ABLATION STUDY VARIANTS ===
    "no_project_context": {
        "name": "No Project Context",
        "description": "MPCO without project-specific information (project name, description, languages)",
        "template": """You are an expert in code optimization. Please generate a prompt that will instruct the target LLM {target_llm} to optimize code for {objective}. Consider the task context, and adapt the prompt complexity and style based on the target LLM's capabilities.

## Task Context
- Description: {task_description}
- Considerations: {task_considerations}

## Target LLM Context
- Target Model: {target_llm}
- For cost-efficient LLMs (e.g., gpt-4-o-mini, gemini-v15-flash, llama-3-1-8b): these models have limited internal chain-of-thought, so the generated prompt should give short, clear and succinct instructions, without internal reasoning.
- For larger LLMs (e.g., gpt-4-o, claude-v35-sonnet, claude-v37-sonnet): The generated prompt should allow for more complex and extensive internal reasoning, and encourage internal verification of any assumptions related to metrics based on the task description. 

NOTE: Your response should contain only the prompt, without any placeholders for the code, formatting instructions, or additional text. The generated prompt should not contain any additional text like placeholders for the code or formatting instructions.
"""
    },
    "no_task_context": {
        "name": "No Task Context",
        "description": "MPCO without detailed task information (task description and considerations)",
        "template": """You are an expert in code optimization. Please generate a prompt that will instruct the target LLM {target_llm} to optimize code for {objective}. Consider the project context, and adapt the prompt complexity and style based on the target LLM's capabilities.

## Project Context
Project Name: {project_name}
Project Description: {project_description}
Primary Languages: {project_languages}

## Target LLM Context
- Target Model: {target_llm}
- For cost-efficient LLMs (e.g., gpt-4-o-mini, gemini-v15-flash, llama-3-1-8b): these models have limited internal chain-of-thought, so the generated prompt should give short, clear and succinct instructions, without internal reasoning.
- For larger LLMs (e.g., gpt-4-o, claude-v35-sonnet, claude-v37-sonnet): The generated prompt should allow for more complex and extensive internal reasoning, and encourage internal verification of any assumptions related to metrics based on the task description. 

NOTE: Your response should contain only the prompt, without any placeholders for the code, formatting instructions, or additional text. The generated prompt should not contain any additional text like placeholders for the code or formatting instructions.
"""
    },
    "no_llm_context": {
        "name": "No LLM Context",
        "description": "MPCO without LLM-specific adaptation instructions",
        "template": """You are an expert in code optimization. Please generate a prompt that will instruct the target LLM {target_llm} to optimize code for {objective}. Consider the project context and task context.

## Project Context
Project Name: {project_name}
Project Description: {project_description}
Primary Languages: {project_languages}

## Task Context
- Description: {task_description}
- Considerations: {task_considerations}

NOTE: Your response should contain only the prompt, without any placeholders for the code, formatting instructions, or additional text. The generated prompt should not contain any additional text like placeholders for the code or formatting instructions.
"""
    },
    "minimal_context": {
        "name": "Minimal Context",
        "description": "MPCO with minimal context - removes all project, task, and LLM-specific information",
        "template": """You are an expert in code optimization. Please generate a prompt that will instruct the target LLM {target_llm} to optimize code for {objective}.

NOTE: Your response should contain only the prompt, without any placeholders for the code, formatting instructions, or additional text. The generated prompt should not contain any additional text like placeholders for the code or formatting instructions.
"""
    }
}

# Baseline prompting techniques for RQ1 comparison
BASELINE_PROMPTING_TEMPLATES = {
    "direct_prompt": {
        "name": "Direct Prompt",
        "description": "Simple, direct optimization instruction",
        "template": "Improve the performance of the provided code. Try to find ways to reduce runtime, while keeping the main functionality of the code unchanged.\n\n{code}"
    },
    "chain_of_thought": {
        "name": "Chain-of-Thought Prompting", 
        "description": "Explicit reasoning steps for optimization",
        "template": """Let's optimize the following code step by step:

Please follow these reasoning steps:
1. First, analyze the current code to identify performance bottlenecks
2. Consider different optimization strategies (algorithmic, data structure, loop optimization, etc.)
3. Evaluate the trade-offs of each approach
4. Select the best optimization strategy
5. Implement the optimized version

Think through each step, then provide the final optimized code."""
    },
    "few_shot": {
        "name": "Few-Shot Prompting",
        "description": "Examples of optimization patterns",
        "template": """Here are examples of code optimization:

Example 1 - Loop optimization:
Original: for i in range(len(arr)): if arr[i] > threshold: result.append(arr[i])
Optimized: result = [x for x in arr if x > threshold]

Example 2 - Algorithm optimization:
Original: for i in range(n): for j in range(n): if matrix[i][j] > 0: count += 1
Optimized: count = np.sum(matrix > 0)

Example 3 - Data structure optimization:
Original: items = []; for x in data: items.append(x); return sorted(items)
Optimized: return sorted(data)

Now optimize the code for better runtime performance."""
    },
    "metacognitive": {
        "name": "Metacognitive Prompting",
        "description": "Self-reflective optimization process",
        "template": """You are an expert code optimizer. Before optimizing the code, reflect on your optimization process:

1. What optimization strategies do you know?
2. Which strategies are most appropriate for this type of code?
3. How will you verify the optimization is correct?

Now optimize the code for better runtime performance.
"""
    },
    "contextual_prompting": {
        "name": "Contextual Prompting",
        "description": "Direct optimization with full context information (same context as MPCO but without meta-prompting)",
        "template": """You are an expert in code optimization. Please optimize the provided code for {objective}. Consider the project context, task context, and adapt your optimization approach accordingly.

## Project Context
Project Name: {project_name}
Project Description: {project_description}
Primary Languages: {project_languages}

## Task Context
- Description: {task_description}
- Considerations: {task_considerations}

## Optimization Instructions
- Focus on {objective}
- Maintain the same functionality
- Consider the project's specific requirements and constraints
- Provide clear, well-commented optimized code
"""
    }
}

# Combine meta-prompt templates with baseline prompting templates
ALL_PROMPTING_TEMPLATES = {**META_PROMPT_TEMPLATES, **BASELINE_PROMPTING_TEMPLATES}

# Judge prompt template for benchmark comparisons
JUDGE_PROMPT_TEMPLATE = """You are an expert in code optimization and performance analysis. Compare the following two code snippets and determine which one would be better for {objective}.

## Task Context
{task_description}

Code A:
```python
{code_a}
```

Code B:
```python
{code_b}
```

Consider the following aspects specific to {objective}:
{task_considerations}

Respond with ONLY ONE of these exact strings:
- "A" if Code A is likely to be better for {objective}
- "B" if Code B is likely to be better for {objective}
- "TIE" if both codes would have similar performance for {objective}

Your response should contain only A, B, or TIE, nothing else.""" 

# Default project and optimization IDs mapping
DEFAULT_PROJECT_OPTIMISATION_IDS = {
    # Original default projects
    "26ecc1a2-2b9c-4733-9d5d-07d0a6608686": ["0b3c2b1e-ec02-4cb5-9135-48ec1007a611"],  # BitmapPlusPlus
    "28334995-7488-4414-876a-fbbdd1d990f9": ["dd671939-838f-4d83-9a52-257b53cfb141"],  # llama.cpp
    "9f8f7777-f359-4f39-bfa8-6a0f4ebe473c": ["bed565ec-01dc-44d2-9cf5-358750851de3"],  # faster-whisper
    "0126bc6f-57c0-4148-bd3e-3d30ea7c6099": ["87bb29d5-4e12-4190-b2ba-f5dd2e1a8c1b"],  # Langflow
    "a732b310-6ec1-44b5-bf4d-ac4b3618a62d": ["86a1c5b8-eb23-433e-8be9-e24453e0ad8c"],  # csv-parser
    "074babc9-86c9-48c5-ac96-4d350a36c9ad": ["3eec1de1-2f69-429e-9b95-b362dd41fff6"],  # rpcs3

    # Benchmark projects
    # "6c47d53e-7384-44d8-be9d-c186a7af480a": ["eef157cf-c8d4-4e7a-a2e5-79cf2f07be88"],  # Default project 1
    # "114ba2fa-8bae-4e19-8f46-3fbef23b4a98": ["05abf1c8-8ff7-457e-b7cb-25cd89130ff3", "9afc41b2-17f5-4799-90f1-1f1eb3625c42"],  # BitNet-function - claude3.7, gpt4o
    # "1cf9f904-d506-4a27-969f-ae6db943eb55": ["3f9da777-66e4-4b71-958f-abdb7456fadb"],  # Whisper GPU
    # "17789b06-49be-4dec-b2bc-2d741a350328": ["f6eccc1a-6b81-4b40-bd52-5d6464e53e58"],  # QuantLib 2.0
    # "f28e9994-4b44-446c-8973-7ab2037f1f55": ["a46ff34d-7037-4d79-81b1-4d7ab680cd4f"],  # QuantLib
    # "372d1ebb-f420-4580-8da3-17d21f3664f3": ["e8f76f2e-5329-4cba-a122-1992fba209c2"],  # BitmapPlusPlus
    # "cd204583-ca0a-4ee7-b837-e5115712902a": ["25f1f709-0b46-4654-b9dd-1ed187b7a349"],  # BitNet-file
    # "a3d17dee-6bed-40fb-95a1-f704ba5486bd": ["787e1843-6a34-4266-a3c2-de6a82bf6793"],  # AABitNet
    }

# Default batch configuration settings
DEFAULT_BATCH_CONFIG = {
    "max_concurrent": 5,
    "timeout_seconds": 300,
    "retry_failed": True
}

# Default use case configurations
DEFAULT_BATCH_RECOMMENDATIONS_CONFIG = {
    "selected_constructs": [],
    "selected_templates": [],
    "include_baseline": False,
    "meta_prompt_llm": DEFAULT_META_PROMPT_LLM,
    "code_optimization_llm": DEFAULT_CODE_OPTIMIZATION_LLM, 
    "selected_task": "runtime_performance",
    "evaluation_repetitions": 3,
    "generated_recommendations": None,
    "batch_results": []
}

DEFAULT_BATCH_SOLUTIONS_CONFIG = {
    "source_type": "recommendations",  # "recommendations", "prompt_versions", or "original_code"
    "selected_recommendations": [],
    "selected_existing_solutions": [],
    "selected_templates": [],  # For prompt_versions mode (multiple templates)
    "selected_template": None,  # For backward compatibility
    "solution_preview": [],  # For prompt_versions mode
    "optimization_id": None,
    "batch_results": []
}

DEFAULT_BATCH_EVALUATION_CONFIG = {
    "source_type": "solutions",  # "solutions" or "recommendations"
    "selected_solutions": [],
    "selected_recommendations": [],
    "evaluation_config": {
        "repetitions": 10
    },
    "batch_results": []
}

# Default project ID
DEFAULT_PROJECT_ID = "5ff581b8-e40a-4244-a842-2af4e4b8d438" 