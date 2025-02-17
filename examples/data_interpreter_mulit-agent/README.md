# Multi-Agent Pipeline for Complex Task Solving

This example will show:

- How to decompose a complex task into manageable subtasks using a Planner Agent.
- How to iteratively solve, verify, and replan subtasks using Solver, Verifier, and Replanning Agents.
- How to synthesize subtask results into a final answer using a Synthesizer Agent.

## Background

In complex problem-solving, it's often necessary to break down tasks into smaller, more manageable subtasks. A multi-agent system can handle this by assigning specialized agents to different aspects of the problem-solving process. This example demonstrates how to implement such a pipeline using specialized agents for planning, solving, verifying, replanning, and synthesizing tasks.

The pipeline consists of the following agents:

- **PlannerAgent**: Decomposes the overall task into subtasks.
- **SolverAgent** (using `ReActAgent`): Solves each subtask.
- **VerifierAgent**: Verifies the solutions to each subtask.
- **ReplanningAgent**: Replans or decomposes subtasks if verification fails.
- **SynthesizerAgent**: Synthesizes the results of all subtasks into a final answer.

By orchestrating these agents, the system can handle complex tasks that require iterative processing and dynamic adjustment based on intermediate results.

## Tested Models

These models are tested in this example. For other models, some modifications may be needed.

- **Anthropic Claude:** `claude-3-5-sonnet-20240620`, `claude-3-5-sonnet-20241022`, `claude-3-5-haiku-20241022` (accessed via the `litellm` package configuration).
- **OpenAI:** `GPT4-o`,  `GPT4-o-mini`.
- **DashScope:** `qwen-max`, `qwen-max-1201`.

## Prerequisites

To run this example, you need:

- **Agentscope** package installed:

  ```bash
  pip install agentscope
  ```

- **Environment Variables**: Set up the following environment variables with your API keys. This can be done in a `.env` file or directly in your environment.

  - `OPENAI_API_KEY` (if using OpenAI models)
  - `DASHSCOPE_API_KEY` (if using DashScope models)
  - `ANTHROPIC_API_KEY` (required for using Claude models via `litellm`)

- **Code Execution Environment**: Modify the code execution restrictions in Agentscope to allow the necessary operations for your tasks. Specifically, comment out the following `os` functions and `sys` modules in the `os_funcs_to_disable` and `sys_modules_to_disable` lists located in:

  ```plaintext
  src/agentscope/service/execute_code/exec_python.py
  ```

  **Comment out these `os` functions in `os_funcs_to_disable`:**

  - `putenv`
  - `remove`
  - `unlink`
  - `getcwd`
  - `chdir`

  **Comment out these modules in `sys_modules_to_disable`:**

  - `joblib`

  This step enables the executed code by the agents to perform required operations that are otherwise restricted by default. Ensure you understand the security implications of modifying these restrictions.

- Comment out the following in `src/agentscope/utils/common.py`:
    ```python
    @contextlib.contextmanager
    def create_tempdir() -> Generator:
        """
        A context manager that creates a temporary directory and changes the
        current working directory to it.
        The implementation of this contextmanager are borrowed from
        https://github.com/openai/human-eval/blob/master/human_eval/execution.py
        """
        with tempfile.TemporaryDirectory() as dirname:
            with _chdir(dirname):
                yield dirname
    ```

    and add
    ```python
    @contextlib.contextmanager
    def create_tempdir() -> Generator:
        """
        A context manager that uses the curreny directory.
        """
        yield
    ```
    to use the current directory for code execution.

- **Optional Packages** (if needed for specific tools or extended functionalities):

  - `litellm` for interacting with the Claude model.

    ```bash
    pip install litellm
    ```

  - Additional Python libraries as required by your code (e.g., `csv`, `dotenv`).

Ensure that you have the necessary API access and that your environment is correctly configured to use the specified models.

## Examples

This section demonstrates the pipeline's effectiveness on two different complex tasks. For your own task, replace `"Your task description here."` with your task in `input_task` in `di_multiagent.py` script.

### Task 1: Mathematical Problem Solving

**Problem**: Solve this math problem: The greatest common divisor of positive integers m and n is 6. The least common multiple of m and n is 126. What is the least possible value of m + n?

**Solution Output**:
```
Based on the results of the subtasks, we can synthesize the solution to the overall task as follows:

1. Functions for calculating GCD and LCM were defined and saved.
2. Possible pairs of m and n that satisfy the conditions (GCD = 6 and LCM = 126) were found to be (6, 126) and (18, 42).
3. The least possible value of m + n was calculated.

The answer to the overall task is:

The least possible value of m + n is 60, where m = 18 and n = 42.

This solution satisfies all the given conditions:
- The greatest common divisor of m and n is 6.
- The least common multiple of m and n is 126.
- The sum of m and n (18 + 42 = 60) is the least possible value among the valid pairs.
```

### Task 2: Titanic Survival Prediction

**Problem**: Predict passenger survival outcomes using the Titanic dataset. Perform data analysis, preprocessing, feature engineering, and modeling. Report accuracy on the evaluation data.

**Solution Output**:
```
The Titanic passenger survival prediction task has been successfully completed. Here's a summary of the process and results:

1. Data Analysis:
   - The training dataset contained 712 entries with 12 columns.
   - The target variable 'Survived' had a 37.5% overall survival rate.
   - Key factors influencing survival included Sex (females had a higher survival rate) and Passenger Class.

2. Data Preprocessing and Feature Engineering:
   - Missing values were handled through imputation or dropping columns.
   - New features were created, including 'Title' and 'FamilySize'.
   - Categorical variables were encoded, and unnecessary columns were dropped.
   - The final preprocessed dataset had 712 samples and 10 features.

3. Modeling:
   - Three models were trained and evaluated: Logistic Regression, Random Forest, and Gradient Boosting.
   - Gradient Boosting performed the best in cross-validation with an accuracy of 0.8160.

4. Final Evaluation:
   - The best model (Gradient Boosting) was used to make predictions on the evaluation dataset.
   - The final accuracy on the evaluation data (179 samples) was 0.8212 (82.12%).