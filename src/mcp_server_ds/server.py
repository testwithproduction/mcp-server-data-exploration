import logging
from typing import Optional, List
from pydantic import BaseModel

from fastmcp import FastMCP

## import common data analysis libraries
import pandas as pd
import numpy as np
import scipy
import sklearn
import statsmodels.api as sm
from io import StringIO
import sys
import argparse
import os
import random
import string
from datetime import datetime

# Ensure log directories exist
LOG_DIR = os.path.join(os.path.dirname(__file__), "../../logs")
SCRIPTS_DIR = os.path.join(LOG_DIR, "scripts")
os.makedirs(SCRIPTS_DIR, exist_ok=True)

# Set up logging to file and console
log_file = os.path.join(LOG_DIR, "server.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

logger.info("Starting mini data science exploration server")


### Data Models
class LoadCsvRequest(BaseModel):
    csv_path: str
    df_name: Optional[str] = None


class RunScriptRequest(BaseModel):
    script: str
    save_to_memory: Optional[List[str]] = None


### Python (Pandas, NumPy, SciPy) Script Runner
class ScriptRunner:
    def __init__(self):
        self.data = {}
        self.df_count = 0
        self.notes: list[str] = []

    def load_csv(self, csv_path: str, df_name: str = None) -> str:
        self.df_count += 1
        if not df_name:
            df_name = f"df_{self.df_count}"
        try:
            self.data[df_name] = pd.read_csv(csv_path)
            self.notes.append(f"Successfully loaded CSV into dataframe '{df_name}'")
            logger.info(f"Loaded CSV '{csv_path}' as dataframe '{df_name}'")
            return f"Successfully loaded CSV into dataframe '{df_name}'"
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise ValueError(f"Error loading CSV: {str(e)}")

    def run_script(
        self, script: str, save_to_memory: Optional[List[str]] = None
    ) -> str:
        """safely run a script, return the result if valid, otherwise return the error message"""
        # Save script to logs/scripts/ with a random name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand_suffix = "".join(
            random.choices(string.ascii_lowercase + string.digits, k=6)
        )
        script_filename = f"script_{timestamp}_{rand_suffix}.py"
        script_path = os.path.join(SCRIPTS_DIR, script_filename)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script)
        logger.info(f"Saved script to {script_path}")

        local_dict = {
            **{df_name: df for df_name, df in self.data.items()},
        }
        stdout_capture = StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = stdout_capture
            self.notes.append(f"Running script: \n{script}")
            logger.debug(f"Running script: {script_filename}")
            exec(
                script,
                {
                    "pd": pd,
                    "np": np,
                    "scipy": scipy,
                    "sklearn": sklearn,
                    "statsmodels": sm,
                },
                local_dict,
            )
            std_out_script = stdout_capture.getvalue()
        except Exception as e:
            logger.error(f"Error running script {script_filename}: {e}")
            raise ValueError(f"Error running script: {str(e)}")
        finally:
            sys.stdout = old_stdout

        if save_to_memory:
            for df_name in save_to_memory:
                self.notes.append(f"Saving dataframe '{df_name}' to memory")
                self.data[df_name] = local_dict.get(df_name)

        output = std_out_script if std_out_script else "No output"
        self.notes.append(f"Result: {output}")
        logger.info(f"Script {script_filename} executed. Output length: {len(output)}")
        return f"Script executed successfully. Output: {output}"

    def get_notes(self) -> str:
        """Get all notes from the script runner"""
        return "\n".join(self.notes)


### FastMCP Server Setup
app = FastMCP("local-mini-ds")
script_runner = ScriptRunner()


@app.tool(
    name="load_csv",
    description="""
Load CSV File Tool

Purpose:
Load a local CSV file into a DataFrame.

Usage Notes:
	•	If a df_name is not provided, the tool will automatically assign names sequentially as df_1, df_2, and so on.
""",
)
def load_csv(request: LoadCsvRequest) -> str:
    return script_runner.load_csv(request.csv_path, request.df_name)


@app.tool(
    name="run_script",
    description="""
Python Script Execution Tool

Purpose:
Execute Python scripts for specific data analytics tasks.

Allowed Actions
	1.	Print Results: Output will be displayed as the script's stdout.
	2.	[Optional] Save DataFrames: Store DataFrames in memory for future use by specifying a save_to_memory name.

Prohibited Actions
	1.	Overwriting Original DataFrames: Do not modify existing DataFrames to preserve their integrity for future tasks.
	2.	Creating Charts: Chart generation is not permitted.
""",
)
def run_script(request: RunScriptRequest) -> str:
    return script_runner.run_script(request.script, request.save_to_memory)


@app.resource(
    uri="data-exploration://notes",
    name="Data Exploration Notes",
    description="Notes generated by the data exploration server",
    mime_type="text/plain",
)
def get_notes() -> str:
    return script_runner.get_notes()


@app.prompt(
    name="explore-data",
    description="A prompt to explore a csv dataset as a data scientist",
)
def explore_data(csv_path: str, topic: str = "general data exploration") -> str:
    """
    Generate a comprehensive data exploration prompt for the given CSV file and topic.
    """
    prompt_template = """
You are a professional Data Scientist tasked with performing exploratory data analysis on a dataset. Your goal is to provide insightful analysis while ensuring stability and manageable result sizes.

First, load the CSV file from the following path:

<csv_path>
{csv_path}
</csv_path>

Your analysis should focus on the following topic:

<analysis_topic>
{topic}
</analysis_topic>

You have access to the following tools for your analysis:
1. load_csv: Use this to load the CSV file.
2. run_script: Use this to execute Python scripts on the MCP server.

Please follow these steps carefully:

1. Load the CSV file using the load_csv tool.

2. Explore the dataset. Provide a brief summary of its structure, including the number of rows, columns, and data types. Wrap your exploration process in <dataset_exploration> tags, including:
   - List of key statistics about the dataset
   - Potential challenges you foresee in analyzing this data

3. Wrap your thought process in <analysis_planning> tags:
   Analyze the dataset size and complexity:
   - How many rows and columns does it have?
   - Are there any potential computational challenges based on the data types or volume?
   - What kind of questions would be appropriate given the dataset's characteristics and the analysis topic?
   - How can we ensure that our questions won't result in excessively large outputs?

   Based on this analysis:
   - List 10 potential questions related to the analysis topic
   - Evaluate each question against the following criteria:
     * Directly related to the analysis topic
     * Can be answered with reasonable computational effort
     * Will produce manageable result sizes
     * Provides meaningful insights into the data
   - Select the top 5 questions that best meet all criteria

4. List the 5 questions you've selected, ensuring they meet the criteria outlined above.

5. For each question, follow these steps:
   a. Wrap your thought process in <analysis_planning> tags:
      - How can I structure the Python script to efficiently answer this question?
      - What data preprocessing steps are necessary?
      - How can I limit the output size to ensure stability?
      - What type of visualization would best represent the results?
      - Outline the main steps the script will follow
   
   b. Write a Python script to answer the question. Include comments explaining your approach and any measures taken to limit output size.
   
   c. Use the run_script tool to execute your Python script on the MCP server.
   
   d. Render the results returned by the run_script tool as a chart using plotly.js (prefer loading from cdnjs.cloudflare.com). Do not use react or recharts, and do not read the original CSV file directly. Provide the plotly.js code to generate the chart.

6. After completing the analysis for all 5 questions, provide a brief summary of your findings and any overarching insights gained from the data.

Remember to prioritize stability and manageability in your analysis. If at any point you encounter potential issues with large result sets, adjust your approach accordingly.

Please begin your analysis by loading the CSV file and providing an initial exploration of the dataset.
"""
    return prompt_template.format(csv_path=csv_path, topic=topic)


### Main function
async def main(transport: str = "stdio", debug: bool = False):
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        logger.debug("Debug logging enabled.")
    else:
        logging.basicConfig(level=logging.INFO)
        logger.info("Info logging enabled.")
    await app.run_async(transport=transport)


if __name__ == "__main__":
    import asyncio

    parser = argparse.ArgumentParser(description="MCP Data Science Server")
    parser.add_argument(
        "--transport",
        type=str,
        default="stdio",
        choices=["stdio", "sse", "websocket"],
        help="Transport protocol to use (default: stdio)",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Enable debug logging"
    )
    args = parser.parse_args()
    try:
        asyncio.run(main(transport=args.transport, debug=args.debug))
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error running server: {e}")
        print(
            "Note: This server is designed to be run through an MCP client, not directly."
        )
