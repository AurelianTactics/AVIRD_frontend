'''
Use Langraph to get the incident, fault, fault %, and short explanation

Usage
python fault\get_basic_fault.py --fault_version mvp_0.01

For DB insert to work, table must be created first in ...

working

to do
    set up database to hold this

backlog
    save current results if something fails

    can handle multiple report IDs
        have another unique id in addition to the report
        for key in dictionary
        for the data insert
        update the table as well

    graph
        ensuring the output is correct
        maybe not chat based?
        better architecture

    evaluation
    different models
    RAG
        fault information
    better prompts
        try different things
        explain what information is being given in the prompt from the data dictionary
        examples

    better data treating
        check for keys, NA values if keys don't exist

    async/parallel

    more args for better information on
        logging destinations
        logging ids

    use waymo data when available

    use other data when available

    use human evalauation thoughts as well

References
https://langchain-ai.github.io/langgraph/tutorials/reflection/reflection/
https://python.langchain.com/docs/integrations/chat/openai/
https://platform.openai.com/docs/models#gpt-4o
'''

import getpass
import os
import traceback
import json
from sqlalchemy import create_engine, text

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai import OpenAI, ChatOpenAI
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

from langgraph.checkpoint.memory import MemorySaver

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder

from typing import Annotated

import argparse
import copy 
import time
import numpy as np
import pandas as pd
from datetime import datetime


def run_get_fault(argparse_args):
    """
    Run the fault getting process
    """
    # setup
    _set_env("OPENAI_API_KEY")
    _set_env("LANGSMITH_API_KEY")

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "AVIRD_fault_v0.0.2"

    # load and treat data
    sgo_df = load_treat_data(argparse_args)

    # make graph
    fault_graph = make_fault_graph(argparse_args, sgo_df)

    # run through each incident
    fault_results_dict = {}
    config = {"recursion_limit": 10}

    for i in range(len(sgo_df)):
        print(f"Processing incident {i}")
        # run through graph
        incident_message = get_incident_message(sgo_df, i)
        
        events = fault_graph.stream(
            {
                "messages": [
                    HumanMessage(content=incident_message),
                ],
            },
            config,
        )

        for event in events:
            #print(event)
            fault_results_dict = add_to_results_dict(fault_results_dict, event, sgo_df, i)

    # save results
    save_treat_results(argparse_args, fault_results_dict)


def add_to_results_dict(fault_results_dict, event, sgo_df, i):
    '''
    to do:
    better treatment of if sgo id not found/ error
    '''
    
    sgo_id = sgo_df['Report ID'].iloc[i]

    try:
            
        if sgo_id not in fault_results_dict:
        
                fault_results_dict[sgo_id]= {}
                # Clean the message content before parsing
                event_content = clean_ai_message(event['arbitrator']['messages'][0].content)
                parsed_event_content = json.loads(event_content)

                if "is_av_at_fault" in parsed_event_content and isinstance(parsed_event_content["is_av_at_fault"], bool):
                    fault_results_dict[sgo_id]["is_av_at_fault"] = parsed_event_content["is_av_at_fault"]
                else:
                    fault_results_dict[sgo_id]["is_av_at_fault"] = None
                    print(f"Warning: is_av_at_fault not in results or not a bool {parsed_event_content}")

                if "av_fault_percentage" in parsed_event_content and isinstance(parsed_event_content["av_fault_percentage"], (float, int)) \
                    and 0 <= parsed_event_content["av_fault_percentage"] <= 1:
                    fault_results_dict[sgo_id]["av_fault_percentage"] = parsed_event_content["av_fault_percentage"] 
                else:
                    fault_results_dict[sgo_id]["av_fault_percentage"] = None
                    print(f"Warning: av_fault_percentage not in results or not a float between 0 and 1 {parsed_event_content}")

                if "short_explanation_of_decision" in parsed_event_content and isinstance(parsed_event_content["short_explanation_of_decision"], str):
                    fault_results_dict[sgo_id]["short_explanation_of_decision"] = parsed_event_content["short_explanation_of_decision"][:1000]
                else:
                    fault_results_dict[sgo_id]["short_explanation_of_decision"] = None
                    print(f"Warning: short_explanation_of_decision not in results or not a string {parsed_event_content}")
        else:
            print(f"Warning: sgo_id already exists in dictionary {sgo_id}")

    except Exception as e:
        print("Error: unable to get event messasge into needed fromat not found ", str(e))
        trc = traceback.format_exc()
        print(trc)

        fault_results_dict[sgo_id] = {
            "is_av_at_fault": None,
            "av_fault_percentage": None,
            "short_explanation_of_decision": "Error in parse"
        }

    return fault_results_dict


def clean_ai_message(content: str) -> str:
    """Clean the AI message content by removing markdown code blocks."""
    # Remove ```json and ``` markers
    content = content.replace('```json\n', '').replace('\n```', '')
    return content


def get_incident_message(sgo_df, i):

    check_cols = [
        #'Narrative',
        'Lighting', 'Roadway Type', 'Roadway Surface', 'Roadway Description', 'Posted Speed Limit (MPH)',
        'Crash With', 'Weather - Clear',
        'Weather - Snow',
        'Weather - Cloudy',
        'Weather - Fog/Smoke',
        'Weather - Rain',
        'Weather - Severe Wind',
        'Weather - Unknown',
        'Weather - Other',
        'Weather - Other Text',
        'Crash With',

        'CP Pre-Crash Movement',
        'CP Any Air Bags Deployed?',
        'CP Was Vehicle Towed?',
        'CP Contact Area - Rear Left',
        'CP Contact Area - Left',
        'CP Contact Area - Front Left',
        'CP Contact Area - Rear',
        'CP Contact Area - Top',
        'CP Contact Area - Front',
        'CP Contact Area - Rear Right',
        'CP Contact Area - Right',
        'CP Contact Area - Front Right',
        'CP Contact Area - Bottom',
        'CP Contact Area - Unknown',
        'SV Pre-Crash Movement',
        'SV Any Air Bags Deployed?',
        'SV Was Vehicle Towed?',
        #'SV Were All Passengers Belted?',
        'SV Precrash Speed (MPH)',
        'SV Pre-crash Speed - Unknown',
        'SV Contact Area - Rear Left',
        'SV Contact Area - Left',
        'SV Contact Area - Front Left',
        'SV Contact Area - Rear',
        'SV Contact Area - Top',
        'SV Contact Area - Front',
        'SV Contact Area - Rear Right',
        'SV Contact Area - Right',
        'SV Contact Area - Front Right',
        'SV Contact Area - Bottom',
        'SV Contact Area - Unknown',
        'City',
        'State',
        'Incident Time (24:00)',
        'Make',
        'Model',]

    subset_df = sgo_df[check_cols].iloc[[i]].copy()

    subset_df.rename(columns={
        'Make': 'Autnomous Vehicle Make',
        'Model': 'Autonomous Vehicle Model',
        'CP Pre-Crash Movement': 'Crash Partner Pre-Crash Movement',
        'CP Any Air Bags Deployed?': 'Crash Partner Any Air Bags Deployed?',
        'CP Was Vehicle Towed?': 'Crash Partner Was Vehicle Towed?',
        'CP Contact Area - Rear Left': 'Crash Partner Contact Area - Rear Left',
        'CP Contact Area - Left': 'Crash Partner Contact Area - Left',
        'CP Contact Area - Front Left': 'Crash Partner Contact Area - Front Left',
        'CP Contact Area - Rear': 'Crash Partner Contact Area - Rear',
        'CP Contact Area - Top': 'Crash Partner Contact Area - Top',
        'CP Contact Area - Front': 'Crash Partner Contact Area - Front',
        'CP Contact Area - Rear Right': 'Crash Partner Contact Area - Rear Right',
        'CP Contact Area - Right': 'Crash Partner Contact Area - Right',
        'CP Contact Area - Front Right': 'Crash Partner Contact Area - Front Right',
        'CP Contact Area - Bottom': 'Crash Partner Contact Area - Bottom',
        'CP Contact Area - Unknown': 'Crash Partner Contact Area - Unknown',
        'SV Pre-Crash Movement': 'Autonomous Vehicle Pre-Crash Movement',
        'SV Any Air Bags Deployed?': 'Autonomous Vehicle Any Air Bags Deployed?',
        'SV Was Vehicle Towed?': 'Autonomous Vehicle Was Vehicle Towed?',
        'SV Precrash Speed (MPH)': 'Autonomous Vehicle Precrash Speed (MPH)',
        'SV Pre-crash Speed - Unknown': 'Autonomous Vehicle Pre-crash Speed - Unknown',
        'SV Contact Area - Rear Left': 'Autonomous Vehicle Contact Area - Rear Left',
        'SV Contact Area - Left': 'Autonomous Vehicle Contact Area - Left',
        'SV Contact Area - Front Left': 'Autonomous Vehicle Contact Area - Front Left',
        'SV Contact Area - Rear': 'Autonomous Vehicle Contact Area - Rear',
        'SV Contact Area - Top': 'Autonomous Vehicle Contact Area - Top',
        'SV Contact Area - Front': 'Autonomous Vehicle Contact Area - Front',
        'SV Contact Area - Rear Right': 'Autonomous Vehicle Contact Area - Rear Right',
        'SV Contact Area - Right': 'Autonomous Vehicle Contact Area - Right',
        'SV Contact Area - Front Right': 'Autonomous Vehicle Contact Area - Front Right',
        'SV Contact Area - Bottom': 'Autonomous Vehicle Contact Area - Bottom',
        'SV Contact Area - Unknown': 'Autonomous Vehicle Contact Area - Unknown',
        }, inplace=True
    )
    #print(subset_df.iloc[0])

    narrative = sgo_df['Narrative'].iloc[i]


    incident_message = f'''
        Incident Information:
        - Narrative:
        {narrative}

        - Other Information:
    '''

    for k, v in subset_df.iloc[0].to_dict().items():
        if v is None or pd.isnull(v) or (isinstance(v, str) and v.strip() == '') or \
            (isinstance(v, float) and np.isnan(v)):
            continue

        if v == 'Y':
            treated_v = 'Yes'
        elif v == 'N':
            treated_v = 'No'
        else:
            treated_v = copy.deepcopy(v)
        incident_message += f'{k}: {treated_v}\n'
    #print(incident_message)
    return incident_message


class State(TypedDict):
    messages: Annotated[list, add_messages]


def make_fault_graph(argparse_args, sgo_df): 
    # Create the chain for arbitrating
    system_prompt = """
    You are a neutral insurance adjuster. Your job is to determine who is at fault for the incident.
    You will provide your answer in JSON format with the following keys: 
    "is_av_at_fault: bool"
    "av_fault_percentage: float in [0, 1]"
    "short_explanation_of_decision: string"
    """

    fault_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])

    llm = ChatOpenAI(
        model_name="gpt-4o-mini"
        #model_name="gpt-4o"
        #model_name="gpt-3.5-turbo"
        #o1, o1 mini
        # temperature, max tokens
        )

    arbitrator_chain = fault_prompt | llm

    def arbitrator_node(state: State) -> State:
        # Use invoke instead of ainvoke for synchronous operation
        return {"messages": [arbitrator_chain.invoke(state["messages"])]}

    workflow = StateGraph(State)

    # Define the nodes
    workflow.add_node("arbitrator", arbitrator_node)

    # Add the edges
    workflow.set_entry_point("arbitrator")
    workflow.set_finish_point("arbitrator")

    # Compile
    return workflow.compile()


def make_fault_prompt(sgo_df):
    fault_prompt = ChatPromptTemplate.from_messages([
        ("system",
        "You are a neutral insurance adjuster. Your job is to determine who is at fault for the incident."
        "You will provide your answer in JSON format with the following keys: "
            "is_av_at_fault: bool"
            "av_fault_percentage: float in [0, 1]"
            "short_explanation_of_decision: string"
            
        #"You will be given the following information: "
        ),

        MessagesPlaceholder(variable_name="incident_information"),
    ])
    # fault_prompt = PromptTemplate.from_template(
    #     "You are a neutral insurance adjuster. Your job is to determine who is at fault for the incident. "
    #     "You will p"
    #     "You will be given the following information: "
    # )

    return fault_prompt


def save_treat_results(argparse_args, fault_results_dict):
    # Generate fault version
    fault_version = get_fault_version(argparse_args)
    
    # Save to file
    save_to_file(argparse_args, fault_results_dict)
    
    # Save to database
    save_to_database(fault_results_dict, fault_version)


def get_fault_version(argparse_args):
    """Get the fault version from args or generate a default one."""
    if hasattr(argparse_args, 'fault_version') and argparse_args.fault_version:
        return argparse_args.fault_version



def save_to_file(argparse_args, fault_results_dict):
    # Create save directory if it doesn't exist
    os.makedirs(argparse_args.save_dir, exist_ok=True)
    
    # Save to JSON file
    save_path = os.path.join(argparse_args.save_dir, 'fault_analysis_results.json')
    with open(save_path, 'w') as f:
        json.dump(fault_results_dict, f, indent=4)
    print(f"Results saved to: {save_path}")


def save_to_database(fault_results_dict, fault_version):
    """Insert fault analysis results into the database."""
    try:
        # Get database connection from environment
        db_url = os.getenv('POSTGRES_DB_URL')
        engine = create_engine(db_url)

        # Prepare the data for insertion
        records = []
        for report_id, data in fault_results_dict.items():
            records.append({
                'report_id': report_id,
                'fault_version': fault_version,
                'is_av_at_fault': data.get('is_av_at_fault'),
                'av_fault_percentage': data.get('av_fault_percentage'),
                'short_explanation_of_decision': data.get('short_explanation_of_decision'),
            })

        # Insert records
        with engine.connect() as conn:
            for record in records:
                query = """
                    INSERT INTO fault_analysis (
                        report_id,
                        fault_version,
                        is_av_at_fault, 
                        av_fault_percentage, 
                        short_explanation_of_decision
                    ) 
                    VALUES (
                        :report_id,
                        :fault_version,
                        :is_av_at_fault, 
                        :av_fault_percentage, 
                        :short_explanation_of_decision
                    );
                """
                conn.execute(text(query), record)
                conn.commit()
        
        print(f"Successfully saved {len(records)} records to database with version {fault_version}")
        
    except Exception as e:
        print(f"Error saving to database: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise


def treat_results(graph_results_dict):
    treated_graph_results_dict = {}

    if isinstance(graph_results_dict, dict):
        treated_graph_results_dict = copy.deepcopy(graph_results_dict)
    
    return graph_results_dict, treated_graph_results_dict
    

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


def load_treat_data(argparse_args):
    sgo_df = pd.read_csv(argparse_args.sgo_data_path)

    sgo_df = treat_data(sgo_df)

    return sgo_df


def treat_data(input_df):
    treated_df = input_df[(input_df['Automation System Engaged?'] == 'ADS')].copy()

    return treated_df


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sgo_data_path", type=str, default='data\\nhtsa_sgo\\2024_11\\SGO-2021-01_Incident_Reports_ADS.csv')
    parser.add_argument("--save_dir", type=str, default='data/fault_analysis', 
                       help="Directory to save fault analysis results. Will be created if it doesn't exist.")
    parser.add_argument("--fault_version", type=str, required=True,
                       help="Version identifier for this fault analysis run. If not provided, will generate based on timestamp and model.")
    args = parser.parse_args()

    run_get_fault(args)