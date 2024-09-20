import phoenix as px
from phoenix.trace.dsl import SpanQuery
from phoenix.trace import SpanEvaluations

import json
import pandas as pd

def query_vectara_spans():
    query = SpanQuery().select(
        "output.value",
        "parent_id",
        "name"
    )

    # Execute the query and return the results
    client = px.Client()
    return client.query_spans(query, project_name="vectara-agentic")

def extract_fcs_value(output):
    try:
        # Convert output to JSON
        output_json = json.loads(output)
        
        # Try to extract metadata.fcs
        if 'metadata' in output_json and 'fcs' in output_json['metadata']:
            return output_json['metadata']['fcs']
    except json.JSONDecodeError:
        print(f"Failed to parse JSON: {output}")
    except KeyError:
        print(f"'fcs' not found in: {output_json}")
    return None

def find_top_level_parent_id(row, all_spans):
    current_id = row['parent_id']
    while current_id is not None:
        parent_row = all_spans[all_spans.index == current_id]
        if parent_row.empty:
            break
        new_parent_id = parent_row['parent_id'].iloc[0]
        if new_parent_id == current_id:
            break
        if new_parent_id is None:
            return current_id
        current_id = new_parent_id
    return current_id

def add_top_level_parent_id(vectara_spans, all_spans):
    vectara_spans['top_level_parent_id'] = vectara_spans.apply(lambda row: find_top_level_parent_id(row, all_spans), axis=1)
    return vectara_spans


if __name__ == "__main__":
    all_spans = query_vectara_spans()
    vectara_spans = all_spans[all_spans['name'] == 'VectaraQueryEngine._query']
    vectara_spans = add_top_level_parent_id(vectara_spans, all_spans)
    vectara_spans['score'] = vectara_spans['output.value'].apply(lambda x: extract_fcs_value(x))
    
    vectara_spans.reset_index(inplace=True)
    top_level_spans = vectara_spans.copy()
    top_level_spans['context.span_id'] = top_level_spans['top_level_parent_id']
    vectara_spans = pd.concat([vectara_spans, top_level_spans], ignore_index=True)
    vectara_spans.set_index('context.span_id', inplace=True)
    
    px.Client().log_evaluations(
        SpanEvaluations(
            dataframe=vectara_spans,
            eval_name="Vectara FCS",
        ),
    )
