import yaml
import time
from typing import Dict
def generate_config(output_path: str, params: Dict):
    timestamp_comment = f"# Generated on {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}\n"
    with open(output_path, 'w') as file:
        file.write(timestamp_comment)
        yaml.dump(params, file, default_flow_style=False, sort_keys=False)