from pydantic import BaseModel
from typing import List

class QA(BaseModel):
    signifiers: str
    question:str
    answer:str 

class ParserConfig(BaseModel):
    fields: dict

class Parser():
    def __init__(self, config: ParserConfig) -> None:
        self.config = config
        self.field_configs = config.fields


    def parse(self, qa_pairs: List[QA]):
        """Formats an example dictionary into a model input string using parser_config."""
        text_results ={"text":[]}

        for qa in qa_pairs:
            parts = []
            for key in self.field_configs:
                if key == "system_prompt" and qa.signifiers == "":
                    text_template = self.field_configs[key]["baseline"]
                else:
                    text_template = self.field_configs[key]["text"]
                try:
                    # Format using keys in the example
                    filled_text = text_template.format(**qa.__dict__)
                except KeyError as e:
                    raise ValueError(f"Missing key {e} in example: {qa}")
                
                parts.append(filled_text)
            text_results["text"].append("\n".join(parts))

        return text_results




