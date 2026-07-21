from pydantic import BaseModel
from typing import List

class QA(BaseModel):
    signifiers: str
    question:str
    answer:str

class ParserConfig(BaseModel):
    fields: dict

class Parser():
    def __init__(self, config: ParserConfig, use_signifiers: bool = True) -> None:
        self.config = config
        self.field_configs = config.fields

        system_prompt_template = self.field_configs.get("system_prompt", {}).get("text", "")
        if not use_signifiers:
            if "{signifiers}" in system_prompt_template:
                raise ValueError(
                    "new_special_tokens is empty but the system prompt contains a "
                    "{signifiers} placeholder. Remove the placeholder to train with "
                    "a natural language system prompt."
                )
            print("Training with a natural language system prompt for all data.")

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
