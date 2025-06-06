from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import match
from inspect_ai.solver import generate, prompt_template, system_message
from datasets import load_dataset

import asyncio
from typing import Any, Dict, List
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, GenerateConfig, Model, ModelAPI, ModelOutput, ModelUsage
from inspect_ai.tool import Tool, ToolChoice
from martian_apart_hack_sdk import martian_client, utils 
import openai
from inspect_ai.model import ChatMessageUser, ChatMessageSystem, ChatMessage
from martian_apart_hack_sdk.models import router_constraints
from inspect_ai import eval
from inspect_ai.model._registry import modelapi
from martian_apart_hack_sdk.models.llm_models import DEEPSEEK_R1, DEEPSEEK_V3

@modelapi(name="martian_base_model")
class MartianBaseModel(ModelAPI):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.config = utils.load_config()
        self.client = martian_client.MartianClient(
            api_url = self.config.api_url,
            api_key = self.config.api_key,
        )
        self.openai_client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_url + "/openai/v2"
        )

    async def generate(self, input:list[ChatMessage], tools: list[Tool], tool_choice: ToolChoice, config: GenerateConfig) -> ModelOutput:
        response = await self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=[ChatMessageUser(role="user", content=x.content) for x in input],
        )

        model_output=  ModelOutput(
            model=self.model_name,
            choices=[ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content=response.choices[0].message.content,
                    role="assistant"
                )
            )],
            usage=ModelUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            ),
            stop_reason=response.choices[0].finish_reason,
            error=response.choices[0].finish_reason
        )
        return model_output

@modelapi(name="martial_router_model") 
class MartialRouterModel(ModelAPI):

    def __init__(self, model_name: str, max_tokens: int = 1000):
        super().__init__(model_name)
        self.config = utils.load_config()
        self.client = martian_client.MartianClient(
            api_url = self.config.api_url,
            api_key = self.config.api_key,
        )
        #self.router = self.client.routers.get(router_name)
        self.router_name = model_name
        self.openai_client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_url + "/openai/v2"
        )
        self.max_tokens = max_tokens

        

    async def generate(self, input:list[ChatMessage], tools: list[Tool], tool_choice: ToolChoice, config: GenerateConfig) -> ModelOutput:
        cost_constraint = router_constraints.RoutingConstraint(
            cost_constraint=router_constraints.CostConstraint(
                value=router_constraints.ConstraintValue(numeric_value=0.1)
             )

        )
        # response = await async_wrapper(self.client.routers.run,
        #     router=self.router,
        #     routing_constraint=cost_constraint,
        #     completion_request=input,
        # )
        response = await self.openai_client.chat.completions.create(
            model=self.router_name,
            messages=[ChatMessageUser(role="user", content=x.content) for x in input],
            extra_body=router_constraints.render_extra_body_router_constraint(cost_constraint),
            max_tokens=self.max_tokens
        )
        model_output=  ModelOutput(
            model=self.router_name,
            choices=[ChatCompletionChoice(
                message=ChatMessageAssistant(
                    content=response.choices[0].message.content,
                    role="assistant"
                )
            )],
            usage=ModelUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens
            ),
            stop_reason=response.choices[0].finish_reason,
            error=response.choices[0].finish_reason
        )
        return model_output

# setup for problem + instructions for providing answer
MATH_PROMPT_TEMPLATE = """
Solve the following math problem step by step. The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.

Reasoning:
""".strip()


@task
def gsm8k(fewshot: int = 10, fewshot_seed: int = 42) -> Task:
    """Inspect Task definition for the GSM8K benchmark

    Args:
        fewshot (int): The number of few shots to include
        fewshot_seed (int): The seed for generating few shots
    """
    # build solver dynamically (may or may not be doing fewshot)
    solver = [prompt_template(MATH_PROMPT_TEMPLATE), generate()]
    if fewshot:
        fewshots = hf_dataset(
            path="gsm8k",
            data_dir="main",
            split="train",
            sample_fields=record_to_sample,
            auto_id=True,
            shuffle=True,
            seed=fewshot_seed,
            limit=fewshot,
        )
        solver.insert(
            0,
            system_message(
                "\n\n".join([sample_to_fewshot(sample) for sample in fewshots])
            ),
        )

    # define task
    return Task(
        dataset=hf_dataset(
            path="gsm8k",
            data_dir="main",
            split="test",
            sample_fields=record_to_sample,
        ),
        solver=solver,
        scorer=match(numeric=True),
    )


def record_to_sample(record: dict[str, Any]) -> Sample:
    DELIM = "####"
    input = record["question"]
    answer = record["answer"].split(DELIM)
    target = answer.pop().strip()
    reasoning = DELIM.join(answer)
    return Sample(input=input, target=target, metadata={"reasoning": reasoning.strip()})


def sample_to_fewshot(sample: Sample) -> str:
    if sample.metadata:
        return (
            f"{sample.input}\n\nReasoning:\n"
            + f"{sample.metadata['reasoning']}\n\n"
            + f"ANSWER: {sample.target}"
        )
    else:
        return ""

# class MartianModel(Model):
#     def __init__(self, model_name: str):
#         super().__init__(api=MartianBaseModel(model_name=model_name), config=GenerateConfig(max_tokens=100))
    
#     async def generate(self, input:list[ChatMessage], tools: list[Tool], tool_choice: ToolChoice, config: GenerateConfig) -> ModelOutput:
#         return await self.api.generate(input)

if __name__ == "__main__":
    task = gsm8k(fewshot=0)
    eval(task, model=Model(api=MartianBaseModel(model_name=DEEPSEEK_R1), config=GenerateConfig(max_tokens=1000)))
    # eval(task, model=Model(api=MartialRouterModel(model_name="organizations/386aa70e-f2d9-44e6-a067-e04ff02cd125/routers/reasoning-router"), config=GenerateConfig(max_tokens=1000)))