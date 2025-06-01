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
from martian_apart_hack_sdk.models.llm_models import DEEPSEEK_R1, DEEPSEEK_V3
import openai
from inspect_ai.model import ChatMessageUser, ChatMessageSystem, ChatMessage
from martian_apart_hack_sdk.models import router_constraints
from inspect_ai import eval
from inspect_ai.model._registry import modelapi

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

from typing import Any

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import match
from inspect_ai.solver import (
    Solver,
    generate,
    prompt_template,
)

USER_PROMPT_TEMPLATE = """
Solve the following math problem step by step.
The last line of your response should be of the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem.

{prompt}

Remember to put your answer on its own line at the end in the form "ANSWER: $ANSWER" (without quotes) where $ANSWER is the answer to the problem, and you do not need to use a \\boxed command.
""".strip()


@task
def aime2024() -> Task:
    """Inspect Task implementation for the AIME 2024 benchmark."""
    dataset = hf_dataset(
        path="Maxwell-Jia/AIME_2024",
        split="train",
        trust=True,
        sample_fields=record_to_sample,
    )

    return Task(
        dataset=dataset,
        solver=aime2024_solver(),
        scorer=[
            match(),
        ],
    )


def aime2024_solver() -> list[Solver]:
    """Build solver for AIME 2024 task."""
    solver = [prompt_template(USER_PROMPT_TEMPLATE), generate()]
    return solver


def record_to_sample(record: dict[str, Any]) -> Sample:
    sample = Sample(
        id=record["ID"],
        input=record["Problem"],
        target=str(record["Answer"]),
        metadata={
            "solution": record["Solution"],
        },
    )
    return sample

if __name__ == "__main__":
    task = aime2024()
    eval(task, model=Model(api=MartianBaseModel(model_name=DEEPSEEK_R1), config=GenerateConfig(max_tokens=1000)))
    # eval(task, model=Model(api=MartialRouterModel(model_name="organizations/386aa70e-f2d9-44e6-a067-e04ff02cd125/routers/reasoning-router"), config=GenerateConfig(max_tokens=1000)))