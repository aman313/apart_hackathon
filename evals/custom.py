import asyncio
from typing import Dict, List
from inspect_ai.model import ChatCompletionChoice, ChatMessageAssistant, ModelAPI, ModelOutput, ModelUsage
from martian_apart_hack_sdk import martian_client, utils 
import openai
from inspect_ai.model import ChatMessageUser, ChatMessageSystem, ChatMessage
from martian_apart_hack_sdk.models import router_constraints



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

    async def generate(self, input:list[ChatMessage]) -> ModelOutput:
        response = await self.openai_client.chat.completions.create(
            model=self.model_name,
            messages=input,
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
    
class MartialRouterModel(ModelAPI):

    def __init__(self, router_name: str="organizations/386aa70e-f2d9-44e6-a067-e04ff02cd125/routers/reasoning-router", max_tokens: int = 100):
        super().__init__(router_name)
        self.config = utils.load_config()
        self.client = martian_client.MartianClient(
            api_url = self.config.api_url,
            api_key = self.config.api_key,
        )
        #self.router = self.client.routers.get(router_name)
        self.router_name = router_name
        self.openai_client = openai.AsyncOpenAI(
                api_key=self.config.api_key,
                base_url=self.config.api_url + "/openai/v2"
        )
        self.max_tokens = max_tokens

        

    async def generate(self, input:list[ChatMessage]) -> ModelOutput:
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
            messages=input,
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


if __name__ == "__main__":
    #model = MartianBaseModel(model_name="gpt-4o-mini")
    model = MartialRouterModel(router_name="organizations/386aa70e-f2d9-44e6-a067-e04ff02cd125/routers/reasoning-router")
    input = [ChatMessageUser(role="user",content="Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?")]
    async def main():
        result = await model.generate(input=input)
        print(result)
    
    asyncio.run(main())
