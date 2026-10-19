from dotenv import load_dotenv
import os
load_dotenv()

from langchain_openai import ChatOpenAI
from typing import Optional, Literal
from pydantic import Field, BaseModel

llm = ChatOpenAI(
    model = "openai/gpt-oss-20b",
    api_key = os.getenv("GROK_API_KEY"),
    base_url = "https://api.groq.com/openai/v1",
)

# schema
class Review(BaseModel):
    key_themes: list[str] = Field(
        description="Write down all the key themes discussed in the review in a list"
    )

    summary: str = Field(
        description="A brief summary of the review"
    )

    sentiment: Literal["pos", "neg"] = Field(
        description="Return sentiment of the review either negative, positive or neutral"
    )

    pros: Optional[list[str]] = Field(
        default=None,
        description="Write down all the pros inside a list"
    )

    cons: Optional[list[str]] = Field(
        default=None,
        description="Write down all the cons inside a list"
    )

    name: Optional[str] = Field(
        default=None,
        description="Write the name of the reviewer"
    )

structured_model = llm.with_structured_output(Review)

result = structured_model.invoke(
    "I bought these wireless noise-cancelling headphones recently and I really like them. The sound quality is excellent and the noise cancellation works really well, especially while travelling. They are comfortable even after wearing them for a few hours and the battery lasts a long time. The design also looks stylish and premium. The only problems are that the microphone isn't very good in noisy places and the headphones are a little expensive compared to other similar options. Overall, I think they are a great choice if you want good sound and noise cancellation and don't mind spending a bit more."
)

print(result)