from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()


def main():
    print("Hello from agentfromscratch!")
    information = """A shepherd boy got bored while watching over the village sheep grazing on the hillside. To entertain himself, he sang out, “Wolf! Wolf! The wolf is chasing the sheep!”
When the villagers heard the cry, they ran up the hill to drive the beast away. But when they arrived, they saw no wolf. The boy just laughed at their angry faces.
“Don’t scream wolf when there is no wolf, boy!” the villagers warned. They angrily went back down the hill.
Later, the shepherd boy cried out once again, “Wolf! Wolf! The wolf is chasing the sheep!” To his amusement, the villagers came running up the hill to scare the wolf away.
As they saw there was no wolf, they said strictly, “Save your frightened cry for when there really is a wolf! Don’t cry ‘wolf’ when there is no wolf!” But the boy grinned at their words while they walked, grumbling down the hill once more.
Eventually, the boy saw a real wolf sneaking around his flock. Alarmed, he jumped to his feet and cried out as loud as he could, “Wolf! Wolf!” But the villagers thought he was fooling them again, and they didn’t come to help.
At sunset, the villagers went looking for the boy who hadn’t returned with their sheep. When they went up the hill, they found him weeping.
“There was a wolf here! The flock is gone! I cried out, ‘Wolf!’ but you didn’t come,” he wailed.
An older man went to comfort the boy. As he put his arm around him, he said, “Nobody believes a liar, even when he is telling the truth.”
"""

    summary_template = """
given the infomration {information}, summarize the story in one sentence"""

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    llm = ChatOpenAI(model="gpt-5.4-nano", temperature=0)

    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(type(response))
    print(response.content)


if __name__ == "__main__":
    main()
